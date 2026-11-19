"""本机 HTTP 服务：把 voicematch 引擎暴露成小程序可调用的接口。

启动
----
    python server/app.py --port 8000

端点
----
- ``GET  /health``          服务与库的版本自检（含队列状态）
- ``GET  /singers?gender=``  库内歌手列表
- ``POST /compare``         上传录音，默认 **受理后立即返回 202 + task_id**
- ``GET  /task/<task_id>``  轮询任务进度与结果
- ``DELETE /task/<task_id>`` 取消任务（排队中立刻丢弃，运行中丢弃结果）

异步流程
--------
推理在 CPU 上要几秒，同步接口会把连接占住、用户也看不到进度。所以 ``/compare``
默认只做受理：录音落到临时文件 → 建任务 → 返回 ``task_id``。客户端拿 ``task_id``
轮询 ``/task/<id>``，``state=done`` 时 ``result`` 就是与 CLI 同构的文档。

调试时想一把拿到结果，加 ``?mode=sync`` 回到同步语义（curl 用着更顺手）。
``/task/<id>?wait=20`` 是长轮询：最多阻塞 20 秒，任务一结束立刻返回。

``/compare`` 接受三种请求体
---------------------------
1. ``multipart/form-data``，文件字段名 ``file``，表单字段 ``gender``
   （``wx.uploadFile`` 默认就是这种）。
2. 原始音频字节（``Content-Type: audio/*`` 或 ``application/octet-stream``），
   性别走查询串 ``?gender=female``，便于 ``curl --data-binary`` 调试。
3. ``application/json`` + ``audio_base64`` 字段。

为什么手写 multipart 解析
-------------------------
Python 3.13 移除了 ``cgi`` 模块，不能再用 ``cgi.FieldStorage``。这里只解析
本服务需要的最小子集（一个文件字段 + 若干文本字段），不做通用 MIME 解析。

安全与边界
----------
- 默认只监听 ``127.0.0.1``，不对外网开放。
- 请求体上限 12 MB（60 秒 16 kHz mp3 远小于此）。
- 上传内容写入临时文件，任务处理结束即删除（成功、失败都删）；不落库、不打日志。
- 任务 id 是随机串，不可枚举；结果在内存里保留 15 分钟后自动清理。
- 任何失败都返回一行 JSON（``status="error"`` + 稳定 ``reason``），
  不返回 traceback。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# 允许从仓库根目录直接运行（src/ 是 src-layout，未安装时也能导入）
_ROOT = Path(__file__).resolve().parent.parent
for _candidate in (str(_ROOT / "src"), str(_ROOT)):
    if Path(_candidate).is_dir() and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from server.engine import EngineError, LibraryBundle  # noqa: E402
from server.tasks import QueueFullError, Task, TaskQueue  # noqa: E402

#: 请求体上限：12 MB。
MAX_BODY_BYTES: int = 12 * 1024 * 1024

#: 长轮询的最大阻塞秒数（客户端传更大的值也只等这么久）。
MAX_POLL_WAIT_SEC: float = 25.0

#: 允许的音频扩展名（soundfile 能解码的常见格式）。
_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"}


class VoiceMatchHandler(BaseHTTPRequestHandler):
    """一个极简的 JSON over HTTP 处理器。"""

    #: 由 ``main`` 注入的库实例。
    bundle: LibraryBundle
    #: 由 ``main`` 注入的任务队列。
    tasks: TaskQueue
    server_version = "VoiceMatchServer/1.1"

    # ---- 路由 -----------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/health":
                payload = self.bundle.health()
                payload["queue"] = self.tasks.snapshot()
                self._json(200, payload)
            elif parsed.path == "/singers":
                gender = (query.get("gender") or [None])[0]
                if gender not in (None, "female", "male"):
                    raise EngineError("bad_gender", "gender 只能是 female/male")
                self._json(200, {"status": "ok", "singers": self.bundle.singers(gender)})
            elif parsed.path.startswith("/task/"):
                self._handle_task(parsed.path[len("/task/"):], query)
            else:
                self._json(404, {"status": "error", "reason": "not_found", "detail": parsed.path})
        except EngineError as exc:
            self._json(400, _error_payload(exc.reason, exc.detail))
        except Exception as exc:  # noqa: BLE001 - HTTP 边界统一收口
            self._json(500, _error_payload(type(exc).__name__, str(exc)))

    def _handle_task(self, task_id: str, query: dict[str, list[str]]) -> None:
        """``GET /task/<id>``：查一次任务快照，可选长轮询。"""
        task_id = task_id.strip("/")
        if not task_id:
            self._json(400, _error_payload("bad_request", "缺少 task_id"))
            return

        wait_raw = (query.get("wait") or ["0"])[0]
        try:
            wait = max(0.0, float(wait_raw))
        except ValueError:
            self._json(400, _error_payload("bad_request", f"wait 不是数字：{wait_raw!r}"))
            return
        wait = min(wait, MAX_POLL_WAIT_SEC)

        task = self.tasks.wait(task_id, wait) if wait > 0 else self.tasks.get(task_id)
        if task is None:
            # 过期或拼错：让客户端知道该重录，而不是无限轮询
            self._json(404, _error_payload("task_not_found", task_id))
            return
        self._json(200, task.public())

    def do_DELETE(self) -> None:  # noqa: N802
        """``DELETE /task/<id>``：取消任务。

        排队中的任务会被立刻丢弃（临时录音同时删除）；已经在推理中的只能标记，
        等这一轮算完再丢结果——HuBERT 无法安全中断。两种情况都返回终态文档，
        客户端据此停止轮询即可。
        """
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/task/"):
            self._json(404, {"status": "error", "reason": "not_found", "detail": parsed.path})
            return
        task_id = parsed.path[len("/task/"):].strip("/")
        if not task_id:
            self._json(400, _error_payload("bad_request", "缺少 task_id"))
            return
        try:
            task = self.tasks.cancel(task_id)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, _error_payload(type(exc).__name__, str(exc)))
            return
        if task is None:
            self._json(404, _error_payload("task_not_found", task_id))
            return
        self._json(200, task.public())

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/compare":
            self._json(404, {"status": "error", "reason": "not_found", "detail": parsed.path})
            return

        query = parse_qs(parsed.query)
        mode = (query.get("mode") or ["async"])[0].strip().lower()
        if mode not in ("sync", "async"):
            self._json(400, _error_payload("bad_request", f"mode 只能是 sync/async，收到 {mode!r}"))
            return

        temp_path: Path | None = None
        try:
            body = self._read_body()
            gender, audio_bytes = self._extract(body, parsed.query)
            if not audio_bytes:
                raise EngineError("missing_audio", "请求里没有音频数据")

            temp_path = _write_temp_audio(audio_bytes)

            if mode == "sync":
                payload = self.bundle.compare(temp_path, gender)
                self._json(200, payload)
                return

            task = self.tasks.submit(gender, temp_path)
            # 文件所有权已交给任务队列，处理结束时由它删除
            temp_path = None
            document = task.public()
            document["status"] = "accepted"
            document["poll_after_ms"] = 500
            self._json(202, document)
        except QueueFullError as exc:
            # 503 + Retry-After：客户端知道是"稍后再来"，不是参数错
            self.send_response(503)
            body = json.dumps(
                _error_payload("queue_full", str(exc)), ensure_ascii=False
            ).encode("utf-8")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Retry-After", "5")
            self._cors()
            self.end_headers()
            self.wfile.write(body)
        except EngineError as exc:
            self._json(400, _error_payload(exc.reason, exc.detail))
        except ValueError as exc:
            self._json(400, _error_payload("bad_request", str(exc)))
        except Exception as exc:  # noqa: BLE001
            # 服务端内部错误：日志留完整栈，响应只给类型名和消息
            traceback.print_exc()
            self._json(500, _error_payload(type(exc).__name__, str(exc)))
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    # ---- 请求解析 -------------------------------------------------

    def _read_body(self) -> bytes:
        """读取请求体，超过上限直接拒绝。"""
        length_header = self.headers.get("Content-Length")
        if not length_header:
            raise EngineError("empty_body", "缺少 Content-Length")
        try:
            length = int(length_header)
        except ValueError as exc:
            raise EngineError("bad_content_length", length_header) from exc
        if length <= 0:
            raise EngineError("empty_body", "请求体为空")
        if length > MAX_BODY_BYTES:
            raise EngineError(
                "body_too_large",
                f"请求体 {length} 字节超过上限 {MAX_BODY_BYTES} 字节",
            )
        return self.rfile.read(length)

    def _extract(self, body: bytes, query: str) -> tuple[str, bytes]:
        """从请求里取出 (gender, 音频字节)，支持 multipart / raw / base64 JSON。"""
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()

        if content_type == "multipart/form-data":
            fields, files = _parse_multipart(body, self.headers.get("Content-Type", ""))
            gender = (fields.get("gender") or "female").strip()
            audio = files.get("file") or files.get("audio")
            return _validated_gender(gender), audio or b""

        if content_type == "application/json":
            try:
                document = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EngineError("bad_json", "请求体不是合法 JSON") from exc
            gender = _validated_gender(str(document.get("gender", "female")))
            encoded = document.get("audio_base64") or ""
            try:
                return gender, base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as exc:
                raise EngineError("bad_base64", "audio_base64 解码失败") from exc

        # raw 音频：性别从 query string 取
        gender = (parse_qs(query).get("gender") or ["female"])[0]
        return _validated_gender(gender), body

    # ---- 响应 -----------------------------------------------------

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt: str, *args) -> None:
        """缩短默认日志格式，只保留方法、路径与状态。"""
        sys.stderr.write(
            f"[voicematch] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n"
        )


def _validated_gender(value: str) -> str:
    gender = (value or "").strip().lower()
    if gender not in ("female", "male"):
        raise EngineError("bad_gender", f"gender 只能是 female/male，收到 {value!r}")
    return gender


def _default_library() -> Path:
    """选默认库：优先真实 HuBERT 库，其次演示库。

    真实库优先是刻意的：用户要的是真实音色结果，演示库的 HashEncoder 只是
    没有真实素材时的链路替身。两者都由 manifest.json 是否存在来判定。
    """
    candidates = [
        _ROOT / "material" / "m4singer_3take" / "lib",
        _ROOT / "server" / "demo_library",
    ]
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    return candidates[-1]


def _write_temp_audio(data: bytes) -> Path:
    """把上传字节写进临时文件，返回路径（调用方负责删除）。

    扩展名统一用 ``.wav`` 之外的猜测意义不大：soundfile/librosa 按内容嗅探，
    这里给一个中性后缀，避免把 mp3 内容误标成 wav 造成解码器混淆。
    """
    handle, name = tempfile.mkstemp(prefix="voicematch-", suffix=".audio")
    with os.fdopen(handle, "wb") as stream:
        stream.write(data)
    return Path(name)


def _error_payload(reason: str, detail: str) -> dict:
    return {
        "status": "error",
        "reason": reason,
        "detail": detail,
        "assumed": "a_cappella",
        "truncated": False,
        "results": [],
    }


# ---- multipart 解析（Python 3.13 无 cgi 模块） ----------------------


def _parse_multipart(
    body: bytes, content_type: str
) -> tuple[dict[str, str], dict[str, bytes]]:
    """解析 multipart/form-data，返回 (文本字段, 文件字段)。

    只支持本服务需要的最小子集：每个 part 一段 ``Content-Disposition``，
    文本字段按 UTF-8 解码，文件字段保留原始字节。分隔符前的 CRLF 只去掉
    一次，避免误伤音频末尾恰好是 CRLF 的字节。
    """
    boundary = _boundary(content_type)
    delimiter = b"--" + boundary
    fields: dict[str, str] = {}
    files: dict[str, bytes] = {}

    for chunk in body.split(delimiter):
        if not chunk or chunk in (b"--", b"--\r\n", b"\r\n"):
            continue
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        if chunk.endswith(b"\r\n"):
            chunk = chunk[:-2]
        if not chunk or chunk == b"--":
            continue
        head, separator, data = chunk.partition(b"\r\n\r\n")
        if not separator:
            continue

        disposition = _header_value(head, b"content-disposition")
        name = _disposition_param(disposition, "name")
        if not name:
            continue

        filename = _disposition_param(disposition, "filename")
        if filename is not None:
            files[name] = data
        else:
            fields[name] = data.decode("utf-8", errors="replace")

    return fields, files


def _boundary(content_type: str) -> bytes:
    match = re.search(r'boundary="?([^";]+)"?', content_type, flags=re.IGNORECASE)
    if not match:
        raise EngineError("bad_multipart", "multipart 请求缺少 boundary")
    return match.group(1).encode("ascii", errors="ignore")


def _header_value(head: bytes, name: bytes) -> str:
    for line in head.split(b"\r\n"):
        key, separator, value = line.partition(b":")
        if separator and key.strip().lower() == name:
            return value.strip().decode("utf-8", errors="replace")
    return ""


def _disposition_param(disposition: str, key: str) -> str | None:
    match = re.search(
        rf'{key}="([^"]*)"', disposition, flags=re.IGNORECASE
    ) or re.search(rf"{key}=([^;]+)", disposition, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _lan_addresses() -> list[str]:
    """列出本机的局域网 IPv4，用于真机联调时填 API_BASE。

    先按主机名解析（多数家用/办公网络够用），再用一个不做实际发包的 UDP
    socket 探测默认出口网卡，覆盖多网卡/虚拟网卡顺序不确定的情况。
    """
    import socket

    found: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            found.add(sock.getsockname()[0])
    except OSError:
        pass
    return sorted(ip for ip in found if not ip.startswith("127."))


def _make_runner(bundle: LibraryBundle):
    """把引擎包成任务队列能消费的 runner。

    进度回调直接挂到 ``task.report``，引擎每到一个阶段就把进度写进任务，
    轮询端读到的就是真实阶段而不是估算值。
    """

    def run(task: Task) -> dict:
        if task.audio_path is None:
            raise EngineError("task_expired", "任务的临时音频已被清理")
        return bundle.compare(task.audio_path, task.gender, on_progress=task.report)

    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="voicematch-server",
        description="把 voicematch 引擎包装成本机 HTTP 服务，供小程序联调。",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=None,
        help=(
            "已构建的歌手库目录。默认优先用 material/m4singer_3take/lib"
            "（真实 HuBERT 库），不存在时回退 server/demo_library（演示库）"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="打分工作线程数。CPU 推理不并行，默认 1；调大只对 I/O 阶段有意义",
    )
    args = parser.parse_args(argv)
    library = args.library or _default_library()

    print(f"[voicematch] 加载库：{library}", file=sys.stderr)
    try:
        bundle = LibraryBundle(library)
    except EngineError as exc:
        print(f"[voicematch] 启动失败（{exc.reason}）：{exc.detail}", file=sys.stderr)
        return 1

    health = bundle.health()
    print(
        "[voicematch] 就绪：encoder={encoder} model={model_id} "
        "library={library_version} singers={singers}".format(
            encoder=health["encoder"],
            model_id=health["model_id"],
            library_version=health["library_version"],
            singers=health["singers"],
        ),
        file=sys.stderr,
    )

    tasks = TaskQueue(_make_runner(bundle), workers=args.workers)
    VoiceMatchHandler.bundle = bundle
    VoiceMatchHandler.tasks = tasks
    server = ThreadingHTTPServer((args.host, args.port), VoiceMatchHandler)
    print(
        f"[voicematch] 监听 http://{args.host}:{args.port} "
        f"（异步任务，{args.workers} 个工作线程）",
        file=sys.stderr,
    )

    # 真机联调最容易卡在"填哪个 IP"。这里直接把可用地址和 config 改法打出来。
    if args.host not in ("127.0.0.1", "localhost"):
        addresses = _lan_addresses()
        if addresses:
            print("[voicematch] 手机可访问（需同一 Wi-Fi）：", file=sys.stderr)
            for address in addresses:
                print(f"[voicematch]   http://{address}:{args.port}", file=sys.stderr)
            print(
                "[voicematch] 小程序改 miniprogram/config.js："
                f"API_BASE: 'http://{addresses[0]}:{args.port}'",
                file=sys.stderr,
            )
        else:
            print(
                "[voicematch] 未探测到局域网 IPv4，请用 ipconfig 手动确认后填 API_BASE",
                file=sys.stderr,
            )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[voicematch] 已停止", file=sys.stderr)
    finally:
        tasks.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
