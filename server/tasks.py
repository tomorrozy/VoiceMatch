"""异步打分任务队列：把 CPU 推理从 HTTP 请求线程里挪出来。

为什么需要
----------
HuBERT 推理在本机 CPU 上要几秒，长录音的前处理还要更久。同步接口有三个真实
问题：

1. HTTP 连接被长时间占住。移动网络下这种长连接容易中途断开，而服务端其实已经
   算完了 —— 结果白算，用户还得重录重传。
2. 用户看不到任何进度，只能盯着转圈。
3. 并发请求全部堵在引擎锁上，后到的请求也不知道自己排在第几。

所以这里把推理交给固定数量的工作线程，请求只负责"受理"并返回 task_id，客户端
拿它轮询进度和结果。

设计取舍
--------
- **单工作线程（默认）**：HuBERT 在 CPU 上本来就不并行，多开线程只会互抢核心，
  还要在编码器外面加锁。``workers`` 可调，但收益有限。
- **内存态、无持久化**：本机联调服务，进程重启丢任务是可接受的；不落库也就不会
  留下录音痕迹。
- **临时文件由队列接管**：处理结束（成功或失败）立刻删除，不等请求结束。
- **取消分两种**：还在排队 → 立刻判终态并删文件；已经开始 → 只能标记后丢弃
  结果（HuBERT 不能安全中断），客户端侧体感仍是"立即取消"。
- **惰性清理**：过期任务在 submit/get/health 时顺手清掉，不起常驻定时器。
- **任务 id 用不可猜的随机串**：本地服务也没有鉴权，别让 id 可枚举。
"""

from __future__ import annotations

import queue
import secrets
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: 任务结果在内存里的保留时长。客户端断线后仍可凭 task_id 取回结果。
TASK_TTL_SEC: int = 15 * 60

#: 内存内任务总数上限，超出时优先淘汰最老的已完成任务。
MAX_TASKS: int = 200

#: 排队上限。每个排队任务都占一个临时音频文件，不能无限堆。
MAX_PENDING: int = 20

#: 终态集合：只有这些状态的任务会被过期/超量清理，也是轮询可以停止的信号。
TERMINAL_STATES: tuple[str, ...] = ("done", "failed", "cancelled")


class QueueFullError(RuntimeError):
    """排队已满，服务当前忙不过来。"""


@dataclass
class Task:
    """一个打分任务的内存记录。

    字段读写跨越两个线程（工作线程写、HTTP 线程读），但都是单条属性赋值，
    CPython 下是原子的，因此不再额外加锁；不追求强一致，只要求读到一个自洽的
    快照（进度落后一帧无所谓）。
    """

    task_id: str
    gender: str
    audio_path: Path | None = field(default=None, repr=False)
    state: str = "queued"          # queued / running / done / failed / cancelled
    stage: str = "queued"
    message: str = "排队中"
    progress: int = 0
    queue_position: int = 0
    cancel_requested: bool = False
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    # ---- 工作线程侧：上报进度 ------------------------------------

    def report(self, stage: str, progress: int, message: str) -> None:
        """引擎在关键节点回调，把进度写进任务（线程安全：单条赋值）。"""
        self.stage = stage
        self.progress = max(0, min(100, int(progress)))
        self.message = message
        self.updated_at = time.monotonic()

    # ---- HTTP 线程侧：序列化 --------------------------------------

    @property
    def elapsed_ms(self) -> int:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return int((end - self.created_at) * 1000)

    def public(self) -> dict[str, Any]:
        """返回给客户端的任务文档（不含音频路径等本地信息）。"""
        document: dict[str, Any] = {
            "task_id": self.task_id,
            "state": self.state,
            "stage": self.stage,
            "message": self.message,
            "progress": self.progress,
            "elapsed_ms": self.elapsed_ms,
            "result": self.result,
            "error": self.error,
        }
        if self.state == "queued":
            document["queue_position"] = self.queue_position
        return document


class TaskQueue:
    """固定工作线程 + 内存任务表。

    ``runner(task)`` 由调用方提供，负责真正干活并抛 ``EngineError`` 一类的
    可预期错误（本模块按鸭子类型取 ``reason`` / ``detail``，不反向依赖引擎）。
    """

    def __init__(
        self,
        runner: Callable[[Task], dict[str, Any]],
        *,
        workers: int = 1,
        ttl: float = TASK_TTL_SEC,
        max_tasks: int = MAX_TASKS,
        max_pending: int = MAX_PENDING,
    ) -> None:
        self._runner = runner
        self._ttl = ttl
        self._max_tasks = max_tasks
        self._max_pending = max_pending
        self._tasks: dict[str, Task] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._closed = False
        self._threads = [
            threading.Thread(
                target=self._work, name=f"voicematch-worker-{index}", daemon=True
            )
            for index in range(max(1, workers))
        ]
        for thread in self._threads:
            thread.start()

    # ---- 对外接口 -------------------------------------------------

    def submit(self, gender: str, audio_path: Path) -> Task:
        """受理一个任务，返回任务记录（此时还在排队）。"""
        with self._lock:
            self._reap_locked()
            pending = sum(
                1 for task in self._tasks.values() if task.state in ("queued", "running")
            )
            if pending >= self._max_pending:
                raise QueueFullError(
                    f"排队任务已达上限 {self._max_pending} 个，请稍后重试"
                )
            task = Task(
                task_id=secrets.token_urlsafe(9),
                gender=gender,
                audio_path=Path(audio_path),
                queue_position=pending,
            )
            self._tasks[task.task_id] = task
            self._order.append(task.task_id)
            self._trim_locked()
        self._queue.put(task.task_id)
        return task

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            self._reap_locked()
            return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> Task | None:
        """取消任务。两种情形分开处理，因为 CPU 推理无法中途打断：

        - **还在排队**：立刻判终态并删掉临时录音，工作线程弹出这个 id 时看到
          状态不是 ``queued`` 就直接跳过，一秒都不会浪费。
        - **已经在跑**：只能标记 ``cancel_requested``，等这一轮推理结束再丢弃
          结果（HuBERT 不能安全中断）。客户端本来就已停止轮询，所以体感是
          "立刻取消"，只是后台多算了几秒。
        """
        with self._lock:
            self._reap_locked()
            task = self._tasks.get(task_id)
            if task is None or task.state in TERMINAL_STATES:
                return task

            if task.state == "queued":
                _unlink(task.audio_path)
                task.audio_path = None
                task.cancel_requested = True
                _mark_cancelled(task)
            else:
                task.cancel_requested = True
                task.message = "正在取消…"
            return task

    def wait(self, task_id: str, timeout: float) -> Task | None:
        """等到任务结束或超时，返回当时的任务快照。

        给 ``/task/<id>?wait=`` 用：客户端一次长轮询能省掉多次短请求。
        超时返回的仍然是 ``queued``/``running`` 状态，不是错误。
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            task = self.get(task_id)
            if task is None or task.state in TERMINAL_STATES:
                return task
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return task
            time.sleep(min(0.05, remaining))

    def snapshot(self) -> dict[str, Any]:
        """队列现状，挂到 /health 里便于排查"为什么慢"。"""
        with self._lock:
            self._reap_locked()
            queued = sum(1 for task in self._tasks.values() if task.state == "queued")
            running = sum(1 for task in self._tasks.values() if task.state == "running")
            return {
                "workers": len(self._threads),
                "queued": queued,
                "running": running,
                "tracked": len(self._tasks),
                "max_pending": self._max_pending,
            }

    def shutdown(self, timeout: float = 2.0) -> None:
        """让工作线程退出；正在跑的任务允许跑完（daemon 线程不阻塞进程）。"""
        if self._closed:
            return
        self._closed = True
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=timeout)

    # ---- 工作线程 -------------------------------------------------

    def _work(self) -> None:
        while True:
            task_id = self._queue.get()
            try:
                if task_id is None:
                    return
                task = self._tasks.get(task_id)
                if task is None:  # 已被淘汰（正常流程不会发生）
                    continue
                if task.state != "queued":
                    # 排队期间被取消：文件已删，直接跳过，不浪费一次推理
                    continue
                self._run(task)
            finally:
                self._queue.task_done()

    def _run(self, task: Task) -> None:
        task.state = "running"
        task.queue_position = 0
        task.report("start", 2, "开始分析")
        try:
            result = self._runner(task)
        except Exception as exc:  # noqa: BLE001 - 任务边界统一收口
            self._cleanup(task)
            self._fail(task, exc)
        else:
            # 先删临时文件再翻状态：客户端看到 state=done 时，录音已经不在磁盘上，
            # 隐私文案里的"处理完即删除"因此是严格成立的，不靠时间差
            self._cleanup(task)
            if task.cancel_requested:
                _mark_cancelled(task)
                return
            task.result = result
            task.report("done", 100, "分析完成")
            task.state = "done"
            task.finished_at = time.monotonic()
            task.updated_at = task.finished_at

    @staticmethod
    def _cleanup(task: Task) -> None:
        _unlink(task.audio_path)
        task.audio_path = None

    def _fail(self, task: Task, exc: Exception) -> None:
        # 取消导致的失败按"已取消"收尾，别让用户看到一条假的错误
        if task.cancel_requested:
            _mark_cancelled(task)
            return

        reason = getattr(exc, "reason", None) or type(exc).__name__
        detail = getattr(exc, "detail", None) or str(exc)
        task.error = {"reason": reason, "detail": detail}
        task.state = "failed"
        task.stage = "failed"
        task.message = "分析失败"
        task.finished_at = time.monotonic()
        task.updated_at = task.finished_at

        # 可预期错误（音质不达标等）不打栈；意外错误留完整栈便于排查
        if getattr(exc, "reason", None) is None:
            traceback.print_exc()
        sys.stderr.write(f"[voicematch] 任务 {task.task_id} 失败：{reason} - {detail}\n")

    # ---- 清理（调用方持有 self._lock） -----------------------------

    def _reap_locked(self) -> None:
        if not self._tasks:
            return
        now = time.monotonic()
        expired = [
            task_id
            for task_id, task in self._tasks.items()
            if task.state in TERMINAL_STATES
            and task.finished_at is not None
            and now - task.finished_at > self._ttl
        ]
        for task_id in expired:
            self._pop_locked(task_id)

    def _trim_locked(self) -> None:
        """总量超限时按创建顺序淘汰最老的已完成任务。"""
        if len(self._tasks) <= self._max_tasks:
            return
        for task_id in list(self._order):
            if len(self._tasks) <= self._max_tasks:
                break
            task = self._tasks.get(task_id)
            if task is not None and task.state in TERMINAL_STATES:
                self._pop_locked(task_id)

    def _pop_locked(self, task_id: str) -> None:
        task = self._tasks.pop(task_id, None)
        if task_id in self._order:
            self._order.remove(task_id)
        if task is not None:
            _unlink(task.audio_path)
            task.audio_path = None


def _mark_cancelled(task: Task) -> None:
    """把任务判为终态 cancelled（调用方保证已清理临时文件）。"""
    task.state = "cancelled"
    task.stage = "cancelled"
    task.message = "已取消"
    task.progress = 0
    task.queue_position = 0
    task.finished_at = time.monotonic()
    task.updated_at = task.finished_at


def _unlink(path: Path | None) -> None:
    """删除临时录音；失败不影响任务结果（系统会自己清临时目录）。"""
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
