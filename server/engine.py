"""引擎适配层：把 voicematch 的打分链路包装成 HTTP 可用的结果。

为什么需要这一层
----------------
CLI 输出的是"一行 JSON"，但小程序报告页还需要三样 CLI 不提供（或不适合直接给用户）的东西：

1. **四维分位**（``dimensions``）——引擎的 ``to_percentiles`` 输出的是
   *每个 descriptor* 的分位（pitch 3 个、timbre 17 个、resonance 5 个、
   style 5 个）。报告页要的是四个维度各一个 0-100 的分数，所以这里按
   ``DIMENSION_OF`` 把同维度 descriptor 的分位取平均；任何一个维度没有
   可用的有限分位时返回 ``None``，绝不填 0（0 会凭空造出"最不像"）。
2. **归因四键**（``attribution``）——``ScoredSinger.attribution`` 是五个键
   ``cos/pitch/timbre/resonance/style``，合计 1.0。``cos`` 是嵌入余弦，
   不是声学维度，报告页的四条归因条只需要后四个，所以这里去掉 ``cos``
   并把四维重新归一化；原始五键保留在 ``attribution_raw`` 便于排查。
3. **配对状态文案**（``pairing.notice``）——CLI 已经给出 ``pairing.status``
   （R29），但状态 token 不该泄漏到用户界面：这里补一句中文说明，页面直接展示。
   合规时是空串，页面据此决定是否显示提示。

其余字段与 ``voicematch.cli`` 的 compare 输出保持同名同义，前端在 Mock 与
真实后端之间切换时不需要改字段名。
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from voicematch.audio.io import load_audio
from voicematch.audio.preprocess import PREPROCESS_QUERY, preprocess
from voicematch.audio.quality import QualityReport, assess_quality
from voicematch.audio.windows import select_best_window
from voicematch.calibrate.model import CalibrationModel, fit_calibration
from voicematch.calibrate.pairs import build_pairs
from voicematch.calibrate.score import NonFinitePrototypeError, rank_singers
from voicematch.config import SAMPLE_RATE
from voicematch.embed.encoder import HashEncoder, HubertEncoder, VocalEncoder
from voicematch.eval.run import calibration_version, truncate_to_input_contract
from voicematch.features.extract import DIMENSION_OF, FeatureVector, extract_features
from voicematch.features.percentile import PercentileModel, fit_percentiles, to_percentiles
from voicematch.library import (
    EMPTY_LIBRARY,
    SINGLE_RECORDING,
    UNVERIFIABLE_LEGACY,
    PairingStatus,
    pairing_status,
)
from voicematch.library.schema import LibraryIndex
from voicematch.library.store import load_library

#: 四个声学维度，与 ``FEATURE_NAMES`` 去掉 ``cos`` 后的顺序一致。
DIMENSIONS: tuple[str, ...] = ("pitch", "timbre", "resonance", "style")

#: 报告页使用的中文维度名。
DIM_LABELS: dict[str, str] = {
    "pitch": "音高",
    "timbre": "音色",
    "resonance": "共鸣",
    "style": "唱腔",
}

#: 排名深度，与设计文档的 Top-3 一致。
TOP_K: int = 3

#: v1 只接受清唱（CLI 同款声明）。
ASSUMED: str = "a_cappella"

#: 进度回调签名：(阶段 token, 0-100 进度, 中文阶段说明)。
ProgressFn = Callable[[str, int, str], None]

#: 库记录的 model_id -> (命令行名, 编码器类)。
_ENCODER_BY_MODEL_ID: dict[str, tuple[str, type[VocalEncoder]]] = {
    "HUBERT_BASE": ("hubert", HubertEncoder),
    "HashEncoder": ("hash", HashEncoder),
}


class EngineError(RuntimeError):
    """可预期地失败（库损坏、性别池为空等），带稳定的 reason token。"""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class LibraryBundle:
    """一个已加载的歌手库：索引、校准模型、分位参照池、编码器。

    这些对象都是只读的，加载一次可服务所有请求；``compare`` 用一把锁串行化，
    因为 HuBERT 推理在 CPU 上并不保证线程安全，而本机服务不需要并发打分。
    HTTP 层的异步任务队列（``server/tasks.py``）也用单工作线程，两层锁因此很少
    真正争用。
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        manifest = self.root / "manifest.json"
        if not manifest.is_file():
            raise EngineError(
                "library_not_found",
                f"库目录缺少 manifest.json：{self.root}。"
                "先用 tools/make_demo_library.py 生成演示库，或按 "
                "python -m voicematch.cli build --catalog ... 建真实库。",
            )
        self.index = load_library(self.root)
        # R29：这个库能否训练出"跨录音"的校准配对。加载时算一次即可——索引
        # 是只读的，请求期间不会变化。结果同时出现在 /health 与每次打分里，
        # 让前端能说明"分数只是排序，不是校准相似度"，而不是只有一个
        # calibrated=false 让人猜原因。
        self.pairing = pairing_status(self.index)
        self.model = self._fit_calibration()
        self.percentiles: dict[str, PercentileModel] = {
            gender: fit_percentiles(self._pool(gender))
            for gender in ("female", "male")
        }
        self._configure_torch_cache()
        self.encoder_name, self.encoder = self._make_encoder()
        self._lock = threading.Lock()
        self._warm_encoder = False

    # ---- 构造辅助 -------------------------------------------------

    def _fit_calibration(self) -> CalibrationModel:
        """用库自身的 segment 配对拟合校准层（与 CLI 完全一致）。"""
        X, y, groups = build_pairs(self.index, self.index.embeddings)
        return fit_calibration(X, y, groups)

    def _pool(self, gender: str) -> list[FeatureVector]:
        """返回某一性别池里每个 segment 的特征向量，用于拟合分位。"""
        return [
            self.index.features[segment.segment_id]
            for segment in self.index.segments
            if self.index.singers[segment.singer_id].gender == gender
        ]

    def _configure_torch_cache(self) -> None:
        """如果库目录旁边有 ``hf_cache``，就用它当 torch 缓存。

        HuBERT 主干是 360 MB 的一次性下载。真实素材目录（如
        ``material/m4singer_3take/``）通常把模型缓存在自己的 ``hf_cache/`` 里，
        不设置 ``TORCH_HOME`` 的话服务启动会去 ``~/.cache/torch`` 重新找、找不到
        就再下一次。已经显式设置过 ``TORCH_HOME`` 的按用户设置来。
        """
        if os.environ.get("TORCH_HOME"):
            return
        candidate = self.root.parent / "hf_cache"
        if candidate.is_dir():
            os.environ["TORCH_HOME"] = str(candidate)

    def _make_encoder(self) -> tuple[str, VocalEncoder]:
        """按库记录的 model_id 选择编码器，避免两个向量空间被混用（R24）。"""
        pair = _ENCODER_BY_MODEL_ID.get(self.index.model_id)
        if pair is None:
            raise EngineError(
                "unknown_encoder",
                f"库记录的 model_id={self.index.model_id!r} 不在已知编码器表里，"
                "拒绝用未知向量空间打分。",
            )
        name, encoder_class = pair
        try:
            encoder = encoder_class()
        except ImportError as exc:
            raise EngineError(
                "missing_ml_extra",
                f"库由 {self.index.model_id} 构建，但当前环境缺少 torch/torchaudio："
                f"{exc}。请安装 pip install -e \".[ml]\"，或改用 HashEncoder 演示库。",
            ) from exc
        return name, encoder

    # ---- 对外接口 -------------------------------------------------

    def health(self) -> dict[str, Any]:
        """返回服务与库的版本信息，供小程序启动时自检。"""
        return {
            "status": "ok",
            "library": str(self.root),
            "model_id": self.index.model_id,
            "library_version": self.index.version,
            "calibration_version": calibration_version(self.model),
            "calibrated": bool(self.model.is_calibrated),
            "pairing": _pairing_payload(self.index),
            "encoder": self.encoder_name,
            "singers": {
                gender: sum(
                    1
                    for singer in self.index.singers.values()
                    if singer.gender == gender
                )
                for gender in ("female", "male")
            },
            "segments": len(self.index.segments),
        }

    def singers(self, gender: str | None = None) -> list[dict[str, Any]]:
        """列出库里可比的歌手（小程序端可用于"歌手池"展示）。"""
        return [
            {
                "singer_id": singer.singer_id,
                "name": singer.name,
                "gender": singer.gender,
                "style": singer.style_tag,
            }
            for singer in sorted(self.index.singers.values(), key=lambda s: s.singer_id)
            if gender is None or singer.gender == gender
        ]

    def compare(
        self,
        audio_path: Path,
        gender: str,
        on_progress: ProgressFn | None = None,
    ) -> dict[str, Any]:
        """对一段录音打一次分，返回与 CLI 同构的 JSON 文档。

        ``status`` 三态与 CLI 一致：``ok`` / ``rejected`` / ``no_usable_window``。
        可预期的失败抛 :class:`EngineError`，由 HTTP 层转成 error 信封。

        ``on_progress`` 在关键节点被调用（阶段 token、进度、中文说明）。它跑在
        调用方线程里，本模块不做任何调度；回调抛异常会被吞掉，绝不影响打分。
        """
        if gender not in ("female", "male"):
            raise EngineError("bad_gender", f"gender 只能是 female/male，收到 {gender!r}")

        def emit(stage: str, progress: int, message: str) -> None:
            if on_progress is None:
                return
            try:
                on_progress(stage, progress, message)
            except Exception:  # noqa: BLE001 - 进度上报失败不能连累打分
                pass

        with self._lock:
            emit("decode", 8, "读取并解析音频")
            decoded, truncated = truncate_to_input_contract(load_audio(Path(audio_path)))
            emit("quality", 20, "评估录音质量")
            report = assess_quality(decoded, SAMPLE_RATE)

            payload: dict[str, Any] = {
                "model_id": self.index.model_id,
                "library_version": self.index.version,
                "calibration_version": calibration_version(self.model),
                "calibrated": bool(self.model.is_calibrated),
                "pairing": _pairing_payload(self.index),
                "encoder": self.encoder_name,
                "query": {"gender": gender, "path": Path(audio_path).name},
                "assumed": ASSUMED,
                "quality": _quality_payload(report),
                "status": "rejected" if report.rejected else "ok",
                "reason": report.reasons[0] if report.rejected else None,
                "truncated": truncated,
                "results": [],
            }
            if report.rejected:
                emit("done", 100, "录音质量不达标")
                return payload

            emit("preprocess", 34, "预处理音频")
            processed = preprocess(decoded, PREPROCESS_QUERY)
            emit("window", 42, "选取最佳片段")
            window = select_best_window(processed, SAMPLE_RATE)
            if window is None:
                payload["status"] = "no_usable_window"
                payload["reason"] = "no_usable_window"
                emit("done", 100, "没有可用的发声片段")
                return payload

            start, end = window
            sample_start = int(round(start * SAMPLE_RATE))
            sample_end = int(round(end * SAMPLE_RATE))
            segment = processed[sample_start:sample_end]
            emit("features", 56, "提取声学特征")
            features = extract_features(segment, SAMPLE_RATE)
            # HuBERT 是整条链路最慢的一步，进度权重给得最大
            emit("embedding", 76, "生成声线嵌入")
            embedding = self.encoder.encode(segment, SAMPLE_RATE)

            try:
                emit("scoring", 92, "校准排序")
                scored = rank_singers(
                    features, embedding, self.index, self.model, gender, top_k=TOP_K
                )
            except NonFinitePrototypeError as exc:
                raise EngineError("non_finite_prototype", str(exc)) from exc
            except LookupError as exc:
                raise EngineError("no_singers_in_pool", str(exc)) from exc

            payload["results"] = [
                self._render(entry, features, gender) for entry in scored
            ]
            emit("done", 100, "分析完成")
            return payload

    def _render(
        self, entry: Any, features: FeatureVector, gender: str
    ) -> dict[str, Any]:
        """把一条 ``ScoredSinger`` 渲染成报告页需要的结构。"""
        singer = self.index.singers[entry.singer_id]
        attribution = _normalised_attribution(entry.attribution)
        top_dim = max(DIMENSIONS, key=lambda key: attribution.get(key, 0.0))
        return {
            "singer_id": entry.singer_id,
            "name": entry.name,
            "style": singer.style_tag,
            "score": round(float(entry.score), 4),
            "calibrated": bool(entry.calibrated),
            "attribution": attribution,
            "attribution_raw": {
                key: round(float(value), 4)
                for key, value in entry.attribution.items()
            },
            "attributionText": (
                f"主要因为{DIM_LABELS[top_dim]}接近"
                if max(attribution.values(), default=0.0) > 0
                else "四个维度的差异都不明显"
            ),
            "dimensions": _dimension_percentiles(
                features, self.percentiles[gender]
            ),
        }


def _pairing_payload(index: LibraryIndex) -> dict[str, Any]:
    """R29 配对状态 + 一句可直接展示的中文说明。

    ``status`` / ``single_material_singers`` 与 ``voicematch.cli compare`` 完全
    同名同义，前端不必区分数据来源；``notice`` 是本适配层补的展示文案（空串
    表示无需提示），因为让每个页面各自维护一份状态映射必然会走样。
    """
    status = pairing_status(index)
    return {
        "status": status.status,
        "single_material_singers": list(status.single_material_singers),
        "notice": _pairing_notice(status),
    }


def _pairing_notice(status: PairingStatus) -> str:
    """把配对状态翻译成用户能懂的一句话；合规状态返回空串。"""
    if status.status == SINGLE_RECORDING:
        count = len(status.single_material_singers)
        return (
            f"该歌手库有 {count} 位歌手只有一份录音，校准没能从他们身上学到"
            "跨录音的声线；分数仍可用于排序，但不代表校准后的相似度。"
        )
    if status.status == UNVERIFIABLE_LEGACY:
        return (
            "该歌手库由旧版本构建，无法验证校准配对是否跨录音；"
            "分数仅作余弦排序参考，不代表校准后的相似度。"
        )
    if status.status == EMPTY_LIBRARY:
        return "歌手库为空，暂时无法给出排序。"
    return ""


def _quality_payload(report: QualityReport) -> dict[str, Any]:
    """质量报告转 JSON；非有限值（静音的 -inf SNR）按 None 输出。"""
    return {
        "speech_sec": _finite(report.speech_sec),
        "snr_db": _finite(report.snr_db),
        "clipped_ratio": _finite(report.clipped_ratio),
        "voiced_ratio": _finite(report.voiced_ratio),
        "rejected": bool(report.rejected),
        "reasons": list(report.reasons),
    }


def _finite(value: float) -> float | None:
    """非有限浮点转 ``None``：JSON 没有 NaN/Infinity 的合法表示。"""
    number = float(value)
    return number if math.isfinite(number) else None


def _normalised_attribution(raw: dict[str, float]) -> dict[str, float]:
    """去掉 ``cos`` 并把四维归因重新归一化为合计 1.0。

    归因为全零时返回四维全 0（模型认为四个维度都没有推动判断），不编造均分。
    """
    shares = {key: max(0.0, float(raw.get(key, 0.0))) for key in DIMENSIONS}
    total = math.fsum(shares.values())
    if total <= 0.0:
        return {key: 0.0 for key in DIMENSIONS}
    return {key: round(value / total, 3) for key, value in shares.items()}


def _dimension_percentiles(
    features: FeatureVector, model: PercentileModel
) -> dict[str, float | None]:
    """把 descriptor 级分位聚合成四个维度各一个 0-100 分位。

    一个维度内只平均**有限**的分位值；``nan``（该 descriptor 测不出或池里
    没有参照）被跳过而不是当 0。整个维度没有可用值时返回 ``None``。
    """
    per_descriptor = to_percentiles(features, model)
    aggregated: dict[str, float | None] = {}
    for dimension in DIMENSIONS:
        values = [
            per_descriptor[name]
            for name, owner in DIMENSION_OF.items()
            if owner == dimension
            and name in per_descriptor
            and math.isfinite(per_descriptor[name])
        ]
        aggregated[dimension] = (
            round(math.fsum(values) / len(values), 1) if values else None
        )
    return aggregated
