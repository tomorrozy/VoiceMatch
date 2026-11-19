"""导出候选试听样本 + 素材可诊断性（设计 6.2）。

盲听标注要问"这段清唱最像谁"，标注者必须能试听候选歌手。本模块从歌手库
素材里为每位歌手切一段试听 wav，并生成 samples.json（singer_id -> 试听文件
+ 姓名 + 性别）供离线标注界面加载。

为什么必须用 library 路径，而不是 query 路径
----------------------------------------------
试听样本要代表歌手**在库中的声音**，所以必须用 ``build_library`` 同一条路径：
``preprocess(PREPROCESS_LIBRARY)`` + ``select_windows``，且**不套 60 s 输入上限**。
一个真实教训：M4Singer 的 soprano_2 素材长 64 s、质量门通过、库里也切出了
(54,64) 的窗；早期版本误用 query 路径（先截到 60 s 再选窗）把唯一可用窗截掉，
于是错误地把它报成"无可发声窗"。样本必须与库同源同口径，这个 bug 正是后果。

诊断与替换建议
--------------
:func:`diagnose_material` 报告一位歌手每个素材的时长、质量门结论与可用窗数，
:func:`diagnose_catalog` 据此给出结构化替换建议。工具链把"素材不行"从一句
告警变成可执行的判断依据。

素材权利
--------
样本直接来自 catalog 声明的歌手素材（研究/内部用途），不对外分发原唱片段；
samples.json 只含歌手 id、展示名、性别与本地文件名。MANIFEST 的 rights 声明
随基准集走，标注界面不得把样本另行发布。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from soundfile import LibsndfileError

from voicematch.audio.io import load_audio
from voicematch.audio.preprocess import PREPROCESS_LIBRARY, preprocess
from voicematch.audio.quality import assess_quality
from voicematch.audio.windows import select_windows
from voicematch.config import SAMPLE_RATE
from voicematch.eval.run import _sample_index
from voicematch.library.catalog import load_catalog

#: 每位歌手试听样本时长上限（秒）。从库中实际切出的窗里截取，不超过窗长。
SAMPLE_SEC: float = 10.0

#: 试听样本文件名前缀，避免与查询录音混淆。
SAMPLE_PREFIX: str = "sample_"


@dataclass(frozen=True)
class SampleInfo:
    """一位歌手的试听样本元数据。"""

    singer_id: str
    name: str
    gender: str
    file: str

    def to_record(self) -> dict:
        return {
            "singer_id": self.singer_id,
            "name": self.name,
            "gender": self.gender,
            "file": self.file,
        }


@dataclass(frozen=True)
class MaterialDiagnosis:
    """一位歌手单个素材的可诊断性报告。"""

    singer_id: str
    material: str
    duration_sec: float
    rejected: bool
    reject_reasons: tuple[str, ...]
    n_windows: int

    @property
    def usable(self) -> bool:
        return self.n_windows > 0

    def to_record(self) -> dict:
        return {
            "singer_id": self.singer_id,
            "material": self.material,
            "duration_sec": round(self.duration_sec, 2),
            "rejected": self.rejected,
            "reject_reasons": list(self.reject_reasons),
            "n_windows": self.n_windows,
            "usable": self.usable,
        }


@dataclass(frozen=True)
class ReplacementSuggestion:
    """一位歌手的素材替换建议。"""

    singer_id: str
    name: str
    gender: str
    usable_material: str | None
    status: str  # "ok" | "needs_replacement"
    diagnosis: list[MaterialDiagnosis] = field(default_factory=list)
    advice: str = ""

    def to_record(self) -> dict:
        return {
            "singer_id": self.singer_id,
            "name": self.name,
            "gender": self.gender,
            "usable_material": self.usable_material,
            "status": self.status,
            "diagnosis": [item.to_record() for item in self.diagnosis],
            "advice": self.advice,
        }


def export_samples(
    catalog_path: Path,
    out_dir: Path,
    *,
    secs: float = SAMPLE_SEC,
) -> tuple[list[SampleInfo], list[str]]:
    """为 catalog 里每位歌手导出一段试听样本，返回 (样本清单, 告警)。

    每位歌手取其素材里第一个可切出窗的录音，用**库路径**切窗（与 build_library
    一致），截到 ``secs`` 秒写盘。全部素材都不可用时跳过该歌手并记告警。
    同一 singer_id 只导出一段（去重），文件名确定性，可重复运行覆盖。

    Raises ``CatalogError``：catalog 本身非法（由 :func:`load_catalog` 抛出）。
    """
    if secs <= 0:
        raise ValueError(f"secs must be positive, got {secs}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    materials = _materials_by_singer(catalog_path)
    samples: list[SampleInfo] = []
    warnings: list[str] = []
    for singer_id in sorted(materials):
        name, gender, paths = materials[singer_id]
        window_audio = _first_usable_window(paths, secs)
        if window_audio is None:
            warnings.append(
                f"singer {singer_id!r}: no usable window in any material; sample skipped"
            )
            continue
        filename = f"{SAMPLE_PREFIX}{singer_id}.wav"
        sf.write(out_dir / filename, window_audio, SAMPLE_RATE)
        samples.append(
            SampleInfo(singer_id=singer_id, name=name, gender=gender, file=filename)
        )
    return samples, warnings


def diagnose_catalog(catalog_path: Path) -> list[ReplacementSuggestion]:
    """逐歌手、逐素材诊断可用性，并给出替换建议。

    这是"素材不行"的可执行化：每条建议含每个素材的时长、质量门结论、可用窗
    数，以及 status/advice。``status=ok`` 表示至少一份素材可用；
    ``needs_replacement`` 表示素材存在但都切不出窗。
    """
    materials = _materials_by_singer(catalog_path)
    result: list[ReplacementSuggestion] = []
    for singer_id in sorted(materials):
        name, gender, paths = materials[singer_id]
        diagnoses = [diagnose_material(singer_id, path) for path in paths]
        usable = [item for item in diagnoses if item.usable]
        chosen = usable[0].material if usable else None
        if usable:
            status = "ok"
            advice = f"usable: {chosen}; {len(usable)} of {len(diagnoses)} materials yield windows"
        else:
            status = "needs_replacement"
            advice = (
                "no material yields a usable window. Replace with a longer (>=10 s), "
                "clean a-cappella take, or re-export from a different song/session. "
                "Check duration and reject_reasons per material below."
            )
        result.append(
            ReplacementSuggestion(
                singer_id=singer_id,
                name=name,
                gender=gender,
                usable_material=chosen,
                status=status,
                diagnosis=diagnoses,
                advice=advice,
            )
        )
    return result


def diagnose_material(singer_id: str, path: Path) -> MaterialDiagnosis:
    """诊断单个素材：时长、质量门结论、按库路径能切出多少窗。"""
    path = Path(path)
    try:
        audio = load_audio(path)
    except (OSError, ValueError, LibsndfileError) as exc:
        return MaterialDiagnosis(
            singer_id=singer_id, material=str(path), duration_sec=0.0,
            rejected=True, reject_reasons=(f"unreadable: {exc}",), n_windows=0,
        )
    report = assess_quality(audio, SAMPLE_RATE)
    processed = preprocess(audio, PREPROCESS_LIBRARY)
    windows = select_windows(processed, SAMPLE_RATE, k=5)
    return MaterialDiagnosis(
        singer_id=singer_id,
        material=str(path),
        duration_sec=len(audio) / SAMPLE_RATE,
        rejected=report.rejected,
        reject_reasons=tuple(report.reasons),
        n_windows=len(windows),
    )


def write_manifest(path: Path, samples: list[SampleInfo]) -> None:
    """写 samples.json（singer_id -> 试听文件），供标注界面加载。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([sample.to_record() for sample in samples], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def write_diagnosis(path: Path, suggestions: list[ReplacementSuggestion]) -> None:
    """写素材诊断与替换建议 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([item.to_record() for item in suggestions], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _materials_by_singer(catalog_path: Path) -> dict[str, tuple[str, str, list[Path]]]:
    """把 catalog 的 (Singer, path) 对聚合成 singer_id -> (name, gender, paths)。"""
    materials: dict[str, tuple[str, str, list[Path]]] = {}
    for singer, material_path in load_catalog(catalog_path):
        if singer.singer_id not in materials:
            materials[singer.singer_id] = (singer.name, singer.gender, [])
        materials[singer.singer_id][2].append(material_path)
    return materials


def _first_usable_window(paths: list[Path], secs: float) -> np.ndarray | None:
    """用 library 路径返回第一个素材的可用窗（截到 secs），都不可用返回 None。"""
    limit = int(secs * SAMPLE_RATE)
    for path in paths:
        try:
            audio = load_audio(path)
        except (OSError, ValueError, LibsndfileError):
            continue
        processed = preprocess(audio, PREPROCESS_LIBRARY)
        windows = select_windows(processed, SAMPLE_RATE, k=5)
        if not windows:
            continue
        start, end = windows[0]
        segment = processed[_sample_index(start) : _sample_index(end)]
        return segment[:limit]
    return None
