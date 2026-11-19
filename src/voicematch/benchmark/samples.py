"""导出候选试听样本：让标注者能听到 8 位歌手各一段清唱（设计 6.2）。

盲听标注要问"这段清唱最像谁"，标注者必须能试听候选歌手。本模块从歌手库
素材里为每位歌手切一段**与产品同一条选窗规则**得到的干净发声窗，写成独立
wav，并生成 samples.json（singer_id -> 试听文件 + 姓名 + 性别）供离线标注界面
加载。

为什么要和建库共用选窗规则
--------------------------
试听样本必须代表歌手在库中的声音。若人工精挑漂亮乐句，标注者记住的是"最好听
的一面"，与用户随手一段查询不对称，标注结果会系统性偏移。用 ``select_best_window``
这条自动规则，样本与库内片段同源同口径。

素材与权利
----------
样本直接来自 catalog 声明的歌手素材（研究/内部用途），不对外分发原唱片段；
samples.json 只含歌手 id、展示名、性别与本地文件名。MANIFEST 的 rights 声明
随基准集走，标注界面不得把样本另行发布。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from voicematch.audio.io import load_audio
from voicematch.audio.preprocess import PREPROCESS_QUERY, preprocess
from voicematch.audio.quality import assess_quality
from voicematch.audio.windows import select_best_window
from voicematch.config import SAMPLE_RATE
from voicematch.eval.run import _sample_index, truncate_to_input_contract
from voicematch.library.catalog import load_catalog

#: 每位歌手的试听样本时长（秒）。10 s 与产品选窗一致，足够判断音色。
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


def export_samples(
    catalog_path: Path,
    out_dir: Path,
    *,
    secs: float = SAMPLE_SEC,
) -> tuple[list[SampleInfo], list[str]]:
    """为 catalog 里每位歌手导出一段试听样本，返回 (样本清单, 告警)。

    ``catalog_path`` 是人工维护的歌手 catalog（相对素材路径按 catalog 所在目录
    解析）。每位歌手取其素材里**第一段**能通过质量门且有有效窗的录音，用共享
    选窗规则切出最长 ``secs`` 秒写盘；全部素材都不可用时跳过该歌手并记告警。

    同一 singer_id 只导出一段（去重），文件名 ``sample_<singer_id>.wav``，
    确定性，可重复运行覆盖同名文件。

    Raises ``CatalogError``：catalog 本身非法（由 :func:`load_catalog` 抛出）。
    """
    if secs <= 0:
        raise ValueError(f"secs must be positive, got {secs}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 每位歌手的素材路径，按 singer_id 聚合（同一歌手可有多 take）。
    materials: dict[str, tuple[str, str, list[Path]]] = {}
    for singer, material_path in load_catalog(catalog_path):
        if singer.singer_id not in materials:
            materials[singer.singer_id] = (singer.name, singer.gender, [])
        materials[singer.singer_id][2].append(material_path)

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


def write_manifest(path: Path, samples: list[SampleInfo]) -> None:
    """写 samples.json（singer_id -> 试听文件），供标注界面加载。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([sample.to_record() for sample in samples], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _first_usable_window(paths: list[Path], secs: float) -> np.ndarray | None:
    """返回第一位可用素材的干净发声窗（截到 secs），都不可用返回 None。"""
    limit = int(secs * SAMPLE_RATE)
    for path in paths:
        try:
            audio, _ = truncate_to_input_contract(load_audio(path))
        except (OSError, ValueError):
            continue
        if assess_quality(audio, SAMPLE_RATE).rejected:
            continue
        processed = preprocess(audio, PREPROCESS_QUERY)
        window = select_best_window(processed, SAMPLE_RATE)
        if window is None:
            continue
        start, end = window
        segment = processed[_sample_index(start) : _sample_index(end)]
        return segment[:limit]
    return None
