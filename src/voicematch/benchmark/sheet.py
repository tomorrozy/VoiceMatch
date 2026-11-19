"""标注清单生成：把真人录音变成可盲听的候选集（设计 6.2）。

一条真人 query 要问出"你最像谁"，标注清单必须满足两个同时成立的性质：

- **系统答案在场**：清单里含系统的 Top-3，否则 Top-K 命中率无从计算；
- **系统答案不可读**：8 位歌手打乱展示，且清单文件绝不写出哪 3 位是系统
  预测。标注者只能靠耳朵判断，不会被系统排序锚定。

这两点由 :func:`voicematch.eval.candidates.candidate_set` 同时保证：它把
Top-3 与 5 位随机同性别歌手混合后用种子打乱。本模块只负责——对每条录音
跑**与产品完全相同**的查询管线得到嵌入与特征，调用 candidate_set，然后把
结果写成两份文件：

- ``sheet.json``：发给标注者的清单，只有 query_id / path / gender /
  candidates / 采集元数据（subject_id、device_id），**没有 system_top**；
- ``provenance.json``：留在内部，记录 system_top 与种子，用于事后审计
  "系统当时预测了谁"而不影响标注。两份文件物理分离是本模块的核心约束。

质量门与受理率的口径
--------------------
清单为**所有**录音生成候选，包括质量门会拒绝、或没有有效窗口的录音——
这些录音虽不可打分，但必须留在冻结的 eval set 里，才能被 ``run_eval``
计入 ``acceptance_rate`` 的分母。若在这里把它们剔除，受理率会虚高成 1.0。
不可打分时系统的 Top-3 无法计算，候选取而代之以随机的 k 位同性别歌手——
它们依旧是合法的盲听清单，只是这次 query 没有"系统答案在场"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voicematch.audio.io import load_audio
from voicematch.audio.preprocess import PREPROCESS_QUERY, preprocess
from voicematch.audio.quality import assess_quality
from voicematch.audio.windows import select_best_window
from voicematch.calibrate.model import CalibrationModel
from voicematch.config import SAMPLE_RATE
from voicematch.embed.encoder import VocalEncoder
from voicematch.eval.candidates import SYSTEM_TOP3, candidate_set
from voicematch.eval.dataset import GENDERS
from voicematch.eval.run import _sample_index, truncate_to_input_contract
from voicematch.features.extract import extract_features
from voicematch.library.schema import LibraryIndex

#: 标注清单每条 query 展示的候选歌手数（设计 6.2：Top-3 + 5 随机 = 8）。
SHEET_CANDIDATES: int = 8


@dataclass(frozen=True)
class SheetEntry:
    """发给一位标注者的一条盲听任务。

    ``candidates`` 是打乱后的歌手 id 顺序，标注者据此展示（配歌手名映射由
    CLI 从库补齐）。``subject_id``/``device_id`` 是 G3 采集元数据，随 query
    走，供冻结后的 eval set 计算跨设备稳定性。
    """

    query_id: str
    path: str
    gender: str
    candidates: tuple[str, ...]
    subject_id: str | None = None
    device_id: str | None = None

    def to_record(self) -> dict:
        """序列化为 sheet.json 的一条记录——刻意不含 system_top。"""
        record: dict = {
            "query_id": self.query_id,
            "path": self.path,
            "gender": self.gender,
            "candidates": list(self.candidates),
        }
        if self.subject_id is not None:
            record["subject_id"] = self.subject_id
        if self.device_id is not None:
            record["device_id"] = self.device_id
        return record


@dataclass(frozen=True)
class SheetProvenance:
    """内部审计记录：系统当时对每条 query 的预测，绝不发给标注者。"""

    query_id: str
    system_top: tuple[str, ...]
    scorable: bool

    def to_record(self) -> dict:
        return {
            "query_id": self.query_id,
            "system_top": list(self.system_top),
            "scorable": self.scorable,
        }


def build_candidate_sheet(
    specs: list[dict],
    index: LibraryIndex,
    model: CalibrationModel,
    encoder: VocalEncoder,
    *,
    seed: int = 0,
    k: int = SHEET_CANDIDATES,
) -> tuple[list[SheetEntry], list[SheetProvenance]]:
    """为每条录音生成盲听候选清单与内部审计记录。

    ``specs`` 是每个录音的字典，必需键 ``query_id``（唯一）、``path``（相对
    本函数 ``base_dir`` 或绝对路径）、``gender``（female/male）；可选
    ``subject_id``/``device_id``。种子按 query 下标派生（``seed + i``），使
    同一批输入可复现，又不会让所有 query 用同一个打乱序列。

    Raises ``ValueError``：query_id 重复、gender 非法、k 小于 1、spec 缺键。
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    seen: set[str] = set()
    for spec in specs:
        for required in ("query_id", "path", "gender"):
            if required not in spec:
                raise ValueError(f"sheet spec missing required key {required!r}")
        qid = spec["query_id"]
        if qid in seen:
            raise ValueError(f"duplicate query_id {qid!r} in sheet specs")
        seen.add(qid)
        if spec["gender"] not in GENDERS:
            raise ValueError(
                f"query {qid!r}: gender must be one of {sorted(GENDERS)}, "
                f"got {spec['gender']!r}"
            )

    entries: list[SheetEntry] = []
    provenance: list[SheetProvenance] = []
    for position, spec in enumerate(specs):
        scorable = _is_scorable(Path(spec["path"]))
        system_top: list[str] = []
        if scorable:
            fv, embedding = _query_features(Path(spec["path"]), encoder)
            candidates = candidate_set(
                embedding,
                fv,
                index,
                model,
                spec["gender"],
                k=k,
                seed=seed + position,
            )
            # 只有可打分时系统答案才在场：从打乱后的候选里找出系统 Top-3，
            # 写进 provenance 供审计；sheet 本身不携带这个信息。
            ranked = [entry.singer_id for entry in _rank_top(
                fv, embedding, index, model, spec["gender"], k=k
            )]
            system_top = [s for s in ranked[:SYSTEM_TOP3]]
        else:
            candidates = _random_candidates(index, spec["gender"], k, seed + position)

        entries.append(
            SheetEntry(
                query_id=spec["query_id"],
                path=spec["path"],
                gender=spec["gender"],
                candidates=tuple(candidates),
                subject_id=spec.get("subject_id"),
                device_id=spec.get("device_id"),
            )
        )
        provenance.append(
            SheetProvenance(
                query_id=spec["query_id"],
                system_top=tuple(system_top),
                scorable=scorable,
            )
        )
    return entries, provenance


def write_sheet(
    path: Path, entries: list[SheetEntry], provenance_path: Path,
    provenance: list[SheetProvenance],
) -> None:
    """把清单与审计记录分别写到两个文件，返回 None。

    先写临时文件再替换，避免中途失败留下半截 sheet。provenance 与 sheet
    分开落盘：发给标注者的只有 sheet 文件。
    """
    _write_json_atomic(path, [entry.to_record() for entry in entries])
    _write_json_atomic(
        provenance_path,
        {
            "note": "INTERNAL ONLY - never share with annotators; holds the system's Top-3",
            "queries": [item.to_record() for item in provenance],
        },
    )


def _is_scorable(path: Path) -> bool:
    """录音是否通过质量门且有可打分窗口（与 run_eval 的判据一致）。"""
    audio = load_audio(path)
    if assess_quality(audio, SAMPLE_RATE).rejected:
        return False
    processed = preprocess(audio, PREPROCESS_QUERY)
    return select_best_window(processed, SAMPLE_RATE) is not None


def _query_features(path: Path, encoder: VocalEncoder):
    """跑查询管线得到 (FeatureVector, embedding)，与 run_eval._rank_one 同构。"""
    audio, _ = truncate_to_input_contract(load_audio(path))
    processed = preprocess(audio, PREPROCESS_QUERY)
    window = select_best_window(processed, SAMPLE_RATE)
    if window is None:  # _is_scorable 已确认， defense in depth
        raise ValueError(f"no usable window in {path} although marked scorable")
    start, end = window
    segment = processed[_sample_index(start) : _sample_index(end)]
    return extract_features(segment, SAMPLE_RATE), encoder.encode(segment, SAMPLE_RATE)


def _rank_top(fv, embedding, index, model, gender: str, k: int):
    """取系统排序前 k 位，用于 provenance 审计。"""
    from voicematch.calibrate.score import rank_singers

    return rank_singers(fv, embedding, index, model, gender, top_k=k)


def _random_candidates(
    index: LibraryIndex, gender: str, k: int, seed: int
) -> list[str]:
    """不可打分 query 的候选：同性别随机 k 位、打乱、无重复。"""
    pool = sorted(
        singer.singer_id
        for singer in index.singers.values()
        if singer.gender == gender
    )
    if not pool:
        raise LookupError(
            f"no singers of gender {gender!r} in library; cannot build a sheet"
        )
    generator = np.random.default_rng(seed)
    chosen = generator.choice(
        pool, size=min(k, len(pool)), replace=False
    )
    order = generator.permutation(len(chosen))
    return [str(chosen[i]) for i in order]


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)
