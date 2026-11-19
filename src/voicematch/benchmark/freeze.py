"""基准集冻结：把多数裁决结果锁成可复现、可审计的 eval set（设计 6.1/6.3）。

标注完成后，本模块把三样东西合成一个冻结的基准集：

1. 标注清单（sheet）——每条 query 的候选集与采集元数据；
2. 多数裁决（adjudicate）——每条 query 的 ``picked`` 与 ``annotator_count``；
3. 一致性摘要（agreement_report）——kappa，作为"问法是否稳定"的证据。

产出两个文件：

- ``eval_set.json``：JSON list，字段严格对齐 :func:`voicematch.eval.dataset.
  load_eval_set` 的契约，可被 ``run_eval`` 直接消费；
- ``MANIFEST.json``：基准集版本（内容哈希）、标注人数、kappa、争议 query、
  权利/用途声明（label_source = blind human annotation）。

为什么不变量写在这里
--------------------
- **冻结即校验**：写盘后立刻用 ``load_eval_set`` 读回。一个损坏的基准集若
  被静默加载，会在评估中途改变分母，产出一个看起来"测过"的数字。校验失败
  必须让冻结失败，而不是留一个坏文件。
- **真值来源必须诚实**：MANIFEST 明确 ``label_source`` 是人类盲听多数，与
  ``tools/holdout_eval.py`` 的数据集 ground-truth 代理指标区分开。任何报告
  引用本基准集的 G1 时，才能声称是"产品准确率"而非"歌手识别自检"。
- **版本由内容决定**：benchmark_version 是 eval_set 规范内容的 SHA-256 前
  16 位（与 library_version 同形）。同一批 query + 标注永远得到同一版本。
- **争议不删**：kappa 低或首选不一致的 query 保留在 eval set 里，只在
  MANIFEST 的 ``disputed_queries`` 列出，供二次校验。删除它们会改变受理率
  与命中率的分母，把"标注不稳"伪装成"指标变好"——这正是设计 6.2 要防的。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from voicematch.benchmark.agreement import Adjudication, AgreementReport
from voicematch.benchmark.sheet import SheetEntry
from voicematch.eval.dataset import load_eval_set

#: MANIFEST 标注的真值来源——与 holdout 的 dataset ground truth 明确区分。
BLIND_LABEL_SOURCE: str = "blind_human_majority_annotation"

#: 冻结产物文件名。
EVAL_SET_FILENAME: str = "eval_set.json"
MANIFEST_FILENAME: str = "MANIFEST.json"


def build_eval_records(
    entries: list[SheetEntry],
    adjudications: dict[str, Adjudication],
) -> list[dict]:
    """把清单条目与裁决合成 eval_set 记录（JSON-ready）。

    ``picked`` 由多数裁决给出，``annotator_count`` 取裁决记录的真实人数。
    Raises ``ValueError``：清单里的 query 缺少裁决、裁决含清单候选之外的
    picked、picked 为空（"都不像"是合法的，故空集允许）。
    """
    records: list[dict] = []
    for entry in entries:
        adjudication = adjudications.get(entry.query_id)
        if adjudication is None:
            raise ValueError(
                f"query {entry.query_id!r} has no adjudication; freeze needs one "
                "per sheet entry"
            )
        unknown = sorted(adjudication.picked - set(entry.candidates))
        if unknown:
            raise ValueError(
                f"query {entry.query_id!r}: adjudicated picked {unknown} are "
                "not among the shown candidates"
            )
        if adjudication.annotator_count < 1:
            raise ValueError(
                f"query {entry.query_id!r}: annotator_count must be >= 1"
            )
        record: dict = {
            "query_id": entry.query_id,
            "path": entry.path,
            "gender": entry.gender,
            "candidates": list(entry.candidates),
            "picked": sorted(adjudication.picked),
            "annotator_count": adjudication.annotator_count,
        }
        if entry.subject_id is not None:
            record["subject_id"] = entry.subject_id
        if entry.device_id is not None:
            record["device_id"] = entry.device_id
        records.append(record)
    return records


def benchmark_version(records: list[dict]) -> str:
    """eval_set 规范内容的 16 位 SHA-256，作为基准集版本号。"""
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def freeze(
    out_dir: Path,
    entries: list[SheetEntry],
    adjudications: dict[str, Adjudication],
    agreement: AgreementReport,
    *,
    library_version: str,
    model_id: str,
    calibration_version: str,
    rights: dict | None = None,
    kappa_threshold: float = 0.6,
    label_method: str = "majority",
) -> dict:
    """冻结基准集并写盘，返回 MANIFEST 字典。

    写入 eval_set 后用 ``load_eval_set`` 读回校验（相对路径按 out_dir 解析），
    校验失败则抛 ``ValueError`` 且不写 MANIFEST。MANIFEST 里的争议 query、
    kappa 与权利声明使这份基准集可被审计、不会被误当成自检指标。
    """
    records = build_eval_records(entries, adjudications)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / EVAL_SET_FILENAME
    eval_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 冻结即校验：任何损坏都在此暴露，不让坏集流入评估。
    load_eval_set(eval_path)

    version = benchmark_version(records)
    disputed = sorted(
        entry.query_id
        for entry in entries
        if entry.query_id in adjudications and adjudications[entry.query_id].disputed
    )
    manifest = {
        "benchmark_version": version,
        "label_source": BLIND_LABEL_SOURCE,
        "label_method": label_method,
        "n_queries": len(records),
        "n_answerable": sum(1 for record in records if record["picked"]),
        "n_none_like": sum(1 for record in records if not record["picked"]),
        "annotator_count_min": min(
            (a.annotator_count for a in adjudications.values()), default=0
        ),
        "agreement": {
            "kappa_mean": _jsonable_float(agreement.kappa_mean),
            "n_pairs": agreement.n_pairs,
            "n_measurable_pairs": agreement.n_measurable_pairs,
            "kappa_threshold": kappa_threshold,
            "below_threshold": agreement.is_below(kappa_threshold),
        },
        "disputed_queries": disputed,
        "eval_set_file": EVAL_SET_FILENAME,
        "library_version": library_version,
        "model_id": model_id,
        "calibration_version": calibration_version,
        "rights": rights or {},
        "caveat": (
            "G1 from this set is product accuracy vs blind human majority; "
            "distinct from holdout proxy metrics (dataset ground-truth labels)."
        ),
    }
    (out_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _jsonable_float(value: float) -> float | None:
    """Non-finite kappa -> null at the JSON boundary (invariant 4)."""
    import math

    return value if math.isfinite(value) else None
