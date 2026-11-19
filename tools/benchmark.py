"""真人盲听基准集采集工具链（设计 6.1/6.2）。

主链子命令，真人环节之外全部自动：

    diagnose   素材可用性诊断 + 替换建议（时长/质量门/可用窗数）
    samples    从库素材导出每位歌手试听样本 + samples.json
    sheet      真人录音 + 歌手库 -> 盲听清单（系统答案分离到 provenance）
    adjudicate 标注文件 + 清单 -> kappa 检验 + 严格多数裁决
    freeze     清单 + 裁决 -> 版本化 eval_set.json + MANIFEST.json
    evaluate   冻结的 eval_set -> run_eval 的 G1 指标（产品准确率）

当 adjudicate 报 kappa 低于门槛（默认 0.6，见 KAPPA_THRESHOLD_DEFAULT）时，
按设计 6.2 改用 pairwise 比较式提问重标这条退路：

    pairwise-sheet       争议 query -> 两两比较任务清单
    pairwise-adjudicate  比较标注 -> 与 adjudicate 同格式的裁决（可直接 freeze）

并在 --adjudicated 文件里把 label_method 标为 pairwise，MANIFEST 随裁决透传，
便于审计"为何重标"。

与 :mod:`voicematch.eval` 的边界：本工具负责"标注从哪来、稳不稳、怎么锁版"，
G1 数字由 :func:`voicematch.eval.run.run_eval` 计算，本工具只负责喂给它正确的
eval set 并如实声明真值来源（MANIFEST.label_source）。

Usage:
    python tools/benchmark.py sheet --specs specs.json --specs-dir audio/ \\
        --library lib/ --encoder hubert --out sheet.json --provenance prov.json
    python tools/benchmark.py adjudicate --annotations ann.json --sheet sheet.json \\
        --out adjudicated.json --kappa-threshold 0.6
    python tools/benchmark.py freeze --sheet sheet.json --adjudicated adjudicated.json \\
        --out frozen/ --library lib/ --encoder hubert
    python tools/benchmark.py evaluate --eval-set frozen/eval_set.json \\
        --library lib/ --encoder hubert
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np  # noqa: E402

from voicematch.benchmark.agreement import (  # noqa: E402
    KAPPA_THRESHOLD_DEFAULT,
    AgreementReport,
    Annotation,
    adjudicate,
    agreement_report,
)
from voicematch.benchmark.freeze import freeze  # noqa: E402
from voicematch.benchmark.pairwise import (  # noqa: E402
    PairAnnotation,
    adjudicate_pairwise,
    build_pairwise_sheet,
)
from voicematch.benchmark.samples import (  # noqa: E402
    diagnose_catalog,
    export_samples,
    write_diagnosis,
    write_manifest,
)
from voicematch.benchmark.sheet import build_candidate_sheet, write_sheet  # noqa: E402
from voicematch.calibrate.model import fit_calibration  # noqa: E402
from voicematch.calibrate.pairs import build_pairs  # noqa: E402
from voicematch.embed.encoder import HashEncoder, HubertEncoder  # noqa: E402
from voicematch.eval.dataset import load_eval_set  # noqa: E402
from voicematch.eval.run import calibration_version, run_eval  # noqa: E402
from voicematch.library.store import load_library  # noqa: E402


def _jsonable(value):
    """Map non-finite floats to null at the JSON boundary (invariant 4)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _emit(payload) -> None:
    print(json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_stack(library: Path, encoder_name: str):
    """Load library + matching encoder + calibration re-derived from the library."""
    index = load_library(library)
    expected = "HUBERT_BASE" if encoder_name == "hubert" else "HashEncoder"
    if index.model_id != expected:
        raise ValueError(
            f"encoder {encoder_name!r} (model_id {expected!r}) does not match "
            f"library model_id {index.model_id!r}"
        )
    encoder = HubertEncoder() if encoder_name == "hubert" else HashEncoder()
    X, y, groups = build_pairs(index, index.embeddings)
    model = fit_calibration(X, y, groups)
    return index, model, encoder


def _resolve_specs(specs: list[dict], specs_dir: Path) -> list[dict]:
    """Resolve each spec's path against specs_dir, keeping the original in the sheet."""
    resolved = []
    for spec in specs:
        item = dict(spec)
        raw = Path(spec["path"])
        if not raw.is_absolute():
            item["path"] = str((specs_dir / raw).resolve())
        resolved.append(item)
    return resolved


def cmd_samples(args) -> int:
    """导出候选试听样本 + samples.json，供离线标注界面加载。"""
    samples, warnings = export_samples(
        args.catalog, args.out_dir, secs=args.secs
    )
    manifest_path = args.out_dir / "samples.json"
    write_manifest(manifest_path, samples)
    _emit({
        "status": "ok",
        "command": "samples",
        "n_samples": len(samples),
        "warnings": warnings,
        "samples_dir": str(args.out_dir),
        "manifest": str(manifest_path),
        "note": "internal listening samples; do not redistribute source vocals",
    })
    return 0


def cmd_diagnose(args) -> int:
    """诊断素材可用性并给出替换建议。"""
    suggestions = diagnose_catalog(args.catalog)
    needs = [s for s in suggestions if s.status != "ok"]
    path = args.out
    write_diagnosis(path, suggestions)
    _emit({
        "status": "ok",
        "command": "diagnose",
        "n_singers": len(suggestions),
        "n_ok": len(suggestions) - len(needs),
        "n_needs_replacement": len(needs),
        "needs_replacement": [s.singer_id for s in needs],
        "out": str(path),
        "note": "materials are diagnosed with the library path (PREPROCESS_LIBRARY + select_windows)",
    })
    return 0


def cmd_sheet(args) -> int:
    specs = _resolve_specs(_read_json(args.specs), args.specs_dir)
    index, model, encoder = _load_stack(args.library, args.encoder)
    entries, provenance = build_candidate_sheet(
        specs, index, model, encoder, seed=args.seed, k=args.k
    )
    write_sheet(args.out, entries, args.provenance, provenance)
    _emit({
        "status": "ok",
        "command": "sheet",
        "n_queries": len(entries),
        "n_scorable": sum(1 for p in provenance if p.scorable),
        "sheet": str(args.out),
        "provenance": str(args.provenance),
        "note": "provenance is INTERNAL ONLY; never share with annotators",
    })
    return 0


def cmd_adjudicate(args) -> int:
    raw = _read_json(args.annotations)
    if not isinstance(raw, list):
        raise ValueError("annotations file must be a JSON list of annotation objects")
    annotations = [
        Annotation(
            query_id=item["query_id"],
            annotator_id=item["annotator_id"],
            picked=tuple(item.get("picked", [])),
        )
        for item in raw
    ]
    sheet = _read_json(args.sheet)
    allowed = {item["query_id"]: item["candidates"] for item in sheet}

    decisions = adjudicate(annotations, allowed)
    agreement = agreement_report(annotations)
    disputed = sorted(qid for qid, d in decisions.items() if d.disputed)

    payload = {
        "label_method": "majority",
        "adjudications": [
            {
                "query_id": qid,
                "picked": sorted(d.picked),
                "annotator_count": d.annotator_count,
                "disputed": d.disputed,
            }
            for qid, d in sorted(decisions.items())
        ],
        "agreement": {
            "kappa_mean": agreement.kappa_mean,
            "pairwise": agreement.pairwise,
            "n_pairs": agreement.n_pairs,
            "n_measurable_pairs": agreement.n_measurable_pairs,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _emit({
        "status": "ok",
        "command": "adjudicate",
        "n_queries": len(decisions),
        "kappa_mean": agreement.kappa_mean,
        "n_pairs": agreement.n_pairs,
        "kappa_threshold": args.kappa_threshold,
        "below_threshold": agreement.is_below(args.kappa_threshold),
        "disputed_queries": disputed,
        "out": str(args.out),
        "guidance": (
            "kappa below threshold: the question is unstable; switch to pairwise "
            "annotation rather than lowering the threshold."
            if agreement.is_below(args.kappa_threshold)
            else "kappa at or above threshold."
        ),
    })
    return 0


def _agreement_from_payload(payload: dict) -> AgreementReport:
    data = payload["agreement"]
    return AgreementReport(
        kappa_mean=data["kappa_mean"],
        pairwise=data["pairwise"],
        n_pairs=data["n_pairs"],
        n_measurable_pairs=data["n_measurable_pairs"],
    )


def cmd_pairwise_sheet(args) -> int:
    """kappa 低于门槛：对争议 query 生成两两比较任务。"""
    adjudicated = _read_json(args.adjudicated)
    disputed = {
        item["query_id"]
        for item in adjudicated["adjudications"]
        if item.get("disputed")
    }
    # 争议集可显式覆盖（例如全部重标），否则只用 disputed 的 query。
    if args.queries:
        wanted = set(_read_json(args.queries))
        disputed = disputed & wanted if args.intersect else wanted
    if not disputed:
        _emit({
            "status": "ok",
            "command": "pairwise-sheet",
            "n_tasks": 0,
            "note": "no disputed queries; nothing to re-annotate",
            "out": str(args.out),
        })
        return 0

    sheet = _read_json(args.sheet)
    by_qid = {item["query_id"]: item for item in sheet}
    contested: dict[str, list[str]] = {}
    for qid in sorted(disputed):
        entry = by_qid.get(qid)
        if entry is None:
            raise ValueError(f"disputed query {qid!r} not found in sheet")
        candidates = list(entry["candidates"])
        if args.top is not None and len(candidates) > args.top:
            candidates = candidates[: args.top]  # 限定比较规模，控制标注负担
        contested[qid] = candidates

    tasks = build_pairwise_sheet(contested)
    payload = {
        "label_method": "pairwise",
        "reason": "kappa below threshold; switched to pairwise per design 6.2",
        "n_queries": len(contested),
        "tasks": [
            {
                "query_id": task.query_id,
                "pair_id": task.pair_id,
                "singer_a": task.singer_a,
                "singer_b": task.singer_b,
            }
            for task in tasks
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _emit({
        "status": "ok",
        "command": "pairwise-sheet",
        "n_queries": len(contested),
        "n_tasks": len(tasks),
        "out": str(args.out),
        "instruction": "for each pair, pick A or B that sounds more like the query; null = cannot tell",
    })
    return 0


def cmd_pairwise_adjudicate(args) -> int:
    """比较标注 -> 与 adjudicate 同格式的裁决，可继续 freeze。"""
    from voicematch.benchmark.agreement import Adjudication

    pair_payload = _read_json(args.pairs)
    raw_tasks = pair_payload["tasks"]
    from voicematch.benchmark.pairwise import PairTask

    tasks = [
        PairTask(
            query_id=item["query_id"],
            pair_id=item["pair_id"],
            singer_a=item["singer_a"],
            singer_b=item["singer_b"],
        )
        for item in raw_tasks
    ]
    raw_annotations = _read_json(args.annotations)
    annotations = [
        PairAnnotation(
            query_id=item["query_id"],
            pair_id=item["pair_id"],
            annotator_id=item["annotator_id"],
            winner=item.get("winner"),
        )
        for item in raw_annotations
    ]
    decisions = adjudicate_pairwise(tasks, annotations)

    payload = {
        "label_method": "pairwise",
        "adjudications": [
            {
                "query_id": qid,
                "picked": sorted(d.picked),
                "annotator_count": d.annotator_count,
                "disputed": d.disputed,
            }
            for qid, d in sorted(decisions.items())
        ],
        # Pairwise re-labeling has no Cohen's-kappa-over-multiselect; keep the
        # original agreement as provenance so MANIFEST records why re-labeling
        # happened. A reviewer reads label_method=pairwise below.
        "agreement": {
            "kappa_mean": None,
            "pairwise": {},
            "n_pairs": 0,
            "n_measurable_pairs": 0,
        },
        "reannotation_of": str(args.pairs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _emit({
        "status": "ok",
        "command": "pairwise-adjudicate",
        "n_queries": len(decisions),
        "n_disputed": sum(1 for d in decisions.values() if d.disputed),
        "out": str(args.out),
    })
    return 0


def _load_rights(library: Path, explicit: Path | None) -> dict:
    """权利声明的查找顺序：显式 --rights > library/catalog/ > 同级 catalog/。

    ingest 生成的 RIGHTS.json 与 catalog.json 同目录，而 build 产物在 lib/；
    freeze 原先只查 library/catalog/，导致 MANIFEST 的 rights 恒为空。这里
    兼容两种布局，保持旧行为（找不到返回 {}）不变。
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.append(library / "catalog" / "RIGHTS.json")
    candidates.append(library.parent / "catalog" / "RIGHTS.json")
    for path in candidates:
        if path.is_file():
            return _read_json(path)
    return {}


def cmd_freeze(args) -> int:
    from voicematch.benchmark.agreement import Adjudication
    from voicematch.benchmark.sheet import SheetEntry

    sheet = _read_json(args.sheet)
    entries = [
        SheetEntry(
            query_id=item["query_id"],
            path=item["path"],
            gender=item["gender"],
            candidates=tuple(item["candidates"]),
            subject_id=item.get("subject_id"),
            device_id=item.get("device_id"),
        )
        for item in sheet
    ]
    payload = _read_json(args.adjudicated)
    adjudications = {
        item["query_id"]: Adjudication(
            query_id=item["query_id"],
            picked=frozenset(item["picked"]),
            annotator_count=item["annotator_count"],
            disputed=item["disputed"],
        )
        for item in payload["adjudications"]
    }
    agreement = _agreement_from_payload(payload)

    index, model, encoder = _load_stack(args.library, args.encoder)
    rights = _load_rights(args.library, args.rights)
    manifest = freeze(
        args.out,
        entries,
        adjudications,
        agreement,
        library_version=index.version,
        model_id=index.model_id,
        calibration_version=calibration_version(model),
        rights=rights,
        kappa_threshold=args.kappa_threshold,
        label_method=payload.get("label_method", "majority"),
    )
    _emit({"status": "ok", "command": "freeze", "manifest": manifest, "out": str(args.out)})
    return 0


def cmd_evaluate(args) -> int:
    index, model, encoder = _load_stack(args.library, args.encoder)
    queries = load_eval_set(args.eval_set)
    result = run_eval(queries, index, model, encoder)
    manifest_path = args.eval_set.parent / "MANIFEST.json"
    label_source = (
        _read_json(manifest_path).get("label_source")
        if manifest_path.is_file()
        else "UNKNOWN (no MANIFEST.json beside eval_set)"
    )
    _emit({
        "status": "ok",
        "command": "evaluate",
        "label_source": label_source,
        "caveat": (
            "G1 here is product accuracy vs blind human majority annotation; "
            "distinct from holdout proxy metrics (dataset ground-truth labels)."
        ),
        "metrics": result,
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sheet = sub.add_parser("sheet", help="generate a blind annotation sheet")
    p_sheet.add_argument("--specs", type=Path, required=True)
    p_sheet.add_argument("--specs-dir", type=Path, default=Path("."))
    p_sheet.add_argument("--library", type=Path, required=True)
    p_sheet.add_argument("--encoder", choices=("hubert", "hash"), default="hubert")
    p_sheet.add_argument("--out", type=Path, required=True)
    p_sheet.add_argument("--provenance", type=Path, required=True)
    p_sheet.add_argument("--seed", type=int, default=0)
    p_sheet.add_argument("--k", type=int, default=8)
    p_sheet.set_defaults(func=cmd_sheet)

    p_samples = sub.add_parser(
        "samples", help="export per-singer listening samples for annotators"
    )
    p_samples.add_argument("--catalog", type=Path, required=True)
    p_samples.add_argument("--out-dir", type=Path, required=True)
    p_samples.add_argument("--secs", type=float, default=10.0)
    p_samples.set_defaults(func=cmd_samples)

    p_diag = sub.add_parser(
        "diagnose", help="diagnose material usability and suggest replacements"
    )
    p_diag.add_argument("--catalog", type=Path, required=True)
    p_diag.add_argument("--out", type=Path, required=True)
    p_diag.set_defaults(func=cmd_diagnose)

    p_adj = sub.add_parser("adjudicate", help="kappa check + majority adjudication")
    p_adj.add_argument("--annotations", type=Path, required=True)
    p_adj.add_argument("--sheet", type=Path, required=True)
    p_adj.add_argument("--out", type=Path, required=True)
    p_adj.add_argument("--kappa-threshold", type=float, default=KAPPA_THRESHOLD_DEFAULT)
    p_adj.set_defaults(func=cmd_adjudicate)

    p_pws = sub.add_parser(
        "pairwise-sheet", help="generate pairwise comparison tasks for contested queries"
    )
    p_pws.add_argument("--adjudicated", type=Path, required=True)
    p_pws.add_argument("--sheet", type=Path, required=True)
    p_pws.add_argument("--out", type=Path, required=True)
    p_pws.add_argument(
        "--queries", type=Path, default=None,
        help="optional JSON list of query_ids to re-annotate",
    )
    p_pws.add_argument(
        "--intersect", action="store_true",
        help="with --queries, restrict to queries that are both listed and disputed",
    )
    p_pws.add_argument(
        "--top", type=int, default=None,
        help="compare only the first N candidates of each query (bounds annotator load)",
    )
    p_pws.set_defaults(func=cmd_pairwise_sheet)

    p_pwa = sub.add_parser(
        "pairwise-adjudicate", help="adjudicate pairwise comparisons into picked sets"
    )
    p_pwa.add_argument("--pairs", type=Path, required=True)
    p_pwa.add_argument("--annotations", type=Path, required=True)
    p_pwa.add_argument("--out", type=Path, required=True)
    p_pwa.set_defaults(func=cmd_pairwise_adjudicate)

    p_freeze = sub.add_parser("freeze", help="freeze a versioned eval set")
    p_freeze.add_argument("--sheet", type=Path, required=True)
    p_freeze.add_argument("--adjudicated", type=Path, required=True)
    p_freeze.add_argument("--out", type=Path, required=True)
    p_freeze.add_argument("--library", type=Path, required=True)
    p_freeze.add_argument("--encoder", choices=("hubert", "hash"), default="hubert")
    p_freeze.add_argument(
        "--rights", type=Path, default=None,
        help="explicit RIGHTS.json; otherwise searched in library/catalog/ then sibling catalog/",
    )
    p_freeze.add_argument("--kappa-threshold", type=float, default=KAPPA_THRESHOLD_DEFAULT)
    p_freeze.set_defaults(func=cmd_freeze)

    p_eval = sub.add_parser("evaluate", help="run G1 metrics on a frozen eval set")
    p_eval.add_argument("--eval-set", type=Path, required=True)
    p_eval.add_argument("--library", type=Path, required=True)
    p_eval.add_argument("--encoder", choices=("hubert", "hash"), default="hubert")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, LookupError, KeyError, FileNotFoundError) as exc:
        _emit({"status": "error", "reason": type(exc).__name__, "detail": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
