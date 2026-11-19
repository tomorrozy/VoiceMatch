"""Paired comparison of two hold-out runs over the same queries.

``tools/holdout_eval.py --per-query-out`` writes one row per query; running it
twice (calibrated and ``--cosine-only``) produces two files that differ only in
the ranking.  Comparing their aggregates is an *unpaired* reading of the same
data and hides the fact that both runs answered identical queries; this tool
joins them by ``query_id`` and reports the paired evidence instead:

- top1/top3 for both runs and their difference;
- the discordant counts and an exact McNemar test (two-sided binomial on
  ``b + c`` discordant queries, no normal approximation -- n is small);
- a paired bootstrap CI of the difference (10 000 draws, fixed seed).  The
  resampling unit defaults to the query; ``--cluster-by singer_id`` resamples
  whole singers instead, which is required when one singer contributes several
  queries (they share a prototype and the same calibration fit);
- the mean rank of the true singer, which moves even when the Top-1 hit does
  not.

Scope: a research diagnostic on dataset-folder labels (one query per singer),
not a product accuracy claim.  Exit status is 0 for a completed comparison and
2 when the two files do not describe the same query set -- comparing different
queries would produce a number about nothing.

Usage:
    python tools/paired_compare.py cal.json cos.json \\
        --label-a calibrated --label-b cosine
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261119


def _load(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_id = {row["query_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit(f"{path}: duplicate query_id")
    return by_id


def _rate(rows: list[dict], key: str) -> float:
    return sum(1 for row in rows if row[key]) / len(rows) if rows else float("nan")


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: binomial on the discordant pairs."""
    n = b + c
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _bootstrap_delta(
    a_hits: np.ndarray,
    b_hits: np.ndarray,
    cluster_ids: list[str] | None = None,
) -> tuple[float, float]:
    """Paired bootstrap CI of the hit-rate difference.

    Without ``cluster_ids`` the resampling unit is the query.  With them, whole
    clusters are drawn (every query of a resampled singer enters together),
    which is the honest unit when one singer contributes several queries: those
    queries share a prototype and the same calibration fit, so treating them as
    independent understates the variance.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = a_hits.size
    if cluster_ids is None:
        draws = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
        deltas = a_hits[draws].mean(axis=1) - b_hits[draws].mean(axis=1)
    else:
        groups: dict[str, list[int]] = {}
        for position, cluster in enumerate(cluster_ids):
            groups.setdefault(str(cluster), []).append(position)
        keys = sorted(groups)
        deltas = np.empty(BOOTSTRAP_DRAWS)
        for draw in range(BOOTSTRAP_DRAWS):
            picked = rng.integers(0, len(keys), size=len(keys))
            indices = np.concatenate([groups[keys[k]] for k in picked])
            deltas[draw] = a_hits[indices].mean() - b_hits[indices].mean()
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path, help="per-query JSON of the first run")
    parser.add_argument("b", type=Path, help="per-query JSON of the second run")
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument(
        "--cluster-by",
        choices=("query", "singer_id"),
        default="query",
        help=(
            "bootstrap resampling unit. Use singer_id when one singer "
            "contributes several queries: those queries share a prototype and "
            "the same calibration fit, so query-level resampling is optimistic"
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="QUERY_ID",
        help=(
            "drop one query from the comparison (repeatable). For sensitivity: "
            "does the conclusion survive removing a known-problem singer?"
        ),
    )
    args = parser.parse_args(argv)

    first, second = _load(args.a), _load(args.b)
    if set(first) != set(second):
        only_a = sorted(set(first) - set(second))
        only_b = sorted(set(second) - set(first))
        print(
            f"query sets differ: only in {args.label_a}={only_a}, "
            f"only in {args.label_b}={only_b}",
            file=sys.stderr,
        )
        return 2

    # Only queries that both runs actually ranked: a query one side skipped
    # cannot say which model ranked the singer higher.
    paired = [
        (first[qid], second[qid])
        for qid in sorted(first)
        if first[qid]["ranking"] and second[qid]["ranking"]
    ]
    if not paired:
        print("no query was ranked by both runs", file=sys.stderr)
        return 2

    excluded = sorted(set(args.exclude))
    unknown = [qid for qid in excluded if qid not in first]
    if unknown:
        print(f"--exclude names unknown queries: {unknown}", file=sys.stderr)
        return 2
    if excluded:
        paired = [pair for pair in paired if pair[0]["query_id"] not in set(excluded)]

    rows_a = [pair[0] for pair in paired]
    rows_b = [pair[1] for pair in paired]
    cluster_ids = (
        [row["singer_id"] for row in rows_a]
        if args.cluster_by == "singer_id"
        else None
    )
    result: dict[str, object] = {
        "status": "ok",
        "command": "paired_compare",
        "n_paired": len(paired),
        "bootstrap_unit": args.cluster_by,
        "n_clusters": len(set(cluster_ids)) if cluster_ids else len(paired),
        "excluded": excluded,
        "labels": {"a": args.label_a, "b": args.label_b},
        "rates": {
            args.label_a: {"top1": _rate(rows_a, "top1"), "top3": _rate(rows_a, "top3")},
            args.label_b: {"top1": _rate(rows_b, "top1"), "top3": _rate(rows_b, "top3")},
        },
        "caveat": "dataset-folder labels (who sang the audio), not blind listening; research diagnostic, not product accuracy",
        "mcnemar_unit": "query (assumes independence; descriptive when a singer contributes several queries)",
    }

    for key in ("top1", "top3"):
        a_hits = np.array([1.0 if row[key] else 0.0 for row in rows_a])
        b_hits = np.array([1.0 if row[key] else 0.0 for row in rows_b])
        both = int(np.sum((a_hits == 1) & (b_hits == 1)))
        only_a = int(np.sum((a_hits == 1) & (b_hits == 0)))
        only_b = int(np.sum((a_hits == 0) & (b_hits == 1)))
        neither = int(np.sum((a_hits == 0) & (b_hits == 0)))
        low, high = _bootstrap_delta(a_hits, b_hits, cluster_ids)
        result[key] = {
            "delta": float(a_hits.mean() - b_hits.mean()),
            "delta_ci95": [low, high],
            "agreement": {"both_hit": both, "only_a": only_a, "only_b": only_b, "neither": neither},
            "mcnemar_exact_p": _mcnemar_exact(only_a, only_b),
        }

    ranks_a = [row["rank_of_truth"] for row in rows_a if row["rank_of_truth"]]
    ranks_b = [row["rank_of_truth"] for row in rows_b if row["rank_of_truth"]]
    result["mean_rank_of_truth"] = {
        args.label_a: float(np.mean(ranks_a)) if ranks_a else None,
        args.label_b: float(np.mean(ranks_b)) if ranks_b else None,
    }

    # Pool size is per gender, so a single aggregate can hide one pool carrying
    # the result.  Descriptive only: ten queries per pool is too few for a test.
    groups: dict[str, dict] = {}
    for gender in sorted({row["gender"] for row in rows_a}):
        subset = [
            (row_a, row_b)
            for row_a, row_b in paired
            if row_a["gender"] == gender
        ]
        groups[gender] = {
            "n_paired": len(subset),
            args.label_a: {
                "top1": _rate([pair[0] for pair in subset], "top1"),
                "top3": _rate([pair[0] for pair in subset], "top3"),
            },
            args.label_b: {
                "top1": _rate([pair[1] for pair in subset], "top1"),
                "top3": _rate([pair[1] for pair in subset], "top3"),
            },
        }
    result["by_gender"] = groups

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
