"""Scorer bake-off: which way of turning similarity into a ranking wins?

Runs one fixed evaluation set through several scorers that share everything
except the final scoring rule:

1. ``cosine`` -- the frozen embedding's cosine to the singer prototype.
2. ``calibration`` -- the current logistic layer on the five pair features.
3. ``fixed`` -- a hand-set linear rule on the same z-scored features
   (cosine dominant, four deltas as penalties), no fitted weights.
4. ``projection`` -- LDA fitted on the library's segment embeddings, then
   cosine in the projected space (a CPU-feasible stand-in for a stronger
   identity representation; full HuBERT fine-tuning is out of scope).
5. ``per_singer`` -- cosine standardised by each singer's own cosine
   distribution against the rest of the pool (per-singer stratification).

Why one script: the expensive step is HuBERT encoding each query; encoding
once and scoring five ways is the only apples-to-apples comparison, and it
keeps the query/pool/window path identical across scorers.

The cosine scorer is checked against a reference ``--per-query-out`` file from
``tools/holdout_eval.py``.  A mismatch means this script's replay of the query
path has drifted, and it exits without writing results instead of publishing a
bake-off about a different pipeline.

This is a research diagnostic on dataset-folder labels, not a product accuracy
benchmark.  The design and decision rule were written before the run in
``docs/2026-11-19-scorer-bakeoff.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from holdout_eval import build_holdout_set
from voicematch.audio.io import load_audio
from voicematch.audio.preprocess import PREPROCESS_QUERY, preprocess
from voicematch.audio.quality import assess_quality
from voicematch.audio.windows import select_best_window
from voicematch.calibrate.model import CalibrationModel, fit_calibration
from voicematch.calibrate.pairs import build_pairs, pair_features
from voicematch.calibrate.score import _mean_feature_vector, _segment_features_by_singer
from voicematch.config import SAMPLE_RATE
from voicematch.embed.encoder import HubertEncoder
from voicematch.eval.dataset import load_eval_set
from voicematch.eval.run import truncate_to_input_contract
from voicematch.features.extract import FeatureVector, extract_features
from voicematch.library.store import load_library

#: Pre-registered fixed weights on the z-scored pair features
#: [cos, pitch, timbre, resonance, style].  Cosine is the identity signal;
#: pitch and timbre are the two most identity-bearing deltas; resonance and
#: style are weaker evidence.  Not fitted, not tuned on the evaluation set.
FIXED_WEIGHTS: np.ndarray = np.array([1.0, -0.5, -0.5, -0.25, -0.25])

#: Ranking depth the evaluation measures (same as run_eval).
TOP_K = 3


@dataclass
class LibraryScorer:
    """Everything the five scoring rules need, fitted from the library alone."""

    model: CalibrationModel
    names: dict[str, str]
    gender_of: dict[str, str]
    protos: dict[str, np.ndarray]
    singer_vectors: dict[str, FeatureVector]
    cos_stats: dict[str, tuple[float, float]]
    lda: LinearDiscriminantAnalysis | None
    projected_protos: dict[str, np.ndarray]
    projected_leave_one_take_out_top1: float | None


def _normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def _fit_library(index) -> LibraryScorer:
    """Fit every library-side object once: calibration, LDA, cosine statistics."""
    pairs = build_pairs(index, index.embeddings)
    model = fit_calibration(*pairs)

    features_of = _segment_features_by_singer(index)
    protos = {sid: np.asarray(vec, dtype=np.float64) for sid, vec in index.prototypes.items()}
    # The singer side of a pair row is the per-descriptor mean over its finite
    # segments -- the same aggregation ``rank_singers`` uses.
    singer_vectors = {
        sid: _mean_feature_vector(vectors) for sid, vectors in features_of.items()
    }

    # Per-singer cosine statistics: how "central" is this prototype in the
    # gender pool?  Computed against every segment of the other singers (not
    # against its own segments), so the statistic is about confusability.
    emb = np.asarray(index.embeddings, dtype=np.float64)
    singer_of = [segment.singer_id for segment in index.segments]
    cos_stats: dict[str, tuple[float, float]] = {}
    for sid, proto in protos.items():
        others = [
            float(np.dot(row, proto))
            for i, row in enumerate(emb)
            if singer_of[i] != sid
            and index.singers[singer_of[i]].gender == index.singers[sid].gender
        ]
        if len(others) >= 2:
            cos_stats[sid] = (float(np.mean(others)), float(np.std(others)))
        else:
            cos_stats[sid] = (float(np.mean(others)) if others else 0.0, 1.0)

    labels = np.array(singer_of)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda.fit(emb, labels)
    projected = np.asarray(lda.transform(emb), dtype=np.float64)
    projected = projected / np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)
    projected_protos = {}
    for sid in sorted(set(singer_of)):
        rows_of = [i for i, owner in enumerate(singer_of) if owner == sid]
        projected_protos[sid] = _normalise(projected[rows_of].mean(axis=0))

    return LibraryScorer(
        model=model,
        names={sid: singer.name for sid, singer in index.singers.items()},
        gender_of={sid: singer.gender for sid, singer in index.singers.items()},
        protos=protos,
        singer_vectors=singer_vectors,
        cos_stats=cos_stats,
        lda=lda,
        projected_protos=projected_protos,
        projected_leave_one_take_out_top1=_leave_one_take_out_top1(index, emb, singer_of),
    )


def _leave_one_take_out_top1(index, emb: np.ndarray, singer_of: list[str]) -> float | None:
    """Cross-recording diagnostic for the projection: train on 2 takes, test the 3rd.

    This is not the headline metric (the hold-out songs are); it answers whether
    the projection separates singers *across recordings* at all, or only fits
    the recording identity.  Segment-level nearest centroid, chance = 1/20.
    """
    materials = [segment.material_id for segment in index.segments]
    takes = sorted(set(materials))
    if len(takes) < 2:
        return None
    correct = 0
    total = 0
    for take in takes:
        train = [i for i, material in enumerate(materials) if material != take]
        test = [i for i, material in enumerate(materials) if material == take]
        if not test or len(set(singer_of[i] for i in train)) < 2:
            continue
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
        lda.fit(emb[train], [singer_of[i] for i in train])
        z_train = np.asarray(lda.transform(emb[train]), dtype=np.float64)
        centroids: dict[str, np.ndarray] = {}
        for row, position in enumerate(train):
            centroids.setdefault(singer_of[position], []).append(z_train[row])
        centroids = {
            sid: _normalise(np.mean(vectors, axis=0)) for sid, vectors in centroids.items()
        }
        z_test = np.asarray(lda.transform(emb[test]), dtype=np.float64)
        z_test = z_test / np.maximum(np.linalg.norm(z_test, axis=1, keepdims=True), 1e-12)
        for position, z in zip(test, z_test):
            predicted = max(
                centroids.items(), key=lambda item: (float(np.dot(z, item[1])), item[0])
            )[0]
            correct += int(predicted == singer_of[position])
            total += 1
    return correct / total if total else None


@dataclass
class QueryContext:
    """The query-side objects the scorers share."""

    emb: np.ndarray
    fv: FeatureVector
    cos: dict[str, float]
    rows: dict[str, np.ndarray]
    projected_cos: dict[str, float]


def _query_context(query, scorer: LibraryScorer, encoder: HubertEncoder) -> QueryContext | None:
    """Replay the product query path; ``None`` means no usable window."""
    audio, _truncated = truncate_to_input_contract(load_audio(query.path))
    if assess_quality(audio, SAMPLE_RATE).rejected:
        return None
    processed = preprocess(audio, PREPROCESS_QUERY)
    window = select_best_window(processed, SAMPLE_RATE)
    if window is None:
        return None
    start, end = window
    segment = processed[
        int(round(start * SAMPLE_RATE)) : int(round(end * SAMPLE_RATE))
    ]
    fv = extract_features(segment, SAMPLE_RATE)
    emb = np.asarray(encoder.encode(segment, SAMPLE_RATE), dtype=np.float64)

    cos = {sid: float(np.dot(emb, proto)) for sid, proto in scorer.protos.items()}
    rows = {
        sid: pair_features(cos[sid], fv, scorer.singer_vectors[sid])
        for sid in scorer.protos
    }
    projected_cos: dict[str, float] = {}
    if scorer.lda is not None:
        z = np.asarray(scorer.lda.transform(emb.reshape(1, -1))[0], dtype=np.float64)
        z = _normalise(z)
        projected_cos = {
            sid: float(np.dot(z, proto)) for sid, proto in scorer.projected_protos.items()
        }
    return QueryContext(emb=emb, fv=fv, cos=cos, rows=rows, projected_cos=projected_cos)


def _scores(ctx: QueryContext, scorer: LibraryScorer, model) -> dict[str, dict[str, float]]:
    """Return every scorer's score for one query, keyed by scorer name."""
    fixed = {}
    calibration = {}
    per_singer = {}
    for sid, row in ctx.rows.items():
        z = (row - model.mean) / model.scale
        fixed[sid] = float(FIXED_WEIGHTS @ z)
        calibration[sid] = float(model.score(row))
        mu, sd = scorer.cos_stats[sid]
        per_singer[sid] = float((ctx.cos[sid] - mu) / (sd if sd > 1e-9 else 1.0))
    return {
        "cosine": dict(ctx.cos),
        "calibration": calibration,
        "fixed": fixed,
        "projection": dict(ctx.projected_cos),
        "per_singer": per_singer,
    }


def _rank(
    scores: dict[str, float], truth: str, gender_of: dict[str, str], gender: str
) -> tuple[list[str], list[float], int | None]:
    """Rank within the query's gender pool, as ``rank_singers`` does.

    The product path only compares a query against singers of its own gender
    (``rank_singers`` filters the pool before scoring).  Scoring every singer
    in the library is fine for the bake-off, but the ranking -- and therefore
    the truth's rank, top1/top3 and the random baseline -- must be taken over
    the gender pool only.  Ranking the whole library lets a singer of the other
    gender slip into the top-3 and silently changes the question being asked.
    """
    pool = {sid: value for sid, value in scores.items() if gender_of.get(sid) == gender}
    order = sorted(pool.items(), key=lambda item: (-item[1], item[0]))
    ids = [sid for sid, _ in order]
    rank = ids.index(truth) + 1 if truth in ids else None
    return ids[:TOP_K], [round(pool[sid], 6) for sid in ids[:TOP_K]], rank


def _load_reference(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["query_id"]: row for row in rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--queries-per-singer", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reference-per-query",
        type=Path,
        default=None,
        help="holdout_eval per-query file whose cosine ranking this run must reproduce",
    )
    args = parser.parse_args(argv)

    index = load_library(args.library)
    if index.model_id != "HUBERT_BASE":
        print(f"expected a HuBERT library, got {index.model_id}", file=sys.stderr)
        return 2

    scorer = _fit_library(index)
    encoder = HubertEncoder()
    print(
        f"library {index.version}: {len(index.singers)} singers, "
        f"{len(index.segments)} segments, LDA takes 3rd-take top1="
        f"{scorer.projected_leave_one_take_out_top1}",
        flush=True,
    )

    eval_path, _provenance = build_holdout_set(args.material, args.queries_per_singer)
    queries = load_eval_set(eval_path)

    rows_by_scorer: dict[str, list[dict]] = {
        name: [] for name in ("cosine", "calibration", "fixed", "projection", "per_singer")
    }
    for query in queries:
        context = _query_context(query, scorer, encoder)
        truth = min(query.picked) if query.picked else None
        base = {
            "query_id": query.query_id,
            "singer_id": truth,
            "gender": query.gender,
            "accepted": context is not None,
            "skipped": context is None,
            "ranking": [],
            "scores": [],
            "rank_of_truth": None,
            "top1": False,
            "top3": False,
        }
        if context is None:
            for name in rows_by_scorer:
                rows_by_scorer[name].append(dict(base))
            continue
        all_scores = _scores(context, scorer, scorer.model)
        for name, scores in all_scores.items():
            ranking, top_scores, rank = (
                _rank(scores, truth, scorer.gender_of, query.gender)
                if truth
                else ([], [], None)
            )
            row = dict(base)
            row.update(
                ranking=ranking,
                scores=top_scores,
                rank_of_truth=rank,
                top1=rank == 1,
                top3=rank is not None and rank <= TOP_K,
            )
            rows_by_scorer[name].append(row)

    if args.reference_per_query is not None:
        reference = _load_reference(args.reference_per_query)
        mismatches = []
        for row in rows_by_scorer["cosine"]:
            ref = reference.get(row["query_id"])
            if ref is None:
                mismatches.append((row["query_id"], "missing in reference"))
            elif ref["ranking"] != row["ranking"]:
                mismatches.append((row["query_id"], f"{row['ranking']} != {ref['ranking']}"))
        if mismatches:
            print(
                f"cosine ranking does not reproduce the reference for "
                f"{len(mismatches)} queries, e.g. {mismatches[:3]}",
                file=sys.stderr,
            )
            return 3

    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "status": "ok",
        "command": "scorer_bakeoff",
        "library_version": index.version,
        "n_total": len(queries),
        "n_answerable": sum(1 for row in rows_by_scorer["cosine"] if row["ranking"]),
        "n_skipped": sum(1 for row in rows_by_scorer["cosine"] if row["skipped"]),
        "projection_leave_one_take_out_top1": scorer.projected_leave_one_take_out_top1,
        "fixed_weights": FIXED_WEIGHTS.tolist(),
        "caveat": "dataset-folder labels (who sang the audio), not blind listening; research diagnostic",
        "scorers": {},
    }
    for name, rows in rows_by_scorer.items():
        ranked = [row for row in rows if row["ranking"] and row["singer_id"]]
        summary["scorers"][name] = {
            "top1": sum(1 for row in ranked if row["top1"]) / len(ranked) if ranked else None,
            "top3": sum(1 for row in ranked if row["top3"]) / len(ranked) if ranked else None,
            "mean_rank_of_truth": float(np.mean([row["rank_of_truth"] for row in ranked]))
            if ranked
            else None,
            "n": len(ranked),
        }
        (args.out / f"per_query_{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
