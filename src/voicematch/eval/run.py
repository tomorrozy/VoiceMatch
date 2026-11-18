"""End-to-end evaluation: annotated recordings in, the headline numbers out.

The runner walks every query through exactly the query path the product uses --
``load_audio -> assess_quality -> preprocess(PREPROCESS_QUERY) ->
select_best_window -> extract_features -> encoder.encode -> rank_singers`` --
and reports the numbers design doc 6.3 requires.  This is the first place every
stage meets, so it also owns the definitions the numbers are reported under.
The quality gate measures the decoded signal (before the LUFS gain); the window
that is scored is the best-ranked one the shared rule admits, not the earliest.

Denominators (G1)
-----------------
The result always carries both headline numbers *and* the counts behind them:

- ``top1``/``top3`` are computed by
  :func:`~voicematch.eval.metrics.top_k_hit` over **accepted, answerable**
  queries only.  ``n_answerable`` is that denominator, so it never has to be
  reconstructed by a reader.
- ``acceptance_rate`` is over **all** queries, measured by
  :func:`~voicematch.eval.metrics.acceptance_rate` from the one quality report
  per query.  A rejected query is not ranked but stays in this denominator.
- ``n_none_like`` counts accepted, ranked queries annotated "都不像"
  (``picked == []``).  Ruling R9 says that is an answer about library coverage,
  not a ranking miss; ``top_k_hit`` already keeps empty label sets out of its
  denominator, so the runner counts them but does not re-implement the rule.
  (A query that is both "都不像" and unscorable is counted once, in
  ``n_skipped``: it never reached the Top-K question at all.)
- ``n_skipped`` counts accepted queries that could not be scored: no usable
  window, or a ranking where one candidate's score is non-finite (an upstream
  prototype with a ``nan``).  One bad row must not abort an evaluation run, and
  it must not be reported as a miss either, so the query leaves the Top-K
  denominator and the count says so.
- ``n_truncated`` counts queries whose recording is longer than the 60 s input
  contract.  Their audio is cut to the first ``MAX_INPUT_SEC`` seconds and then
  evaluated normally (ruling R23): truncation is *not* a rejection, so a
  truncated query still counts as accepted and stays in every denominator it
  would otherwise be in.
- ``n_total`` counts every query, so the ledger adds up:
  ``n_total == n_accepted + n_rejected`` and
  ``n_answerable == n_accepted - n_none_like - n_skipped`` (``n_truncated`` is
  informational: truncated queries are neither removed nor reclassified).

None of the metrics is a plain average of everything: when nothing is
measurable, ``top1``/``top3``/``acceptance_rate`` are ``nan`` (the metrics'
contract), never ``0.0``.

Serialisation is the caller's boundary
--------------------------------------
The returned dict is not JSON-ready by design: the metric values can be
``nan``, and ``QualityReport.snr_db`` is ``-inf`` for a digitally silent
recording (the runner does not put quality fields in the result, but the same
rule applies to whatever an owner of this dict adds).  ``NaN``/``Infinity`` are
not valid JSON, so whoever serialises this report must map non-finite floats to
``null``; the CLI (:mod:`voicematch.cli`) does exactly that.

The calibration model is fitted from the library's own segment pairs
-------------------------------------------------------------------
:func:`calibration_version` names the fitted model by a short hash of its five
fields.  This phase does not persist a calibration model: it is re-derived
deterministically from the library with ``build_pairs`` +
``fit_calibration`` (G4), so a result can be reproduced from the library and
the code revision alone.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence

import numpy as np

from voicematch.audio.io import load_audio
from voicematch.audio.preprocess import PREPROCESS_QUERY, preprocess
from voicematch.audio.quality import QualityReport, assess_quality
from voicematch.audio.windows import select_best_window
from voicematch.calibrate.model import CalibrationModel
from voicematch.calibrate.score import ScoredSinger, rank_singers
from voicematch.config import MAX_INPUT_SEC, SAMPLE_RATE
from voicematch.embed.encoder import VocalEncoder
from voicematch.eval.dataset import EvalQuery
from voicematch.eval.metrics import acceptance_rate, top_k_hit
from voicematch.features.extract import extract_features
from voicematch.library.schema import LibraryIndex

#: Ranking depth the evaluation measures.  Top-3 is the deepest headline number
#: (design doc 6.3), so one ranking of this depth produces both ``top1`` and
#: ``top3``.
RANK_TOP_K: int = 3


def run_eval(
    queries: Sequence[EvalQuery],
    index: LibraryIndex,
    model: CalibrationModel,
    encoder: VocalEncoder,
) -> dict[str, float]:
    """Evaluate ``queries`` against ``index``/``model``/``encoder``.

    Returns the two headline numbers (``top1``, ``top3``) with their acceptance
    rate and every denominator count; see the module docstring for the exact
    definition of each.  The ``n_*`` entries are plain ``int`` counts and the
    three metric entries are plain ``float`` values; the metric values may be
    ``nan`` when nothing was measurable, and must be sanitised at any JSON
    boundary.  A recording longer than the 60 s input contract is cut to
    ``MAX_INPUT_SEC`` and counted in ``n_truncated``; it is not rejected (R23).

    A query's candidate pool is selected by *its own* ``gender``.  An eval set
    that asks for a gender the library does not contain is a data problem, not
    a bad row: ``rank_singers`` raises ``LookupError`` with
    ``no_singers_in_pool`` and this function lets it propagate rather than
    silently skipping queries the library cannot answer.

    The query's ``candidates`` are annotation provenance -- the singers the
    annotator was shown -- not a ranking restriction: the runner ranks the
    library's whole gender pool, which is what the product does for a live
    query.  ``picked`` is a subset of ``candidates`` and is the label the Top-K
    number is measured against.
    """
    reports: list[QualityReport] = []
    predictions: list[list[str]] = []
    labels: list[set[str]] = []
    n_skipped = 0
    n_none_like = 0
    n_truncated = 0

    for query in queries:
        audio, truncated = _truncate_to_input_contract(load_audio(query.path))
        if truncated:
            n_truncated += 1
        # The gate measures the decoded signal, before the LUFS gain: clipping
        # is a property of the source, and the pipeline's limiter must not be
        # able to read as source clipping (finding C2).
        report = assess_quality(audio, SAMPLE_RATE)
        reports.append(report)
        if report.rejected:
            continue

        processed = preprocess(audio, PREPROCESS_QUERY)
        window = select_best_window(processed, SAMPLE_RATE)
        if window is None:
            n_skipped += 1
            continue
        scored = _rank_one(query, processed, window, index, model, encoder)
        if scored is None:
            n_skipped += 1
            continue
        if not query.picked:
            n_none_like += 1
        predictions.append([entry.singer_id for entry in scored])
        labels.append(set(query.picked))

    n_total = len(reports)
    n_rejected = sum(1 for report in reports if report.rejected)
    n_accepted = n_total - n_rejected
    return {
        "top1": top_k_hit(predictions, labels, k=1),
        "top3": top_k_hit(predictions, labels, k=RANK_TOP_K),
        "acceptance_rate": acceptance_rate(reports),
        "n_total": n_total,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "n_none_like": n_none_like,
        "n_skipped": n_skipped,
        "n_truncated": n_truncated,
        "n_answerable": len(predictions) - n_none_like,
    }


def _truncate_to_input_contract(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    """Cut ``audio`` to the 60 s input contract, returning ``(audio, truncated)``.

    Ruling R23: the cap belongs on the audio, not in the eval schema.  A longer
    recording is evaluated on its first ``MAX_INPUT_SEC`` seconds, and the
    caller counts it as truncated -- not as rejected.  The boundary is exact:
    exactly ``MAX_INPUT_SEC`` seconds is within the contract and is not cut.
    """
    limit = int(MAX_INPUT_SEC * SAMPLE_RATE)
    if audio.size <= limit:
        return audio, False
    return audio[:limit], True


def _rank_one(
    query: EvalQuery,
    processed: np.ndarray,
    window: tuple[float, float],
    index: LibraryIndex,
    model: CalibrationModel,
    encoder: VocalEncoder,
) -> list[ScoredSinger] | None:
    """Rank one accepted query, or return ``None`` if it cannot be scored.

    Only the best-ranked ``window`` selected by the shared automatic rule is
    scored (the product's query path).  The whole gender pool is ranked and then
    cut to ``RANK_TOP_K``, so a non-finite score anywhere in the pool skips the
    query: a ``nan`` row can sort unpredictably and must not be passed to
    ``top_k_hit`` as a measured miss just because it did not land in the Top-3.
    """
    start, end = window
    segment = processed[_sample_index(start) : _sample_index(end)]
    features = extract_features(segment, SAMPLE_RATE)
    embedding = encoder.encode(segment, SAMPLE_RATE)
    pool_size = sum(
        1 for singer in index.singers.values() if singer.gender == query.gender
    )
    scored = rank_singers(
        features,
        embedding,
        index,
        model,
        query.gender,
        top_k=max(RANK_TOP_K, pool_size),
    )
    if not all(math.isfinite(entry.score) for entry in scored):
        return None
    return scored[:RANK_TOP_K]


def _sample_index(seconds: float) -> int:
    """Convert a window boundary in seconds to a sample index.

    The same rounding as the library builder's segment slicing, so a window
    selected by the shared rule is cut to the same samples on both sides.
    """
    return int(round(seconds * SAMPLE_RATE))


def calibration_version(model: CalibrationModel) -> str:
    """Return a short hash naming ``model`` by its five stored fields.

    The formula is deliberately the whole model and nothing else: a SHA-256
    over the canonical JSON

        ``{"coef": [5 floats], "mean": [5 floats], "scale": [5 floats],
        "intercept": float, "auc": float or null}``

    with sorted keys, no whitespace and ASCII encoding, truncated to the first
    16 hex characters -- the same shape as the library version.  Every value is
    written as a Python ``float``; a non-finite value (``auc == nan`` for the
    canonical cosine-only fallback) is written as ``null``, because ``NaN`` is
    not a JSON token and two uncalibrated models must still hash.  Five
    numbers are all ``CalibrationModel.score`` reads, so equal versions mean
    equal scoring behaviour; the version does not claim the model was trained
    on particular data.
    """
    payload = {
        "auc": _canonical_float(model.auc),
        "coef": _canonical_vector(model.coef),
        "intercept": _canonical_float(model.intercept),
        "mean": _canonical_vector(model.mean),
        "scale": _canonical_vector(model.scale),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _canonical_vector(values: np.ndarray) -> list[float | None]:
    """Return one model field as a list of floats, non-finite entries as null."""
    return [_canonical_float(value) for value in np.asarray(values).reshape(-1)]


def _canonical_float(value: float) -> float | None:
    """Return ``value`` as a float, or ``None`` when it is non-finite."""
    number = float(value)
    return number if math.isfinite(number) else None
