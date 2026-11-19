"""Evaluation metrics: the numbers that decide whether the product is "准" at all.

Design doc 6.3 (验收) hangs on four numbers: Top-1/Top-3 命中率, 受理率,
Spearman correlation with human preference, and cross-device Top-3 overlap.
Every one of them has a degenerate input, and this module's whole reason to
exist is that these inputs must not be laundered into confident-looking
numbers:

- **No data is ``nan``, never ``0.0``.**  ``0.0`` is a *measurement*: "we scored
  every query and every one was wrong".  An empty eval set, a ranking that was
  never produced, a flat score sequence -- none of them are that.  A caller (or
  a report) that cannot tell the two apart will read "模型全错" out of an
  evaluation that did not run, so every metric here returns ``float("nan")``
  when it has nothing to measure.
- **Top-K is measured only where the question had an answer.**  A query whose
  annotation is "都不像" says the library does not contain their match; design
  doc 6.2 calls that a library-coverage signal, not a ranking miss.  Counting it
  as a miss would make the Top-K number depend on how many people answered
  "none of these" -- a different question -- so :func:`top_k_hit` keeps those
  queries out of its denominator (ruling R9).  The rule lives here, at the leaf,
  so the eval runner and the metric cannot disagree about the denominator.
- **A flat ranking has no rank correlation.**  ``spearman`` over a constant
  sequence is undefined (0/0); returning ``1.0`` would claim agreement with an
  order the scores cannot distinguish.  The guard is explicit rather than left
  to ``scipy`` so the result does not depend on the installed SciPy version's
  warning behaviour.
- **Broken input is not thin data.**  ``ValueError`` is raised for a misaligned
  prediction/label pair, a ``k`` below 1, or a non-finite score: those are caller
  bugs, and silently pairing by position (or correlating against a ``nan``)
  would produce a number about nothing.  ``nan`` is reserved for "the data
  cannot answer this", not for "the code cannot".

Serialisation is the runner's boundary, not this module's.  :class:`QualityReport`
carries ``snr_db == -inf`` for digital silence (see its docstring), which is not
valid strict JSON; every metric here reads only the field(s) it measures
(``acceptance_rate`` reads ``rejected`` alone), so no metric can carry a
non-finite measurement into its result, and this module deliberately exposes no
JSON helper: ``run_eval``'s output dict is the only JSON shape, and sanitising
``-inf`` at that boundary is Task 14's responsibility.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr

from voicematch.audio.quality import QualityReport

__all__ = ["acceptance_rate", "spearman", "top3_overlap", "top_k_hit"]


def top_k_hit(
    pred: Sequence[Sequence[str]], labels: Sequence[set[str]], k: int
) -> float:
    """Fraction of answerable queries whose top-``k`` contains an acceptable artist.

    ``pred[i]`` is the ranked artist ids returned for query ``i`` and
    ``labels[i]`` is that query's set of acceptable artists -- one artist, or the
    1-2 a majority of annotators picked (design doc 6.2).  Query ``i`` is a hit
    when any of its acceptable artists appears in ``pred[i][:k]``; the result is
    the mean over queries.

    Two deliberate choices decide what is measured and what counts as a hit:

    * A query with an empty ``labels`` set is annotated "都不像" and is left out
      of the denominator (ruling R9): it is evidence about library coverage, and
      folding it in as a miss would report a ranking failure the ranking did not
      cause.  When no query has an acceptable artist, nothing was measured and
      the result is ``nan`` -- not ``0.0``.
    * ``pred[i]`` is read as a *set* of the first ``k`` ids, so a ranking that
      repeats an artist cannot spend two of its ``k`` slots on one hit.  An empty
      ranking for a query that *does* have an acceptable artist is a miss, not
      ``nan``: nothing was surfaced, so nothing matched.  (This is deliberately
      unlike :func:`top3_overlap`, where a missing ranking means the agreement
      question itself cannot be asked.)  ``k`` larger than the ranking is not an
      error -- a small gender pool simply offers fewer than ``k`` candidates.

    Raises ``ValueError`` when ``k`` is below 1 or when ``pred`` and ``labels``
    have different lengths.  Either one mis-pairs queries with annotations, and
    a fraction computed from the wrong pairing says nothing about the model.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    predictions = list(pred)
    annotations = [set(label_set) for label_set in labels]
    if len(predictions) != len(annotations):
        raise ValueError(
            f"pred and labels must be the same length, got "
            f"{len(predictions)} predictions and {len(annotations)} label sets"
        )

    answerable = [i for i, acceptable in enumerate(annotations) if acceptable]
    if not answerable:
        return float("nan")
    hits = sum(
        1 for i in answerable if annotations[i] & set(predictions[i][:k])
    )
    return hits / len(answerable)


def acceptance_rate(reports: Sequence[QualityReport]) -> float:
    """Fraction of queries the quality gate did not reject.

    Design doc 6.3 requires the acceptance rate to be reported *together with*
    the Top-K hit rate (and at >= 85%), because a hit rate over the queries the
    gate happened to accept is only as meaningful as the fraction it covers.

    Only ``rejected`` is read.  That is also why a report whose ``snr_db`` is
    ``-inf`` -- what a digitally silent recording produces, per
    :class:`QualityReport` -- is a perfectly ordinary input here: this metric
    neither raises on it nor has to sanitise it, and the value is still available
    to whatever boundary owns serialisation.

    An empty sequence measured nothing, so the result is ``nan``, never ``0.0``.
    """
    items = list(reports)
    if not items:
        return float("nan")
    accepted = sum(1 for report in items if not report.rejected)
    return accepted / len(items)


def spearman(
    pred_scores: Sequence[float], human_scores: Sequence[float]
) -> float:
    """Rank correlation between the model's scores and human preference.

    Design doc 6.3 asks for this only over queries with at least two distinct
    first choices -- with a single preference there is no order to correlate
    against.  This function enforces the part of that rule it can see: if either
    sequence is constant (fewer than two distinct values, which also covers empty
    and single-point input), the correlation is undefined and the result is
    ``nan``.  It is *not* ``1.0``: a flat score sequence agrees with no ordering,
    and a flat annotation sequence contains no ordering to agree with.

    The zero-variance case is checked before SciPy is called, so the contract
    does not depend on the installed SciPy version's ``ConstantInputWarning``
    behaviour and a warnings-as-errors test run stays clean.

    Raises ``ValueError`` for a length mismatch, a non-1-D input, or any
    non-finite score: a ``nan``/``inf`` score is a broken measurement chain, and
    letting it yield ``nan`` would make it indistinguishable from "not enough
    annotations" -- the distinction this module exists to keep.
    """
    predicted = _finite_scores(pred_scores, "pred_scores")
    human = _finite_scores(human_scores, "human_scores")
    if predicted.size != human.size:
        raise ValueError(
            f"pred_scores and human_scores must be the same length, got "
            f"{predicted.size} and {human.size}"
        )
    if _has_no_variance(predicted) or _has_no_variance(human):
        return float("nan")
    return float(spearmanr(predicted, human).statistic)


def top3_overlap(a: Sequence[str], b: Sequence[str]) -> float:
    """Cross-device stability: how many of the same artists two Top-3s share.

    Design doc 6.3 fixes the denominator at 3 -- "Top-3 重合 >= 2/3" is the
    stability criterion -- so the result is ``len(intersection) / 3``.  Only the
    first three ids of each ranking are compared (a caller may hand over the
    whole scored pool), so a longer ranking cannot push the result above 1.0, and
    ids are deduplicated so a repeated artist cannot count twice.

    A missing ranking -- either side empty -- returns ``nan``: "device B produced
    no ranking" is not the same measurement as "device B's ranking disagreed",
    and ``0.0`` would report measured instability where nothing was measured.
    """
    first = list(a)[:3]
    second = list(b)[:3]
    if not first or not second:
        return float("nan")
    return len(set(first) & set(second)) / 3.0


def _finite_scores(values: Sequence[float], name: str) -> np.ndarray:
    """Return ``values`` as a 1-D float64 array, rejecting non-finite scores."""
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError(
            f"{name} must be finite; a nan/inf score has no rank to correlate"
        )
    return scores


def _has_no_variance(scores: np.ndarray) -> bool:
    """Whether ``scores`` has fewer than two distinct values (or none at all)."""
    return np.unique(scores).size < 2
