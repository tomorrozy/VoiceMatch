"""Ranking: one query in, a ranked, explained top-k of singers out.

The frozen embedding says who is close; the calibration model says how close a
human would call it.  This module joins the two for a single query: the query's
embedding is compared with each candidate singer's prototype, the resulting
cosine and the four feature deltas against that singer are turned into one
calibration row, and the model scores and explains that row.

Ranking is gender-scoped, and an empty pool is an error
-------------------------------------------------------
The candidate pool is every catalogued singer whose ``gender`` equals the
query's (``male``/``female``); no other singer's prototype or features are
compared.  When the filter leaves nothing -- a female query against an all-male
library -- this module raises ``LookupError`` carrying the token
``no_singers_in_pool``.  It deliberately does **not** return an empty list: an
empty list means "ranked, and nothing made the cut", whereas an empty pool
means the question could not be asked of this library, and upstream would
otherwise render it as "you look like 0 singers".

Cosine is the dot product, against a unit-norm contract
-------------------------------------------------------
Both sides are unit-norm by construction -- the encoder L2-normalises its
output, and prototypes are built from unit-norm segment vectors -- so
``dot(emb, prototype)`` *is* the cosine, and no renormalisation is applied
here.  Leaving the arithmetic alone keeps a broken encoder visible instead of
hiding it behind a repair, and it is what lets a zero prototype (two mutually
antipodal segments) score exactly ``0.0`` rather than divide by zero.  The query
embedding is still validated: a wrong shape or a non-finite value raises
``ValueError``, because either one silently ranks against the wrong
coordinates.  Candidate prototypes get the same check (finding I9): the index
validates only their shape, so a ``nan`` prototype is a legal index, and
``nan`` cosine values would sort unpredictably and be compared as if they said
something.  One is refused with ``NonFinitePrototypeError`` before any row is
scored.

A singer's feature vector is the mean of its finite segments
------------------------------------------------------------
The library stores one ``FeatureVector`` per segment, but the calibration model
consumes one pair row per singer: comparing the query with every one of a
singer's segments would produce one score per segment with no defined way to
merge them into one ranked entry.  The singer's feature vector is therefore,
**per descriptor, the mean over the segments in which that descriptor is
finite**.  A ``nan`` measurement -- the features layer's "could not measure" --
is skipped rather than read as zero (which would drag the mean toward the
bottom of the pool), and a descriptor finite in none of the singer's segments
stays ``nan``, which :func:`~voicematch.calibrate.pairs.pair_features` treats as
"no comparable evidence" and reports as a delta of ``0.0``.  The within-singer
spread is thus deliberately absent from this comparison: one singer is
represented by its central value, matching the model's one-row-per-candidate
input.  A singer with no segments aggregates to all-``nan`` and is ranked on
cosine alone, which is the same rule as a singer whose descriptors were never
measurable.

The score is a probability only when the model is calibrated
-------------------------------------------------------------
``CalibrationModel.score`` is a calibrated same-singer probability when the
model was fitted on enough singers (``model.is_calibrated`` is true).  When it
is not -- the canonical cosine-only fallback that a too-small or one-class
library produces -- the score is still a strict, monotone ranking score, but it
must not be rendered as a percentage.  Every :class:`ScoredSinger` therefore
carries a ``calibrated`` flag, so a caller cannot accidentally present a
cosine-only ranking as a calibrated probability.

Ordering and ties
-----------------
Results are sorted by descending score, ties broken by ascending ``singer_id``.
Two runs over the same library and query return the same order even when two
singers score exactly the same, and no dependence on dictionary iteration order
leaks into the result.  ``top_k`` must be at least 1 -- a caller bug raises
``ValueError`` instead of returning an unrenderable empty ranking -- and the
result is capped by the pool size, so ``top_k=3`` against two candidates returns
two.

``ScoredSinger.attribution`` is the model's attribution of the same pair row
that produced ``score``: the five keys ``cos/pitch/timbre/resonance/style``,
shares summing to 1.0 when a column moved the decision, all zero when nothing
did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from voicematch.calibrate.model import CalibrationModel
from voicematch.calibrate.pairs import FEATURE_NAMES, pair_features
from voicematch.config import EMBED_DIM
from voicematch.features.extract import DIMENSION_OF, FeatureVector
from voicematch.library.schema import LibraryIndex, Singer

#: The four acoustic dimensions, in ``FEATURE_NAMES`` order after ``cos``.
_DIMENSIONS: tuple[str, ...] = FEATURE_NAMES[1:]

#: Token identifying the "gender filter left no candidate" failure.  Upstream
#: matches on it, so the token is defined once as a constant rather than buried
#: in a longer message.
NO_SINGERS_IN_POOL: str = "no_singers_in_pool"

#: Token identifying the "a candidate prototype is not finite" failure (I9).
NON_FINITE_PROTOTYPE: str = "non_finite_prototype"


class NonFinitePrototypeError(ValueError):
    """A candidate singer's prototype is not finite.

    A ``ValueError`` because it is not content the models could not measure --
    it is a broken measurement chain, exactly like a non-finite query
    embedding.  It is its own type so the evaluation runner can report the
    query as unscorable (``n_skipped``) instead of aborting the whole run, and
    so the CLI can name it in its error envelope without matching on a string
    (finding I9).
    """


@dataclass(frozen=True)
class ScoredSinger:
    """One ranked candidate: who, how close, why, and how to read the score.

    ``score`` comes from ``CalibrationModel.score`` on the pair row built from
    the query embedding and the query's feature deltas against this singer.
    ``attribution`` explains that same row (``cos/pitch/timbre/resonance/style``
    shares; see ``CalibrationModel.attribution``).  ``calibrated`` says whether
    ``score`` is a calibrated probability (see the module docstring); it
    defaults to ``False`` so an entry constructed by hand cannot accidentally
    claim a calibrated percentage it does not carry.
    """

    singer_id: str
    name: str
    score: float
    attribution: dict[str, float]
    calibrated: bool = False


def rank_singers(
    fv: FeatureVector,
    emb: np.ndarray,
    index: LibraryIndex,
    model: CalibrationModel,
    gender: Literal["male", "female"],
    top_k: int = 3,
) -> list[ScoredSinger]:
    """Return the ``top_k`` singers of ``gender`` closest to one query, scored.

    ``fv`` is the query's feature vector and ``emb`` its ``(EMBED_DIM,)``
    encoder output.  For every catalogued singer of ``gender`` the pair row is
    ``[cos(emb, prototype), Δpitch, Δtimbre, Δresonance, Δstyle]`` against the
    singer's aggregate feature vector (the per-descriptor mean over its finite
    segments; see the module docstring), and the model's ``score`` and
    ``attribution`` of that row become the result.

    Results are ordered by descending score with ties broken by ascending
    ``singer_id``.  An empty gender pool raises ``LookupError`` with the token
    ``no_singers_in_pool``; a malformed ``emb`` or a ``top_k`` below 1 raises
    ``ValueError``, and a candidate prototype with a non-finite value raises
    :class:`NonFinitePrototypeError` (a ``ValueError``; I9) instead of sorting
    the pool on ``nan``.
    """
    embedding = _validated_embedding(emb)
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    pool = [singer for singer in index.singers.values() if singer.gender == gender]
    if not pool:
        raise LookupError(
            f"{NO_SINGERS_IN_POOL}: no singers with gender {gender!r} in the library"
        )
    for singer in pool:
        _validated_prototype(singer, index)

    features_of = _segment_features_by_singer(index)
    scored = [
        _score_singer(singer, embedding, fv, index, model, features_of)
        for singer in pool
    ]
    # Descending score; equal scores tie-break on the id, so two runs on equal
    # scores return one order instead of leaking dictionary insertion order.
    scored.sort(key=lambda entry: (-entry.score, entry.singer_id))
    return scored[:top_k]


def _score_singer(
    singer: Singer,
    embedding: np.ndarray,
    fv: FeatureVector,
    index: LibraryIndex,
    model: CalibrationModel,
    features_of: dict[str, list[FeatureVector]],
) -> ScoredSinger:
    """Build one candidate's pair row, then score and explain it."""
    prototype = np.asarray(index.prototypes[singer.singer_id], dtype=np.float64)
    cosine = float(np.dot(embedding, prototype))
    singer_vector = _mean_feature_vector(features_of.get(singer.singer_id, []))
    row = pair_features(cosine, fv, singer_vector)
    return ScoredSinger(
        singer_id=singer.singer_id,
        name=singer.name,
        score=float(model.score(row)),
        attribution=model.attribution(row),
        calibrated=model.is_calibrated,
    )


def _validated_embedding(emb: np.ndarray) -> np.ndarray:
    """Return the query embedding as float64 after shape/finite validation."""
    values = np.asarray(emb, dtype=np.float64)
    if values.shape != (EMBED_DIM,):
        raise ValueError(
            f"expected a ({EMBED_DIM},) query embedding, got shape {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "query embedding must be finite; a nan cosine would poison the ranking"
        )
    return values


def _validated_prototype(singer: Singer, index: LibraryIndex) -> None:
    """Refuse a candidate whose prototype cannot produce a finite cosine.

    The index only checks prototype *shape* (a ``nan`` prototype is a legal
    index: one test builds one on purpose), so the finiteness check has to
    happen where the vectors are compared.  Raising before any row is scored
    means a broken library fails with one clear error instead of sorting the
    pool on ``nan`` and returning rows the CLI would render as ``"score":
    null`` in a meaningless order (finding I9).
    """
    prototype = np.asarray(index.prototypes[singer.singer_id], dtype=np.float64)
    if not np.all(np.isfinite(prototype)):
        raise NonFinitePrototypeError(
            f"{NON_FINITE_PROTOTYPE}: prototype for singer "
            f"{singer.singer_id!r} must be finite; a nan cosine would poison the "
            "ranking"
        )


def _segment_features_by_singer(
    index: LibraryIndex,
) -> dict[str, list[FeatureVector]]:
    """Group the index's per-segment feature vectors by singer, in index order.

    The grouping walks ``index.segments``, so the list for each singer is in the
    index's own segment order: the mean below adds in the same order on every
    run, which keeps the ranking a deterministic function of the library (G4).
    """
    grouped: dict[str, list[FeatureVector]] = {}
    for segment in index.segments:
        grouped.setdefault(segment.singer_id, []).append(
            index.features[segment.segment_id]
        )
    return grouped


def _mean_feature_vector(vectors: list[FeatureVector]) -> FeatureVector:
    """Mean one singer's per-segment features, descriptor by descriptor.

    Each descriptor is the mean of its finite values over ``vectors``.  A value
    of ``nan`` -- "could not measure" -- is skipped, not read as zero, so one
    unmeasurable segment cannot drag the singer's value toward the pool bottom.
    A descriptor finite in none of the segments stays ``nan``: the mean has no
    evidence, and ``pair_features`` reports the resulting delta as ``0.0``
    rather than inventing a value.
    """
    groups: dict[str, dict[str, float]] = {
        dimension: {} for dimension in _DIMENSIONS
    }
    for name, dimension in DIMENSION_OF.items():
        values = [
            group[name]
            for group in (getattr(vector, dimension) for vector in vectors)
            if name in group and math.isfinite(group[name])
        ]
        groups[dimension][name] = (
            math.fsum(values) / len(values) if values else float("nan")
        )
    return FeatureVector(**groups)
