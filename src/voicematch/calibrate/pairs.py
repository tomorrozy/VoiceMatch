"""Calibration pair construction: what two segments contribute to the model.

The design (spec 5.4) trains a Bradley-Terry style layer on **in-library
pairs**: two segments of the same singer are a positive pair, two segments of
different singers are a negative pair.  This module builds those pairs and
reduces each one to the five numbers the model consumes.

Why five numbers and not the 256-d embedding
--------------------------------------------
The frozen embedding already answers "who is close"; the calibration layer
answers "how close would a human call that".  Its inputs are therefore the
embedding cosine plus one delta per scoring dimension (pitch, timbre,
resonance, style), so a fitted coefficient says which kind of difference moved
a pair apart.  Ruling R2: the prototypes are 256-d vectors and cannot produce a
four-dimension delta, so ``pair_features`` takes only the two feature vectors.
The embedding cosine still enters as the first column, computed from the
per-segment embedding matrix.

The deltas are aggregated through ``DIMENSION_OF`` rather than a name list
written here: a new descriptor registered in that table enters the calibration
with no edit to this module, and a descriptor that is *not* registered cannot
silently contribute.

A delta of 0.0 means "no evidence of a difference in this dimension"
---------------------------------------------------------------------
Per dimension the delta is the mean absolute difference over the descriptors
that are finite in **both** feature vectors.  A descriptor one side could not
measure is skipped, not filled in; if no descriptor in a dimension is
comparable on both sides, the delta is ``0.0`` -- "we saw no difference" -- and
never ``nan``.  A single missing measurement must not make the pair unusable,
and ``nan`` would poison the whole logistic fit.

Fold partition (ruling R21)
---------------------------
Singers are sorted by id and assigned ``fold = index % n_folds`` with
``n_folds = min(5, max(2, n_singers // 2))``.  A negative pair is formed **only
between two singers in the same fold**; a positive pair is two different
segments of one singer (necessarily in one fold).  ``groups`` is the fold id.
Both singers of every pair live in the same fold, so the out-of-fold AUC is
singer-disjoint.

The accepted cost is real pair volume: 10 singers give 5 folds of 2, so each
singer's negatives can only come from a single partner, and a library with one
singer per fold (including any 2-singer library) yields no pairs at all.  A
correctly measured "does this generalise to an unseen singer" number is worth
more than pair volume; the pair builder says so by returning empty arrays
rather than inventing a comparison across the fold boundary.

Per-singer balance
------------------
Left alone, one singer with many segments contributes ``C(m, 2)`` positives
while a short partner contributes few negatives, so the class ratio would be an
artefact of segment counts.  For each singer both candidate lists are sampled
down to the smaller of the two, using one RNG seeded with a fixed constant
(``_PAIR_SAMPLE_SEED``), in a deterministic iteration order.  Both singers of a
negative pair generate the pair from their own side; it is emitted once, so a
row is never counted twice.  Collapsing those duplicates can leave slightly
fewer negatives than positives in a fold -- the balance is per singer before
collapsing, which is where the sample cap acts.

The caller may pass the library's own ``embeddings`` matrix or an equivalent
``(n_segments, dims)`` matrix; rows are aligned with ``index.segments`` and a
mismatch is refused rather than sliced.
"""

from __future__ import annotations

import math

import numpy as np

from voicematch.features.extract import DIMENSION_OF, FeatureVector
from voicematch.library.schema import LibraryIndex

#: Column order of a calibration row: the embedding cosine, then one delta per
#: scoring dimension.  ``CalibrationModel`` uses the same order for its
#: coefficients and attribution keys, so this tuple is the interface.
FEATURE_NAMES: tuple[str, ...] = ("cos", "pitch", "timbre", "resonance", "style")

#: The four acoustic dimensions, in ``FEATURE_NAMES`` order.
_DIMENSIONS: tuple[str, ...] = FEATURE_NAMES[1:]

#: Fixed seed for the per-singer positive/negative balance.  A fixed seed makes
#: the emitted pairs a deterministic function of the library (G4); the cost is
#: that the sample is not independent of the library version, which is fine for
#: a training set that is regenerated with the library anyway.
_PAIR_SAMPLE_SEED: int = 0


def pair_features(
    cos: float, fv_a: FeatureVector, fv_b: FeatureVector
) -> np.ndarray:
    """Return ``[cos, Δpitch, Δtimbre, Δresonance, Δstyle]`` for one pair.

    ``cos`` is passed through as column 0.  Each delta is the mean absolute
    difference of that dimension's descriptors over the descriptors finite in
    both vectors; a dimension with no such descriptor reports ``0.0`` (no
    evidence of a difference), never ``nan``.  The aggregation is driven by
    ``DIMENSION_OF``, so adding a descriptor needs no change here.

    Ruling R2: there are deliberately no prototype arguments.  A 256-d
    prototype cannot be compared per scoring dimension, and the embedding's
    contribution is already the cosine.
    """
    row = np.empty(len(FEATURE_NAMES), dtype=np.float64)
    row[0] = float(cos)
    for column, dimension in enumerate(_DIMENSIONS, start=1):
        row[column] = _dimension_delta(dimension, fv_a, fv_b)
    return row


def _dimension_delta(
    dimension: str, fv_a: FeatureVector, fv_b: FeatureVector
) -> float:
    """Mean absolute difference over descriptors measured on both sides."""
    group_a = getattr(fv_a, dimension)
    group_b = getattr(fv_b, dimension)
    total = 0.0
    count = 0
    for name, owner in DIMENSION_OF.items():
        if owner != dimension:
            continue
        if name not in group_a or name not in group_b:
            continue
        value_a = group_a[name]
        value_b = group_b[name]
        if not (math.isfinite(value_a) and math.isfinite(value_b)):
            continue
        total += abs(value_a - value_b)
        count += 1
    if count == 0:
        return 0.0
    return total / count


def build_pairs(
    index: LibraryIndex, embeddings: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(X, y, groups)`` from the library's own segment pairs.

    ``X`` is ``(n_pairs, 5)`` (see :func:`pair_features`), ``y`` is ``1`` for a
    same-singer pair and ``0`` for a different-singer pair, and ``groups`` is
    the fold id (R21).  Rows are emitted in a deterministic order, so the same
    library always builds the same matrix.

    A negative pair only joins two singers that share a fold.  Both singers of
    every pair live in the same fold, so the out-of-fold AUC is
    singer-disjoint.  A fold with fewer than two singers that have segments
    contributes nothing; if that leaves fewer than two usable folds the result
    is empty and ``fit_calibration`` reports ``nan``, not a fabricated zero.

    ``embeddings`` is ``(n_segments, dims)`` row-aligned with ``index.segments``
    and must be finite; a misaligned or non-finite matrix raises ``ValueError``
    because either one silently attributes a cosine to the wrong pair.  Cosine
    is computed as ``dot(a, b) / (||a|| ||b||)`` (the stored vectors are
    unit-norm in production, but this function does not assume it), and a
    zero-norm row gives cosine ``0.0`` -- finite and last, like the prototype
    rule in Task 9.
    """
    values = _validated_embeddings(embeddings, index)
    rows_of: dict[str, list[int]] = {}
    for segment_row, segment in enumerate(index.segments):
        rows_of.setdefault(segment.singer_id, []).append(segment_row)

    singer_ids = sorted(index.singers)
    n_folds = min(5, max(2, len(singer_ids) // 2))
    fold_of = {
        singer_id: position % n_folds
        for position, singer_id in enumerate(singer_ids)
    }

    norms = np.linalg.norm(values, axis=1)
    rng = np.random.default_rng(_PAIR_SAMPLE_SEED)
    positives: list[tuple[int, int]] = []
    negatives: list[tuple[int, int]] = []

    for fold in range(n_folds):
        fold_singers = [
            singer_id
            for singer_id in singer_ids
            if fold_of[singer_id] == fold and rows_of.get(singer_id)
        ]
        if len(fold_singers) < 2:
            continue
        for singer_id in fold_singers:
            own_rows = rows_of[singer_id]
            positive_candidates = [
                (own_rows[first], own_rows[second])
                for first in range(len(own_rows))
                for second in range(first + 1, len(own_rows))
            ]
            negative_candidates = [
                (own_row, other_row)
                for own_row in own_rows
                for other_singer in fold_singers
                if other_singer != singer_id
                for other_row in rows_of[other_singer]
            ]
            keep = min(len(positive_candidates), len(negative_candidates))
            positives.extend(_sample(positive_candidates, keep, rng))
            negatives.extend(_sample(negative_candidates, keep, rng))

    negative_rows = _deduplicate(negatives)
    pairs = [(first, second, 1) for first, second in positives]
    pairs.extend((first, second, 0) for first, second in negative_rows)

    X = np.empty((len(pairs), len(FEATURE_NAMES)), dtype=np.float64)
    y = np.empty(len(pairs), dtype=np.int64)
    groups = np.empty(len(pairs), dtype=np.int64)
    for row, (first, second, label) in enumerate(pairs):
        segment_a = index.segments[first]
        segment_b = index.segments[second]
        X[row] = pair_features(
            _cosine(values, norms, first, second),
            index.features[segment_a.segment_id],
            index.features[segment_b.segment_id],
        )
        y[row] = label
        groups[row] = fold_of[segment_a.singer_id]
    return X, y, groups


def _validated_embeddings(
    embeddings: np.ndarray, index: LibraryIndex
) -> np.ndarray:
    """Return the embedding matrix as float64 after row/finite validation."""
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            "expected a 2-D (segments, dims) embedding array, got shape "
            f"{values.shape}"
        )
    if values.shape[0] != len(index.segments):
        raise ValueError(
            f"expected one embedding row per segment, got {values.shape[0]} "
            f"rows and {len(index.segments)} segments"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "embeddings must be finite; a nan cosine would poison the "
            "calibration fit"
        )
    return values


def _sample(
    items: list[tuple[int, int]], count: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Take ``count`` items deterministically, preserving canonical order.

    The chosen positions are sorted before selection so the emitted rows do not
    depend on the sampling order; the result is a deterministic function of the
    candidate list and the seed.
    """
    if count >= len(items):
        return list(items)
    positions = np.sort(rng.choice(len(items), size=count, replace=False))
    return [items[int(position)] for position in positions]


def _deduplicate(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse a negative pair generated from both of its singers.

    A negative pair is generated once by each of its two singers (each needs it
    against its own positive count).  It is one observation, not two, so it is
    emitted once, keeping the first orientation for determinism.
    """
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for first, second in pairs:
        key = (first, second) if first < second else (second, first)
        if key in seen:
            continue
        seen.add(key)
        unique.append((first, second))
    return unique


def _cosine(
    values: np.ndarray, norms: np.ndarray, first: int, second: int
) -> float:
    """Cosine of two rows, with a zero-norm row scoring ``0.0``."""
    denominator = float(norms[first]) * float(norms[second])
    if denominator == 0.0:
        return 0.0
    return float(np.dot(values[first], values[second]) / denominator)
