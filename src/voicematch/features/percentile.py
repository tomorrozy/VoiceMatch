"""Map raw descriptors onto 0-100 percentiles of a same-gender reference pool.

The report never shows the physical value of a descriptor: a dimension score of
87 has to mean "you sit above 87% of the reference pool", otherwise the number
cannot be explained and cannot stay stable across devices.  So the percentile
is defined relative to a fixed pool, and the pool is fitted once per gender.

Ruling R4 (overrides the plan text): the pool minimum ranks 0 and the maximum
ranks 100, via ``searchsorted(side="left") / (n - 1) * 100``.  The plan's
``side="right" / n * 100`` never yields 0 for the minimum and contradicts the
plan's own expected values.  Queries outside the pool clamp to 0/100, and a
missing (``nan``) descriptor stays ``nan``: missing is "not comparable", neither
"most similar" nor "least similar".
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from voicematch.features.extract import FeatureVector

#: The ``FeatureVector`` field names, in report order.  Descriptors themselves
#: are discovered from the groups, so the four dimensions are covered here and a
#: group that grows a descriptor needs no change to this module.
_DIMENSIONS = ("pitch", "timbre", "resonance", "style")


@dataclass(frozen=True)
class PercentileModel:
    """The sorted reference values of one same-gender pool, keyed by descriptor.

    Each array holds every finite observed value of that descriptor in the pool,
    sorted ascending, so it can be searched with ``np.searchsorted``.
    """

    reference: dict[str, np.ndarray]


def fit_percentiles(vectors: Sequence[FeatureVector]) -> PercentileModel:
    """Fit the reference distribution from one same-gender pool of vectors.

    ``vectors`` are the feature vectors of every library window in a single
    gender pool.  The same-gender pool is a property of the library, not of the
    fit function: the caller chooses the pool and fits one model per gender, so
    there is deliberately no ``gender`` parameter here.  A percentile is only
    meaningful against the pool it was fitted on.

    Non-finite pool values are dropped: a descriptor one pool member could not
    measure is not a rank in that member's distribution.  A descriptor with no
    finite value anywhere in the pool is absent from the model, and later maps
    to ``nan`` rather than to a fabricated 0 or 100.
    """
    samples: dict[str, list[float]] = {}
    for vector in vectors:
        for dimension in _DIMENSIONS:
            for name, value in getattr(vector, dimension).items():
                samples.setdefault(name, []).append(float(value))

    reference: dict[str, np.ndarray] = {}
    for name, values in samples.items():
        finite = np.asarray(
            [value for value in values if np.isfinite(value)], dtype=np.float64
        )
        if finite.size:
            reference[name] = np.sort(finite)
    return PercentileModel(reference=reference)


def to_percentiles(fv: FeatureVector, model: PercentileModel) -> dict[str, float]:
    """Return each descriptor of ``fv`` as a 0-100 percentile of its pool.

    The result has one entry per descriptor the vector carries.  A descriptor
    the pool never measured has no rank and maps to ``nan``; a non-finite input
    value stays ``nan``.  A one-member pool cannot rank anything, so every value
    is the middle of the scale (50).
    """
    return {
        name: _percentile(model.reference.get(name), float(value))
        for dimension in _DIMENSIONS
        for name, value in getattr(fv, dimension).items()
    }


def _percentile(reference: np.ndarray | None, value: float) -> float:
    """Return ``value``'s rank within ``reference`` on a 0-100 scale."""
    if reference is None or reference.size == 0:
        # The pool has no usable value for this descriptor: no rank exists.
        return float("nan")
    if not np.isfinite(value):
        # A missing measurement is not a position in the distribution.
        return float("nan")
    if reference.size == 1:
        # A single reference member has no spread, only a middle.
        return 50.0

    index = int(np.searchsorted(reference, value, side="left"))
    percentile = index / (reference.size - 1) * 100.0
    # A query above the pool ceiling ranks past the last member; the report
    # promises 0-100, so clamp instead of returning more than 100.
    return min(percentile, 100.0)
