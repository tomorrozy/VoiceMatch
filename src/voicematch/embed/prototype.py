"""One prototype vector per singer: the thing a query embedding is compared to.

A singer's library material is several scored segments, each with its own
unit-norm embedding.  Ranking needs one vector per singer, so the segment
embeddings are averaged and the mean is re-normalised: the prototype is the
singer's dominant direction, independent of how many segments were sampled.

Decision on the degenerate mean: unit vectors can sum to exactly zero (two
antipodal segments), and that mean has no direction to normalise.  Rather than
dividing by zero (``nan``) or raising, an exactly-zero mean is returned as the
zero vector, unnormalised.  Downstream ranking computes
``cos = dot(query_embedding, prototype)`` (Task 12), so such a singer scores
0.0 -- finite, and last in any ranking -- while a single conflicting singer
never aborts a whole library build.  Because every input is unit-norm the mean
has norm <= 1, so every other prototype is unit-norm and every cosine stays in
``[-1, 1]``.  Only an exact zero has no direction to divide out: a tiny but
non-zero mean is normalised like any other, and no fallback vector is invented
for either case.

The result is built by sorted singer id, because Task 10 writes the prototypes
into a versioned manifest whose hash must be stable: iteration order is part of
this function's contract, not an accident of input order.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def singer_prototypes(
    embeddings: np.ndarray, singer_ids: Sequence[str]
) -> dict[str, np.ndarray]:
    """Collapse per-segment embeddings into one unit-norm prototype per singer.

    ``embeddings`` is ``(n_segments, dims)`` and ``singer_ids[i]`` is the singer
    of row ``i``.  Each singer's prototype is ``mean(rows) / ||mean(rows)||``,
    returned as ``float32`` like the stored embeddings.  A singer with one
    segment gets that segment's vector back (normalising a unit vector leaves it
    unchanged); no special case is needed.  A singer whose rows sum to exactly
    zero gets the zero vector -- see the module docstring for why that is not an
    error.

    Raises ``ValueError`` if ``embeddings`` is not 2-D or has a different number
    of rows than ``singer_ids`` has ids.  Slicing to the shorter of the two
    would silently attribute segments to the wrong singer, so it is refused.
    """
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            "expected a 2-D (segments, dims) embedding array, got shape "
            f"{values.shape}"
        )
    ids = list(singer_ids)
    if values.shape[0] != len(ids):
        raise ValueError(
            "expected one singer id per embedding row, got "
            f"{values.shape[0]} embeddings and {len(ids)} singer ids"
        )

    rows_of: dict[str, list[int]] = {}
    for row, singer_id in enumerate(ids):
        rows_of.setdefault(singer_id, []).append(row)

    prototypes: dict[str, np.ndarray] = {}
    for singer_id in sorted(rows_of):
        mean = values[rows_of[singer_id]].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 0.0:
            mean = mean / norm
        prototypes[singer_id] = mean.astype(np.float32)
    return prototypes
