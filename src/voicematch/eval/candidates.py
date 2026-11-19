"""The annotation candidate set for blind listening (design 6.2, finding I7).

Design 6.2 fixes the shape of every annotation task: an annotator is shown
**the system's Top-3 plus five random same-gender singers**, eight in total, in
shuffled order, and picks the one or two that sound most like the query (or
"none of these").  Two properties of that construction are what make it a
measurement rather than a demo:

- the system's own answer is always on the list, so a Top-K hit rate can be
  computed from the annotation at all;
- the fillers are random same-gender singers, not the system's next best, so
  the task does not degenerate into "rank these eight" and the annotator cannot
  simply read the system's ordering off the page.

The shuffle is seeded: the same seed regenerates the same eight candidates in
the same order, so an annotation round can be repeated or audited.  The seed is
the caller's (the reporting pipeline stores it with the round); this module does
not own a global one.

Both the embedding *and* the feature vector are inputs because the system's
Top-3 is the product's ranking, and
:func:`~voicematch.calibrate.score.rank_singers` scores a pair row built from
both sides (the design's candidate set is the calibrated ranking, not a cosine
shortlist).
"""

from __future__ import annotations

import numpy as np

from voicematch.calibrate.model import CalibrationModel
from voicematch.calibrate.score import rank_singers
from voicematch.features.extract import FeatureVector
from voicematch.library.schema import LibraryIndex

#: Ranking depth taken from the system (design 6.2: the annotation shows Top-3).
SYSTEM_TOP3: int = 3


def candidate_set(
    query_emb: np.ndarray,
    fv: FeatureVector,
    index: LibraryIndex,
    model: CalibrationModel,
    gender: str,
    k: int = 8,
    seed: int = 0,
) -> list[str]:
    """Return the shuffled annotation candidates for one query.

    The set is the system's Top-3 (``SYSTEM_TOP3``) followed by ``k - 3``
    distinct random singers of ``gender`` that are not already chosen, all
    shuffled with a generator seeded by ``seed``.  A pool smaller than ``k``
    yields the whole pool; a pool smaller than ``SYSTEM_TOP3`` yields whatever
    the ranking returns.  No singer appears twice.

    Raises ``LookupError`` with the ranking module's ``no_singers_in_pool``
    token when the library has no singer of ``gender`` (the annotation question
    cannot be asked), and ``ValueError`` when ``k`` is below 1.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    pool = sorted(
        singer.singer_id
        for singer in index.singers.values()
        if singer.gender == gender
    )
    # Let the ranking raise the shared empty-pool error before sampling, so the
    # annotation path fails with exactly the query path's token.  ``top_k`` is
    # capped by the pool inside the ranking, so it is not clamped here (a
    # ``top_k`` below 1 would raise the wrong error on an empty pool).
    system_top = [
        entry.singer_id
        for entry in rank_singers(
            fv,
            query_emb,
            index,
            model,
            gender,
            top_k=min(SYSTEM_TOP3, k),
        )
    ]

    remaining = [singer_id for singer_id in pool if singer_id not in system_top]
    n_random = min(max(0, k - len(system_top)), len(remaining))
    generator = np.random.default_rng(seed)
    if n_random:
        fillers = [
            str(singer_id)
            for singer_id in generator.choice(remaining, size=n_random, replace=False)
        ]
    else:
        fillers = []

    candidates = system_top + fillers
    order = generator.permutation(len(candidates))
    return [candidates[position] for position in order]
