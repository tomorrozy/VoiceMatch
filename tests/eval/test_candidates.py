"""Tests for the annotation candidate set (design 6.2, finding I7).

Every evaluation query is shown to annotators as **the system's Top-3 plus five
random same-gender singers**, shuffled, so a Top-K hit rate is measurable while
the annotation task is not just "rate the system's answer" (design 6.2).  The
plan assigned generating that set to this package and nothing did it, so the
properties pinned here are:

- the system's Top-3 is always in the set;
- the rest are distinct singers of the query's gender, and no singer repeats;
- a given seed gives the same set in the same order (the shuffle is
  reproducible), so an annotation round can be regenerated exactly.

The library is hand-built (as in the calibration tests) so which singer ranks
where is authored by the test instead of extracted from audio.
"""

import numpy as np
import pytest

from voicematch.calibrate.model import CalibrationModel
from voicematch.calibrate.pairs import FEATURE_NAMES
from voicematch.config import EMBED_DIM
from voicematch.eval.candidates import SYSTEM_TOP3, candidate_set
from voicematch.features.extract import DIMENSION_OF, FeatureVector
from voicematch.library.schema import LibraryIndex, Segment, Singer

#: One singer per prototype coordinate, so the ranking is a total order.
_SINGER_IDS: tuple[str, ...] = tuple(f"f{position:02d}" for position in range(10))

#: The system's Top-3 for a query identical to the first prototype.
_EXPECTED_TOP3: list[str] = ["f00", "f01", "f02"]


def _feature_vector(value: float = 0.5) -> FeatureVector:
    """A ``FeatureVector`` covering exactly the registered descriptors."""
    groups: dict[str, dict[str, float]] = {
        "pitch": {},
        "timbre": {},
        "resonance": {},
        "style": {},
    }
    for name, dimension in DIMENSION_OF.items():
        groups[dimension][name] = value
    return FeatureVector(**groups)


def _index() -> LibraryIndex:
    """Ten female singers whose prototypes are the first ten basis vectors."""
    singers = {
        singer_id: Singer(singer_id, singer_id.upper(), "female", "pop")
        for singer_id in _SINGER_IDS
    }
    segments = tuple(
        Segment(f"{singer_id}-00", singer_id, 0.0, 10.0)
        for singer_id in _SINGER_IDS
    )
    prototypes = {}
    for position, singer_id in enumerate(_SINGER_IDS):
        vector = np.zeros(EMBED_DIM, dtype=np.float32)
        vector[position] = 1.0
        prototypes[singer_id] = vector
    return LibraryIndex(
        version="candidate-fixture",
        model_id="fixture-encoder",
        singers=singers,
        segments=segments,
        prototypes=prototypes,
        features={
            segment.segment_id: _feature_vector() for segment in segments
        },
        embeddings=np.zeros((len(segments), EMBED_DIM), dtype=np.float32),
    )


def _cosine_model() -> CalibrationModel:
    """The canonical cosine-only fallback: score = sigmoid(cosine)."""
    return CalibrationModel(
        coef=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        mean=np.zeros(len(FEATURE_NAMES)),
        scale=np.ones(len(FEATURE_NAMES)),
        intercept=0.0,
        auc=float("nan"),
    )


def _query():
    """The first singer's prototype and a feature vector with zero deltas."""
    index = _index()
    return index, index.prototypes["f00"], _feature_vector()


def test_the_candidate_set_is_the_top3_plus_random_same_gender_singers():
    index, embedding, fv = _query()

    candidates = candidate_set(
        embedding, fv, index, _cosine_model(), "female", k=8, seed=0
    )

    assert len(candidates) == 8
    assert len(set(candidates)) == 8
    assert set(_EXPECTED_TOP3) <= set(candidates)
    assert set(candidates) <= set(index.singers)


def test_the_candidate_set_is_deterministic_for_a_seed():
    index, embedding, fv = _query()

    first = candidate_set(
        embedding, fv, index, _cosine_model(), "female", k=8, seed=7
    )
    second = candidate_set(
        embedding, fv, index, _cosine_model(), "female", k=8, seed=7
    )

    assert first == second
    # The same call with a different seed is deterministic too (its members may
    # legitimately coincide with seed 7's; only reproducibility is promised).
    other = candidate_set(
        embedding, fv, index, _cosine_model(), "female", k=8, seed=8
    )
    assert other == candidate_set(
        embedding, fv, index, _cosine_model(), "female", k=8, seed=8
    )


def test_a_seed_fixes_the_shuffled_order():
    # Golden order for seed 0: pins both determinism and the fact that the set
    # is shuffled (the unshuffled order starts with the Top-3).
    index, embedding, fv = _query()

    candidates = candidate_set(
        embedding, fv, index, _cosine_model(), "female", k=8, seed=0
    )

    assert candidates == [
        "f05",
        "f02",
        "f06",
        "f07",
        "f04",
        "f01",
        "f09",
        "f00",
    ]
    assert SYSTEM_TOP3 == 3


def test_a_smaller_k_returns_the_system_top_k_only():
    index, embedding, fv = _query()

    assert (
        candidate_set(
            embedding, fv, index, _cosine_model(), "female", k=2, seed=0
        )
        == _EXPECTED_TOP3[:2]
    )


def test_the_pool_must_not_be_empty():
    index = _index()

    with pytest.raises(LookupError, match="no_singers_in_pool"):
        candidate_set(
            index.prototypes["f00"],
            _feature_vector(),
            index,
            _cosine_model(),
            "male",
        )


def test_k_below_one_is_a_caller_bug():
    index, embedding, fv = _query()

    with pytest.raises(ValueError, match="k"):
        candidate_set(
            embedding, fv, index, _cosine_model(), "female", k=0
        )
