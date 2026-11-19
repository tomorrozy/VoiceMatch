"""Tests for ranking: one query in, a ranked, explained top-k out.

The brief's test only covers the empty-pool error (ruling R8 says so), so the
rest pin the behaviour Task 14 depends on: the singer whose prototype is nearer
the query comes out first, ranking is scoped to the queried gender, ties are
deterministic, the singer's feature vector is the mean of its finite segment
values, and an uncalibrated model's cosine-only ranking is marked as such
instead of passing for a probability.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from voicematch.calibrate.model import CalibrationModel
from voicematch.calibrate.pairs import FEATURE_NAMES
from voicematch.calibrate.score import ScoredSinger, rank_singers
from voicematch.config import EMBED_DIM

NAN = float("nan")


def _unit(vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` scaled to unit length, as float32 like an embedding."""
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _sigmoid(logit: float) -> float:
    """The logistic function, for hand-computed expected scores."""
    return 1.0 / (1.0 + math.exp(-logit))


def _pitch_only_model() -> CalibrationModel:
    """A model whose score depends on the pitch delta alone.

    With ``coef = [0, 1, 0, 0, 0]``, ``mean = 0``, ``scale = 1`` and
    ``intercept = 0`` the score is exactly ``sigmoid(Δpitch)``, so a test can
    read the aggregated feature vector back out of the returned score.
    """
    return CalibrationModel(
        coef=np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        mean=np.zeros(len(FEATURE_NAMES)),
        scale=np.ones(len(FEATURE_NAMES)),
        intercept=0.0,
        auc=NAN,
    )


def _by_id(ranked: list[ScoredSinger]) -> dict[str, ScoredSinger]:
    """Index a ranking by singer id so a test can assert on one candidate."""
    return {entry.singer_id: entry for entry in ranked}


def test_empty_gender_pool_raises_explicitly(make_index, make_model, fv):
    with pytest.raises(LookupError, match="no_singers_in_pool"):
        rank_singers(
            fv,
            np.ones(256, np.float32) / 16,
            make_index(genders=("male",)),
            make_model(),
            gender="female",
        )


def test_ranks_the_closer_prototype_first(make_index, make_model, fv):
    """Ruling R8: ordering is actually exercised, not just the error path."""
    index = make_index()
    # Between the two orthonormal prototypes but twice as close to s2: cos =
    # 0.894 for s2 versus 0.447 for s1.  The nearer singer is deliberately not
    # the first by id, so a ranking that sorted by id would fail this test.
    query = _unit(0.5 * index.prototypes["s1"] + index.prototypes["s2"])

    ranked = rank_singers(fv, query, index, make_model(), gender="female")

    assert [entry.singer_id for entry in ranked] == ["s2", "s1"]
    assert ranked[0].score > ranked[1].score
    assert ranked[0].name == index.singers["s2"].name


def test_ranks_only_within_the_queried_gender(make_index, make_model, fv):
    index = make_index(genders=("female", "male"))

    # The query is the male singer's prototype, but only the female pool exists.
    ranked = rank_singers(
        fv, index.prototypes["s2"], index, make_model(), gender="female"
    )

    assert [entry.singer_id for entry in ranked] == ["s1"]


def test_default_top_k_is_three_and_is_capped_by_the_pool(make_index, make_model, fv):
    five_singers = make_index(genders=("female",) * 5)

    ranked = rank_singers(
        fv, five_singers.prototypes["s1"], five_singers, make_model(), gender="female"
    )

    assert [entry.singer_id for entry in ranked] == ["s1", "s2", "s3"]

    two_singers = make_index()
    assert (
        len(
            rank_singers(
                fv,
                np.zeros(EMBED_DIM, dtype=np.float32),
                two_singers,
                make_model(),
                gender="female",
                top_k=5,
            )
        )
        == 2
    )


def test_equal_scores_break_ties_by_singer_id(make_index, make_model, fv):
    # Insertion order s2 then s1, so ascending-id order cannot be dict order.
    index = make_index(singer_ids=("s2", "s1"))
    shared = index.prototypes["s1"]
    index = replace(index, prototypes={"s1": shared, "s2": shared})

    ranked = rank_singers(
        fv, shared, index, make_model(), gender="female", top_k=2
    )

    assert [entry.singer_id for entry in ranked] == ["s1", "s2"]
    assert ranked[0].score == ranked[1].score


def test_uncalibrated_model_marks_its_ranking_as_cosine_only(
    make_index, make_model, fv
):
    index = make_index()
    query = index.prototypes["s1"]

    uncalibrated = rank_singers(
        fv, query, index, make_model(calibrated=False), gender="female"
    )

    assert [entry.singer_id for entry in uncalibrated] == ["s1", "s2"]
    assert all(entry.calibrated is False for entry in uncalibrated)
    # The canonical fallback still gives a strict order: sigmoid(cos).
    assert uncalibrated[0].score > uncalibrated[1].score

    calibrated = rank_singers(
        fv, query, index, make_model(calibrated=True), gender="female"
    )

    assert [entry.singer_id for entry in calibrated] == ["s1", "s2"]
    assert all(entry.calibrated is True for entry in calibrated)


def test_singer_vector_is_the_mean_of_its_finite_segment_values(
    make_index, make_fv
):
    descriptor = "median_f0_hz"
    index = make_index(
        features={
            "s1-00": make_fv({descriptor: 1.0}, default=NAN),
            "s1-01": make_fv({descriptor: 3.0}, default=NAN),
        }
    )
    query = make_fv({descriptor: 5.0}, default=NAN)

    ranked = rank_singers(
        query,
        index.prototypes["s1"],
        index,
        _pitch_only_model(),
        gender="female",
        top_k=2,
    )

    # The singer's value is (1 + 3) / 2 = 2, so Δpitch = |5 - 2| = 3.
    assert _by_id(ranked)["s1"].score == pytest.approx(_sigmoid(3.0), rel=1e-12)


def test_singer_vector_skips_unmeasured_segments_instead_of_poisoning_the_mean(
    make_index, make_fv
):
    descriptor = "median_f0_hz"
    index = make_index(
        features={
            "s1-00": make_fv({descriptor: 1.0}, default=NAN),
            "s1-01": make_fv({}, default=NAN),  # nothing measurable
        }
    )
    query = make_fv({descriptor: 5.0}, default=NAN)

    ranked = rank_singers(
        query,
        index.prototypes["s1"],
        index,
        _pitch_only_model(),
        gender="female",
        top_k=2,
    )

    # mean([1.0]) = 1.0, so Δpitch = 4.0; a nan-poisoned mean would give 0.5.
    assert _by_id(ranked)["s1"].score == pytest.approx(_sigmoid(4.0), rel=1e-12)


def test_a_descriptor_never_measured_contributes_no_delta(make_index, make_fv):
    descriptor = "median_f0_hz"
    index = make_index(
        features={
            "s1-00": make_fv({}, default=NAN),
            "s1-01": make_fv({}, default=NAN),
        }
    )
    query = make_fv({descriptor: 5.0}, default=NAN)

    ranked = rank_singers(
        query,
        index.prototypes["s1"],
        index,
        _pitch_only_model(),
        gender="female",
        top_k=2,
    )

    # The singer never measured the descriptor, so there is nothing comparable:
    # Δpitch = 0.0.  Reading the empty mean as 0.0 instead would give Δ = 5.
    assert _by_id(ranked)["s1"].score == pytest.approx(_sigmoid(0.0), rel=1e-12)


def test_attribution_explains_the_same_pair_row_that_was_scored(
    make_index, make_fv
):
    descriptor = "median_f0_hz"
    index = make_index(
        features={
            "s1-00": make_fv({descriptor: 1.0}, default=NAN),
            "s1-01": make_fv({descriptor: 3.0}, default=NAN),
        }
    )
    query = make_fv({descriptor: 5.0}, default=NAN)

    ranked = rank_singers(
        query,
        index.prototypes["s1"],
        index,
        _pitch_only_model(),
        gender="female",
        top_k=2,
    )

    attribution = _by_id(ranked)["s1"].attribution
    assert tuple(attribution) == FEATURE_NAMES
    assert attribution["pitch"] == pytest.approx(1.0)
    assert sum(attribution.values()) == pytest.approx(1.0)


def test_ranking_uses_the_model_score_not_the_raw_cosine(make_index, make_fv):
    descriptor = "median_f0_hz"
    index = make_index(
        features={
            "s1-00": make_fv({descriptor: 5.0}, default=NAN),
            "s1-01": make_fv({descriptor: 5.0}, default=NAN),
            "s2-00": make_fv({descriptor: 0.0}, default=NAN),
            "s2-01": make_fv({descriptor: 0.0}, default=NAN),
        }
    )
    query = make_fv({descriptor: 5.0}, default=NAN)
    # s1 is the cosine winner (the query emb is its own prototype), but the
    # pitch-only model scores s1 at sigmoid(0) and s2 at sigmoid(5): ordering is
    # the model's, proving the deltas actually enter the score and the sort.
    ranked = rank_singers(
        query,
        index.prototypes["s1"],
        index,
        _pitch_only_model(),
        gender="female",
        top_k=2,
    )

    assert [entry.singer_id for entry in ranked] == ["s2", "s1"]
    assert _by_id(ranked)["s2"].score == pytest.approx(_sigmoid(5.0), rel=1e-12)
    assert _by_id(ranked)["s1"].score == pytest.approx(_sigmoid(0.0), rel=1e-12)


@pytest.mark.parametrize(
    "embedding",
    [
        np.ones(EMBED_DIM - 1, dtype=np.float32),
        np.full(EMBED_DIM, np.nan, dtype=np.float32),
    ],
    ids=["wrong-shape", "not-finite"],
)
def test_invalid_query_embedding_raises(
    make_index, make_model, fv, embedding
):
    with pytest.raises(ValueError, match="embedding"):
        rank_singers(
            fv,
            embedding,
            make_index(),
            make_model(),
            gender="female",
        )


def test_top_k_below_one_raises(make_index, make_model, fv):
    with pytest.raises(ValueError, match="top_k"):
        rank_singers(
            fv,
            np.zeros(EMBED_DIM, dtype=np.float32),
            make_index(),
            make_model(),
            gender="female",
            top_k=0,
        )


# --- I9: non-finite prototypes are refused at the source --------------------


def _with_non_finite_prototype(index, singer_id: str):
    """Return ``index`` with one singer's prototype set to ``nan``."""
    prototypes = {
        key: np.array(vector, copy=True)
        for key, vector in index.prototypes.items()
    }
    prototypes[singer_id] = np.full(EMBED_DIM, np.nan, dtype=np.float32)
    return replace(index, prototypes=prototypes)


def test_a_non_finite_prototype_is_rejected_at_the_source(make_index, make_model, fv):
    # The index only shape-checks prototypes, so a nan prototype is a legal
    # index: the cosine against it is nan and the sort would silently be about
    # nothing.  Ranking must refuse it the way it already refuses a non-finite
    # query embedding.
    index = _with_non_finite_prototype(make_index(), "s2")

    with pytest.raises(ValueError, match="non_finite_prototype"):
        rank_singers(
            fv, index.prototypes["s1"], index, make_model(), gender="female"
        )


def test_a_non_finite_prototype_outside_the_pool_does_not_matter(
    make_index, make_model, fv
):
    # The guard is scoped to the candidates that are actually compared: a
    # broken male prototype must not stop a female query from being ranked.
    index = _with_non_finite_prototype(
        make_index(genders=("female", "male")), "s2"
    )

    ranked = rank_singers(
        fv, index.prototypes["s1"], index, make_model(), gender="female"
    )

    assert [entry.singer_id for entry in ranked] == ["s1"]
