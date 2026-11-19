"""Tests for per-singer prototype vectors.

The prototype is the mean of a singer's segment embeddings, re-normalised to
unit length.  The task leaves one corner open: two unit vectors can sum to
exactly zero (antipodal segments), and that mean has no direction, so "mean
then L2-normalise" would divide by zero and emit ``nan``.

The decision pinned here: an exactly-zero mean is returned as the zero vector,
unnormalised and not raised.  Downstream ranking computes
``cos = dot(query_embedding, prototype)`` (Task 12), so such a singer scores
0.0 -- finite and last in any ranking -- instead of poisoning a score with
``nan`` or aborting a whole library build over one conflicting singer.  Every
input is unit-norm, so every other mean has norm <= 1 and normalises to a
unit-norm prototype.

Also pinned: deterministic (sorted) key order, because Task 10 writes the
prototypes into a versioned manifest whose hash must be stable; ``float32``
output, matching the stored embeddings; no mutation of the input; and refusal
of mismatched inputs rather than silently truncating.
"""

import warnings

import numpy as np
import pytest

from voicematch.config import EMBED_DIM
from voicematch.embed.prototype import singer_prototypes


def _unit_rows(n, dim=8, seed=0):
    """``n`` random unit-norm float32 vectors, like encoder output."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 1.0, (n, dim)).astype(np.float64)
    return (values / np.linalg.norm(values, axis=1, keepdims=True)).astype(np.float32)


def _reference(embeddings, singer_ids):
    """Literal restatement of the rule, used as the oracle."""
    e = np.asarray(embeddings, dtype=np.float64)
    out = {}
    for singer_id in sorted(set(singer_ids)):
        rows = e[[i for i, owner in enumerate(singer_ids) if owner == singer_id]]
        mean = rows.mean(axis=0)
        norm = np.linalg.norm(mean)
        out[singer_id] = (mean / norm if norm > 0.0 else mean).astype(np.float32)
    return out


def test_prototypes_average_within_singer_and_are_normalized():
    e = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32)
    p = singer_prototypes(e, ["a", "a", "b"])
    assert set(p) == {"a", "b"}
    assert abs(np.linalg.norm(p["a"]) - 1.0) < 1e-5
    assert np.dot(p["a"], e[0]) > np.dot(p["a"], e[2])


def test_prototype_is_mean_of_its_rows_then_normalized():
    embeddings = _unit_rows(7, seed=1)
    ids = ["a", "b", "a", "c", "b", "a", "c"]

    prototypes = singer_prototypes(embeddings, ids)
    reference = _reference(embeddings, ids)

    assert set(prototypes) == set(reference)
    for singer_id, expected in reference.items():
        np.testing.assert_allclose(
            prototypes[singer_id], expected, rtol=1e-6, atol=1e-6
        )


def test_single_segment_singer_gets_its_own_vector_back():
    embeddings = _unit_rows(3, seed=2)

    prototypes = singer_prototypes(embeddings, ["solo", "other", "other"])

    np.testing.assert_allclose(prototypes["solo"], embeddings[0], rtol=1e-6, atol=1e-6)
    assert abs(np.linalg.norm(prototypes["solo"]) - 1.0) < 1e-6


def test_prototypes_are_unit_norm_float32_of_the_embedding_dimension():
    embeddings = _unit_rows(6, dim=EMBED_DIM, seed=3)

    prototypes = singer_prototypes(embeddings, ["a", "a", "a", "b", "b", "c"])

    assert set(prototypes) == {"a", "b", "c"}
    for vector in prototypes.values():
        assert vector.shape == (EMBED_DIM,)
        assert vector.dtype == np.float32
        assert abs(np.linalg.norm(vector) - 1.0) < 1e-5


def test_keys_are_sorted_for_stable_manifest_hashing():
    embeddings = _unit_rows(4, seed=4)

    assert list(singer_prototypes(embeddings, ["m", "a", "z", "a"])) == ["a", "m", "z"]

    # The key order is the sorted ids, not an artefact of input row order.
    shuffled = singer_prototypes(
        embeddings[[2, 0, 3, 1]], ["z", "m", "a", "a"]
    )
    assert list(shuffled) == ["a", "m", "z"]


def test_exactly_zero_mean_returns_the_zero_vector_without_nan_or_warning():
    antipodal = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prototypes = singer_prototypes(antipodal, ["singer", "singer"])

    assert np.array_equal(prototypes["singer"], np.zeros(2, np.float32))
    assert np.all(np.isfinite(prototypes["singer"]))
    assert not np.isnan(prototypes["singer"]).any()


def test_one_conflicting_singer_does_not_break_the_others():
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )

    prototypes = singer_prototypes(embeddings, ["opposed", "opposed", "fine"])

    assert np.array_equal(prototypes["opposed"], np.zeros(3, np.float32))
    np.testing.assert_allclose(prototypes["fine"], [0.0, 1.0, 0.0], atol=1e-7)


def test_repeated_calls_are_deterministic():
    embeddings = _unit_rows(9, seed=5)
    ids = ["b", "a"] * 4 + ["c"]

    first = singer_prototypes(embeddings, ids)
    second = singer_prototypes(embeddings, ids)

    assert list(first) == list(second)
    for singer_id in first:
        assert np.array_equal(first[singer_id], second[singer_id])


def test_mismatched_lengths_raise_instead_of_truncating():
    embeddings = _unit_rows(3, seed=6)

    with pytest.raises(ValueError, match="singer id"):
        singer_prototypes(embeddings, ["a", "b"])
    with pytest.raises(ValueError, match="singer id"):
        singer_prototypes(embeddings[:1], ["a", "b"])


def test_embeddings_must_be_two_dimensional():
    with pytest.raises(ValueError, match="2-D"):
        singer_prototypes(np.array([1.0, 0.0], np.float32), ["a", "b"])


def test_input_embeddings_are_not_mutated():
    embeddings = _unit_rows(5, seed=7)
    before = embeddings.copy()

    singer_prototypes(embeddings, ["a", "b", "a", "b", "a"])

    np.testing.assert_array_equal(embeddings, before)
