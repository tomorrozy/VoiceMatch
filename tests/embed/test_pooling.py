"""Tests for deterministic frame pooling.

The brief lists this file but supplies no test code (ruling R13), and its
description of the reduction ("mean+std 拼接后降维到 EMBED_DIM") does not say
how.  Ruling R3 fixes it: the first 128 dimensions of the frame mean followed by
the first 128 dimensions of the frame std, exactly ``EMBED_DIM`` values, with no
random projection — a seeded projection would silently change every stored
embedding between runs and break G4.

These tests pin that reduction as a literal oracle, the weighted variant, and
the degenerate cases a real HuBERT window reaches: no frames, one frame, and
all-identical frames (zero std, and therefore no warnings and no ``nan``).
"""

import warnings

import numpy as np
import pytest

from voicematch.config import EMBED_DIM
from voicematch.embed.pooling import pool_frames

HALF = EMBED_DIM // 2


def _frames(n_frames, n_dims=HALF + 8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (n_frames, n_dims)).astype(np.float32)


def _reference_pool(frames, weights=None):
    """Literal restatement of R3, used as the oracle."""
    f = np.asarray(frames, dtype=np.float64)
    if weights is None:
        mean = f.mean(axis=0)
        std = f.std(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        mean = (w[:, None] * f).sum(axis=0) / w.sum()
        std = np.sqrt((w[:, None] * (f - mean) ** 2).sum(axis=0) / w.sum())
    return np.concatenate([mean[:HALF], std[:HALF]])


def test_pool_takes_mean_first_half_then_std_first_half():
    frames = _frames(7)
    pooled = pool_frames(frames)

    assert pooled.shape == (EMBED_DIM,)
    np.testing.assert_allclose(pooled, _reference_pool(frames), rtol=1e-5, atol=1e-5)


def test_pool_returns_float32():
    assert pool_frames(_frames(3)).dtype == np.float32


def test_pool_is_deterministic():
    frames = _frames(9, seed=3)

    assert np.array_equal(pool_frames(frames), pool_frames(frames))


def test_trailing_feature_dimensions_are_ignored():
    # The reduction is a fixed prefix, not a projection: changing dimensions
    # past the first HALF must not move a single output value.
    frames = _frames(5, n_dims=HALF + 16, seed=4)
    changed = frames.copy()
    changed[:, HALF:] += 10.0

    assert np.array_equal(pool_frames(frames), pool_frames(changed))


def test_single_frame_pools_to_itself_plus_zero_std():
    frame = _frames(1, seed=5)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pooled = pool_frames(frame.reshape(1, -1))

    np.testing.assert_allclose(pooled[:HALF], frame[:HALF], rtol=1e-6, atol=1e-6)
    assert np.all(pooled[HALF:] == 0.0)


def test_identical_frames_have_zero_std_without_warnings():
    frame = _frames(1, seed=6)[0]
    frames = np.tile(frame, (12, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pooled = pool_frames(frames)

    np.testing.assert_allclose(pooled[:HALF], frame[:HALF], rtol=1e-6, atol=1e-6)
    assert np.all(pooled[HALF:] == 0.0)


def test_no_frames_raises():
    # Pooling nothing has no mean; returning zeros would produce a vector with
    # no direction instead of failing the caller that supplied no audio.
    with pytest.raises(ValueError, match="zero frames"):
        pool_frames(np.zeros((0, HALF + 8), np.float32))


def test_weights_reweight_the_mean_and_std():
    frames = _frames(2, seed=7)
    weights = np.array([3.0, 1.0])

    pooled = pool_frames(frames, weights)

    np.testing.assert_allclose(
        pooled, _reference_pool(frames, weights), rtol=1e-5, atol=1e-5
    )
    assert not np.allclose(pooled[:HALF], frames.mean(axis=0)[:HALF])


def test_uniform_weights_match_weights_none():
    frames = _frames(6, seed=8)

    np.testing.assert_allclose(
        pool_frames(frames, np.full(6, 0.37)), pool_frames(frames), rtol=1e-5, atol=1e-5
    )


def test_weights_must_match_the_frame_count():
    with pytest.raises(ValueError, match="weights"):
        pool_frames(_frames(5), np.ones(4))


def test_weights_must_be_non_negative_and_finite():
    frames = _frames(3)
    with pytest.raises(ValueError, match="weights"):
        pool_frames(frames, np.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError, match="weights"):
        pool_frames(frames, np.array([1.0, np.nan, 1.0]))


def test_weights_must_sum_to_a_positive_value():
    with pytest.raises(ValueError, match="weights"):
        pool_frames(_frames(3), np.zeros(3))


def test_frames_must_be_two_dimensional():
    with pytest.raises(ValueError, match="2-D"):
        pool_frames(np.zeros(HALF + 8, np.float32))


def test_frames_need_at_least_half_the_embedding_dimensions():
    # 128 mean + 128 std is the whole reduction; a narrower frame tensor
    # cannot fill EMBED_DIM and must not be silently padded.
    with pytest.raises(ValueError, match="feature"):
        pool_frames(np.zeros((3, HALF - 1), np.float32))
