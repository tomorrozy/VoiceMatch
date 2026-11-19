"""Tests for the explainable timbre descriptors.

The brief pins the key names; the tests below pin the full key set and the
directions of the descriptors that have an unambiguous physical meaning.
"""

import math
import warnings

import numpy as np
import pytest

from voicematch.features.timbre import timbre_features

SR = 24000

MFCC_KEYS = {f"mfcc{i}_mean" for i in range(1, 14)}
TIMBRE_KEYS = MFCC_KEYS | {
    "spectral_centroid_hz",
    "spectral_contrast_mean",
    "ltas_slope",
}


def _white(secs=3.0, sr=SR):
    return np.random.default_rng(3).normal(0, 0.1, int(sr * secs)).astype(np.float32)


def _dark(white):
    """Low-pass the noise with a 32-tap moving average."""
    return np.convolve(white, np.ones(32) / 32, mode="same").astype(np.float32)


def test_bright_noise_has_higher_centroid_than_dark():
    rng = np.random.default_rng(3)
    white = rng.normal(0, 0.1, 24000 * 3).astype(np.float32)
    dark = np.convolve(white, np.ones(32) / 32, mode="same").astype(np.float32)
    assert (timbre_features(white, 24000)["spectral_centroid_hz"]
            > timbre_features(dark, 24000)["spectral_centroid_hz"] * 2)


def test_feature_keys_are_stable():
    keys = set(timbre_features(np.random.default_rng(4).normal(0, .1, 24000*3).astype(np.float32), 24000))
    assert {"spectral_centroid_hz", "spectral_contrast_mean", "ltas_slope"} <= keys


# --- contract details the brief leaves implicit ------------------------------


def test_keys_are_exactly_the_thirteen_mfccs_plus_three_descriptors():
    # Later stages read these names by name, so no extra keys may leak out.
    assert set(timbre_features(_white(), SR)) == TIMBRE_KEYS


def test_darkening_lowers_the_ltas_slope():
    # The long-term average spectrum of the low-passed noise falls off faster
    # with frequency, so its dB-per-octave slope is clearly the smaller one.
    white = _white()
    assert (timbre_features(_dark(white), SR)["ltas_slope"]
            < timbre_features(white, SR)["ltas_slope"] - 1.0)


def test_empty_audio_returns_nan_for_every_key_without_warnings():
    # No frames, no spectrum: same all-NaN contract as the pitch descriptors,
    # and no librosa warning about a zero-length STFT.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feats = timbre_features(np.zeros(0, np.float32), SR)

    assert set(feats) == TIMBRE_KEYS
    assert all(math.isnan(value) for value in feats.values())


def test_non_mono_audio_raises_value_error():
    with pytest.raises(ValueError):
        timbre_features(np.zeros((2, SR * 3), np.float32), SR)
    with pytest.raises(ValueError):
        timbre_features(_white(), 0)
