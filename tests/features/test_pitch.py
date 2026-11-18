"""Tests for the explainable pitch descriptors.

The pitch range must agree with the window selector: ``windows.py`` ranks
windows with the same pYIN call, so a frame has to count as voiced here exactly
when it counts as voiced there.
"""

import math
import warnings

import numpy as np
import pytest

from voicematch.features.pitch import pitch_features

SR = 24000


def _tone(f0, secs=3.0, sr=SR):
    t = np.arange(int(sr * secs)) / sr
    return (0.3 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def _glide(f0=200.0, f1=400.0, secs=3.0, sr=SR):
    """Exponential frequency sweep: exactly one octave over ``secs``."""
    t = np.arange(int(sr * secs)) / sr
    f = f0 * (f1 / f0) ** (t / secs)
    return (0.3 * np.sin(2 * np.pi * np.cumsum(f) / sr)).astype(np.float32)


def test_median_f0_matches_synthesized_tone():
    assert abs(pitch_features(_tone(220.0), 24000)["median_f0_hz"] - 220.0) < 5.0


def test_jitter_is_near_zero_for_pure_tone():
    assert pitch_features(_tone(220.0), 24000)["f0_jitter"] < 0.01


# --- contract details the brief leaves implicit ------------------------------


def test_keys_are_exactly_the_three_descriptors():
    # Later stages (feature vector, percentiles, calibration) read these names,
    # so the key set is an interface, not an implementation detail.
    assert set(pitch_features(_tone(220.0), SR)) == {
        "median_f0_hz",
        "range_semitones",
        "f0_jitter",
    }


def test_range_semitones_recovers_the_synthesized_glide():
    # A 200 Hz -> 400 Hz sweep spans one octave.  The 5th/95th percentiles trim
    # the tracker's edge frames, so the measured span is slightly under 12 st.
    assert 9.0 < pitch_features(_glide(), SR)["range_semitones"] < 13.0


def test_unvoiced_audio_returns_nan_for_every_key_without_warnings():
    # Digital silence has no f0 frames at all.  The feature layer must not
    # raise, and must not let numpy warn about mean/std/percentile of an empty
    # slice: the quality gate owns the rejection decision, not this module.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feats = pitch_features(np.zeros(SR * 3, np.float32), SR)

    assert set(feats) == {"median_f0_hz", "range_semitones", "f0_jitter"}
    assert all(math.isnan(value) for value in feats.values())


def test_empty_audio_is_unvoiced_not_an_exception():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feats = pitch_features(np.zeros(0, np.float32), SR)

    assert all(math.isnan(value) for value in feats.values())


def test_non_mono_audio_raises_value_error():
    with pytest.raises(ValueError):
        pitch_features(np.zeros((2, SR * 3), np.float32), SR)
    with pytest.raises(ValueError):
        pitch_features(_tone(220.0), 0)
