"""Tests for the aggregated feature vector.

The brief lists this file but supplied no test code (ruling R13).  These tests
pin the interface later tasks rely on: the vector carries exactly the
descriptors its four producers emit, a value a producer could not measure stays
``nan`` instead of being zero-filled, and ``DIMENSION_OF`` covers every
descriptor so no feature silently drops out of the dimension aggregation.

``DIMENSION_OF`` is checked against the literal name sets below rather than
against the producers' private key tuples: a renamed descriptor must fail here,
because the map is a hand-written interface, not something derived from the
producer code.
"""

import math
import warnings

import numpy as np
import pytest

import voicematch.features.extract as extract_module
from voicematch.features.extract import DIMENSION_OF, FeatureVector, extract_features
from voicematch.features.pitch import pitch_features
from voicematch.features.resonance import resonance_features
from voicematch.features.style import style_features
from voicematch.features.timbre import timbre_features

SR = 24000

PITCH_KEYS = {"median_f0_hz", "range_semitones", "f0_jitter"}
TIMBRE_KEYS = {f"mfcc{i}_mean" for i in range(1, 14)} | {
    "spectral_centroid_hz",
    "spectral_contrast_mean",
    "ltas_slope",
}
RESONANCE_KEYS = {"f1_hz", "f2_hz", "f3_hz", "low_band_ratio", "hnr_db"}
STYLE_KEYS = {
    "vibrato_rate_hz",
    "vibrato_extent_cents",
    "slur_ratio",
    "onset_hardness",
    "breathiness",
}


def _tone(f0=220.0, secs=1.5, sr=SR):
    t = np.arange(int(sr * secs)) / sr
    return (0.3 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def test_vector_carries_exactly_the_producer_keys():
    fv = extract_features(_tone(), SR)

    assert isinstance(fv, FeatureVector)
    assert set(fv.pitch) == PITCH_KEYS
    assert set(fv.timbre) == TIMBRE_KEYS
    assert set(fv.resonance) == RESONANCE_KEYS
    assert set(fv.style) == STYLE_KEYS


def test_vector_values_are_the_producer_values_unchanged():
    # ``extract_features`` only assembles; it must not rescale, round or drop
    # any value the producers computed on the same window.
    x = _tone()
    fv = extract_features(x, SR)

    assert fv.pitch == pitch_features(x, SR)
    assert fv.timbre == timbre_features(x, SR)
    assert fv.resonance == resonance_features(x, SR)
    assert fv.style == style_features(x, SR)


def test_nan_from_a_sub_extractor_survives_without_zero_filling(monkeypatch):
    # A descriptor the pitch producer could not measure must reach the vector
    # as ``nan``.  Filling it with 0 (or 100) would manufacture a position in
    # the percentile pool out of missing data.
    nan = float("nan")
    monkeypatch.setattr(
        extract_module, "pitch_features", lambda x, sr: {k: nan for k in PITCH_KEYS}
    )

    fv = extract_features(_tone(), SR)

    assert all(math.isnan(value) for value in fv.pitch.values())


def test_unanalysable_audio_keeps_nan_in_the_vector():
    # Digital silence is a real (if extreme) input: the pitch, resonance and
    # style producers report "not measurable", and the vector must say the same
    # rather than fail or substitute a default.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fv = extract_features(np.zeros(SR, np.float32), SR)

    assert all(math.isnan(value) for value in fv.pitch.values())
    assert all(math.isnan(value) for value in fv.resonance.values())
    assert all(math.isnan(value) for value in fv.style.values())


def test_caller_bug_still_raises():
    # The producers reject a non-mono array as a caller bug; assembly must let
    # that through instead of swallowing it.
    with pytest.raises(ValueError):
        extract_features(np.zeros((2, SR), np.float32), SR)


def test_dimension_of_covers_every_descriptor():
    expected = PITCH_KEYS | TIMBRE_KEYS | RESONANCE_KEYS | STYLE_KEYS
    assert set(DIMENSION_OF) == expected


def test_dimension_of_puts_each_descriptor_in_its_producer_group():
    for key in PITCH_KEYS:
        assert DIMENSION_OF[key] == "pitch"
    for key in TIMBRE_KEYS:
        assert DIMENSION_OF[key] == "timbre"
    for key in RESONANCE_KEYS:
        assert DIMENSION_OF[key] == "resonance"
    for key in STYLE_KEYS:
        assert DIMENSION_OF[key] == "style"


def test_dimension_of_uses_only_the_four_dimensions():
    assert set(DIMENSION_OF.values()) == {"pitch", "timbre", "resonance", "style"}
