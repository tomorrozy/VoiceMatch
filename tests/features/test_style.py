"""Tests for the explainable singing-style descriptors.

The brief supplied the first two tests; the rest pin the key set, the unusable
input contract, and the direction of the remaining descriptors.

One ruling on the brief's numbers: Step 3 defines
``vibrato_extent_cents = 2 * sqrt(2) * std(filtered)``, which is a *peak-to-peak*
estimate (a sinusoid of amplitude A has std A/sqrt(2), so 2*sqrt(2)*std = 2A).
The planned assertion ``50 < extent < 150`` was written against the amplitude-only
reading, and the plan was never executed, so it contradicts the algorithm the
same plan prescribes.  The algorithm is implemented exactly as written (and its
docstring says peak-to-peak); only the assertion bound is corrected.  The
scaling test below pins the peak-to-peak semantics: a ±50-cent vibrato must
measure half a ±100-cent one, so a formula that is off by a constant factor
cannot pass by happening to fit one absolute range.
"""

import math
import warnings

import numpy as np
import pytest

from voicematch.features.style import style_features

SR = 24000

STYLE_KEYS = {
    "vibrato_rate_hz",
    "vibrato_extent_cents",
    "slur_ratio",
    "onset_hardness",
    "breathiness",
}


def _vibrato(base=220.0, rate=6.0, cents=100.0, secs=3.0, sr=24000):
    t = np.arange(int(sr * secs)) / sr
    f = base * 2 ** ((cents / 1200) * np.sin(2 * np.pi * rate * t))
    return (0.3 * np.sin(2 * np.pi * np.cumsum(f) / sr)).astype(np.float32)


def _tone(f0=220.0, secs=3.0, sr=SR):
    t = np.arange(int(sr * secs)) / sr
    return (0.3 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def test_vibrato_rate_and_extent_are_recovered():
    r = style_features(_vibrato(), 24000)
    assert abs(r["vibrato_rate_hz"] - 6.0) < 1.0
    # A ±100-cent vibrato is 200 cents peak-to-peak; the band-passed track
    # measures 180 because the filter's passband and the frame grid trim the
    # extremes.  (The brief's 50-150 bound assumed an amplitude-only reading.)
    assert 150 < r["vibrato_extent_cents"] < 250


def test_steady_tone_has_negligible_extent():
    t = np.arange(72000) / 24000
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    assert style_features(x, 24000)["vibrato_extent_cents"] < 10


# --- contract details the brief leaves implicit ------------------------------


def test_vibrato_extent_scales_with_vibrato_depth():
    # Pins the peak-to-peak semantics: the descriptor is linear in depth, so a
    # formula that is off by a constant factor cannot pass this test.
    shallow = style_features(_vibrato(cents=50.0), SR)["vibrato_extent_cents"]
    deep = style_features(_vibrato(cents=100.0), SR)["vibrato_extent_cents"]
    assert 0.35 < shallow / deep < 0.65


def test_steady_tone_records_zero_vibrato_rate():
    # A pitch track too flat for a rate (extent below 10 cents) must not report
    # the noise-peak frequency of the band-passed jitter.
    assert style_features(_tone(), SR)["vibrato_rate_hz"] == 0.0


def test_keys_are_exactly_the_five_descriptors():
    # Later stages read these names by name, so no extra keys may leak out.
    assert set(style_features(_tone(), SR)) == STYLE_KEYS


def test_slur_ratio_is_zero_for_a_steady_tone():
    assert style_features(_tone(), SR)["slur_ratio"] < 0.1


def test_slur_ratio_recovers_a_legato_glissando():
    # 6 semitones per second for 2 s: a connected portamento, well inside the
    # descriptor's 2-12 st/s window and far from any note change.  The unit is
    # semitones per second (not per frame), so the value does not depend on the
    # hop length.
    t = np.arange(int(SR * 2.0)) / SR
    f = 220.0 * 2.0 ** (6.0 * t / 12.0)
    x = (0.3 * np.sin(2 * np.pi * np.cumsum(f) / SR)).astype(np.float32)
    assert style_features(x, SR)["slur_ratio"] > 0.5


def _gated_tone(ramp_sec, secs=2.0, sr=SR):
    """220 Hz tone gated on at 0.5 s, with a linear attack of ``ramp_sec``."""
    t = np.arange(int(sr * secs)) / sr
    envelope = np.zeros_like(t)
    start = int(0.5 * sr)
    length = max(1, int(ramp_sec * sr))
    envelope[start : start + length] = np.linspace(0.0, 1.0, length)
    envelope[start + length :] = 1.0
    return (0.3 * np.sin(2 * np.pi * 220.0 * t) * envelope).astype(np.float32)


def test_hard_attack_is_harder_than_a_slow_fade():
    hard = style_features(_gated_tone(0.005), SR)["onset_hardness"]
    soft = style_features(_gated_tone(0.5), SR)["onset_hardness"]
    assert hard > 0.03
    assert soft < 0.01


def test_breathiness_is_higher_for_broadband_noise_than_for_a_low_tone():
    white = np.random.default_rng(3).normal(0, 0.1, SR * 3).astype(np.float32)
    noisy = style_features(white, SR)["breathiness"]
    tonal = style_features(_tone(), SR)["breathiness"]
    assert noisy > 1.0
    assert tonal < 0.01


def test_silence_returns_nan_for_every_key_without_warnings():
    # Digital silence has no f0 frames, no envelope rise and no band energy:
    # the feature layer must return the NaN dictionary, not raise or warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feats = style_features(np.zeros(SR * 3, np.float32), SR)

    assert set(feats) == STYLE_KEYS
    assert all(math.isnan(value) for value in feats.values())


def test_short_audio_returns_nan_without_warnings():
    # A window shorter than one analysis frame has no frame grid; librosa would
    # warn about n_fft > signal length, so the module must short-circuit.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feats = style_features(_tone()[:100], SR)

    assert all(math.isnan(value) for value in feats.values())


def test_empty_audio_returns_nan_for_every_key():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feats = style_features(np.zeros(0, np.float32), SR)

    assert set(feats) == STYLE_KEYS
    assert all(math.isnan(value) for value in feats.values())


def test_non_mono_audio_raises_value_error():
    with pytest.raises(ValueError):
        style_features(np.zeros((2, SR * 3), np.float32), SR)
    with pytest.raises(ValueError):
        style_features(_tone(), 0)
