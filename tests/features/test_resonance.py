"""Tests for the explainable resonance descriptors.

``f1_hz``/``f2_hz``/``f3_hz`` come from Praat's Burg formant tracker, so the
synthetic fixture is an additive vowel: a 120 Hz pulse train whose harmonics are
shaped by three formant resonances.  That gives the tracker a real structure to
find instead of asking it to invent formants for a pure tone.

``low_band_ratio`` and ``hnr_db`` are checked by direction: a low-frequency tone
must put a larger share of its energy under 2 kHz than a bright tone, and a
clean periodic vowel must measure a higher harmonics-to-noise ratio than the same
vowel buried in noise.
"""

import math
import warnings

import numpy as np
import pytest

from voicematch.features.resonance import resonance_features

SR = 24000

RESONANCE_KEYS = {"f1_hz", "f2_hz", "f3_hz", "low_band_ratio", "hnr_db"}

#: (centre frequency Hz, bandwidth Hz) of the three formants in the fixture.
FORMANTS = ((700.0, 80.0), (1200.0, 90.0), (2500.0, 120.0))


def _vowel(f0=120.0, secs=2.0, sr=SR, noise=0.0):
    """Additive vowel: harmonic amplitudes shaped by the formant resonances."""
    t = np.arange(int(sr * secs)) / sr
    y = np.zeros_like(t)
    for k in range(1, int(sr / 2.0 / f0) + 1):
        frequency = k * f0
        amplitude = sum(
            1.0 / (1.0 + ((frequency - centre) / (bandwidth / 2.0)) ** 2)
            for centre, bandwidth in FORMANTS
        )
        y += amplitude * np.sin(2 * np.pi * frequency * t + 0.3 * k)
    y = 0.3 * y / np.max(np.abs(y))
    if noise:
        y = y + noise * np.random.default_rng(7).normal(0.0, 1.0, y.size)
    return y.astype(np.float32)


def test_vowel_yields_a_plausible_first_formant():
    r = resonance_features(_vowel(), SR)
    # Not zero, not NaN: the tracker must find the ~700 Hz resonance.
    assert 300.0 < r["f1_hz"] < 900.0
    assert r["f2_hz"] > r["f1_hz"]
    assert r["f3_hz"] > r["f2_hz"]


# --- contract details the brief leaves implicit ------------------------------


def test_keys_are_exactly_the_five_descriptors():
    # The key set is the interface later stages read; no extra keys may leak.
    assert set(resonance_features(_vowel(), SR)) == RESONANCE_KEYS


def test_low_band_ratio_ranks_low_frequency_energy_above_bright_energy():
    t = np.arange(int(SR * 2.0)) / SR
    low = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    bright = (0.3 * np.sin(2 * np.pi * 6000.0 * t)).astype(np.float32)

    low_ratio = resonance_features(low, SR)["low_band_ratio"]
    bright_ratio = resonance_features(bright, SR)["low_band_ratio"]

    assert low_ratio > 0.9
    assert bright_ratio < 0.01
    assert low_ratio > bright_ratio


def test_clean_vowel_has_a_higher_hnr_than_the_same_vowel_in_noise():
    clean = resonance_features(_vowel(), SR)["hnr_db"]
    noisy = resonance_features(_vowel(noise=0.3), SR)["hnr_db"]
    assert clean > noisy + 20.0


def test_silence_returns_nan_for_every_key_without_warnings():
    # Praat reports its -200 dB floor for silent audio; the feature layer must
    # instead return the NaN dictionary and let the quality gate reject the clip.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feats = resonance_features(np.zeros(SR * 3, np.float32), SR)

    assert set(feats) == RESONANCE_KEYS
    assert all(math.isnan(value) for value in feats.values())


def test_short_audio_returns_nan_without_warnings():
    # A window shorter than one analysis frame has no frame grid.  This also
    # keeps the Praat C++ layer away from sub-frame Sounds, which can abort the
    # interpreter rather than raise.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feats = resonance_features(_vowel()[:100], SR)

    assert all(math.isnan(value) for value in feats.values())


def test_empty_audio_returns_nan_for_every_key():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feats = resonance_features(np.zeros(0, np.float32), SR)

    assert set(feats) == RESONANCE_KEYS
    assert all(math.isnan(value) for value in feats.values())


def test_non_mono_audio_raises_value_error():
    with pytest.raises(ValueError):
        resonance_features(_vowel().reshape(2, -1), SR)
    with pytest.raises(ValueError):
        resonance_features(_vowel(), 0)
