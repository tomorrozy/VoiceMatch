"""Tests for the quality gate: which queries are scored, and why others are not."""

import math
import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from voicematch.audio.quality import QualityReport, assess_quality

SR = 24000


def test_silence_is_rejected_with_reason():
    r = assess_quality(np.zeros(24000 * 10, np.float32), 24000)
    assert r.rejected and "too_short_speech" in r.reasons


def test_clipping_is_rejected():
    x = np.sign(np.sin(np.arange(24000 * 10) * 0.05)).astype(np.float32)
    r = assess_quality(x, 24000)
    assert r.rejected and "clipped" in r.reasons


def test_clean_speech_is_accepted():
    # Syllabic signal: 22 tone bursts of 0.4 s separated by 0.1 s near-silent
    # gaps (~8.8 s of voiced audio).  A stationary pure sine has near-constant
    # frame energy, so it offers the frame-energy noise floor nothing to
    # measure and every frame-percentile SNR lands at ~0 dB.
    rng = np.random.default_rng(1)
    burst, gap = int(0.4 * SR), int(0.1 * SR)
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(burst) / SR)
    parts = []
    for _ in range(22):
        parts.append(tone)
        parts.append(rng.normal(0, 1e-4, gap))
    x = np.concatenate(parts).astype(np.float32)

    r = assess_quality(x, SR)
    assert not r.rejected and r.speech_sec >= 8.0 and r.snr_db > 10


# --- behaviour the brief demands beyond its three sample tests ---------------


def test_silence_never_raises_and_reports_every_missing_criterion():
    # Silence has no defined SNR (0/0), so the gate must guard the division
    # instead of letting numpy emit a RuntimeWarning and turn it into NaN.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = assess_quality(np.zeros(SR * 10, np.float32), SR)

    assert r.rejected
    assert not math.isnan(r.snr_db)
    assert r.snr_db == float("-inf")
    assert r.speech_sec == 0.0
    assert r.voiced_ratio == 0.0
    # Every hit is collected in rule order, not just the first one.
    assert r.reasons == ("too_short_speech", "low_snr", "low_voiced_ratio")


def test_all_violations_are_collected_in_rule_order():
    # One second of full-scale square wave followed by nine seconds of digital
    # silence: too little speech, heavily clipped, and only ~10% voiced frames.
    burst = np.zeros(SR * 10, np.float32)
    burst[: SR] = np.sign(np.sin(2 * np.pi * 220 * np.arange(SR) / SR))
    r = assess_quality(burst, SR)

    assert r.rejected
    assert r.reasons == ("too_short_speech", "clipped", "low_voiced_ratio")
    assert r.speech_sec < 8.0
    assert r.clipped_ratio == pytest.approx(0.1, abs=1e-3)
    assert r.voiced_ratio < 0.3


def test_low_snr_is_reported_alone_when_other_checks_pass():
    # Constant-envelope, unclipped signal: enough speech and fully voiced, but
    # no dynamic range, so the SNR check is the only one that fires.
    x = (0.5 * np.sign(np.sin(np.arange(SR * 10) * 0.05))).astype(np.float32)
    r = assess_quality(x, SR)

    assert r.rejected
    assert r.reasons == ("low_snr",)
    assert r.speech_sec >= 8.0
    assert r.clipped_ratio == 0.0


def test_report_is_a_frozen_dataclass_with_tuple_reasons():
    r = assess_quality(np.zeros(SR * 10, np.float32), SR)
    assert isinstance(r, QualityReport)
    assert isinstance(r.reasons, tuple)
    with pytest.raises(FrozenInstanceError):
        r.rejected = False  # type: ignore[misc]


def test_empty_audio_is_rejected_instead_of_crashing():
    r = assess_quality(np.zeros(0, np.float32), SR)
    assert r.rejected
    assert "too_short_speech" in r.reasons


def test_non_mono_audio_is_rejected_with_value_error():
    with pytest.raises(ValueError):
        assess_quality(np.zeros((2, SR * 10), np.float32), SR)


def test_non_positive_sample_rate_is_rejected_with_value_error():
    with pytest.raises(ValueError):
        assess_quality(np.zeros(SR * 10, np.float32), 0)
