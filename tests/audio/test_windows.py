"""Tests for deterministic automatic window selection.

The singer library and user queries are cut with this one rule, so the tests use
material that a pitch tracker can actually follow.  The plan's original input
for the determinism test was white noise, which has no voiced frames at all and
therefore yields no windows by construction (ruling R5); it is replaced by a
synthesized harmonic tone.
"""

import numpy as np
import pytest

from voicematch.audio.quality import assess_quality
from voicematch.audio.windows import select_best_window, select_windows

SR = 24000


def _harmonic_tone(f0=220.0, secs=40.0, sr=SR):
    """Voice-like signal: fundamental plus two harmonics at amplitude 0.3."""
    t = np.arange(int(sr * secs)) / sr
    wave = (
        np.sin(2 * np.pi * f0 * t)
        + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    return (0.3 * wave).astype(np.float32)


def test_windows_are_non_overlapping_and_deterministic():
    x = _harmonic_tone()
    a, b = select_windows(x, SR, k=3), select_windows(x, SR, k=3)
    assert a == b and len(a) == 3
    for (s1, e1), (s2, e2) in zip(a, a[1:]):
        assert e1 <= s2
    assert all(e - s > 8.0 for s, e in a)


def test_silence_yields_no_windows():
    assert select_windows(np.zeros(24000 * 30, np.float32), 24000) == []


# --- contract details the brief leaves implicit ------------------------------


def test_windows_have_the_requested_length_and_stay_inside_the_recording():
    x = _harmonic_tone()
    wins = select_windows(x, SR, k=2, window_sec=12.0)

    assert len(wins) == 2
    for start, end in wins:
        assert start >= 0.0 and end <= 40.0
        assert end - start == pytest.approx(12.0)


def test_recording_shorter_than_window_is_covered_by_one_short_window():
    # R15: a 9 s clip can satisfy the 8 s input contract; a fixed 10 s window
    # would silently select nothing and the query would never be scored.
    x = _harmonic_tone(secs=9.0)
    wins = select_windows(x, SR, k=3)

    assert len(wins) == 1
    start, end = wins[0]
    assert start == pytest.approx(0.0)
    assert end == pytest.approx(9.0)


def test_windows_with_less_than_min_speech_are_not_selected():
    # A 5 s burst in the middle of 30 s of silence is voiced but shorter than
    # MIN_SPEECH_SEC, so no 10 s window contains enough usable audio.
    x = np.zeros(SR * 30, np.float32)
    x[SR * 10 : SR * 15] = _harmonic_tone(secs=5.0)

    assert select_windows(x, SR, k=5) == []


# --- R17: the energy-based speech definition decides admission ---------------


def test_nine_second_clip_with_one_second_of_silence_yields_one_window():
    x = np.concatenate([np.zeros(SR, np.float32), _harmonic_tone(secs=8.0)])
    wins = select_windows(x, SR, k=3)

    assert len(wins) == 1
    assert wins[0] == pytest.approx((0.0, 9.0))


def test_nine_second_clip_with_noisy_lead_in_yields_one_window():
    noise = np.random.default_rng(11).normal(0, 1e-3, int(0.5 * SR)).astype(np.float32)
    x = np.concatenate([noise, _harmonic_tone(secs=8.5)])
    wins = select_windows(x, SR, k=3)

    assert len(wins) == 1
    assert wins[0] == pytest.approx((0.0, 9.0))


def test_seven_second_clip_is_still_refused():
    # Below the threshold even when the energy split counts all 6 s of tone.
    x = np.concatenate([np.zeros(SR, np.float32), _harmonic_tone(secs=6.0)])

    assert select_windows(x, SR, k=3) == []


def test_gate_accepted_clip_with_unpitched_breath_is_not_dropped():
    # The defect R17 fixes: the energy gate counts 0.8 s of audible (but
    # unpitched) breath as speech, so the clip is accepted, while pitch-based
    # admission sees only 7.2 s of f0 frames and used to return no window.
    rng = np.random.default_rng(7)
    breath = rng.normal(0, 0.05, int(0.8 * SR)).astype(np.float32)
    x = np.concatenate([np.zeros(SR, np.float32), breath, _harmonic_tone(secs=7.2)])

    assert not assess_quality(x, SR).rejected
    wins = select_windows(x, SR, k=3)
    assert len(wins) == 1
    assert wins[0] == pytest.approx((0.0, 9.0))


def test_empty_audio_yields_no_windows():
    assert select_windows(np.zeros(0, np.float32), SR) == []


def test_invalid_input_raises_value_error():
    with pytest.raises(ValueError):
        select_windows(np.zeros((2, SR * 10), np.float32), SR)
    with pytest.raises(ValueError):
        select_windows(np.zeros(SR * 10, np.float32), 0)
    with pytest.raises(ValueError):
        select_windows(np.zeros(SR * 10, np.float32), SR, window_sec=0.0)


# --- I1: the query path scores the best-ranked window, not the first --------


def test_select_best_window_is_the_highest_ranked_admitted_window():
    # The cleanest stretch is late: 25 s of a quiet harmonic tone, then 10 s
    # more audible, then 5 s clearly loudest, so only the window starting at
    # 30 s contains the best-scoring audio.  ``select_windows`` keeps its
    # result ordered by start time for the library builder, so its first entry
    # is the earliest chosen window -- the query path must score the
    # best-ranked one instead (I1).
    quiet = 0.05 * _harmonic_tone(secs=25.0)
    medium = 0.4 * _harmonic_tone(secs=10.0)
    loud = 1.2 * _harmonic_tone(secs=5.0)
    x = np.concatenate([quiet, medium, loud]).astype(np.float32)

    best = select_best_window(x, SR)
    windows = select_windows(x, SR, k=5)

    assert best is not None
    assert best[0] == pytest.approx(30.0)
    assert best in windows
    assert windows[0][0] < best[0]


def test_select_best_window_is_none_when_no_window_is_admitted():
    assert select_best_window(np.zeros(SR * 30, np.float32), SR) is None
    assert select_best_window(np.zeros(0, np.float32), SR) is None
    with pytest.raises(ValueError):
        select_best_window(np.zeros((2, SR * 10), np.float32), SR)
