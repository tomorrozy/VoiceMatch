"""Tests for deterministic automatic window selection.

The singer library and user queries are cut with this one rule, so the tests use
material that a pitch tracker can actually follow.  The plan's original input
for the determinism test was white noise, which has no voiced frames at all and
therefore yields no windows by construction (ruling R5); it is replaced by a
synthesized harmonic tone.
"""

import numpy as np
import pytest

from voicematch.audio.windows import select_windows

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


def test_empty_audio_yields_no_windows():
    assert select_windows(np.zeros(0, np.float32), SR) == []


def test_invalid_input_raises_value_error():
    with pytest.raises(ValueError):
        select_windows(np.zeros((2, SR * 10), np.float32), SR)
    with pytest.raises(ValueError):
        select_windows(np.zeros(SR * 10, np.float32), 0)
    with pytest.raises(ValueError):
        select_windows(np.zeros(SR * 10, np.float32), SR, window_sec=0.0)
