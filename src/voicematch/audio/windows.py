"""Deterministic automatic selection of the scoring windows.

Both the singer library and user queries are cut with this single rule, so the
library cannot be hand-picked into looking better than a spontaneous user
recording (design doc §5.5).  The rule rewards windows that are continuously
voiced, pitch-stable and energetic; nothing here uses random state, so the same
audio always produces the same windows.

Window admission and window ranking deliberately use different measurements:
whether a window contains enough usable audio is decided by the energy-based
``speech_intervals`` shared with the quality gate, while the pitch track only
ranks the admitted windows.  That way a recording the gate accepts is not
silently dropped just because a breath or a consonant lowered its pitched frame
count (ruling R17).

The window length defaults to ``WINDOW_SEC`` (10 s) and is shortened to the
recording length for shorter clips, so a 9 s recording that the quality gate
accepts still yields its whole-clip window.

Two entry points, one ranking
-----------------------------
:func:`select_windows` returns the chosen windows **ordered by start time**,
which is what the library builder wants: its segment list follows the material
in order.  :func:`select_best_window` returns the single **highest-ranked**
admitted window, which is what the query path scores.  Taking ``[0]`` of the
start-sorted list instead would score the earliest capable stretch rather than
the cleanest one (finding I1).
"""

import numpy as np

from voicematch.audio.pitch import F0_MAX_HZ, F0_MIN_HZ, pitch_track
from voicematch.audio.quality import speech_intervals
from voicematch.config import HOP_LENGTH, MIN_SPEECH_SEC, WINDOW_SEC

# Candidate windows start every second (plan Task 4, Step 3).  With a 10 s
# window, neighbouring candidates share 90 % of their audio, so the greedy pass
# has to skip overlaps instead of stacking near-duplicates.
SLIDE_SEC = 1.0

# ``F0_MIN_HZ``/``F0_MAX_HZ`` are re-exported from :mod:`voicematch.audio.pitch`:
# they are the range the ranking score, the quality gate and the f0 descriptors
# all use, so "voiced" means the same thing everywhere.


def select_windows(
    x: np.ndarray,
    sr: int,
    k: int = 5,
    window_sec: float = WINDOW_SEC,
) -> list[tuple[float, float]]:
    """Return at most ``k`` non-overlapping ``(start_sec, end_sec)`` windows.

    Candidate windows are ``window_sec`` long, shortened to the recording length
    when the recording is shorter, and are laid on ``x`` every second.  Each is
    scored with ``voiced_ratio * pitch_stability * rms``:

    - ``voiced_ratio`` — share of f0 frames inside the window that a pitch
      tracker marked as voiced;
    - ``pitch_stability`` — ``1 / (1 + std(f0) / mean(f0))`` over those voiced
      frames, i.e. 1.0 for a perfectly steady pitch;
    - ``rms`` — root-mean-square level of the window.

    Admission is separate from ranking: a window is eligible only when the
    energy-based speech intervals shared with the quality gate cover at least
    ``MIN_SPEECH_SEC`` inside it (ruling R17).  The pitch track never decides
    acceptance, and an all-zero input yields no windows without a dedicated code
    path.  A ``window_sec`` shorter than ``MIN_SPEECH_SEC`` can never qualify.

    Eligible windows are greedily taken in descending score order and any
    window overlapping an already selected one is skipped, until ``k`` windows
    are chosen or no candidates are left.  Because candidates are taken by score
    rather than packed to maximise their count, fewer than ``k`` windows can be
    returned even when more non-overlapping windows would fit; the result is
    ordered by start time (ascending) for the library builder, so it is *not* a
    ranking -- use :func:`select_best_window` to score the best window.  It is
    empty when the recording is empty or contains no usable speech stretch
    (including clips shorter than ``MIN_SPEECH_SEC``, whose only window is the
    clip itself).
    """
    y = _validated_audio(x, sr, window_sec)
    if k <= 0:
        return []
    chosen: list[tuple[float, float]] = []
    for (start, end), _score in _ranked_windows(y, sr, window_sec):
        overlaps = any(
            start < chosen_end and chosen_start < end
            for chosen_start, chosen_end in chosen
        )
        if overlaps:
            continue
        chosen.append((start, end))
        if len(chosen) == k:
            break
    chosen.sort()
    return chosen


def select_best_window(
    x: np.ndarray, sr: int, window_sec: float = WINDOW_SEC
) -> tuple[float, float] | None:
    """Return the highest-ranked admitted window, or ``None`` if none qualifies.

    This is the query path's window: one recording is scored on the best stretch
    the shared automatic rule admits, not on the earliest one.  ``None`` means
    the recording passed the gate but has no window with enough usable speech --
    the caller's ``no_usable_window`` state, which is distinct from a quality
    rejection.  Invalid input (non-mono audio, a non-positive rate or window)
    raises ``ValueError`` exactly as :func:`select_windows` does.
    """
    y = _validated_audio(x, sr, window_sec)
    ranked = _ranked_windows(y, sr, window_sec)
    return ranked[0][0] if ranked else None


def _validated_audio(
    x: np.ndarray, sr: int, window_sec: float
) -> np.ndarray:
    """Validate the input arguments shared by both entry points."""
    y = np.asarray(x, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"expected mono 1-D audio, got shape {y.shape}")
    if sr <= 0:
        raise ValueError(f"sample rate must be positive, got {sr}")
    if window_sec <= 0:
        raise ValueError(f"window_sec must be positive, got {window_sec}")
    return y


def _ranked_windows(
    y: np.ndarray, sr: int, window_sec: float
) -> list[tuple[tuple[float, float], float]]:
    """Return every admitted window with its score, best first.

    ``[(window, score), ...]`` sorted by descending score with ties broken by
    ascending start, so both entry points rank windows identically.  The window
    length is shortened to the recording length for short clips (R15).
    """
    duration = y.size / sr
    if duration <= 0.0:
        return []
    window_len = min(window_sec, duration)
    starts = _candidate_starts(duration, window_len)
    if starts.size == 0:
        return []

    f0 = pitch_track(y, sr)
    frame_times = np.arange(f0.size) * HOP_LENGTH / sr
    intervals = speech_intervals(y, sr)
    ranked: list[tuple[tuple[float, float], float]] = []
    for start in starts:
        start_sec = float(start)
        if _speech_seconds(intervals, start_sec, window_len, sr) < MIN_SPEECH_SEC:
            continue
        score = _window_score(y, sr, f0, frame_times, start_sec, window_len)
        ranked.append(((start_sec, start_sec + window_len), score))

    ranked.sort(key=lambda entry: (-entry[1], entry[0][0]))
    return ranked


def _candidate_starts(duration_sec: float, window_sec: float) -> np.ndarray:
    """Return the start time of every full window that fits in the recording."""
    max_start = duration_sec - window_sec
    if max_start < 0.0:
        return np.empty(0, dtype=np.float64)
    # The tolerance admits a window that lands exactly on the end of the
    # recording despite floating-point round-off in ``duration_sec``.
    count = int(np.floor(max_start / SLIDE_SEC + 1e-9)) + 1
    return np.arange(count, dtype=np.float64) * SLIDE_SEC


def _speech_seconds(
    intervals: list[tuple[int, int]], start_sec: float, window_sec: float, sr: int
) -> float:
    """Return the energy-based speech duration inside one window, in seconds.

    ``intervals`` are the sample ranges from the shared ``speech_intervals``
    helper; only their overlap with the window counts.  The total is clamped to
    the window length, so a fully voiced window cannot report more speech than
    it contains.
    """
    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + window_sec) * sr)
    speech_samples = 0
    for seg_start, seg_end in intervals:
        overlap = min(seg_end, end_sample) - max(seg_start, start_sample)
        if overlap > 0:
            speech_samples += overlap
    return min(speech_samples, end_sample - start_sample) / sr


def _window_score(
    y: np.ndarray,
    sr: int,
    f0: np.ndarray,
    frame_times: np.ndarray,
    start_sec: float,
    window_sec: float,
) -> float:
    """Return ``voiced_ratio * pitch_stability * rms`` for one candidate window.

    The pitch track is used for ranking only; whether a window is admitted at all
    is decided by ``_speech_seconds`` (ruling R17).  Frames are associated with
    the window by their centre because the pitch track is centred on the frame
    grid.
    """
    end_sec = start_sec + window_sec
    segment = y[int(start_sec * sr) : int(end_sec * sr)].astype(np.float64)
    rms = float(np.sqrt(np.mean(segment**2))) if segment.size else 0.0

    inside = (frame_times >= start_sec) & (frame_times < end_sec)
    f0_in_window = f0[inside]
    voiced = f0_in_window[np.isfinite(f0_in_window)]
    if voiced.size == 0:
        return 0.0

    voiced_ratio = voiced.size / f0_in_window.size
    pitch_stability = 1.0 / (1.0 + float(np.std(voiced)) / float(np.mean(voiced)))
    return float(voiced_ratio * pitch_stability * rms)
