"""Quality gate: decide whether a query may be scored, and why it is refused.

The product rule (design doc G5) is that a low-quality recording is refused
with structured reasons instead of being scored anyway.  Downstream the
acceptance rate is computed from these reports, so ``rejected`` and ``reasons``
are part of the API contract: every violated rule is collected, in rule order,
so the user learns about all the problems at once.

``speech_intervals`` is the pipeline's single definition of effective speech:
the gate measures ``speech_sec`` with it and the window selector admits windows
with it, so "usable audio" cannot mean two different things (ruling R17).
"""

from dataclasses import dataclass

import librosa
import numpy as np

from voicematch.config import FRAME_LENGTH, HOP_LENGTH, MIN_SPEECH_SEC

# Rejection policy (plan Task 3, Step 3).  Kept together so the thresholds can
# be audited without reading the gate logic.
SPLIT_TOP_DB = 30.0
CLIP_LEVEL = 0.999
MAX_CLIPPED_RATIO = 0.01
MIN_SNR_DB = 10.0
MIN_VOICED_RATIO = 0.3


@dataclass(frozen=True)
class QualityReport:
    """Measured quality of one query recording.

    ``snr_db`` is ``-inf`` when no voiced frame carries energy: the ratio is
    undefined for digital silence and is kept explicitly non-finite rather than
    replaced by an arbitrary finite number.  Every other field is finite.
    """

    speech_sec: float
    snr_db: float
    clipped_ratio: float
    voiced_ratio: float
    rejected: bool
    reasons: tuple[str, ...]


def assess_quality(x: np.ndarray, sr: int) -> QualityReport:
    """Measure mono audio ``x`` at ``sr`` and list every violated quality rule.

    ``speech_sec`` is measured from :func:`speech_intervals`, the same definition
    the window selector uses to admit windows.

    ``reasons`` is ordered like the rules themselves: ``too_short_speech``,
    ``clipped``, ``low_snr``, ``low_voiced_ratio``.  It lists every hit instead
    of stopping at the first one.
    """
    y = np.asarray(x, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"expected mono 1-D audio, got shape {y.shape}")
    if sr <= 0:
        raise ValueError(f"sample rate must be positive, got {sr}")

    clipped_ratio = float(np.mean(np.abs(y) >= CLIP_LEVEL)) if y.size else 0.0
    intervals = speech_intervals(y, sr)
    speech_samples = sum(end - start for start, end in intervals)
    speech_sec = speech_samples / sr
    energies = _frame_energies(y)
    voiced = _voiced_frame_mask(energies.size, intervals)
    voiced_ratio = float(voiced.mean()) if voiced.size else 0.0
    snr_db = _snr_db(energies, voiced)

    reasons: list[str] = []
    if speech_sec < MIN_SPEECH_SEC:
        reasons.append("too_short_speech")
    if clipped_ratio > MAX_CLIPPED_RATIO:
        reasons.append("clipped")
    if snr_db < MIN_SNR_DB:
        reasons.append("low_snr")
    if voiced_ratio < MIN_VOICED_RATIO:
        reasons.append("low_voiced_ratio")

    return QualityReport(
        speech_sec=speech_sec,
        snr_db=snr_db,
        clipped_ratio=clipped_ratio,
        voiced_ratio=voiced_ratio,
        rejected=bool(reasons),
        reasons=tuple(reasons),
    )


def speech_intervals(x: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Return the speech segments of mono audio ``x`` at ``sr``, in samples.

    Each entry is a ``(start, end)`` sample range (end exclusive) on the pipeline
    frame grid.  This is the pipeline's only definition of "有效发声" (ruling
    R17): ``assess_quality`` measures ``speech_sec`` from it and
    ``select_windows`` admits a window only when these intervals cover
    ``MIN_SPEECH_SEC`` inside it, so the two stages cannot disagree about which
    audio is usable.

    ``sr`` is validated for symmetry with the other pipeline entry points; the
    split itself is defined on the sample grid by ``FRAME_LENGTH`` and
    ``HOP_LENGTH``.

    ``librosa.effects.split`` uses the peak frame level as its 0 dB reference,
    so an all-zero signal is reported as one non-silent interval.  A recording
    with no non-zero sample, or one shorter than the analysis frame, therefore
    has no speech interval at all.
    """
    y = np.asarray(x, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"expected mono 1-D audio, got shape {y.shape}")
    if sr <= 0:
        raise ValueError(f"sample rate must be positive, got {sr}")
    if y.size < FRAME_LENGTH or not np.any(y):
        return []
    intervals = librosa.effects.split(
        y, top_db=SPLIT_TOP_DB, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    return [(int(start), int(end)) for start, end in intervals]


def _frame_energies(y: np.ndarray) -> np.ndarray:
    """Return the mean-square energy of every frame on the pipeline grid.

    Frames start every ``HOP_LENGTH`` samples and are not padded, so a signal
    shorter than ``FRAME_LENGTH`` has no frames at all.
    """
    if y.size < FRAME_LENGTH:
        return np.empty(0, dtype=np.float64)
    return np.mean(
        librosa.util.frame(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        .astype(np.float64) ** 2,
        axis=0,
    )


def _voiced_frame_mask(n_frames: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    """Mark the frames that overlap one of ``intervals``.

    ``voiced_ratio`` and the SNR numerator use this same frame set.
    """
    starts = np.arange(n_frames) * HOP_LENGTH
    voiced = np.zeros(n_frames, dtype=bool)
    for start, end in intervals:
        voiced |= (starts < end) & (starts + FRAME_LENGTH > start)
    return voiced


def _snr_db(energies: np.ndarray, voiced: np.ndarray) -> float:
    """Voiced-frame mean energy over the 5th-percentile frame energy, in dB.

    The 5th percentile of all frame energies estimates the noise floor.  The
    denominator is floored at ``eps`` so a recording whose quietest frames are
    exactly zero cannot divide by zero or take ``log(0)``; such a recording
    reports a very high finite SNR rather than ``+inf``.  A signal with no
    voiced frame at all has no defined ratio and reports ``-inf``.
    """
    voiced_energy = float(energies[voiced].mean()) if voiced.any() else 0.0
    if voiced_energy <= 0.0:
        return float("-inf")
    noise_energy = max(
        float(np.percentile(energies, 5)), float(np.finfo(np.float64).eps)
    )
    return float(10.0 * np.log10(voiced_energy / noise_energy))
