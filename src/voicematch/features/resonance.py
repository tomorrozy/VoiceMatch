"""Explainable resonance descriptors: where the voice rings, and how cleanly.

``f1_hz``/``f2_hz``/``f3_hz`` are the formant centres that map to the "胸腔/口腔/
头腔倾向" reading of a voice: a low F1 with close higher formants is a darker,
chestier resonance.  ``low_band_ratio`` is the share of the spectrum's energy
below 2 kHz, and ``hnr_db`` is Praat's harmonics-to-noise ratio, which separates
a clean phonation from a breathy or rough one.

Formants and HNR are Praat's own analyses (Burg formant tracker, Harmonicity),
pinned to explicit settings so a stored feature vector is reproducible from the
code revision alone.  The analysis range starts at the pipeline's shared
``F0_MIN_HZ``, so a low male voice is not analysed with a higher-pitch model
than the pitch track uses.

Like the other feature modules, this layer never throws for content it cannot
analyse: a window shorter than one analysis frame, or digital silence, gets
``nan`` descriptors and no warnings.  The quality gate owns the decision to
refuse such audio.
"""

import librosa
import numpy as np
import parselmouth

from voicematch.audio.windows import F0_MIN_HZ
from voicematch.config import FRAME_LENGTH, HOP_LENGTH

#: The descriptor names.  Part of the feature-vector interface: later stages and
#: the explanation report look them up by name.
_FORMANT_KEYS = ("f1_hz", "f2_hz", "f3_hz")
_RESONANCE_KEYS = _FORMANT_KEYS + ("low_band_ratio", "hnr_db")

#: Praat formant-analysis settings.  These are Praat's defaults for a 24 kHz
#: Sound, pinned here so a parselmouth upgrade cannot silently change them.
_MAX_NUMBER_OF_FORMANTS = 5
_MAX_FORMANT_HZ = 5500.0
_FORMANT_WINDOW_SEC = 0.025
_PRE_EMPHASIS_FROM_HZ = 50.0

#: Upper edge of the "low band" whose share of the total energy is reported.
_LOW_BAND_HZ = 2000.0


def resonance_features(x: np.ndarray, sr: int) -> dict[str, float]:
    """Return the resonance descriptors of one audio window.

    - ``f1_hz``, ``f2_hz``, ``f3_hz`` — median of Praat's Burg formant tracks
      over the analysed frames (the median is robust to the occasional frame
      where the tracker jumps).
    - ``low_band_ratio`` — 0-2 kHz energy divided by total energy, i.e. the
      share of the window's energy in the low band (the "denominator share" of
      the breathiness ratio).
    - ``hnr_db`` — median harmonics-to-noise ratio in dB from Praat's
      Harmonicity analysis; a clean, periodic phonation is high, a breathy or
      rough one is low.

    A non-mono array or a non-positive sample rate raises ``ValueError`` instead:
    that is a caller bug, not content.  Audio shorter than one analysis frame,
    or digital silence, returns ``nan`` for every key.
    """
    y = np.asarray(x, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"expected mono 1-D audio, got shape {y.shape}")
    if sr <= 0:
        raise ValueError(f"sample rate must be positive, got {sr}")
    if y.size < FRAME_LENGTH or not np.any(y):
        # Too short for a frame grid, or no signal at all: Praat would either
        # emit its -200 dB silence floor or (for sub-frame Sounds) abort.  The
        # feature layer reports "not measurable" instead.
        return {key: float("nan") for key in _RESONANCE_KEYS}

    sound = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    features = _formant_medians(sound)
    features["low_band_ratio"] = _low_band_ratio(y, sr)
    features["hnr_db"] = _median_harmonicity(sound)
    return features


def _formant_medians(sound: parselmouth.Sound) -> dict[str, float]:
    """Return the median F1-F3 over every analysed frame."""
    try:
        formant = sound.to_formant_burg(
            max_number_of_formants=_MAX_NUMBER_OF_FORMANTS,
            maximum_formant=_MAX_FORMANT_HZ,
            window_length=_FORMANT_WINDOW_SEC,
            pre_emphasis_from=_PRE_EMPHASIS_FROM_HZ,
        )
    except parselmouth.PraatError:
        return {key: float("nan") for key in _FORMANT_KEYS}

    times = formant.xs()
    medians: dict[str, float] = {}
    for number, key in enumerate(_FORMANT_KEYS, start=1):
        values = np.asarray(
            [formant.get_value_at_time(number, float(time)) for time in times],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        medians[key] = float(np.median(finite)) if finite.size else float("nan")
    return medians


def _median_harmonicity(sound: parselmouth.Sound) -> float:
    """Return the median harmonics-to-noise ratio in dB.

    ``minimum_pitch`` is the shared pitch floor, so harmonicity is analysed over
    the same vocal range as the pitch track instead of Praat's 75 Hz default.
    """
    try:
        harmonicity = sound.to_harmonicity(minimum_pitch=F0_MIN_HZ)
    except parselmouth.PraatError:
        return float("nan")
    values = np.asarray(harmonicity.values, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _low_band_ratio(y: np.ndarray, sr: int) -> float:
    """Return the share of the frame-grid energy below 2 kHz."""
    spectrum = (
        np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)) ** 2
    )
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=FRAME_LENGTH)
    total = float(spectrum.sum())
    if total <= 0.0:
        return float("nan")
    return float(spectrum[frequencies <= _LOW_BAND_HZ].sum()) / total
