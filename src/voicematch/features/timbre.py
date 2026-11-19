"""Explainable timbre descriptors: brightness, texture, spectral tilt.

Every spectrum is taken on the pipeline frame grid (``FRAME_LENGTH`` /
``HOP_LENGTH``), the same grid the pitch track covers, so all four dimensions
describe the same frames of the same window.

Like the pitch module, this layer never throws for content it cannot analyse: a
window with no frames has no spectrum, so every descriptor is ``nan``.  The
quality gate owns the decision to refuse such audio.
"""

import librosa
import numpy as np

from voicematch.config import FRAME_LENGTH, HOP_LENGTH

#: Number of MFCC coefficients exposed as descriptors.
N_MFCC = 13

#: The descriptor names.  Part of the feature-vector interface: later stages and
#: the explanation report look them up by name.
_TIMBRE_KEYS = tuple(f"mfcc{i}_mean" for i in range(1, N_MFCC + 1)) + (
    "spectral_centroid_hz",
    "spectral_contrast_mean",
    "ltas_slope",
)


def timbre_features(x: np.ndarray, sr: int) -> dict[str, float]:
    """Return the timbre descriptors of one audio window.

    - ``mfcc1_mean`` … ``mfcc13_mean`` — mean cepstral coefficients; the coarse
      spectral-envelope shape that carries most of the "what it sounds like"
      information.
    - ``spectral_centroid_hz`` — mean energy centroid; brightness.
    - ``spectral_contrast_mean`` — mean peak/valley contrast across bands;
      how much the spectrum is articulated rather than smooth.
    - ``ltas_slope`` — least-squares slope of the long-term average power
      spectrum over log2 frequency, in dB per octave; negative means the energy
      is concentrated in the low end (a "darker" timbre).

    A non-mono array or a non-positive sample rate raises ``ValueError``
    instead: that is a caller bug, not content.
    """
    y = np.asarray(x, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"expected mono 1-D audio, got shape {y.shape}")
    if sr <= 0:
        raise ValueError(f"sample rate must be positive, got {sr}")
    if y.size == 0:
        # No frames, no spectrum: return the NaN dictionary explicitly instead
        # of asking librosa for a zero-length STFT.
        return {key: float("nan") for key in _TIMBRE_KEYS}

    spectrum = np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
    # Reuse the one STFT: the log mel spectrogram below is exactly what
    # librosa.feature.mfcc(y=...) would build internally.
    mel = librosa.feature.melspectrogram(
        S=spectrum**2, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=1.0), n_mfcc=N_MFCC)

    features = {
        f"mfcc{i + 1}_mean": float(np.mean(mfcc[i])) for i in range(N_MFCC)
    }
    features["spectral_centroid_hz"] = float(
        np.mean(
            librosa.feature.spectral_centroid(
                S=spectrum, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
            )
        )
    )
    features["spectral_contrast_mean"] = float(
        np.mean(
            librosa.feature.spectral_contrast(
                S=spectrum, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
            )
        )
    )
    features["ltas_slope"] = _ltas_slope(spectrum, sr)
    return features


def _ltas_slope(spectrum: np.ndarray, sr: int) -> float:
    """Return the long-term average spectrum's slope in dB per octave.

    The LTAS is the frame-mean power spectrum; its dB values are fitted by least
    squares against log2 frequency.  ``top_db=None`` keeps the fit from being
    flattened by ``power_to_db``'s 80 dB display floor, and ``power_to_db``
    floors exact zeros instead of emitting ``log10(0)`` warnings.  The DC bin is
    excluded because it carries no octave information.
    """
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=FRAME_LENGTH)
    mean_power = np.mean(spectrum**2, axis=1)
    decibels = librosa.power_to_db(mean_power, ref=1.0, top_db=None)
    positive = frequencies > 0
    return float(
        np.polyfit(np.log2(frequencies[positive]), decibels[positive], 1)[0]
    )
