"""Decode audio files into the pipeline's canonical waveform."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from voicematch.config import SAMPLE_RATE


def load_audio(path: Path) -> np.ndarray:
    """Read ``path`` and return mono float32 audio at ``SAMPLE_RATE``.

    Channels are averaged, the DC offset is removed, and the signal is
    resampled. Every downstream stage therefore sees the same representation
    regardless of the source file's sample rate or channel layout.

    A valid file with zero frames (an empty recording) decodes to an empty
    array instead of raising or emitting a numpy warning: removing a DC offset
    from nothing has no mean to compute (``Mean of empty slice``/``nan``), and
    "no samples" is content the quality gate must be able to report as
    ``too_short_speech`` rather than a decoding failure (CLI finding C1).
    """
    data, sr = sf.read(path, always_2d=True)
    if data.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    mono = data.mean(axis=1)
    mono = mono - mono.mean()
    mono = librosa.resample(
        mono, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="soxr_hq"
    )
    return np.ascontiguousarray(mono, dtype=np.float32)
