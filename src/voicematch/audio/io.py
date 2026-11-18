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
    """
    data, sr = sf.read(path, always_2d=True)
    mono = data.mean(axis=1)
    mono = mono - mono.mean()
    mono = librosa.resample(
        mono, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="soxr_hq"
    )
    return np.ascontiguousarray(mono, dtype=np.float32)
