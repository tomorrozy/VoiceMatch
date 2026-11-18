"""Symmetric preprocessing shared by library segments and user queries."""

from dataclasses import dataclass

import librosa
import numpy as np
import pyloudnorm as pyln

from voicematch.config import FRAME_LENGTH, HOP_LENGTH, LUFS_TARGET, SAMPLE_RATE


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings that define one preprocessing pipeline."""

    separate_vocals: bool
    dereverb: bool
    lufs_target: float


# Query and library audio must be processed identically; the contract test
# compares ``pipeline_signature`` of these two objects. The flags are off by
# default because Demucs (the optional ``ml`` extra) is required for separation
# and dereverb is not implemented yet.
PREPROCESS_QUERY = PreprocessConfig(
    separate_vocals=False, dereverb=False, lufs_target=LUFS_TARGET
)
PREPROCESS_LIBRARY = PreprocessConfig(
    separate_vocals=False, dereverb=False, lufs_target=LUFS_TARGET
)


def pipeline_signature(
    cfg: PreprocessConfig,
) -> tuple[int, bool, bool, int, int, float]:
    """Return everything that must match between query and library pipelines.

    Any future preprocessing choice that could make the two sides diverge has
    to be added here; the parity contract test reads this tuple. Every entry is
    the value ``preprocess`` actually uses, so a divergent config cannot hide
    behind a pinned module constant.
    """
    return (
        SAMPLE_RATE,
        cfg.separate_vocals,
        cfg.dereverb,
        FRAME_LENGTH,
        HOP_LENGTH,
        cfg.lufs_target,
    )


def preprocess(x: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Apply the configured pipeline to mono audio at ``SAMPLE_RATE``."""
    y = np.asarray(x, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"expected mono 1-D audio, got shape {y.shape}")
    if cfg.dereverb:
        raise NotImplementedError("dereverb is not implemented")
    if cfg.separate_vocals:
        y = separate_vocals(y, SAMPLE_RATE)

    # Silence (or a signal shorter than the loudness meter's block) has no
    # meaningful integrated loudness; leave it untouched instead of
    # multiplying by an infinite gain or raising from the meter.  A signal
    # shorter than the block must reach the quality gate, which reports it as
    # ``too_short_speech``; pyloudnorm otherwise raises "Audio must have length
    # greater than the block size" before any product rule can run (C1).
    meter = pyln.Meter(SAMPLE_RATE)
    if y.size < meter.block_size * SAMPLE_RATE:
        return y
    loudness = meter.integrated_loudness(y.astype(np.float64))
    if not np.isfinite(loudness):
        return y
    gain = 10.0 ** ((cfg.lufs_target - loudness) / 20.0)
    return (y * gain).astype(np.float32)


def separate_vocals(x: np.ndarray, sr: int) -> np.ndarray:
    """Return the vocal stem of ``x`` (mono, at ``sr``).

    Demucs is an optional dependency of the ``ml`` extra; the import is
    deferred so that the rest of the package works without it.
    """
    try:
        import torch
        from demucs.api import Separator
    except ImportError as exc:
        raise ImportError("install voicematch[ml]") from exc

    separator = Separator(model="htdemucs")
    wav = torch.from_numpy(np.asarray(x, dtype=np.float32))
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav = wav.repeat(separator.audio_channels, 1)
    # Demucs resamples the input to the model's rate and returns every stem at
    # that rate, so the vocals track is mapped back to the pipeline rate.
    _, stems = separator.separate_tensor(wav, sr)
    vocals = stems["vocals"].mean(dim=0).cpu().numpy()
    vocals = librosa.resample(
        vocals, orig_sr=separator.samplerate, target_sr=sr, res_type="soxr_hq"
    )
    return np.ascontiguousarray(vocals, dtype=np.float32)
