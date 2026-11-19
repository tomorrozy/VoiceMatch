"""Shared fixtures for benchmark tests: a real accepted recording on disk."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from voicematch.config import SAMPLE_RATE


def _voiced_wav(secs: float = 12.0, f0: float = 220.0) -> np.ndarray:
    """Voiced-burst pattern the quality gate accepts and yields a window."""
    period = 0.6
    n = int(SAMPLE_RATE * period)
    t = np.arange(n) / SAMPLE_RATE
    burst = (0.3 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    burst[int(SAMPLE_RATE * 0.5) :] = 0.0
    out = np.zeros(int(SAMPLE_RATE * secs), dtype=np.float32)
    pos = 0
    while pos + n <= out.size:
        out[pos : pos + n] = burst
        pos += n
    return out


@pytest.fixture
def accepted_wav(tmp_path):
    """Write a clean recording that passes the gate; return its path."""
    path = tmp_path / "query.wav"
    sf.write(path, _voiced_wav(), SAMPLE_RATE)
    return path
