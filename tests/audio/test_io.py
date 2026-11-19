import numpy as np
import soundfile as sf

from voicematch.audio.io import load_audio
from voicematch.config import SAMPLE_RATE


def test_stereo_44100_is_downmixed_and_resampled(tmp_path):
    sr, n = 44100, 44100
    x = np.stack([np.sin(np.arange(n) * 0.1), np.sin(np.arange(n) * 0.2)], axis=1).astype(np.float32)
    p = tmp_path / "a.wav"
    sf.write(p, x, sr)
    out = load_audio(p)
    assert out.ndim == 1 and out.dtype == np.float32
    assert abs(len(out) / SAMPLE_RATE - n / sr) < 0.05


def test_dc_offset_is_removed(tmp_path):
    sr = 48000
    x = (np.ones(sr, dtype=np.float32) * 0.3)  # 纯直流
    p = tmp_path / "dc.wav"
    sf.write(p, x, sr)
    assert abs(load_audio(p).mean()) < 1e-3
