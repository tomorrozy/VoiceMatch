"""Tests for listening-sample export."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from voicematch.benchmark.samples import (
    SAMPLE_PREFIX,
    export_samples,
    write_manifest,
)
from voicematch.config import SAMPLE_RATE


def _voiced(secs=16.0, f0=220.0, amp=0.3) -> np.ndarray:
    period_samples = int(SAMPLE_RATE * 0.6)
    t = np.arange(period_samples) / SAMPLE_RATE
    burst = (amp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    burst[int(SAMPLE_RATE * 0.5) :] = 0.0
    out = np.zeros(int(SAMPLE_RATE * secs), dtype=np.float32)
    pos = 0
    while pos + period_samples <= out.size:
        out[pos : pos + period_samples] = burst
        pos += period_samples
    return out


def _write_catalog(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for sid, f0 in (("f0", 220.0), ("m0", 140.0)):
        sf.write(audio_dir / f"{sid}.wav", _voiced(f0=f0), SAMPLE_RATE)
    catalog = {
        "singers": [
            {
                "singer_id": "f0", "name": "Female 0", "gender": "female",
                "style_tag": "pop", "materials": ["audio/f0.wav"],
            },
            {
                "singer_id": "m0", "name": "Male 0", "gender": "male",
                "style_tag": "rock", "materials": ["audio/m0.wav"],
            },
        ]
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


def test_export_samples_writes_wav_and_manifest(tmp_path):
    catalog = _write_catalog(tmp_path)
    out = tmp_path / "samples"
    samples, warnings = export_samples(catalog, out, secs=8.0)
    assert warnings == []
    assert len(samples) == 2
    by_id = {s.singer_id: s for s in samples}
    assert by_id["f0"].name == "Female 0"
    assert by_id["f0"].gender == "female"
    assert by_id["f0"].file == f"{SAMPLE_PREFIX}f0.wav"
    assert (out / f"{SAMPLE_PREFIX}f0.wav").is_file()
    assert (out / f"{SAMPLE_PREFIX}m0.wav").is_file()

    manifest_path = out / "samples.json"
    write_manifest(manifest_path, samples)
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {r["singer_id"] for r in records} == {"f0", "m0"}


def test_export_samples_skips_silent_material(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sf.write(audio_dir / "f0.wav", _voiced(), SAMPLE_RATE)
    sf.write(
        audio_dir / "silent.wav",
        np.zeros(int(SAMPLE_RATE * 16), dtype=np.float32),
        SAMPLE_RATE,
    )
    catalog = {
        "singers": [
            {"singer_id": "f0", "name": "F0", "gender": "female",
             "style_tag": "pop", "materials": ["audio/f0.wav"]},
            {"singer_id": "bad", "name": "Bad", "gender": "female",
             "style_tag": "pop", "materials": ["audio/silent.wav"]},
        ]
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")

    samples, warnings = export_samples(path, tmp_path / "samples", secs=8.0)
    assert [s.singer_id for s in samples] == ["f0"]
    assert any("bad" in w for w in warnings)


def test_export_samples_rejects_nonpositive_secs(tmp_path):
    with pytest.raises(ValueError, match="secs must be positive"):
        export_samples(_write_catalog(tmp_path), tmp_path / "s", secs=0)
