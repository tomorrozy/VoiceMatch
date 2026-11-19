"""Tests for listening-sample export."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from voicematch.benchmark.samples import (
    SAMPLE_PREFIX,
    diagnose_catalog,
    diagnose_material,
    export_samples,
    write_diagnosis,
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


def test_diagnose_catalog_marks_ok_and_needs_replacement(tmp_path):
    catalog = _write_catalog(tmp_path)  # f0 voiced, m0 voiced
    # Add a singer with only silent material.
    audio_dir = tmp_path / "audio"
    sf.write(audio_dir / "silent.wav",
             np.zeros(int(SAMPLE_RATE * 16), dtype=np.float32), SAMPLE_RATE)
    raw = json.loads(catalog.read_text(encoding="utf-8"))
    raw["singers"].append(
        {"singer_id": "bad", "name": "Bad", "gender": "female",
         "style_tag": "pop", "materials": ["audio/silent.wav"]}
    )
    catalog.write_text(json.dumps(raw), encoding="utf-8")

    suggestions = diagnose_catalog(catalog)
    by_id = {s.singer_id: s for s in suggestions}
    assert by_id["f0"].status == "ok"
    assert by_id["m0"].status == "ok"
    assert by_id["bad"].status == "needs_replacement"
    assert "Replace" in by_id["bad"].advice
    assert by_id["bad"].diagnosis[0].n_windows == 0

    out = tmp_path / "diagnosis.json"
    write_diagnosis(out, suggestions)
    records = json.loads(out.read_text(encoding="utf-8"))
    assert {r["status"] for r in records} == {"ok", "needs_replacement"}


def test_long_material_window_beyond_60s_is_kept(tmp_path):
    """Regression: a usable window past 60 s must not be truncated away.

    This is the soprano_2 case: the only clean window sat at 54-64 s; the query
    path's 60 s cap hid it, but the library path (which build actually uses)
    keeps it. Sample export must use the library path.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    # 64 s of voiced material.
    sf.write(audio_dir / "long.wav", _voiced(secs=64.0), SAMPLE_RATE)
    catalog = {
        "singers": [
            {"singer_id": "long", "name": "Long", "gender": "female",
             "style_tag": "pop", "materials": ["audio/long.wav"]}
        ]
    }
    cat = tmp_path / "catalog.json"
    cat.write_text(json.dumps(catalog), encoding="utf-8")

    diagnosis = diagnose_material("long", audio_dir / "long.wav")
    assert diagnosis.duration_sec > 60
    assert diagnosis.usable  # window found via library path

    samples, warnings = export_samples(cat, tmp_path / "samples", secs=10.0)
    assert [s.singer_id for s in samples] == ["long"]
    assert warnings == []


def test_diagnose_material_unreadable(tmp_path):
    d = diagnose_material("x", tmp_path / "missing.wav")
    assert d.rejected and d.n_windows == 0
    assert any("unreadable" in r for r in d.reject_reasons)
