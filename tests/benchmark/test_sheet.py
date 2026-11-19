"""Tests for the blind annotation sheet generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicematch.benchmark.sheet import (
    SHEET_CANDIDATES,
    build_candidate_sheet,
    write_sheet,
)
from voicematch.calibrate.model import fit_calibration
from voicematch.calibrate.pairs import build_pairs
from voicematch.config import SAMPLE_RATE
from voicematch.embed.encoder import HashEncoder
from voicematch.library.build import build_library
from voicematch.library.schema import Singer


def _tone(f0: float, secs: float = 16.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * secs)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def _voiced_query(f0: float = 220.0, secs: float = 12.0) -> np.ndarray:
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


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    """A library with 8 female + 2 male singers + calibration + encoder.

    Eight per gender is exactly design 6.2's sheet size (Top-3 + 5 fillers), so
    ``candidate_set`` returns the full 8 distinct singers for a female query.
    """
    root = tmp_path_factory.mktemp("bench-sheet-material")
    encoder = HashEncoder()
    females = [f"f{i}" for i in range(8)]
    males = ["m0", "m1"]
    f0s = {sid: 180.0 + i * 18 for i, sid in enumerate(females + males)}
    genders = {**{sid: "female" for sid in females}, **{sid: "male" for sid in males}}
    entries = []
    for sid in females + males:
        singer = Singer(sid, f"Singer {sid}", genders[sid], "pop")
        for take, amp in (("a", 0.30), ("b", 0.32)):
            p = root / f"{sid}-{take}.wav"
            sf.write(p, _tone(f0s[sid], amp=amp), SAMPLE_RATE)
            entries.append((singer, p))
    index = build_library(entries, encoder)
    X, y, g = build_pairs(index, index.embeddings)
    model = fit_calibration(X, y, g)
    return index, model, encoder, root


def test_sheet_has_eight_shuffled_candidates(tmp_path, stack):
    index, model, encoder, root = stack
    q = tmp_path / "q.wav"
    sf.write(q, _voiced_query(), SAMPLE_RATE)
    specs = [
        {
            "query_id": "q1",
            "path": str(q),
            "gender": "female",
            "subject_id": "subj-1",
            "device_id": "dev-a",
        }
    ]
    entries, provenance = build_candidate_sheet(specs, index, model, encoder, seed=0)
    assert len(entries) == 1
    assert len(entries[0].candidates) == SHEET_CANDIDATES
    assert len(set(entries[0].candidates)) == SHEET_CANDIDATES
    assert set(entries[0].candidates) == {f"f{i}" for i in range(8)}
    assert entries[0].subject_id == "subj-1"
    assert provenance[0].scorable is True


def test_sheet_reproducible_with_same_seed(tmp_path, stack):
    index, model, encoder, root = stack
    q = tmp_path / "q.wav"
    sf.write(q, _voiced_query(), SAMPLE_RATE)
    specs = [{"query_id": "q1", "path": str(q), "gender": "female"}]
    e1, _ = build_candidate_sheet(specs, index, model, encoder, seed=7)
    e2, _ = build_candidate_sheet(specs, index, model, encoder, seed=7)
    assert e1[0].candidates == e2[0].candidates


def test_sheet_record_never_leaks_system_top(tmp_path, stack):
    index, model, encoder, root = stack
    q = tmp_path / "q.wav"
    sf.write(q, _voiced_query(), SAMPLE_RATE)
    entries, _ = build_candidate_sheet(
        [{"query_id": "q1", "path": str(q), "gender": "female"}],
        index, model, encoder,
    )
    record = entries[0].to_record()
    assert "system_top" not in record
    assert "score" not in record
    assert set(record) <= {
        "query_id", "path", "gender", "candidates", "subject_id", "device_id"
    }


def test_write_sheet_separates_provenance(tmp_path, stack):
    index, model, encoder, root = stack
    q = tmp_path / "q.wav"
    sf.write(q, _voiced_query(), SAMPLE_RATE)
    entries, provenance = build_candidate_sheet(
        [{"query_id": "q1", "path": str(q), "gender": "female"}],
        index, model, encoder,
    )
    sheet_path = tmp_path / "sheet.json"
    prov_path = tmp_path / "provenance.json"
    write_sheet(sheet_path, entries, prov_path, provenance)

    sheet = json.loads(sheet_path.read_text(encoding="utf-8"))
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    assert isinstance(sheet, list)
    assert "system_top" not in json.dumps(sheet, ensure_ascii=False)
    assert "system_top" in json.dumps(prov, ensure_ascii=False)
    assert "INTERNAL ONLY" in prov["note"]


def test_sheet_keeps_unscorable_query(tmp_path, stack):
    index, model, encoder, root = stack
    # Silence -> rejected by quality gate, still must get a candidate list so
    # it stays in the frozen eval set for an honest acceptance rate.
    q = tmp_path / "silent.wav"
    sf.write(q, np.zeros(int(SAMPLE_RATE * 12), dtype=np.float32), SAMPLE_RATE)
    entries, provenance = build_candidate_sheet(
        [{"query_id": "q0", "path": str(q), "gender": "female"}],
        index, model, encoder,
    )
    assert provenance[0].scorable is False
    assert len(entries[0].candidates) > 0


def test_sheet_rejects_duplicate_query_ids(tmp_path, stack):
    index, model, encoder, root = stack
    q = tmp_path / "q.wav"
    sf.write(q, _voiced_query(), SAMPLE_RATE)
    specs = [
        {"query_id": "q1", "path": str(q), "gender": "female"},
        {"query_id": "q1", "path": str(q), "gender": "female"},
    ]
    with pytest.raises(ValueError, match="duplicate query_id"):
        build_candidate_sheet(specs, index, model, encoder)


def test_sheet_rejects_bad_gender(tmp_path, stack):
    index, model, encoder, root = stack
    q = tmp_path / "q.wav"
    sf.write(q, _voiced_query(), SAMPLE_RATE)
    with pytest.raises(ValueError, match="gender must be one of"):
        build_candidate_sheet(
            [{"query_id": "q1", "path": str(q), "gender": "other"}],
            index, model, encoder,
        )
