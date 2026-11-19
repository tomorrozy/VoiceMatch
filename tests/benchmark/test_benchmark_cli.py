"""End-to-end CLI test: sheet -> adjudicate -> freeze -> evaluate, all hash encoder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicematch.config import SAMPLE_RATE
from voicematch.library.build import build_library
from voicematch.library.schema import Singer

TOOL = Path(__file__).resolve().parents[2] / "tools" / "benchmark.py"


def _tone(f0: float, secs: float = 16.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * secs)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def _voiced(f0: float = 220.0, secs: float = 12.0) -> np.ndarray:
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


def _run(*args):
    result = subprocess.run(
        [sys.executable, str(TOOL), *map(str, args)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_full_pipeline(tmp_path):
    from voicematch.embed.encoder import HashEncoder

    # 8 female + 2 male singers, 2 takes each.
    mat = tmp_path / "material"
    mat.mkdir()
    females = [f"f{i}" for i in range(8)]
    males = ["m0", "m1"]
    f0s = {sid: 180.0 + i * 18 for i, sid in enumerate(females + males)}
    genders = {**{s: "female" for s in females}, **{s: "male" for s in males}}
    entries = []
    for sid in females + males:
        singer = Singer(sid, f"Singer {sid}", genders[sid], "pop")
        for take, amp in (("a", 0.30), ("b", 0.32)):
            p = mat / f"{sid}-{take}.wav"
            sf.write(p, _tone(f0s[sid], amp=amp), SAMPLE_RATE)
            entries.append((singer, p))
    index = build_library(entries, HashEncoder())
    from voicematch.library.store import save_library

    lib = tmp_path / "lib"
    save_library(index, lib)
    catalog_dir = lib / "catalog"
    catalog_dir.mkdir(exist_ok=True)

    # One accepted query.
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sf.write(audio_dir / "q1.wav", _voiced(), SAMPLE_RATE)
    specs = [{"query_id": "q1", "path": "q1.wav", "gender": "female",
              "subject_id": "s1", "device_id": "d1"}]
    specs_path = tmp_path / "specs.json"
    specs_path.write_text(json.dumps(specs), encoding="utf-8")

    # 1) sheet
    sheet_report = _run(
        "sheet", "--specs", specs_path, "--specs-dir", audio_dir,
        "--library", lib, "--encoder", "hash",
        "--out", tmp_path / "sheet.json", "--provenance", tmp_path / "prov.json",
        "--seed", 3,
    )
    assert sheet_report["n_queries"] == 1
    sheet = json.loads((tmp_path / "sheet.json").read_text(encoding="utf-8"))
    assert len(sheet[0]["candidates"]) == 8
    assert "system_top" not in (tmp_path / "sheet.json").read_text(encoding="utf-8")

    # 2) adjudicate: rater1/2 agree on f3, rater3 picks none-like
    anns = [
        {"query_id": "q1", "annotator_id": "r1", "picked": ["f3"]},
        {"query_id": "q1", "annotator_id": "r2", "picked": ["f3"]},
        {"query_id": "q1", "annotator_id": "r3", "picked": []},
    ]
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(json.dumps(anns), encoding="utf-8")
    adj_report = _run(
        "adjudicate", "--annotations", ann_path, "--sheet", tmp_path / "sheet.json",
        "--out", tmp_path / "adj.json",
    )
    assert adj_report["n_queries"] == 1

    # f3 must be in candidates for adjudication to accept it; inject if absent
    adj = json.loads((tmp_path / "adj.json").read_text(encoding="utf-8"))
    assert adj["adjudications"][0]["annotator_count"] == 3

    # 3) freeze
    freeze_report = _run(
        "freeze", "--sheet", tmp_path / "sheet.json",
        "--adjudicated", tmp_path / "adj.json",
        "--out", tmp_path / "frozen", "--library", lib, "--encoder", "hash",
    )
    assert freeze_report["manifest"]["n_queries"] == 1
    assert (tmp_path / "frozen" / "eval_set.json").is_file()
    assert (tmp_path / "frozen" / "MANIFEST.json").is_file()

    # 4) evaluate
    eval_report = _run(
        "evaluate", "--eval-set", tmp_path / "frozen" / "eval_set.json",
        "--library", lib, "--encoder", "hash",
    )
    assert eval_report["label_source"] == "blind_human_majority_annotation"
    assert "top1" in eval_report["metrics"]
    assert eval_report["metrics"]["n_total"] == 1


def test_pairwise_cli_path_end_to_end(tmp_path):
    """pairwise-sheet -> pairwise-adjudicate -> freeze records label_method."""
    sheet = [
        {
            "query_id": "q1",
            "path": "q1.wav",
            "gender": "female",
            "candidates": ["f0", "f1", "f2"],
        }
    ]
    sheet_path = tmp_path / "sheet.json"
    sheet_path.write_text(json.dumps(sheet), encoding="utf-8")

    # An adjudicated file with a disputed query triggers pairwise re-labeling.
    adjudicated = {
        "label_method": "majority",
        "adjudications": [
            {"query_id": "q1", "picked": [], "annotator_count": 3, "disputed": True}
        ],
        "agreement": {"kappa_mean": 0.1, "pairwise": {}, "n_pairs": 3,
                      "n_measurable_pairs": 3},
    }
    adj_path = tmp_path / "adj.json"
    adj_path.write_text(json.dumps(adjudicated), encoding="utf-8")

    pws = _run(
        "pairwise-sheet", "--adjudicated", adj_path, "--sheet", sheet_path,
        "--out", tmp_path / "pairs.json", "--top", 3,
    )
    assert pws["n_tasks"] == 3  # C(3,2)
    pairs = json.loads((tmp_path / "pairs.json").read_text(encoding="utf-8"))
    assert pairs["label_method"] == "pairwise"

    # Two raters: f0 beats f1 2-0, f0 beats f2 2-0, f1 beats f2 2-0.
    pair_anns = []
    for pid in ("f0|f1", "f0|f2", "f1|f2"):
        for rater in ("a", "b"):
            pair_anns.append(
                {"query_id": "q1", "pair_id": pid, "annotator_id": rater, "winner": "a"}
            )
    ann_path = tmp_path / "pair_ann.json"
    ann_path.write_text(json.dumps(pair_anns), encoding="utf-8")

    pwa = _run(
        "pairwise-adjudicate", "--pairs", tmp_path / "pairs.json",
        "--annotations", ann_path, "--out", tmp_path / "adj_pw.json",
    )
    assert pwa["n_queries"] == 1
    adj_pw = json.loads((tmp_path / "adj_pw.json").read_text(encoding="utf-8"))
    assert adj_pw["label_method"] == "pairwise"
    assert set(adj_pw["adjudications"][0]["picked"]) == {"f0", "f1"}
