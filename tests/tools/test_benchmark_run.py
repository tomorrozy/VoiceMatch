"""The runbook's job is to stop you before a mistake costs a re-record.

Two failure modes are expensive and invisible until much later:

- two recordings sharing a filename -- the annotation UI matches audio by
  ``basename``, so the second one silently plays the wrong clip;
- a recording shorter than 8 s -- it can never pass the "8 s voiced inside a
  10 s window" admission rule, so it can only ever lower the acceptance rate.

``preflight`` turns both into errors. This module also pins the merge rules
(same rater counted twice would be a double vote) and the pipeline state
machine, whose whole point is telling you the one next command.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicematch.config import SAMPLE_RATE

TOOL = Path(__file__).resolve().parents[2] / "tools" / "benchmark_run.py"


@pytest.fixture(scope="module")
def runner():
    import sys

    spec = importlib.util.spec_from_file_location("benchmark_run_under_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules; register first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_audio(path: Path, seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * SAMPLE_RATE)
    tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(frames) / SAMPLE_RATE)
    sf.write(path, tone.astype("float32"), SAMPLE_RATE)
    return path


def _spec(query_id: str, path: str, gender: str = "female", **extra) -> dict:
    base = {"query_id": query_id, "path": path, "gender": gender}
    base.update(extra)
    return base


def _codes(items: list[dict]) -> set[str]:
    return {item["code"] for item in items}


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------
def test_clean_specs_pass(tmp_path, runner):
    _write_audio(tmp_path / "audio" / "p001-a.wav", 12.0)
    specs = [_spec("p001-a", "audio/p001-a.wav", subject_id="p001", device_id="d1")]
    errors, warnings, stats = runner.validate_specs(specs, tmp_path)
    assert errors == []
    assert stats["n_specs"] == 1
    assert stats["n_subjects"] == 1
    assert "pilot_scale" in _codes(warnings)


def test_duplicate_filename_is_an_error_even_in_different_folders(tmp_path, runner):
    """The UI matches by basename: same name in two folders still cross-talks."""
    _write_audio(tmp_path / "a" / "take.wav", 12.0)
    _write_audio(tmp_path / "b" / "take.wav", 12.0)
    specs = [
        _spec("q1", "a/take.wav", "female"),
        _spec("q2", "b/take.wav", "male"),
    ]
    errors, _warnings, _stats = runner.validate_specs(specs, tmp_path)
    assert "duplicate_filename" in _codes(errors)


def test_duplicate_query_id_is_an_error(tmp_path, runner):
    _write_audio(tmp_path / "audio" / "a.wav", 12.0)
    _write_audio(tmp_path / "audio" / "b.wav", 12.0)
    specs = [_spec("q1", "audio/a.wav"), _spec("q1", "audio/b.wav")]
    errors, _warnings, _stats = runner.validate_specs(specs, tmp_path)
    assert "duplicate_query_id" in _codes(errors)


def test_recording_shorter_than_the_voiced_gate_is_an_error(tmp_path, runner):
    _write_audio(tmp_path / "audio" / "short.wav", 6.5)
    errors, _warnings, stats = runner.validate_specs(
        [_spec("q1", "audio/short.wav")], tmp_path
    )
    assert "too_short" in _codes(errors)
    assert stats["durations_sec"]["q1"] == pytest.approx(6.5, abs=0.05)


def test_over_contract_length_is_only_a_warning(tmp_path, runner):
    _write_audio(tmp_path / "audio" / "long.wav", 61.0)
    errors, warnings, _stats = runner.validate_specs(
        [_spec("q1", "audio/long.wav")], tmp_path
    )
    assert errors == []
    assert "over_contract" in _codes(warnings)


def test_bad_gender_and_missing_file_are_errors(tmp_path, runner):
    errors, _warnings, _stats = runner.validate_specs(
        [_spec("q1", "audio/nope.wav", gender="woman")], tmp_path
    )
    assert {"bad_gender", "missing_audio"} <= _codes(errors)


def test_unreadable_audio_is_an_error(tmp_path, runner):
    broken = tmp_path / "audio" / "broken.wav"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not a wav")
    errors, _warnings, _stats = runner.validate_specs(
        [_spec("q1", "audio/broken.wav")], tmp_path
    )
    assert "unreadable_audio" in _codes(errors)


def test_missing_subject_id_warns_because_g3_needs_it(tmp_path, runner):
    _write_audio(tmp_path / "audio" / "a.wav", 12.0)
    _errors, warnings, _stats = runner.validate_specs([_spec("q1", "audio/a.wav")], tmp_path)
    assert "missing_subject_id" in _codes(warnings)
    assert "single_device" in _codes(warnings)


def test_not_a_list_is_rejected(runner, tmp_path):
    errors, _warnings, _stats = runner.validate_specs({"query_id": "q1"}, tmp_path)
    assert _codes(errors) == {"not_a_list"}


def test_preflight_refuses_an_empty_specs_file(tmp_path, runner):
    (tmp_path / "specs.json").write_text("[]", encoding="utf-8")
    config = {
        "specs": "specs.json",
        "k": 8,
        "encoder": "hubert",
        "library": None,
        "catalog": None,
    }
    payload = runner.preflight(tmp_path, config)
    assert payload["status"] == "error"
    assert _codes(payload["errors"]) == {"empty_specs"}


def test_preflight_catches_an_encoder_library_mismatch(tmp_path, runner):
    _write_audio(tmp_path / "audio" / "a.wav", 12.0)
    (tmp_path / "specs.json").write_text(
        json.dumps([_spec("q1", "audio/a.wav", subject_id="p1", device_id="d1")]),
        encoding="utf-8",
    )
    library = tmp_path / "lib"
    library.mkdir()
    (library / "manifest.json").write_text(
        json.dumps({"model_id": "HashEncoder", "singers": []}), encoding="utf-8"
    )
    config = {
        "specs": "specs.json",
        "k": 8,
        "encoder": "hubert",
        "library": str(library),
        "catalog": None,
    }
    payload = runner.preflight(tmp_path, config)
    assert payload["status"] == "error"
    assert "encoder_mismatch" in _codes(payload["errors"])


# --------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------
def _sheet(tmp_path: Path, query_id: str = "q1") -> None:
    sheet = [
        {
            "query_id": query_id,
            "path": str(tmp_path / "audio" / f"{query_id}.wav"),
            "gender": "female",
            "candidates": ["s1", "s2", "s3"],
        }
    ]
    (tmp_path / "sheet.json").write_text(json.dumps(sheet), encoding="utf-8")


def _annotations(tmp_path: Path, rows: list[dict], name: str = "annotations-r1.json") -> None:
    folder = tmp_path / "annotations"
    folder.mkdir(exist_ok=True)
    (folder / name).write_text(json.dumps(rows), encoding="utf-8")


def test_merge_sums_raters_and_writes_sorted_rows(tmp_path, runner):
    _sheet(tmp_path)
    _annotations(tmp_path, [{"query_id": "q1", "annotator_id": "r1", "picked": ["s1"]}], "a.json")
    _annotations(tmp_path, [{"query_id": "q1", "annotator_id": "r2", "picked": []}], "b.json")
    payload, errors, warnings = runner.merge_annotations(tmp_path, raters=2)
    assert errors == []
    assert payload["queries"] == 1
    assert payload["raters"] == 2
    assert warnings == []


def test_same_rater_twice_is_rejected_as_a_double_vote(tmp_path, runner):
    _sheet(tmp_path)
    _annotations(tmp_path, [{"query_id": "q1", "annotator_id": "r1", "picked": ["s1"]}], "a.json")
    _annotations(tmp_path, [{"query_id": "q1", "annotator_id": "r1", "picked": ["s2"]}], "b.json")
    _payload, errors, _warnings = runner.merge_annotations(tmp_path, raters=2)
    assert "duplicate_annotation" in _codes(errors)


def test_a_pick_outside_the_sheet_is_rejected(tmp_path, runner):
    _sheet(tmp_path)
    _annotations(tmp_path, [{"query_id": "q1", "annotator_id": "r1", "picked": ["s9"]}])
    _payload, errors, _warnings = runner.merge_annotations(tmp_path, raters=1)
    assert "pick_outside_candidates" in _codes(errors)


def test_thin_coverage_is_a_warning_not_a_failure(tmp_path, runner):
    _sheet(tmp_path)
    _annotations(tmp_path, [{"query_id": "q1", "annotator_id": "r1", "picked": ["s1"]}])
    payload, errors, warnings = runner.merge_annotations(tmp_path, raters=3)
    assert errors == []
    assert payload["raters"] == 1
    assert "thin_annotation" in _codes(warnings)


def test_unannotated_query_is_reported(tmp_path, runner):
    _sheet(tmp_path)
    _annotations(tmp_path, [{"query_id": "other", "annotator_id": "r1", "picked": []}])
    _payload, _errors, warnings = runner.merge_annotations(tmp_path, raters=1)
    codes = _codes(warnings)
    assert "unannotated_query" in codes
    assert "unknown_query" in codes


def test_merge_without_exports_says_so(tmp_path, runner):
    _payload, errors, _warnings = runner.merge_annotations(tmp_path, raters=3)
    assert _codes(errors) == {"no_annotation_files"}


# --------------------------------------------------------------------------
# 状态机：告诉人"下一步敲什么"
# --------------------------------------------------------------------------
def _state(runner, tmp_path: Path, step_name: str, config: dict) -> dict:
    step = next(s for s in runner.STEPS if s.name == step_name)
    return runner._step_state(tmp_path, step, config)


def test_empty_specs_file_is_waiting_for_human_not_ready(tmp_path, runner):
    """An empty specs.json must not let `sheet` "succeed" with zero queries."""
    (tmp_path / "specs.json").write_text("[]", encoding="utf-8")
    config = {"library": None, "catalog": None, "specs": "specs.json"}
    state = _state(runner, tmp_path, "sheet", config)
    assert state["state"] == "waiting_for_human"
    assert any("0 条录音" in item for item in state["missing"])


def test_merge_is_ready_once_exports_exist(tmp_path, runner):
    _sheet(tmp_path)
    config = {"library": None, "catalog": None}
    assert _state(runner, tmp_path, "merge", config)["state"] == "waiting_for_human"
    _annotations(tmp_path, [{"query_id": "q1", "annotator_id": "r1", "picked": ["s1"]}])
    assert _state(runner, tmp_path, "merge", config)["state"] == "ready"


def test_missing_library_is_blocked_not_a_human_wait(tmp_path, runner):
    _write_audio(tmp_path / "audio" / "a.wav", 12.0)
    (tmp_path / "specs.json").write_text(
        json.dumps([_spec("q1", "audio/a.wav")]), encoding="utf-8"
    )
    state = _state(runner, tmp_path, "sheet", {"library": None, "catalog": None})
    assert state["state"] == "blocked"
    assert any("library" in item for item in state["missing"])


def test_done_step_is_reported_done(tmp_path, runner):
    (tmp_path / "diagnosis.json").write_text("{}", encoding="utf-8")
    state = _state(runner, tmp_path, "diagnose", {"catalog": "x", "library": None})
    assert state["state"] == "done"


def test_unknown_step_is_rejected_without_running_anything(tmp_path, runner, capsys):
    code = runner.main(["run", "--dir", str(tmp_path), "--steps", "nope"])
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 2
    assert payload["status"] == "error"
    assert payload["known"] == [step.name for step in runner.STEPS]


def test_status_returns_the_ordered_pipeline(tmp_path, runner, capsys):
    code = runner.main(["status", "--dir", str(tmp_path)])
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0
    assert [item["step"] for item in payload["steps"]] == [s.name for s in runner.STEPS]
    assert payload["next_step"] == "diagnose"
