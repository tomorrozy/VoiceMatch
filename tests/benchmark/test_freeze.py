"""Tests for freezing an eval set: validation, versioning, provenance."""

from __future__ import annotations

import json
import math

import pytest

from voicematch.benchmark.agreement import Adjudication, AgreementReport
from voicematch.benchmark.freeze import (
    EVAL_SET_FILENAME,
    MANIFEST_FILENAME,
    build_eval_records,
    benchmark_version,
    freeze,
)
from voicematch.benchmark.sheet import SheetEntry
from voicematch.eval.dataset import load_eval_set


def _entry(qid="q1", path="query.wav"):
    return SheetEntry(
        query_id=qid,
        path=path,
        gender="female",
        candidates=("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"),
        subject_id="subj-1",
        device_id="dev-a",
    )


def _adjudication(picked=("s1",), disputed=False):
    return Adjudication(
        query_id="q1",
        picked=frozenset(picked),
        annotator_count=3,
        disputed=disputed,
    )


def _agreement(kappa=0.8):
    return AgreementReport(
        kappa_mean=kappa,
        pairwise={"a|b": kappa},
        n_pairs=1,
        n_measurable_pairs=1,
    )


def test_build_eval_records_maps_fields():
    records = build_eval_records([_entry()], {"q1": _adjudication()})
    record = records[0]
    assert record["query_id"] == "q1"
    assert record["picked"] == ["s1"]
    assert record["annotator_count"] == 3
    assert record["subject_id"] == "subj-1"
    assert record["device_id"] == "dev-a"
    assert "system_top" not in record


def test_build_eval_records_rejects_missing_adjudication():
    with pytest.raises(ValueError, match="has no adjudication"):
        build_eval_records([_entry()], {})


def test_build_eval_records_rejects_picked_outside_candidates():
    with pytest.raises(ValueError, match="not among the shown candidates"):
        build_eval_records([_entry()], {"q1": _adjudication(picked=("s99",))})


def test_none_like_record_has_empty_picked():
    records = build_eval_records([_entry()], {"q1": _adjudication(picked=())})
    assert records[0]["picked"] == []


def test_benchmark_version_is_content_determined():
    records = build_eval_records([_entry()], {"q1": _adjudication()})
    v1 = benchmark_version(records)
    assert len(v1) == 16
    assert v1 == benchmark_version(records)  # deterministic
    assert v1 != benchmark_version(
        build_eval_records([_entry()], {"q1": _adjudication(picked=("s2",))})
    )


def test_freeze_writes_valid_eval_set_and_manifest(tmp_path, accepted_wav):
    entry = _entry(path=str(accepted_wav))
    manifest = freeze(
        tmp_path / "bench",
        [entry],
        {"q1": _adjudication()},
        _agreement(),
        library_version="lib123",
        model_id="HUBERT_BASE",
        calibration_version="cal456",
    )
    bench = tmp_path / "bench"
    assert (bench / EVAL_SET_FILENAME).is_file()
    assert (bench / MANIFEST_FILENAME).is_file()

    # The frozen eval set must load cleanly with the project's own loader.
    queries = load_eval_set(bench / EVAL_SET_FILENAME)
    assert len(queries) == 1
    assert queries[0].picked == {"s1"}
    assert queries[0].annotator_count == 3

    assert manifest["label_source"] == "blind_human_majority_annotation"
    assert manifest["library_version"] == "lib123"
    assert manifest["n_queries"] == 1
    assert manifest["agreement"]["below_threshold"] is False
    assert manifest["benchmark_version"] == benchmark_version(
        build_eval_records([entry], {"q1": _adjudication()})
    )


def test_freeze_records_disputed_and_none_like(tmp_path, accepted_wav):
    entry = _entry(path=str(accepted_wav))
    manifest = freeze(
        tmp_path / "bench",
        [entry],
        {"q1": _adjudication(disputed=True, picked=())},
        _agreement(kappa=0.2),
        library_version="lib",
        model_id="m",
        calibration_version="c",
        kappa_threshold=0.6,
    )
    assert manifest["disputed_queries"] == ["q1"]
    assert manifest["n_none_like"] == 1
    assert manifest["agreement"]["below_threshold"] is True
    assert manifest["agreement"]["kappa_mean"] == 0.2


def test_freeze_unmeasurable_kappa_is_null_and_not_below(tmp_path, accepted_wav):
    manifest = freeze(
        tmp_path / "bench",
        [_entry(path=str(accepted_wav))],
        {"q1": _adjudication()},
        _agreement(kappa=float("nan")),
        library_version="lib",
        model_id="m",
        calibration_version="c",
    )
    assert manifest["agreement"]["kappa_mean"] is None  # strict JSON, no NaN
    assert manifest["agreement"]["below_threshold"] is False


def test_freeze_rejects_corrupt_eval_set(tmp_path, accepted_wav):
    # picked references a singer not in candidates -> build_eval_records must
    # refuse, and freeze must propagate rather than leave a bogus set on disk.
    with pytest.raises(ValueError, match="not among the shown candidates"):
        freeze(
            tmp_path / "bench",
            [_entry(path=str(accepted_wav))],
            {"q1": _adjudication(picked=("s99",))},
            _agreement(),
            library_version="lib",
            model_id="m",
            calibration_version="c",
        )
