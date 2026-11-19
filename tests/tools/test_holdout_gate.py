"""The hold-out tool must not print a top1 it cannot stand behind.

``fit_calibration`` degrades instead of failing: a library with no measurable
out-of-fold AUC gets the canonical cosine-only model (``auc = nan``).  Scoring
with it is exactly pure cosine, so a top1 from that run is a cosine number
wearing a calibrated label -- the failure mode R29 documented (in-library AUC
1.000, hold-out top1 0.30).  The same applies to a pre-R29 library, whose
positives may be two windows of one recording.

These tests pin the two halves of the gate:

- ``calibration_refusal`` on its own (no library, no audio, no network): which
  states are refused, which reason token they carry, and the two explicit ways
  through (``--cosine-only``, ``--allow-unverifiable-legacy``);
- the entry point (subprocess, real library on disk): a refused run exits 2,
  prints one JSON error document, and never reaches the hold-out set -- the
  material directory does not even exist.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicematch.config import SAMPLE_RATE
from voicematch.embed.encoder import HashEncoder
from voicematch.eval.dataset import EvalQuery
from voicematch.library import (
    CROSS_RECORDING,
    SINGLE_RECORDING,
    UNVERIFIABLE_LEGACY,
    PairingStatus,
    Singer,
    SingleMaterialSingerWarning,
    build_library,
    load_library,
    save_library,
)

TOOL = Path(__file__).resolve().parents[2] / "tools" / "holdout_eval.py"


@pytest.fixture(scope="module")
def gate_module():
    """Load ``tools/holdout_eval.py`` without running ``main``.

    The tool is a script, not an installed module, so it is imported by path.
    Importing it is enough: the gate functions are pure.
    """
    spec = importlib.util.spec_from_file_location("holdout_eval_under_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tone(f0: float, secs: float = 12.0) -> np.ndarray:
    """A pure tone the builder accepts as one 10 s window."""
    times = np.arange(int(SAMPLE_RATE * secs)) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * f0 * times)).astype(np.float32)


def _voiced(f0: float = 220.0, secs: float = 12.0) -> np.ndarray:
    """Voiced bursts with pauses: passes the query quality gate.

    A continuous tone is rejected as ``low_snr`` -- there is no pause from which
    to estimate a noise floor -- so the query fixture has to breathe.
    """
    period = 0.6
    n = int(SAMPLE_RATE * period)
    times = np.arange(n) / SAMPLE_RATE
    burst = (0.3 * np.sin(2 * np.pi * f0 * times)).astype(np.float32)
    burst[int(SAMPLE_RATE * 0.5) :] = 0.0
    out = np.zeros(int(SAMPLE_RATE * secs), dtype=np.float32)
    position = 0
    while position + n <= out.size:
        out[position : position + n] = burst
        position += n
    return out


def _census(**overrides) -> dict[str, int]:
    base = {"total": 0, "positives": 0, "negatives": 0, "folds": 0}
    base.update(overrides)
    return base


def test_pair_census_counts_positives_and_folds(gate_module):
    """Positives decide whether a calibrated number can exist; folds decide
    whether it is measurable at all."""
    pairs = (
        np.zeros((3, 5)),
        np.array([1, 1, 0]),
        np.array([0, 0, 1]),
    )
    assert gate_module.pair_census(pairs) == {
        "total": 3,
        "positives": 2,
        "negatives": 1,
        "folds": 2,
    }


def test_pair_census_of_an_empty_pair_matrix_is_all_zero(gate_module):
    """A library with no pair at all reports zeros, not an error."""
    empty = (np.empty((0, 5)), np.empty(0), np.empty(0))
    assert gate_module.pair_census(empty) == _census()


def test_a_calibrated_cross_recording_library_passes(gate_module):
    """The measurable, verifiable case is not refused."""
    assert (
        gate_module.calibration_refusal(
            calibrated=True,
            pairing=PairingStatus(CROSS_RECORDING, ()),
            census=_census(total=642, positives=331, negatives=311, folds=5),
            library_version="abc",
            calibration_version="def",
            cosine_only=False,
        )
        is None
    )


def test_no_positive_pair_is_refused_with_its_own_token(gate_module):
    """The canonical-fallback case: an uncalibrated model must not print top1."""
    refusal = gate_module.calibration_refusal(
        calibrated=False,
        pairing=PairingStatus(SINGLE_RECORDING, ("s1", "s2")),
        census=_census(),
        library_version="abc",
        calibration_version="def",
        cosine_only=False,
    )
    assert refusal is not None
    assert refusal["status"] == "error"
    assert refusal["command"] == "holdout_eval"
    assert refusal["reason"] == gate_module.UNCALIBRATED_LIBRARY
    assert "no same-singer positive pair" in refusal["detail"]
    assert "--cosine-only" in refusal["hint"]
    assert refusal["pairing"]["single_material_singers"] == ["s1", "s2"]
    # The document is what a pipeline parses, so it must be JSON-ready as is.
    assert json.loads(json.dumps(refusal, sort_keys=True))["reason"] == refusal["reason"]


def test_positives_in_one_fold_are_refused_for_the_measurability_reason(gate_module):
    """Positives alone are not enough: no singer-disjoint AUC, no number."""
    refusal = gate_module.calibration_refusal(
        calibrated=False,
        pairing=PairingStatus(SINGLE_RECORDING, ()),
        census=_census(total=4, positives=4, negatives=0, folds=1),
        library_version="abc",
        calibration_version="def",
        cosine_only=False,
    )
    assert refusal is not None
    assert refusal["reason"] == gate_module.UNCALIBRATED_LIBRARY
    assert "singer fold" in refusal["detail"]


def test_a_legacy_library_is_refused_even_when_auc_is_measured(gate_module):
    """An empty ``material_id`` cannot be verified as cross-recording.

    This is the dangerous case: segment-as-material makes positives out of two
    windows of one take, so the AUC *is* measured (``calibrated=True``) and only
    the pairing status exposes the leak.  R29 section 7 forbids citing it.
    """
    refusal = gate_module.calibration_refusal(
        calibrated=True,
        pairing=PairingStatus(UNVERIFIABLE_LEGACY, ()),
        census=_census(total=77, positives=77, negatives=0, folds=5),
        library_version="abc",
        calibration_version="def",
        cosine_only=False,
    )
    assert refusal is not None
    assert refusal["reason"] == gate_module.UNVERIFIABLE_LIBRARY
    assert "material_id" in refusal["detail"]
    assert "rebuild" in refusal["hint"]


def test_cosine_only_is_the_explicit_way_through(gate_module):
    """The diagnostic flag bypasses both refusals -- that is its whole point."""
    for calibrated, pairing in (
        (False, PairingStatus(SINGLE_RECORDING, ("s1",))),
        (True, PairingStatus(UNVERIFIABLE_LEGACY, ())),
        (False, PairingStatus(SINGLE_RECORDING, ())),
    ):
        assert (
            gate_module.calibration_refusal(
                calibrated=calibrated,
                pairing=pairing,
                census=_census(),
                library_version="abc",
                calibration_version="def",
                cosine_only=True,
            )
            is None
        )


def test_the_legacy_hatch_lifts_exactly_one_refusal(gate_module):
    """``--allow-unverifiable-legacy`` re-measures an old library, nothing else.

    It exists so a pre-R29 number can be reproduced; it must not become a
    general "ignore the gate" switch.  An uncalibratable library stays refused,
    because there the calibrated score is the cosine fallback and no label would
    make the number true.
    """
    legacy = gate_module.calibration_refusal(
        calibrated=True,
        pairing=PairingStatus(UNVERIFIABLE_LEGACY, ()),
        census=_census(total=80, positives=43, negatives=37, folds=5),
        library_version="abc",
        calibration_version="def",
        cosine_only=False,
        allow_unverifiable_legacy=True,
    )
    assert legacy is None

    uncalibrated = gate_module.calibration_refusal(
        calibrated=False,
        pairing=PairingStatus(SINGLE_RECORDING, ("s1",)),
        census=_census(),
        library_version="abc",
        calibration_version="def",
        cosine_only=False,
        allow_unverifiable_legacy=True,
    )
    assert uncalibrated is not None
    assert uncalibrated["reason"] == gate_module.UNCALIBRATED_LIBRARY

    # The states can coincide.  Checking the legacy branch first and returning
    # early would let this combined case through with the cosine fallback.
    legacy_and_uncalibrated = gate_module.calibration_refusal(
        calibrated=False,
        pairing=PairingStatus(UNVERIFIABLE_LEGACY, ()),
        census=_census(),
        library_version="abc",
        calibration_version="def",
        cosine_only=False,
        allow_unverifiable_legacy=True,
    )
    assert legacy_and_uncalibrated is not None
    assert legacy_and_uncalibrated["reason"] == gate_module.UNCALIBRATED_LIBRARY


def test_metric_kind_names_what_the_run_measured(gate_module):
    """Three states, three labels -- the label travels with the numbers."""
    assert gate_module.metric_kind_for(cosine_only=False, r29_verified=True) == "calibrated"
    assert (
        gate_module.metric_kind_for(cosine_only=False, r29_verified=False)
        == "calibrated_unverified_legacy"
    )
    # A cosine run never fitted a calibration, so the pairing state is moot.
    for verified in (True, False):
        assert (
            gate_module.metric_kind_for(cosine_only=True, r29_verified=verified)
            == "cosine_only_diagnostic"
        )


def _two_singer_library(root: Path) -> Path:
    """Build and save a 2-singer, 1-recording-each library.

    Two singers land in two folds of one singer each (``pairs.py``), so
    ``build_pairs`` emits nothing and ``fit_calibration`` returns the canonical
    cosine-only fallback: exactly the state the gate exists for.
    """
    entries = []
    for singer_id, f0 in (("s1", 220.0), ("s2", 330.0)):
        path = root / f"{singer_id}.wav"
        sf.write(path, _tone(f0), SAMPLE_RATE)
        entries.append((Singer(singer_id, singer_id.upper(), "female", "pop"), path))
    # The build says out loud what the fixture is: one recording per singer.
    with pytest.warns(SingleMaterialSingerWarning):
        index = build_library(entries, HashEncoder())
    out = root / "lib"
    save_library(index, out)
    return out


@pytest.fixture(scope="module")
def two_singer_library(tmp_path_factory) -> Path:
    """One build shared by the tests that need a real library on disk."""
    return _two_singer_library(tmp_path_factory.mktemp("holdout-gate"))


def test_entry_point_refuses_before_touching_the_material(
    two_singer_library, tmp_path
):
    """Exit 2 + one JSON error document, and no hold-out download.

    ``--material`` points at a directory that does not exist: if the gate ran
    after ``build_holdout_set`` the run would fail on that path instead of
    reporting the calibration state.  Reaching the gate first is part of the
    contract -- a refused run must not fetch audio.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--library",
            str(two_singer_library),
            "--material",
            str(tmp_path / "no-such-material"),
            "--encoder",
            "hash",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, (result.stdout, result.stderr)
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["status"] == "error"
    assert report["reason"] == "uncalibrated_library"
    assert report["pairs"]["positives"] == 0
    assert report["pairing"]["single_material_singers"] == ["s1", "s2"]
    # No metrics key at all: the refusal withholds numbers rather than nulling
    # them, so a key-by-key consumer cannot read 0.0 as a measured top1.
    assert "metrics" not in report


def test_per_query_rows_carry_the_metric_label(gate_module, two_singer_library, tmp_path):
    """A sidecar outlives the report it came from, so each row is self-labelled.

    The ranking itself is the product path (quality gate -> shared window rule
    -> features -> encoder -> ``rank_singers``); this test only pins that the
    diagnostic label and the per-query truth rank survive into the file.
    """
    index = load_library(two_singer_library)
    model = gate_module.fit_calibration(
        np.empty((0, 5)), np.empty(0), np.empty(0)
    )
    query_path = tmp_path / "query.wav"
    sf.write(query_path, _voiced(), SAMPLE_RATE)
    query = EvalQuery(
        query_id="q1",
        path=query_path,
        gender="female",
        candidates=("s1", "s2"),
        picked={"s1"},
        annotator_count=1,
    )
    rows = gate_module.per_query_rows(
        [query], index, model, HashEncoder(), gate_module.COSINE_ONLY_DIAGNOSTIC
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["metric_kind"] == "cosine_only_diagnostic"
    assert row["accepted"] is True
    assert row["skipped"] is False
    assert len(row["ranking"]) == 2  # the whole gender pool was ranked
    assert row["rank_of_truth"] in (1, 2)
    assert row["top1"] == (row["rank_of_truth"] == 1)
    # JSON-ready as written: no numpy scalars leak into the sidecar.
    json.dumps(rows)

