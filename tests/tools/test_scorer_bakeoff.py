"""Unit tests for the scorer bake-off's deterministic scoring rules.

The expensive query path is exercised on real M4Singer material by the tool
itself; these tests pin the inexpensive parts whose silent drift would change
the comparison: ranking/ties, normalisation, fixed-weight arithmetic and
per-singer standardisation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from voicematch.calibrate.model import CalibrationModel
from voicematch.features.extract import FeatureVector

TOOL = Path(__file__).resolve().parents[2] / "tools" / "scorer_bakeoff.py"


@pytest.fixture(scope="module")
def bakeoff():
    """Import the script without constructing HuBERT or reading any library."""
    # The tool imports its sibling ``holdout_eval.py`` by module name.
    sys.path.insert(0, str(TOOL.parent))
    spec = importlib.util.spec_from_file_location("scorer_bakeoff_under_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _empty_features() -> FeatureVector:
    return FeatureVector(pitch={}, timbre={}, resonance={}, style={})


def _context(bakeoff, rows, projected_cos):
    return bakeoff.QueryContext(
        emb=np.zeros(1),
        fv=_empty_features(),
        cos={"a": rows["a"][0], "b": rows["b"][0]},
        rows=rows,
        projected_cos=projected_cos,
    )


def _scorer(bakeoff):
    return bakeoff.LibraryScorer(
        model=CalibrationModel(
            coef=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            mean=np.zeros(5),
            scale=np.ones(5),
            intercept=0.0,
            auc=0.8,
        ),
        names={},
        gender_of={},
        protos={},
        singer_vectors={},
        cos_stats={"a": (0.7, 0.1), "b": (0.9, 0.2)},
        lda=None,
        projected_protos={},
        projected_leave_one_take_out_top1=None,
    )


def test_normalise_is_unit_norm_and_leaves_zero_finite(bakeoff):
    assert np.allclose(bakeoff._normalise(np.array([3.0, 4.0])), [0.6, 0.8])
    assert np.array_equal(bakeoff._normalise(np.zeros(2)), np.zeros(2))


def _pool(*ids: str) -> dict[str, str]:
    """A one-gender pool: every candidate the ranking is allowed to see."""
    return {sid: "female" for sid in ids}


def test_rank_breaks_equal_scores_by_singer_id(bakeoff):
    scores_in = {"b": 0.5, "a": 0.5, "c": 0.1}
    ranking, scores, rank = bakeoff._rank(scores_in, "b", _pool(*scores_in), "female")
    assert ranking == ["a", "b", "c"]
    assert scores == [0.5, 0.5, 0.1]
    assert rank == 2


def test_rank_reports_a_truth_outside_top_three_by_its_real_position(bakeoff):
    scores_in = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
    ranking, _scores, rank = bakeoff._rank(scores_in, "d", _pool(*scores_in), "female")
    assert ranking == ["a", "b", "c"]
    assert rank == 4


def test_rank_excludes_the_other_gender_from_the_pool(bakeoff):
    """rank_singers scores a query's own gender only; the bake-off must too."""
    scores_in = {"a": 0.9, "m": 0.8, "b": 0.7}
    gender_of = {"a": "female", "b": "female", "m": "male"}
    ranking, _scores, rank = bakeoff._rank(scores_in, "a", gender_of, "female")
    assert ranking == ["a", "b"]
    assert rank == 1


def test_all_five_scorers_use_their_declared_inputs(bakeoff):
    rows = {
        "a": np.array([0.8, 0.0, 0.0, 0.0, 0.0]),
        "b": np.array([0.7, 1.0, 1.0, 1.0, 1.0]),
    }
    scorer = _scorer(bakeoff)
    scores = bakeoff._scores(
        _context(bakeoff, rows, {"a": 0.4, "b": 0.3}), scorer, scorer.model
    )
    assert set(scores) == {"cosine", "calibration", "fixed", "projection", "per_singer"}
    assert scores["cosine"] == {"a": 0.8, "b": 0.7}
    # [1.0, -0.5, -0.5, -0.25, -0.25] . [0.7, 1, 1, 1, 1] = -0.8
    assert scores["fixed"] == {"a": 0.8, "b": pytest.approx(-0.8)}
    assert scores["per_singer"] == {"a": pytest.approx(1.0), "b": pytest.approx(-1.0)}
    assert scores["projection"] == {"a": 0.4, "b": 0.3}
    assert scores["calibration"]["a"] > scores["calibration"]["b"]


def test_per_singer_zero_std_falls_back_to_unit_scale(bakeoff):
    scorer = _scorer(bakeoff)
    scorer.cos_stats["a"] = (0.7, 0.0)
    rows = {"a": np.array([0.8, 0.0, 0.0, 0.0, 0.0]), "b": np.zeros(5)}
    scores = bakeoff._scores(
        _context(bakeoff, rows, {"a": 0.0, "b": 0.0}), scorer, scorer.model
    )
    assert np.isfinite(scores["per_singer"]["a"])
    assert scores["per_singer"]["a"] == pytest.approx(0.1)
