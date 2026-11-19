"""Tests for the zero-cost walkthrough and the freeze rights lookup."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
WALKTHROUGH = TOOLS / "benchmark_walkthrough.py"
BENCHMARK_CLI = TOOLS / "benchmark.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def walk():
    return _load(WALKTHROUGH, "benchmark_walkthrough_under_test")


@pytest.fixture(scope="module")
def cli():
    return _load(BENCHMARK_CLI, "benchmark_cli_under_test")


def _make_material(root: Path, n_female: int, n_male: int) -> tuple[Path, Path]:
    """Minimal catalog + lib manifest with a controlled same-gender pool."""
    lib = root / "lib"
    lib.mkdir()
    singers = {f"f{i}": {"gender": "female"} for i in range(n_female)}
    singers.update({f"m{i}": {"gender": "male"} for i in range(n_male)})
    (lib / "manifest.json").write_text(
        json.dumps({"singers": singers, "model_id": "HUBERT_BASE"}),
        encoding="utf-8",
    )
    catalog_dir = root / "catalog"
    catalog_dir.mkdir()
    catalog = catalog_dir / "catalog.json"
    catalog.write_text(
        json.dumps({"singers": [{"singer_id": sid} for sid in singers]}),
        encoding="utf-8",
    )
    return catalog, lib


def test_preflight_rejects_small_gender_pool(tmp_path, walk):
    """10-singer library (5/5) must be rejected, not silently degraded."""
    catalog, lib = _make_material(tmp_path, 5, 5)
    with pytest.raises(ValueError) as exc:
        walk.preflight(catalog, lib)
    result = exc.value.args[0]
    assert result["status"] == "fail"
    assert result["gender_pools"] == {"female": 5, "male": 5}
    assert any("< 8" in p for p in result["problems"])


def test_preflight_accepts_20_singer_pool(tmp_path, walk):
    catalog, lib = _make_material(tmp_path, 10, 10)
    result = walk.preflight(catalog, lib)
    assert result["status"] == "ok"
    assert result["n_library_singers"] == 20
    assert result["problems"] == []


def test_preflight_rejects_missing_catalog(tmp_path, walk):
    _, lib = _make_material(tmp_path, 10, 10)
    with pytest.raises(ValueError) as exc:
        walk.preflight(tmp_path / "nope.json", lib)
    assert any("catalog" in p for p in exc.value.args[0]["problems"])


def test_fake_annotations_are_valid(walk):
    """Script-fabricated annotations must satisfy the same contract as real ones."""
    sheet = [
        {"query_id": "q1", "candidates": ["a", "b", "c", "d", "e", "f", "g", "h"]},
        {"query_id": "q2", "candidates": ["d", "e", "f", "g", "h", "i", "j", "k"]},
    ]
    provenance = {
        "queries": [
            {"query_id": "q1", "system_top": ["a", "b", "c"], "scorable": True},
            {"query_id": "q2", "system_top": ["d", "e", "f"], "scorable": True},
        ]
    }
    anns = walk._fake_annotations(sheet, provenance)
    assert len(anns) == len(sheet) * 3
    assert {a["annotator_id"] for a in anns} == {"rater1", "rater2", "rater3"}
    allowed = {e["query_id"]: set(e["candidates"]) for e in sheet}
    for ann in anns:
        assert set(ann["picked"]) <= allowed[ann["query_id"]]
        assert len(ann["picked"]) <= 2
    # last query's rater3 deliberately answers "none of these" (label diversity)
    assert [a["picked"] for a in anns
            if a["query_id"] == "q2" and a["annotator_id"] == "rater3"] == [[]]


def test_fake_annotations_handles_unscorable_query(walk):
    """A query with no system Top-3 still gets a legal in-candidates pick."""
    sheet = [{"query_id": "q1", "candidates": ["a", "b", "c"]}]
    provenance = {"queries": [{"query_id": "q1", "system_top": [], "scorable": False}]}
    anns = walk._fake_annotations(sheet, provenance)
    assert all(set(a["picked"]) <= {"a", "b", "c"} for a in anns)


def test_demo_notice_warns_about_synthetic_data(walk):
    assert "合成" in walk.DEMO_NOTICE
    assert "DEMO" in walk.DEMO_NOTICE
    assert "严禁" in walk.DEMO_NOTICE
    assert walk.REQUIRED_POOL == 8


def test_load_rights_sibling_catalog(tmp_path, cli):
    """The real layout: RIGHTS.json sits beside lib/ in catalog/, not lib/catalog/."""
    lib = tmp_path / "lib"
    lib.mkdir()
    sibling = tmp_path / "catalog" / "RIGHTS.json"
    sibling.parent.mkdir()
    rights = {"license": "CC-BY-NC-SA-4.0", "commercial_use": False}
    sibling.write_text(json.dumps(rights), encoding="utf-8")
    assert cli._load_rights(lib, None) == rights


def test_load_rights_precedence(tmp_path, cli):
    """Explicit --rights wins; lib/catalog/ wins over sibling; absent -> {}."""
    lib = tmp_path / "lib"
    (lib / "catalog").mkdir(parents=True)
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"source": "explicit"}), encoding="utf-8")
    (lib / "catalog" / "RIGHTS.json").write_text(
        json.dumps({"source": "in_lib"}), encoding="utf-8"
    )
    sibling = tmp_path / "catalog" / "RIGHTS.json"
    sibling.parent.mkdir(exist_ok=True)
    sibling.write_text(json.dumps({"source": "sibling"}), encoding="utf-8")

    assert cli._load_rights(lib, explicit) == {"source": "explicit"}
    assert cli._load_rights(lib, None) == {"source": "in_lib"}

    # Isolate from tmp_path/catalog so the sibling lookup also misses.
    isolated = tmp_path / "isolated"
    empty_lib = isolated / "lib"
    empty_lib.mkdir(parents=True)
    assert cli._load_rights(empty_lib, None) == {}
