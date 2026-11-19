"""Guard the offline annotation UI's dual-mode contract.

The UI is a standalone HTML file with no build step, so these structural checks
are the only automated guard that both modes exist and export the exact field
names the CLI consumers (adjudicate / pairwise-adjudicate) expect.
"""

from __future__ import annotations

from pathlib import Path

UI = Path(__file__).resolve().parents[2] / "docs" / "benchmark" / "annotation-ui.html"


def _source() -> str:
    return UI.read_text(encoding="utf-8")


def test_ui_exists_and_is_self_contained():
    source = _source()
    assert "<!DOCTYPE html>" in source
    # No network / CDN dependencies: annotation runs fully offline.
    assert "http://" not in source and "https://" not in source
    assert "webkitdirectory" in source


def test_ui_has_both_modes():
    source = _source()
    for token in ("modeMulti", "modePair", "多选标注", "pairwise 比较"):
        assert token in source
    assert "setMode('multi')" in source
    assert "setMode('pair')" in source


def test_ui_multi_export_fields_match_adjudicate():
    source = _source()
    # multi annotations.json -> {query_id, annotator_id, picked}
    assert "query_id: r.query_id" in source
    assert "annotator_id: state.annotator" in source
    assert "picked: state.answers[r.query_id]" in source


def test_ui_pair_export_fields_match_pairwise_adjudicate():
    source = _source()
    # pair annotations -> {query_id, pair_id, annotator_id, winner}
    assert "pair_id: t.pair_id" in source
    assert "winner: state.answers[t.pair_id]" in source
    assert "pair-annotations-" in source


def test_ui_loads_required_json_files():
    source = _source()
    for needed in ('findJson("sheet.json")', 'findJson("samples.json")', 'findJson("pairs.json")'):
        assert needed in source


def test_ui_js_braces_balanced():
    source = _source()
    js = source.split("<script>")[1].split("</script>")[0]
    assert js.count("{") == js.count("}")
    assert js.count("(") == js.count(")")
