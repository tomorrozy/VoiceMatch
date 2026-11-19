"""Tests for the human-edited catalog parser.

The catalog is the one file an operator writes by hand, so the contract here is
that bad input fails *before* any audio is decoded: a typo in the gender, a
wrong path or a duplicate id must name the offending record instead of
producing a library that silently ranks into the wrong pool or scores nothing.
"""

import json

import pytest

from voicematch.library.catalog import CatalogError, load_catalog
from voicematch.library.schema import Singer

_FEMALE = {
    "singer_id": "f01",
    "name": "Singer One",
    "gender": "female",
    "style_tag": "pop",
    "materials": ["f01/take1.wav", "f01/take2.wav"],
}


def _write_catalog(tmp_path, singers, filename="catalog.json"):
    """Write a catalog JSON into ``tmp_path`` and make its material files.

    Only valid relative path strings are created; tests that feed a bad
    ``materials`` value rely on validation rejecting it before any file check.
    """
    for singer in singers:
        materials = singer.get("materials", [])
        if not isinstance(materials, list):
            continue
        for material in materials:
            if not isinstance(material, str) or not material.strip():
                continue
            path = tmp_path / material
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not real audio")
    path = tmp_path / filename
    path.write_text(json.dumps({"singers": singers}), encoding="utf-8")
    return path


def test_resolves_material_paths_relative_to_the_catalog(tmp_path):
    catalog = _write_catalog(tmp_path, [_FEMALE])
    # Invoke from a different working directory: resolution is still relative
    # to the catalog file, not the process cwd.
    entries = load_catalog(catalog)

    assert [(singer.singer_id, path) for singer, path in entries] == [
        ("f01", tmp_path / "f01" / "take1.wav"),
        ("f01", tmp_path / "f01" / "take2.wav"),
    ]
    assert entries[0][0] == Singer("f01", "Singer One", "female", "pop")


def test_one_entry_per_material_so_build_can_merge_them(tmp_path):
    # build_library groups by singer_id; the catalog expands materials into
    # pairs and lets the builder do the merge, rather than pre-merging here.
    catalog = _write_catalog(tmp_path, [_FEMALE])
    assert len(load_catalog(catalog)) == 2


def test_an_absolute_material_path_is_used_as_is(tmp_path):
    absolute = tmp_path / "elsewhere.wav"
    absolute.write_bytes(b"x")
    record = dict(_FEMALE, materials=[str(absolute)])
    catalog = _write_catalog(tmp_path, [record])

    assert load_catalog(catalog)[0][1] == absolute


def test_unknown_keys_are_ignored_so_a_template_can_carry_notes(tmp_path):
    record = dict(_FEMALE, note="replace these paths", extra=123)
    catalog = _write_catalog(tmp_path, [record])

    assert len(load_catalog(catalog)) == 2


def test_multiple_singers_are_kept_in_catalog_order(tmp_path):
    male = dict(
        _FEMALE,
        singer_id="m01",
        name="Singer Two",
        gender="male",
        style_tag="rock",
        materials=["m01/take1.wav"],
    )
    catalog = _write_catalog(tmp_path, [_FEMALE, male])

    ids = [singer.singer_id for singer, _ in load_catalog(catalog)]
    # f01 appears once per material, in catalog order, then the male singer.
    assert ids == ["f01", "f01", "m01"]
    assert list(dict.fromkeys(ids)) == ["f01", "m01"]


# --- validation: every failure names the bad record and decodes nothing -----


@pytest.mark.parametrize(
    "payload, message",
    [
        ([], "'singers'"),
        ({"singers": []}, "'singers'"),
        ({"singers": "not a list"}, "'singers'"),
        ("not an object", "JSON object"),
    ],
)
def test_malformed_top_level_is_rejected(tmp_path, payload, message):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CatalogError, match=message):
        load_catalog(catalog)


def test_missing_catalog_file_is_reported(tmp_path):
    with pytest.raises(CatalogError, match="not found"):
        load_catalog(tmp_path / "does-not-exist.json")


def test_invalid_json_is_reported(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{not json", encoding="utf-8")

    with pytest.raises(CatalogError, match="not valid JSON"):
        load_catalog(catalog)


@pytest.mark.parametrize(
    "field, bad_value, message",
    [
        ("singer_id", "", "singer_id"),
        ("singer_id", 7, "singer_id"),
        ("name", "", "name"),
        ("gender", "F", "gender"),
        ("gender", "other", "gender"),
        ("style_tag", 5, "style_tag"),
        ("materials", [], "materials"),
        ("materials", "a.wav", "materials"),
    ],
)
def test_a_bad_singer_field_is_rejected(tmp_path, field, bad_value, message):
    record = dict(_FEMALE, **{field: bad_value})
    catalog = _write_catalog(tmp_path, [record])

    with pytest.raises(CatalogError, match=message):
        load_catalog(catalog)


def test_a_missing_material_file_is_rejected_before_build(tmp_path):
    # A path that is not a file fails here, not halfway through decoding with
    # an opaque library error.
    record = dict(_FEMALE, materials=["f01/gone.wav"])
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"singers": [record]}), encoding="utf-8")

    with pytest.raises(CatalogError, match="missing file"):
        load_catalog(catalog)


def test_an_empty_material_path_is_rejected(tmp_path):
    record = dict(_FEMALE, materials=["   "])
    catalog = _write_catalog(tmp_path, [record])

    with pytest.raises(CatalogError, match="non-empty path"):
        load_catalog(catalog)


def test_a_duplicate_singer_id_is_rejected(tmp_path):
    catalog = _write_catalog(tmp_path, [_FEMALE, dict(_FEMALE, name="Other")])

    with pytest.raises(CatalogError, match="duplicate singer_id"):
        load_catalog(catalog)


def test_a_non_object_singer_record_is_rejected(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"singers": ["oops"]}), encoding="utf-8")

    with pytest.raises(CatalogError, match="must be an object"):
        load_catalog(catalog)
