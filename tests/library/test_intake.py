"""Tests for material intake and its rights gate.

Intake is the boundary between "we have audio files" and "we have a catalog".
The contract under test is that a commercial declaration backed by a
non-commercial licence — or lacking verified consent — is refused before any
catalog is written, and that a research declaration is stamped non-commercial
so the restriction travels with the material.
"""

import json

import pytest

from voicematch.library.catalog import load_catalog
from voicematch.library.intake import (
    IntakeError,
    check_rights,
    classify_license,
    ingest,
    load_manifest,
    read_catalog_rights,
)


def _manifest(tmp_path, **overrides):
    """Write a manifest plus one wav per singer folder; return its path."""
    singers = overrides.pop(
        "singers",
        [
            {
                "singer_id": "s1",
                "name": "Singer One",
                "gender": "female",
                "style_tag": "pop",
                "dir": "material/s1",
            }
        ],
    )
    for singer in singers:
        folder = tmp_path / singer["dir"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "take1.wav").write_bytes(b"not real audio")
    data = {
        "source": "VocalSet (Zenodo 10.5281/zenodo.1442513)",
        "license": "CC-BY-4.0",
        "intended_use": "research",
        "rights_verified": False,
        "consent_ref": None,
        "singers": singers,
        **overrides,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_license_classification_fails_closed():
    assert classify_license("CC-BY-4.0") == "possibly_commercial"
    assert classify_license("CC0-1.0") == "possibly_commercial"
    for bad in (
        "CC BY-NC-SA 4.0",
        "CC-BY-NC-ND",
        "research-only",
        "unknown",
        "TO_FILL",
        "",
        None,
        5,
    ):
        assert classify_license(bad) == "non_commercial"


def test_research_use_is_allowed_and_stamped_non_commercial(tmp_path):
    manifest = _manifest(tmp_path, license="CC-BY-NC-SA-4.0")
    result = ingest(manifest, tmp_path / "catalog")

    assert result["status"] == "ok"
    rights = result["rights_block"]
    assert rights["intended_use"] == "research"
    assert rights["commercial_use"] is False
    assert rights["license_class"] == "non_commercial"
    sidecar = json.loads((tmp_path / "catalog" / "RIGHTS.json").read_text())
    assert sidecar == rights


def test_generated_catalog_is_loadable_and_carries_rights(tmp_path):
    manifest = _manifest(tmp_path)
    result = ingest(manifest, tmp_path / "catalog")
    catalog_path = tmp_path / "catalog" / "catalog.json"

    entries = load_catalog(catalog_path)
    assert [(s.singer_id, p.name) for s, p in entries] == [("s1", "take1.wav")]
    assert entries[0][1].is_file()
    assert read_catalog_rights(catalog_path)["source"] == result["rights_block"]["source"]


def test_catalog_paths_survive_moving_the_output_folder(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    out = tmp_path / "catalog"
    ingest(manifest, out)
    # Paths are relative to the catalog, so reading it from another cwd works.
    monkeypatch.chdir(tmp_path)
    assert load_catalog(out / "catalog.json")[0][1].is_file()


def test_product_use_with_non_commercial_license_is_refused(tmp_path):
    manifest = _manifest(
        tmp_path,
        license="CC BY-NC 4.0",
        intended_use="product",
        rights_verified=True,
        consent_ref="consent.pdf",
    )
    out = tmp_path / "catalog"

    with pytest.raises(IntakeError, match="forbids commercial use"):
        ingest(manifest, out)
    assert not out.exists()


def test_product_use_without_verified_rights_is_refused(tmp_path):
    manifest = _manifest(
        tmp_path,
        intended_use="product",
        rights_verified=False,
        consent_ref="consent.pdf",
    )
    with pytest.raises(IntakeError, match="rights_verified"):
        ingest(manifest, tmp_path / "catalog")


def test_product_use_without_consent_reference_is_refused(tmp_path):
    manifest = _manifest(
        tmp_path,
        intended_use="product",
        rights_verified=True,
        consent_ref=None,
    )
    with pytest.raises(IntakeError, match="consent_ref"):
        ingest(manifest, tmp_path / "catalog")


def test_product_use_with_full_attestation_is_allowed(tmp_path):
    manifest = _manifest(
        tmp_path,
        intended_use="product",
        rights_verified=True,
        consent_ref="signed/consent-2026.pdf",
    )
    result = ingest(manifest, tmp_path / "catalog")

    assert result["rights_block"]["commercial_use"] is True
    assert result["rights_block"]["consent_ref"] == "signed/consent-2026.pdf"


def test_unknown_license_fails_closed_for_product(tmp_path):
    manifest = _manifest(
        tmp_path,
        license="unknown",
        intended_use="product",
        rights_verified=True,
        consent_ref="consent.pdf",
    )
    with pytest.raises(IntakeError, match="forbids commercial use"):
        ingest(manifest, tmp_path / "catalog")


@pytest.mark.parametrize(
    "override, message",
    [
        ({"source": ""}, "source"),
        ({"license": None}, "license must be a string"),
        ({"intended_use": "internal"}, "intended_use"),
        ({"rights_verified": "yes"}, "rights_verified"),
        ({"singers": []}, "singers"),
    ],
)
def test_malformed_declaration_is_refused(tmp_path, override, message):
    manifest = _manifest(tmp_path, **override)
    with pytest.raises(IntakeError, match=message):
        ingest(manifest, tmp_path / "catalog")


def test_missing_manifest_and_invalid_json_are_reported(tmp_path):
    with pytest.raises(IntakeError, match="not found"):
        load_manifest(tmp_path / "none.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(IntakeError, match="not valid JSON"):
        load_manifest(bad)


def test_a_singer_folder_without_audio_is_refused(tmp_path):
    manifest = _manifest(tmp_path)
    data = json.loads(manifest.read_text())
    data["singers"][0]["dir"] = "empty"
    (tmp_path / "empty").mkdir()
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(IntakeError, match="no audio files"):
        ingest(manifest, tmp_path / "catalog")


def test_a_duplicate_singer_id_is_refused(tmp_path):
    singer = {
        "singer_id": "s1",
        "name": "Singer One",
        "gender": "female",
        "style_tag": "pop",
        "dir": "material/s1",
    }
    manifest = _manifest(tmp_path, singers=[singer, dict(singer)])

    with pytest.raises(IntakeError, match="duplicate singer_id"):
        ingest(manifest, tmp_path / "catalog")


def test_bad_gender_is_refused(tmp_path):
    manifest = _manifest(
        tmp_path,
        singers=[
            {
                "singer_id": "s1",
                "name": "One",
                "gender": "other",
                "style_tag": "pop",
                "dir": "material/s1",
            }
        ],
    )
    folder = tmp_path / "material" / "s1"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "a.wav").write_bytes(b"x")

    with pytest.raises(IntakeError, match="gender"):
        ingest(manifest, tmp_path / "catalog")


def test_manual_catalog_without_rights_reports_none(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"singers": []}), encoding="utf-8")
    assert read_catalog_rights(catalog) is None


def test_check_rights_directly_reports_the_license_class():
    rights = check_rights(
        {
            "source": "self-collected",
            "license": "CC-BY-4.0",
            "intended_use": "product",
            "rights_verified": True,
            "consent_ref": "consent.pdf",
        }
    )
    assert rights["license_class"] == "possibly_commercial"
    assert rights["commercial_use"] is True
