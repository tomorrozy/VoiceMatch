"""Turn a folder of real a-cappella material into a validated catalog.

This is the intake step between "we have audio files" and :mod:`catalog`:
an operator supplies a small manifest naming the source, its licence, the
intended use and one folder per singer.  Intake checks the rights declaration
**before** writing anything, then emits a catalog the build CLI can consume
plus a ``RIGHTS.json`` sidecar that travels with the material.

Why a gate here
---------------
The project's product goal is commercial, but nearly every public singing
corpus is CC BY-NC or research-only.  A permissive-looking dataset can be
silently dragged into a product library and turn into a legal problem that is
found late.  The gate makes that a loud, early failure:

- ``intended_use = "research"`` is always allowed; the output is stamped
  ``commercial_use: false`` so nobody later mistakes the built library for a
  product artefact.
- ``intended_use = "product"`` requires a licence that is **not** known to be
  non-commercial, plus ``rights_verified: true`` and a ``consent_ref`` naming
  the signed performer/release document.  Unknown licences fail closed: an
  absent licence is not permission.

The gate is a declaration check, not legal advice.  It cannot verify that a
consent document is genuine or that a licence truly covers this use; it makes
the responsible person state the facts in a file that is kept next to the
material and echoed into every build result.

Manifest shape (see ``docs/material-manifest.example.json``)::

    {
      "source": "VocalSet (Zenodo 10.5281/zenodo.1442513)",
      "license": "CC-BY-4.0",
      "intended_use": "research",
      "rights_verified": false,
      "consent_ref": null,
      "singers": [
        {"singer_id": "male01", "name": "Male 01", "gender": "male",
         "style_tag": "technique", "dir": "male01"}
      ]
    }

Each ``dir`` is relative to the manifest's parent and must hold at least one
audio file.  Intake writes material paths into the catalog relative to the
catalog file, so the output directory and the material can be moved together.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from voicematch.library.catalog import CatalogError

#: Licence tokens that demonstrably forbid commercial use or derivative works.
#: Matching is deliberately broad: if a licence might be non-commercial, a
#: product build must be blocked rather than argued about later.
_NON_COMMERCIAL_PATTERN = re.compile(
    r"(^|[^a-z])"
    r"(nc|non[-_ ]?commercial|research[-_ ]?only|academic[-_ ]?only|nd|no[-_ ]?deriv)"
    r"([^a-z]|$)",
    re.IGNORECASE,
)

_AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3", ".m4a")


class IntakeError(ValueError):
    """The manifest or its rights declaration is not usable."""


def load_manifest(path: Path) -> dict[str, Any]:
    """Parse a material manifest as a plain dict, failing with clear messages."""
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise IntakeError(f"manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise IntakeError(f"manifest {manifest_path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise IntakeError("manifest must be a JSON object")
    return raw


def classify_license(license_name: object) -> str:
    """Return ``"non_commercial"`` or ``"possibly_commercial"``.

    Unknown or missing licences become ``"non_commercial"`` — fail closed.  An
    explicit CC-BY / CC0 / proprietary string without an NC or ND token is the
    only way to reach ``"possibly_commercial"``.
    """
    if not isinstance(license_name, str) or not license_name.strip():
        return "non_commercial"
    # Explicit placeholders are an admission that nobody checked the licence.
    if license_name.strip().lower() in {"unknown", "unspecified", "to_fill", "tbd", "none"}:
        return "non_commercial"
    if _NON_COMMERCIAL_PATTERN.search(license_name):
        return "non_commercial"
    return "possibly_commercial"


def check_rights(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the rights declaration and return the rights block to persist.

    Raises :class:`IntakeError` when a product use is not backed by a licence
    that could permit it plus an attested consent reference.  Research use is
    allowed for any declared licence and is always stamped non-commercial.
    """
    source = manifest.get("source")
    if not isinstance(source, str) or not source.strip():
        raise IntakeError("manifest.source must name the dataset or collection")

    license_name = manifest.get("license")
    license_class = classify_license(license_name)
    if license_class == "non_commercial" and not isinstance(license_name, str):
        raise IntakeError("manifest.license must be a string (use 'unknown' if unsure)")

    intended_use = manifest.get("intended_use")
    if intended_use not in ("research", "product"):
        raise IntakeError(
            "manifest.intended_use must be 'research' or 'product', "
            f"got {intended_use!r}"
        )

    verified = manifest.get("rights_verified")
    if not isinstance(verified, bool):
        raise IntakeError("manifest.rights_verified must be a boolean")
    consent_ref = manifest.get("consent_ref")

    if intended_use == "product":
        if license_class == "non_commercial":
            raise IntakeError(
                f"license {license_name!r} forbids commercial use; a product "
                "library cannot be built from this material"
            )
        if not verified:
            raise IntakeError(
                "product use requires rights_verified: true — someone must "
                "confirm the licence and performer consent"
            )
        if not isinstance(consent_ref, str) or not consent_ref.strip():
            raise IntakeError(
                "product use requires consent_ref naming the signed consent "
                "or release document"
            )

    return {
        "source": source,
        "license": license_name if isinstance(license_name, str) else "unknown",
        "license_class": license_class,
        "intended_use": intended_use,
        "commercial_use": intended_use == "product",
        "rights_verified": verified,
        "consent_ref": consent_ref if isinstance(consent_ref, str) else None,
    }


def ingest(manifest_path: Path, out_dir: Path) -> dict[str, Any]:
    """Validate ``manifest_path`` and write ``catalog.json`` + ``RIGHTS.json``.

    Returns a summary with the catalog path, rights block, singer count and
    material file count.  No audio is decoded and no embedding is computed;
    this only checks the declaration and that each singer folder has files.
    """
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    rights = check_rights(manifest)

    singers = manifest.get("singers")
    if not isinstance(singers, list) or not singers:
        raise IntakeError("manifest.singers must be a non-empty list")

    base = manifest_path.parent
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog_path = out / "catalog.json"

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    file_count = 0
    for index, singer in enumerate(singers):
        where = f"singers[{index}]"
        if not isinstance(singer, dict):
            raise IntakeError(f"{where} must be an object")
        singer_id = singer.get("singer_id")
        if not isinstance(singer_id, str) or not singer_id.strip():
            raise IntakeError(f"{where}.singer_id must be a non-empty string")
        if singer_id in seen_ids:
            raise IntakeError(f"duplicate singer_id {singer_id!r}")
        seen_ids.add(singer_id)
        name = singer.get("name")
        if not isinstance(name, str) or not name.strip():
            raise IntakeError(f"{where}.name must be a non-empty string")
        if singer.get("gender") not in ("male", "female"):
            raise IntakeError(f"{where}.gender must be 'male' or 'female'")
        if not isinstance(singer.get("style_tag"), str):
            raise IntakeError(f"{where}.style_tag must be a string")
        folder = singer.get("dir")
        if not isinstance(folder, str) or not folder.strip():
            raise IntakeError(f"{where}.dir must be a non-empty folder path")

        singer_dir = base / folder
        if not singer_dir.is_dir():
            raise IntakeError(f"{where}.dir is not a directory: {singer_dir}")
        materials = sorted(
            path
            for path in singer_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES
        )
        if not materials:
            raise IntakeError(
                f"{where}.dir has no audio files ({', '.join(_AUDIO_SUFFIXES)}): "
                f"{singer_dir}"
            )
        # Paths in the catalog are relative to the catalog file, so the output
        # folder and the material can be relocated together.
        file_count += len(materials)
        records.append(
            {
                "singer_id": singer_id,
                "name": name,
                "gender": singer["gender"],
                "style_tag": singer["style_tag"],
                "materials": [
                    Path(os.path.relpath(path, catalog_path.parent)).as_posix()
                    for path in materials
                ],
            }
        )

    catalog = {"singers": records, "_rights": rights}
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rights_path = out / "RIGHTS.json"
    rights_path.write_text(
        json.dumps(rights, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "status": "ok",
        "catalog": str(catalog_path),
        "rights": str(rights_path),
        "singers": len(records),
        "materials": file_count,
        "rights_block": rights,
    }


def read_catalog_rights(catalog_path: Path) -> dict[str, Any] | None:
    """Read the ``_rights`` block from a generated catalog, if present.

    Manually written catalogs (``docs/catalog.example.json``) have no such
    block; the build CLI then reports no rights information rather than
    assuming one.  This keeps the intake boundary visible in the build output.
    """
    try:
        raw = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    rights = raw.get("_rights")
    return rights if isinstance(rights, dict) else None
