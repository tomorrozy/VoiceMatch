"""Read a human-edited singer catalog into ``build_library``'s input shape.

A catalog is a small JSON file that names the singers and points at their
material.  It is the one piece of the pipeline an operator writes by hand, so
it is validated hard before any audio is decoded: a misspelt gender or a wrong
material path must fail here with a clear message, not produce a library that
silently ranks a singer in the wrong pool or builds an empty index.

Shape (see ``docs/catalog.example.json``)::

    {
      "singers": [
        {
          "singer_id": "studio_01",
          "name": "Singer Name",
          "gender": "female",
          "style_tag": "pop",
          "materials": ["audio/studio_01/take1.wav", "audio/studio_01/take2.wav"]
        }
      ]
    }

Material paths are resolved **relative to the catalog file's directory**, not
the process's working directory.  The catalog therefore travels with its audio
as one folder, and invoking the build from anywhere gives the same files.  The
paths themselves never enter the library version hash (``build_library`` hashes
file *bytes*), so moving the folder does not change the identity of the library.

This module only parses and validates; it does not decode audio or fabricate
material.  A catalog that points at missing files is rejected, and an empty
``materials`` list is rejected rather than producing a singer that can never be
scored.  Whether a material actually yields usable windows is decided later by
``build_library``, which skips silent/short material with a warning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voicematch.library.schema import Singer

#: Genders a singer may be catalogued under.  The query pool is chosen by this
#: field, so any other value would strand the singer in neither pool.
_GENDERS = ("male", "female")


class CatalogError(ValueError):
    """The catalog is malformed; no library should be built from it."""


def load_catalog(path: Path) -> list[tuple[Singer, Path]]:
    """Parse ``path`` into the ``(Singer, material path)`` pairs build takes.

    ``path`` must be a JSON object with a non-empty ``singers`` list.  Each
    singer needs a unique non-empty ``singer_id``, a non-empty ``name``, a
    ``gender`` of ``"male"``/``"female"``, a string ``style_tag`` and a non-empty
    ``materials`` list of path strings.  Material paths are resolved relative to
    the catalog file's parent and must exist as files.

    Unknown keys are ignored so a template can carry human notes.  Every
    validation failure raises :class:`CatalogError` naming the offending singer
    or material; nothing is decoded and no partial result is returned.
    """
    catalog_path = Path(path)
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogError(f"catalog not found: {catalog_path}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogError(f"catalog {catalog_path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise CatalogError("catalog must be a JSON object with a 'singers' key")
    singers = raw.get("singers")
    if not isinstance(singers, list) or not singers:
        raise CatalogError("catalog 'singers' must be a non-empty list")

    base = catalog_path.parent
    entries: list[tuple[Singer, Path]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(singers):
        singer, materials = _parse_singer(record, index, base, seen_ids)
        entries.extend((singer, material) for material in materials)
    return entries


def _parse_singer(
    record: Any, index: int, base: Path, seen_ids: set[str]
) -> tuple[Singer, list[Path]]:
    """Validate one singer record and resolve its material paths."""
    where = f"singers[{index}]"
    if not isinstance(record, dict):
        raise CatalogError(f"{where} must be an object")

    singer_id = record.get("singer_id")
    if not isinstance(singer_id, str) or not singer_id.strip():
        raise CatalogError(f"{where}.singer_id must be a non-empty string")
    if singer_id in seen_ids:
        raise CatalogError(f"duplicate singer_id {singer_id!r}")
    seen_ids.add(singer_id)

    name = record.get("name")
    if not isinstance(name, str) or not name.strip():
        raise CatalogError(f"{where}.name must be a non-empty string")

    gender = record.get("gender")
    if gender not in _GENDERS:
        raise CatalogError(
            f"{where}.gender must be one of {_GENDERS}, got {gender!r}"
        )

    style_tag = record.get("style_tag")
    if not isinstance(style_tag, str):
        raise CatalogError(f"{where}.style_tag must be a string")

    materials = record.get("materials")
    if not isinstance(materials, list) or not materials:
        raise CatalogError(f"{where}.materials must be a non-empty list of paths")

    resolved: list[Path] = []
    for material_index, value in enumerate(materials):
        material_where = f"{where}.materials[{material_index}]"
        if not isinstance(value, str) or not value.strip():
            raise CatalogError(f"{material_where} must be a non-empty path string")
        candidate = Path(value)
        path = candidate if candidate.is_absolute() else base / candidate
        if not path.is_file():
            raise CatalogError(f"{material_where} points at a missing file: {path}")
        resolved.append(path)

    singer = Singer(
        singer_id=singer_id,
        name=name,
        gender=gender,
        style_tag=style_tag,
    )
    return singer, resolved
