"""Persist one built singer library as a directory of plain, inspectable files.

Layout of one library directory:

- ``manifest.json`` — the version, the encoder identity (``model_id``, R24), the
  singers, the ordered segment list, the per-segment feature vectors, and the
  resolved versions of the numerical stack the build ran on (``deps``, finding
  I5).  JSON because an operator debugging a bad library can read it and diff
  two builds with ordinary tools.
- ``prototypes.npz`` — one prototype vector per singer id.
- ``embeddings.npy`` — the ``(n_segments, EMBED_DIM)`` ``float32`` matrix whose
  row ``i`` belongs to ``segments[i]``.

The version is stored rather than recomputed: it identifies the material and
pipeline that produced the vectors, which the files on disk cannot show on their
own, and the scoring path writes it into every result (G4).  The encoder
identity is stored separately (``model_id``) because it is not a property of the
files: the query path reads it back and refuses to encode a query with a
different model than the one that built the prototypes (R24).  ``deps`` is
informational only: it records which numpy/librosa/scipy/scikit-learn versions
produced the stored vectors, so a rebuild elsewhere is detectable, but it is not
part of the index identity and does not enter the version hash.

The loader checks the directory's internal agreement — segment list against
embedding rows, prototypes against singers, features against segments — and
refuses a mismatch instead of loading a library that would silently score the
user against the wrong singer.

Feature vectors legitimately contain ``nan`` (a descriptor that could not be
measured on that window).  The manifest stores those values as ``null`` and
restores them as ``nan``, so the JSON stays readable by any standards-compliant
parser instead of relying on Python's non-standard ``NaN`` token.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

import numpy as np

from voicematch.features.extract import FeatureVector
from voicematch.library.schema import LibraryIndex, Segment, Singer

MANIFEST_NAME: str = "manifest.json"
PROTOTYPES_NAME: str = "prototypes.npz"
EMBEDDINGS_NAME: str = "embeddings.npy"

#: Runtime distributions whose resolved versions are recorded in the manifest
#: (finding I5).  They are the numerical stack the stored vectors depend on.
_DEPENDENCY_PACKAGES: tuple[str, ...] = (
    "numpy",
    "librosa",
    "scipy",
    "scikit-learn",
)

#: Feature groups of ``FeatureVector``; the manifest uses the same names.
_GROUPS: tuple[str, ...] = ("pitch", "timbre", "resonance", "style")

#: Top-level manifest keys the loader depends on.
_REQUIRED_KEYS: tuple[str, ...] = (
    "version",
    "model_id",
    "singers",
    "segments",
    "features",
)


def save_library(index: LibraryIndex, out: Path) -> None:
    """Write ``index`` into the directory ``out``, creating it if needed.

    An existing directory is reused, so saving the same library twice is
    idempotent.  Files are replaced, not appended: the three files always
    describe one single library.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    np.savez(
        out / PROTOTYPES_NAME,
        **{
            singer_id: np.asarray(index.prototypes[singer_id], dtype=np.float32)
            for singer_id in sorted(index.prototypes)
        },
    )
    np.save(out / EMBEDDINGS_NAME, np.asarray(index.embeddings, dtype=np.float32))

    manifest = {
        "version": index.version,
        "model_id": index.model_id,
        "deps": _dependency_versions(),
        "singers": [
            asdict(index.singers[singer_id]) for singer_id in sorted(index.singers)
        ],
        "segments": [asdict(segment) for segment in index.segments],
        "features": {
            segment.segment_id: _features_to_json(index.features[segment.segment_id])
            for segment in index.segments
        },
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _dependency_versions() -> dict[str, str]:
    """Return the resolved versions of the numerical stack this build ran on.

    Finding I5: the manifest records ``numpy``/``librosa``/``scipy``/
    ``scikit-learn`` so a rebuild on another machine that silently upgraded one
    of them is detectable.  A distribution whose metadata cannot be found is
    recorded as ``"unknown"`` rather than failing the save: an unwritable
    library is worse than an incomplete build record.
    """
    versions: dict[str, str] = {}
    for distribution in _DEPENDENCY_PACKAGES:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "unknown"
    return versions


def load_library(path: Path) -> LibraryIndex:
    """Read back a library written by :func:`save_library`.

    Raises ``ValueError`` when the files disagree with each other — most
    importantly when the manifest's segment count is not the embedding matrix's
    row count.  Truncating to the shorter of the two would misattribute every
    following row, so it is refused.  The encoder identity (``model_id``) must
    be present: a library that cannot say which model produced its vectors
    cannot be safely queried (R24).
    """
    root = Path(path)
    manifest = _read_manifest(root / MANIFEST_NAME)

    segments = tuple(
        Segment(
            segment_id=str(record["segment_id"]),
            singer_id=str(record["singer_id"]),
            start=float(record["start"]),
            end=float(record["end"]),
            # Older manifests predate the field; "" keeps the historical
            # "each segment its own material" behaviour for them.
            material_id=str(record.get("material_id", "")),
        )
        for record in manifest["segments"]
    )
    embeddings = np.load(root / EMBEDDINGS_NAME, allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(segments):
        raise ValueError(
            f"{MANIFEST_NAME} lists {len(segments)} segments but {EMBEDDINGS_NAME} "
            f"has shape {embeddings.shape}; refusing to load a library whose "
            "per-segment embeddings are not row-aligned with its segments"
        )

    singers = {
        str(record["singer_id"]): Singer(
            singer_id=str(record["singer_id"]),
            name=str(record["name"]),
            gender=record["gender"],
            style_tag=str(record["style_tag"]),
        )
        for record in manifest["singers"]
    }

    feature_records = manifest["features"]
    if set(feature_records) != {segment.segment_id for segment in segments}:
        raise ValueError(
            f"{MANIFEST_NAME}: 'features' must be keyed by exactly the segment ids"
        )
    features = {
        segment.segment_id: _features_from_json(feature_records[segment.segment_id])
        for segment in segments
    }

    return LibraryIndex(
        version=manifest["version"],
        model_id=manifest["model_id"],
        singers=singers,
        segments=segments,
        prototypes=_load_prototypes(root / PROTOTYPES_NAME),
        features=features,
        embeddings=embeddings,
    )


def _read_manifest(path: Path) -> dict:
    """Parse the manifest and check it has the keys the loader needs."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    missing = [key for key in _REQUIRED_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"{path.name} is missing mandatory keys: {missing}")
    if not isinstance(manifest["version"], str):
        raise ValueError(f"{path.name}: 'version' must be a string")
    if not isinstance(manifest["model_id"], str) or not manifest["model_id"]:
        raise ValueError(
            f"{path.name}: 'model_id' must be a non-empty string"
        )
    return manifest


def _load_prototypes(path: Path) -> dict[str, np.ndarray]:
    """Read ``prototypes.npz`` as ``singer_id -> float32 prototype``."""
    with np.load(path, allow_pickle=False) as data:
        return {
            str(singer_id): np.asarray(data[singer_id], dtype=np.float32)
            for singer_id in data.files
        }


def _features_to_json(
    feature_vector: FeatureVector,
) -> dict[str, dict[str, float | None]]:
    """Serialise one feature vector; non-finite values become ``null``."""
    return {
        group: {
            name: value if math.isfinite(value) else None
            for name, value in getattr(feature_vector, group).items()
        }
        for group in _GROUPS
    }


def _features_from_json(record: dict) -> FeatureVector:
    """Rebuild one feature vector; ``null`` values become ``nan``."""
    groups: dict[str, dict[str, float]] = {}
    for group in _GROUPS:
        try:
            values = record[group]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"malformed feature record: missing group {group!r}"
            ) from exc
        groups[group] = {
            name: float("nan") if value is None else float(value)
            for name, value in values.items()
        }
    return FeatureVector(**groups)
