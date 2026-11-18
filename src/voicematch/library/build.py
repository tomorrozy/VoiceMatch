"""Cut singer material into scored segments and stamp them with one version.

The build is deliberately the only place that turns raw material into a library,
so "the material changed, so the version changed" is a property of the code
rather than something an operator has to remember (G4).  For every material
file the pipeline is the same one the query path uses: decode, preprocess, pick
windows with the shared automatic rule, then extract features and one embedding
per window.

The version is a SHA-256 over a canonical JSON dump of every
``(singer_id, segment_id, start, end, material_sha256)`` record plus
``PIPELINE_VERSION`` and ``MODEL_ID``, truncated to 16 hex characters.  Nothing
host-specific enters it: the file path and its name are not part of the hash, so
the same bytes copied elsewhere rebuild to the same version, while any byte
change in the material changes it.  Materials are processed in an order fixed by
their content hash, so the entry order in ``entries`` does not affect the
version either.

Several materials may belong to one singer (the library design is five songs
per singer).  They are merged into one ``singers`` entry, one prototype and a
single segment list.  Material that yields no usable window cannot produce
measurements; it is skipped with a warning instead of aborting the whole build,
and a singer left with no usable material is reported and left out of the index
rather than shipped as a singer that can never be scored.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from voicematch.audio.io import load_audio
from voicematch.audio.preprocess import PREPROCESS_LIBRARY, preprocess
from voicematch.audio.windows import select_windows
from voicematch.config import EMBED_DIM, PIPELINE_VERSION, SAMPLE_RATE
from voicematch.embed.encoder import MODEL_ID, VocalEncoder
from voicematch.embed.prototype import singer_prototypes
from voicematch.features.extract import FeatureVector, extract_features
from voicematch.library.schema import LibraryIndex, Segment, Singer

#: Windows taken from one material file by the shared automatic rule.
WINDOWS_PER_MATERIAL: int = 5

#: Block size for hashing a material file; the files can be large.
_HASH_BLOCK_BYTES: int = 1 << 20

#: Version-hash input record: one segment and the material it was cut from.
_SegmentRecord = tuple[str, str, float, float, str]


def build_library(
    entries: Sequence[tuple[Singer, Path]], encoder: VocalEncoder
) -> LibraryIndex:
    """Build the versioned library from ``(Singer, material path)`` pairs.

    ``entries`` may repeat a ``Singer`` with different material; materials of
    one ``singer_id`` are merged.  All entries of a singer must carry the same
    ``Singer`` metadata, otherwise ``ValueError`` is raised before any audio is
    read (a contradictory catalog entry is a data bug, not something to resolve
    silently).

    Each material is decoded with :func:`~voicematch.audio.io.load_audio`,
    preprocessed with the library preprocessing config, and cut with
    :func:`~voicematch.audio.windows.select_windows` (``k=WINDOWS_PER_MATERIAL``).
    Every window yields one ``Segment``, one ``FeatureVector`` and one
    embedding, in that order, so the returned arrays are row-aligned.

    Material with no usable window (silence, too little speech) is skipped with
    a ``UserWarning`` naming the material, and a singer with no usable material
    at all is additionally reported and omitted from the library.  The index can
    legitimately be empty if every entry is unusable.
    """
    grouped = _group_entries(entries)
    singers: dict[str, Singer] = {}
    segments: list[Segment] = []
    features: dict[str, FeatureVector] = {}
    rows: list[np.ndarray] = []
    records: list[_SegmentRecord] = []

    for singer_id in sorted(grouped):
        singer = grouped[singer_id][0][0]
        # Sort by content hash, not by path: the segment numbering and therefore
        # the version then depend on the material alone, not on the order of
        # ``entries`` or on where the files live.
        materials = sorted(
            ((_file_sha256(path), path) for _singer, path in grouped[singer_id]),
            key=lambda material: material[0],
        )
        usable = 0
        for material_sha256, material_path in materials:
            audio = preprocess(load_audio(material_path), PREPROCESS_LIBRARY)
            windows = select_windows(audio, SAMPLE_RATE, k=WINDOWS_PER_MATERIAL)
            if not windows:
                warnings.warn(
                    f"material {material_path} for singer {singer_id!r} yields no "
                    "usable windows and is skipped",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            for start, end in windows:
                window = audio[_sample_index(start) : _sample_index(end)]
                segment_id = f"{singer_id}-{usable:02d}"
                segments.append(Segment(segment_id, singer_id, start, end))
                features[segment_id] = extract_features(window, SAMPLE_RATE)
                rows.append(encoder.encode(window, SAMPLE_RATE))
                records.append(
                    (singer_id, segment_id, start, end, material_sha256)
                )
                usable += 1

        if usable == 0:
            warnings.warn(
                f"singer {singer_id!r} has no usable segments and is left out of "
                "the library",
                UserWarning,
                stacklevel=2,
            )
            continue
        singers[singer_id] = singer

    embeddings = (
        np.stack(rows).astype(np.float32)
        if rows
        else np.zeros((0, EMBED_DIM), dtype=np.float32)
    )
    prototypes = singer_prototypes(
        embeddings, [segment.singer_id for segment in segments]
    )
    return LibraryIndex(
        version=_library_version(records),
        singers=singers,
        segments=tuple(segments),
        prototypes=prototypes,
        features=features,
        embeddings=embeddings,
    )


def _group_entries(
    entries: Sequence[tuple[Singer, Path]],
) -> dict[str, list[tuple[Singer, Path]]]:
    """Group the entries by ``singer_id``, refusing contradictory metadata."""
    grouped: dict[str, list[tuple[Singer, Path]]] = {}
    for singer, material in entries:
        grouped.setdefault(singer.singer_id, []).append((singer, Path(material)))

    for singer_id, items in grouped.items():
        reference = items[0][0]
        for singer, _material in items[1:]:
            if singer != reference:
                raise ValueError(
                    f"conflicting Singer metadata for {singer_id!r}: "
                    f"{reference!r} != {singer!r}"
                )
    return grouped


def _sample_index(seconds: float) -> int:
    """Convert a segment boundary in seconds to a sample index."""
    return int(round(seconds * SAMPLE_RATE))


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 of one material file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as material:
        for block in iter(lambda: material.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _library_version(records: Sequence[_SegmentRecord]) -> str:
    """Return the 16-hex library version for these segment records.

    The JSON is canonicalised (sorted keys, no whitespace, ASCII only) so the
    hash is stable across platforms and Python versions.
    """
    payload = {
        "model_id": MODEL_ID,
        "pipeline_version": PIPELINE_VERSION,
        "segments": sorted([list(record) for record in records]),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
