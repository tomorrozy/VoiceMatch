"""The singer library's data model: what one built library is made of.

A library is a set of singers, the scored segments cut from their material, one
prototype vector per singer, and the per-segment measurements the scoring path
needs again:

- ``features`` maps **segment id -> FeatureVector**.  It is per segment, not per
  singer: the scoring path compares the user's feature vector against library
  segments, so a singer-level average would destroy the comparison it is meant
  to support.
- ``embeddings`` is the ``(n_segments, EMBED_DIM)`` ``float32`` matrix of
  per-segment embeddings, and row ``i`` belongs to ``segments[i]`` (ruling R1).
  Prototypes cannot be inverted back into segment vectors, and the calibration
  stage builds its pairs from per-segment vectors, so the matrix is part of the
  index instead of being recomputed later.

The invariants are checked in ``__post_init__`` because a misaligned array does
not raise anywhere on its own: it silently scores the user against the wrong
singer.  ``load_library`` re-checks the same invariants against the files on
disk and refuses a corrupt directory with a message naming the file.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from voicematch.config import EMBED_DIM
from voicematch.features.extract import FeatureVector


@dataclass(frozen=True)
class Singer:
    """One catalogued singer."""

    singer_id: str
    name: str
    gender: Literal["male", "female"]
    style_tag: str


@dataclass(frozen=True)
class Segment:
    """One scored stretch cut from a singer's material, in seconds."""

    segment_id: str
    singer_id: str
    start: float
    end: float


@dataclass(frozen=True)
class LibraryIndex:
    """A built library: version, catalogued singers, segments and measurements.

    ``features`` is keyed by segment id (see the module docstring) and
    ``embeddings`` is row-aligned with ``segments``.  ``version`` identifies the
    material and pipeline that produced the index and is stored with it rather
    than recomputed on load.
    """

    version: str
    singers: dict[str, Singer]
    segments: tuple[Segment, ...]
    prototypes: dict[str, np.ndarray]
    features: dict[str, FeatureVector]
    embeddings: np.ndarray

    def __post_init__(self) -> None:
        _check_embeddings(self.embeddings, self.segments)
        _check_segment_singers(self.segments, self.singers)
        _check_prototypes(self.prototypes, self.singers)
        _check_features(self.features, self.segments)


def _check_segment_singers(
    segments: tuple[Segment, ...], singers: dict[str, Singer]
) -> None:
    """Refuse a segment attributed to a singer the library does not contain."""
    unknown = sorted({segment.singer_id for segment in segments} - set(singers))
    if unknown:
        raise ValueError(
            f"segments reference singers that are not catalogued: {unknown}"
        )


def _check_embeddings(embeddings: np.ndarray, segments: tuple[Segment, ...]) -> None:
    """Refuse an embedding matrix that cannot belong to these segments."""
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBED_DIM:
        raise ValueError(
            f"embeddings must have shape (n_segments, {EMBED_DIM}), "
            f"got {embeddings.shape}"
        )
    if embeddings.dtype != np.float32:
        raise ValueError(f"embeddings must be float32, got {embeddings.dtype}")
    if embeddings.shape[0] != len(segments):
        raise ValueError(
            f"embeddings has {embeddings.shape[0]} rows but there are "
            f"{len(segments)} segments; row i is the embedding of segments[i]"
        )


def _check_prototypes(
    prototypes: dict[str, np.ndarray], singers: dict[str, Singer]
) -> None:
    """Refuse a prototype map that cannot cover the catalogued singers."""
    missing = sorted(set(singers) - set(prototypes))
    unexpected = sorted(set(prototypes) - set(singers))
    if missing or unexpected:
        raise ValueError(
            "prototypes must cover exactly the singers: "
            f"missing {missing}, unexpected {unexpected}"
        )
    for singer_id, vector in prototypes.items():
        if vector.shape != (EMBED_DIM,):
            raise ValueError(
                f"prototype for singer {singer_id!r} must have shape "
                f"({EMBED_DIM},), got {vector.shape}"
            )


def _check_features(
    features: dict[str, FeatureVector], segments: tuple[Segment, ...]
) -> None:
    """Refuse a feature map that is not keyed by exactly these segments."""
    segment_ids = {segment.segment_id for segment in segments}
    if len(segment_ids) != len(segments):
        raise ValueError(
            f"there are {len(segments)} segments but only {len(segment_ids)} "
            "distinct segment ids; features are keyed by segment id"
        )
    if set(features) != segment_ids:
        missing = sorted(segment_ids - set(features))
        unexpected = sorted(set(features) - segment_ids)
        raise ValueError(
            "features must be keyed by exactly the segment ids: "
            f"missing {missing}, unexpected {unexpected}"
        )
