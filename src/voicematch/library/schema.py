"""The singer library's data model: what one built library is made of.

A library is a set of singers, the scored segments cut from their material, one
prototype vector per singer, the per-segment measurements the scoring path
needs again, and the identity of the encoder that produced the embeddings:

- ``features`` maps **segment id -> FeatureVector**.  It is per segment, not per
  singer: the scoring path compares the user's feature vector against library
  segments, so a singer-level average would destroy the comparison it is meant
  to support.
- ``embeddings`` is the ``(n_segments, EMBED_DIM)`` ``float32`` matrix of
  per-segment embeddings, and row ``i`` belongs to ``segments[i]`` (ruling R1).
  Prototypes cannot be inverted back into segment vectors, and the calibration
  stage builds its pairs from per-segment vectors, so the matrix is part of the
  index instead of being recomputed later.
- ``model_id`` is the ``VocalEncoder.model_id`` that produced ``embeddings``
  (ruling R24).  Two encoders put the same audio in unrelated vector spaces, so
  the query path refuses to encode against a library built by a different one;
  without this field such a mismatch would rank singers by noise and look
  perfectly normal.

The invariants are checked in ``__post_init__`` because a misaligned array does
not raise anywhere on its own: it silently scores the user against the wrong
singer.  ``load_library`` re-checks the same invariants against the files on
disk and refuses a corrupt directory with a message naming the file.
"""

import math
from dataclasses import dataclass, fields
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
    """One scored stretch cut from a singer's material, in seconds.

    ``material_id`` names the *recording* the stretch was cut from (a material
    file's content hash).  It exists so calibration pairs can require two
    stretches of one singer to come from **different recordings**: a positive
    pair of two windows of the same take teaches "same session = same person",
    which collapses on an unseen song.  It defaults to ``""`` for backwards
    compatibility with libraries built before this field existed and with
    synthetic fixtures; an empty id is treated as "each segment its own
    material", so the historical pairing behaviour is preserved there.
    """

    segment_id: str
    singer_id: str
    start: float
    end: float
    material_id: str = ""


@dataclass(frozen=True, eq=False)
class LibraryIndex:
    """A built library: version, catalogued singers, segments and measurements.

    ``features`` is keyed by segment id (see the module docstring) and
    ``embeddings`` is row-aligned with ``segments``.  ``version`` identifies the
    material and pipeline that produced the index and is stored with it rather
    than recomputed on load; ``model_id`` names the encoder that produced the
    embeddings (R24) and is checked by the query path before any ranking.

    Equality is value equality over every field, with ``nan`` compared equal:
    the features layer legitimately emits ``nan`` for content it cannot analyse,
    so a correct round trip must compare equal to the built index.  The type is
    deliberately **unhashable** — it holds mutable arrays and dictionaries, and
    a ``TypeError`` from ``hash()`` is the correct answer.
    """

    version: str
    model_id: str
    singers: dict[str, Singer]
    segments: tuple[Segment, ...]
    prototypes: dict[str, np.ndarray]
    features: dict[str, FeatureVector]
    embeddings: np.ndarray

    # Deliberately unhashable: the index holds mutable arrays and dictionaries,
    # so there is no stable hash.  This fails cleanly, naming this type, instead
    # of the generated hash failing halfway through on a nested dict.
    __hash__ = None

    def __post_init__(self) -> None:
        _check_model_id(self.model_id)
        _check_embeddings(self.embeddings, self.segments)
        _check_segment_singers(self.segments, self.singers)
        _check_prototypes(self.prototypes, self.singers)
        _check_features(self.features, self.segments)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LibraryIndex):
            return NotImplemented
        return (
            self.version == other.version
            and self.model_id == other.model_id
            and self.singers == other.singers
            and self.segments == other.segments
            and _features_equal(self.features, other.features)
            and _prototypes_equal(self.prototypes, other.prototypes)
            and np.array_equal(self.embeddings, other.embeddings, equal_nan=True)
        )


def _check_model_id(model_id: str) -> None:
    """Refuse an index whose encoder identity is missing or not a string.

    The identity is what keeps a query from being encoded with a different
    model than the library's (R24); an empty or misspelt value would silently
    defeat the check, so it is validated at construction.
    """
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(
            f"model_id must be a non-empty string, got {model_id!r}"
        )


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


def _features_equal(
    left: dict[str, FeatureVector], right: dict[str, FeatureVector]
) -> bool:
    """Compare feature maps value by value, treating ``nan`` as equal."""
    if left.keys() != right.keys():
        return False
    for segment_id, left_vector in left.items():
        right_vector = right[segment_id]
        for feature_field in fields(FeatureVector):
            left_group = getattr(left_vector, feature_field.name)
            right_group = getattr(right_vector, feature_field.name)
            if left_group.keys() != right_group.keys():
                return False
            for name, value in left_group.items():
                if not _numbers_equal(value, right_group[name]):
                    return False
    return True


def _numbers_equal(left: float, right: float) -> bool:
    """Numeric equality in which two ``nan`` values count as equal."""
    if left == right:
        return True
    return math.isnan(left) and math.isnan(right)


def _prototypes_equal(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> bool:
    """Compare prototype maps by key and array value, treating ``nan`` as equal."""
    if left.keys() != right.keys():
        return False
    return all(
        np.array_equal(left[singer_id], right[singer_id], equal_nan=True)
        for singer_id in left
    )
