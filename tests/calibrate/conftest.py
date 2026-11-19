"""Shared fixtures for the calibration tests: a hand-sized singer library.

``make_index`` builds a tiny ``LibraryIndex`` in which every number is authored
by the test rather than extracted from audio: the ranking tests need to control
exactly which prototype is nearest and exactly which descriptors differ.  The
default library holds two same-gender singers whose prototypes are the first two
basis vectors, so "nearer the query" is decided by a single coordinate.

``make_model`` builds a ``CalibrationModel`` two ways: the canonical cosine-only
fallback (``auc = nan``, ``is_calibrated`` false) that ``fit_calibration``
returns when there is nothing to learn, and a fitted-shaped model with a real
AUC and non-zero dimension weights.  Ranking must carry that distinction
through, so tests need both without training anything.

The segment embedding matrix is zero-filled on purpose.  The ranking path reads
only the prototypes and the per-segment feature vectors, and a signal-free
matrix keeps an accidental segment-embedding comparison from looking like a
working ranking.  The calibration pairs tests build their own random matrices.
"""

from __future__ import annotations

import numpy as np
import pytest

from voicematch.calibrate.model import CalibrationModel
from voicematch.calibrate.pairs import FEATURE_NAMES
from voicematch.config import EMBED_DIM
from voicematch.features.extract import DIMENSION_OF, FeatureVector
from voicematch.library.schema import LibraryIndex, Segment, Singer

#: Descriptor value shared by the default fixture library and the default query,
#: so a default ranking has zero dimension deltas and orders on cosine alone.
DEFAULT_DESCRIPTOR: float = 0.5

#: The feature groups ``FeatureVector`` stores.
_GROUPS: tuple[str, ...] = ("pitch", "timbre", "resonance", "style")


def _feature_vector(
    values: dict[str, float] | None = None, *, default: float
) -> FeatureVector:
    """Return a ``FeatureVector`` covering exactly the registered descriptors."""
    overrides = values or {}
    groups: dict[str, dict[str, float]] = {group: {} for group in _GROUPS}
    for name, group in DIMENSION_OF.items():
        value = overrides[name] if name in overrides else default
        groups[group][name] = float(value)
    return FeatureVector(**groups)


def _basis_vector(position: int) -> np.ndarray:
    """A unit-norm ``(EMBED_DIM,)`` float32 prototype that is 1 at one position."""
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    vector[position] = 1.0
    return vector


@pytest.fixture
def make_fv():
    """Factory: one ``FeatureVector`` with every descriptor set explicitly.

    ``values`` overrides descriptors by name; every other descriptor is set to
    ``default``, which a test can pass as ``nan`` to model "could not measure".
    """

    def factory(
        values: dict[str, float] | None = None, *, default: float = DEFAULT_DESCRIPTOR
    ) -> FeatureVector:
        return _feature_vector(values, default=default)

    return factory


@pytest.fixture
def fv(make_fv) -> FeatureVector:
    """The default query feature vector: every descriptor at 0.5."""
    return make_fv()


@pytest.fixture
def make_index():
    """Factory: build a small ``LibraryIndex`` by hand (see module docstring)."""

    def factory(
        *,
        genders: tuple[str, ...] = ("female", "female"),
        singer_ids: tuple[str, ...] | None = None,
        segments_per_singer: int = 2,
        features: dict[str, FeatureVector] | None = None,
        prototypes: dict[str, np.ndarray] | None = None,
    ) -> LibraryIndex:
        if singer_ids is None:
            singer_ids = tuple(
                f"s{position + 1}" for position in range(len(genders))
            )
        if len(singer_ids) != len(genders):
            raise ValueError(
                "singer_ids and genders must name the same number of singers"
            )
        singers = {
            singer_id: Singer(singer_id, f"Singer {singer_id}", gender, "pop")
            for singer_id, gender in zip(singer_ids, genders)
        }
        segments: list[Segment] = []
        segment_features: dict[str, FeatureVector] = {}
        for singer_id in singer_ids:
            for number in range(segments_per_singer):
                segment = Segment(
                    f"{singer_id}-{number:02d}",
                    singer_id,
                    float(number),
                    float(number + 1),
                )
                segments.append(segment)
                segment_features[segment.segment_id] = _feature_vector(
                    None, default=DEFAULT_DESCRIPTOR
                )
        if features:
            segment_features.update(features)
        if prototypes is None:
            prototypes = {
                singer_id: _basis_vector(position)
                for position, singer_id in enumerate(singer_ids)
            }
        return LibraryIndex(
            version="fixture-v1",
            model_id="fixture-encoder",
            singers=singers,
            segments=tuple(segments),
            prototypes=prototypes,
            features=segment_features,
            embeddings=np.zeros((len(segments), EMBED_DIM), dtype=np.float32),
        )

    return factory


@pytest.fixture
def make_model():
    """Factory: a cosine-only fallback model or a fitted-shaped calibrated one."""

    def factory(*, calibrated: bool = False) -> CalibrationModel:
        if calibrated:
            return CalibrationModel(
                coef=np.array([2.0, -0.5, 0.25, 0.0, 0.5]),
                mean=np.zeros(len(FEATURE_NAMES)),
                scale=np.ones(len(FEATURE_NAMES)),
                intercept=-1.0,
                auc=0.87,
            )
        return CalibrationModel(
            coef=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            mean=np.zeros(len(FEATURE_NAMES)),
            scale=np.ones(len(FEATURE_NAMES)),
            intercept=0.0,
            auc=float("nan"),
        )

    return factory
