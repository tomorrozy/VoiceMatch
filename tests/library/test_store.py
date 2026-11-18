"""Tests for saving and loading the versioned singer library.

The round trip is the contract: after ``save_library``/``load_library`` the
ranking and calibration stages must see exactly the version, encoder identity,
singers, segments, prototypes, per-segment embeddings and per-segment features
the build produced.  The loader is also the one place that can notice a
half-written or hand-edited library directory, so a manifest whose segment list
disagrees with the embedding matrix must be refused instead of silently dropping
rows.

The manifest is plain JSON, so feature values outside the JSON number grammar
(``nan`` for a descriptor that could not be measured) are stored explicitly as
``null`` and restored as ``nan``; raw ``NaN``/``Infinity`` tokens would make the
file unreadable to any non-Python JSON reader.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

from voicematch.config import EMBED_DIM
from voicematch.features.extract import FeatureVector
from voicematch.library import LibraryIndex, load_library, save_library

_GROUPS = ("pitch", "timbre", "resonance", "style")


def _assert_feature_vectors_equal(actual, expected) -> None:
    for group in _GROUPS:
        assert getattr(actual, group) == pytest.approx(
            getattr(expected, group), nan_ok=True
        )


def test_saves_the_documented_directory_layout(two_singer_index, tmp_path):
    out = tmp_path / "library"

    save_library(two_singer_index, out)

    assert (out / "manifest.json").is_file()
    assert (out / "prototypes.npz").is_file()
    assert (out / "embeddings.npy").is_file()

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == two_singer_index.version
    assert manifest["model_id"] == two_singer_index.model_id
    assert [record["singer_id"] for record in manifest["singers"]] == ["s1", "s2"]
    assert len(manifest["segments"]) == len(two_singer_index.segments)
    assert set(manifest["features"]) == {
        segment.segment_id for segment in two_singer_index.segments
    }


def test_round_trip_preserves_every_field(two_singer_index, tmp_path):
    out = tmp_path / "library"
    save_library(two_singer_index, out)

    back = load_library(out)

    assert back.version == two_singer_index.version
    assert back.model_id == two_singer_index.model_id
    assert back.singers == two_singer_index.singers
    assert back.segments == two_singer_index.segments
    assert isinstance(back.segments, tuple)

    assert set(back.prototypes) == set(two_singer_index.prototypes)
    for singer_id, prototype in two_singer_index.prototypes.items():
        assert back.prototypes[singer_id].dtype == np.float32
        np.testing.assert_array_equal(back.prototypes[singer_id], prototype)

    assert back.embeddings.dtype == np.float32
    assert back.embeddings.shape == two_singer_index.embeddings.shape
    np.testing.assert_array_equal(back.embeddings, two_singer_index.embeddings)

    assert set(back.features) == set(two_singer_index.features)
    for segment_id, feature_vector in two_singer_index.features.items():
        _assert_feature_vectors_equal(back.features[segment_id], feature_vector)


def test_round_trip_preserves_the_encoder_identity(two_singer_index, tmp_path):
    # R24: the manifest must say which encoder produced the vectors, and the
    # loader must restore it, or the query path cannot refuse a different one.
    out = tmp_path / "library"
    save_library(two_singer_index, out)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == two_singer_index.model_id
    assert load_library(out).model_id == two_singer_index.model_id


def test_loaded_index_equals_the_built_index(two_singer_index, tmp_path):
    # The natural way to state the round-trip contract, and the one downstream
    # G4 checks want to write: save then load must give an *equal* index.
    out = tmp_path / "library"
    save_library(two_singer_index, out)

    loaded = load_library(out)

    assert loaded == two_singer_index


def test_indexes_differing_in_one_embedding_row_are_not_equal(one_singer_index):
    embeddings = one_singer_index.embeddings.copy()
    embeddings[0, 0] = np.float32(embeddings[0, 0] + 0.5)
    changed = replace(one_singer_index, embeddings=embeddings)

    assert changed != one_singer_index
    assert one_singer_index != "not a library"  # foreign types are simply unequal


def test_nan_arrays_compare_equal_in_prototypes_and_embeddings(one_singer_index):
    def with_nans():
        prototypes = {
            singer_id: np.array(vector, copy=True)
            for singer_id, vector in one_singer_index.prototypes.items()
        }
        prototypes["s1"][0] = np.float32(np.nan)
        embeddings = np.array(one_singer_index.embeddings, copy=True)
        embeddings[0, 0] = np.float32(np.nan)
        return replace(one_singer_index, prototypes=prototypes, embeddings=embeddings)

    assert with_nans() == with_nans()


def test_index_is_deliberately_unhashable(one_singer_index):
    # The index holds mutable arrays and dicts, so it must not pretend to be a
    # stable hash key; the failure has to name the type, not a nested dict.
    with pytest.raises(TypeError, match="unhashable type: 'LibraryIndex'"):
        hash(one_singer_index)


def test_load_refuses_a_manifest_that_disagrees_with_the_embedding_rows(
    two_singer_index, tmp_path
):
    out = tmp_path / "library"
    save_library(two_singer_index, out)
    embeddings = np.load(out / "embeddings.npy")
    np.save(out / "embeddings.npy", embeddings[:-1])

    with pytest.raises(ValueError, match="row-aligned"):
        load_library(out)


def test_an_empty_library_round_trips(tmp_path):
    # An all-unusable catalog legitimately builds to an empty index; it must
    # still persist and load instead of tripping over the empty arrays.
    index = LibraryIndex(
        version="0" * 16,
        model_id="test-encoder",
        singers={},
        segments=(),
        prototypes={},
        features={},
        embeddings=np.zeros((0, EMBED_DIM), dtype=np.float32),
    )
    out = tmp_path / "library"

    save_library(index, out)
    back = load_library(out)

    assert back.version == index.version
    assert back.singers == {} and back.segments == ()
    assert back.prototypes == {} and back.features == {}
    assert back.embeddings.shape == (0, EMBED_DIM)
    assert back.embeddings.dtype == np.float32


def test_non_finite_feature_values_survive_the_round_trip(one_singer_index, tmp_path):
    segment = one_singer_index.segments[0]
    feature_vector = one_singer_index.features[segment.segment_id]
    nan_vector = FeatureVector(
        pitch={**feature_vector.pitch, "median_f0_hz": float("nan")},
        timbre=feature_vector.timbre,
        resonance=feature_vector.resonance,
        style=feature_vector.style,
    )
    index = LibraryIndex(
        version=one_singer_index.version,
        model_id=one_singer_index.model_id,
        singers=one_singer_index.singers,
        segments=one_singer_index.segments,
        prototypes=one_singer_index.prototypes,
        features={**one_singer_index.features, segment.segment_id: nan_vector},
        embeddings=one_singer_index.embeddings,
    )
    out = tmp_path / "library"

    save_library(index, out)

    text = (out / "manifest.json").read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text

    back = load_library(out)

    assert np.isnan(back.features[segment.segment_id].pitch["median_f0_hz"])
    _assert_feature_vectors_equal(back.features[segment.segment_id], nan_vector)
    # NaN feature values must compare equal, otherwise a correct round trip of
    # real material (whose features layer emits nan for unanalysable content)
    # would look like a changed library.
    assert back == index
