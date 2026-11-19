"""Tests for building the versioned singer library from raw material.

Two properties are pinned here that the store tests do not cover:

- the build reaches the specified five non-overlapping 10 s windows per singer
  (ruling R14: the fixtures carry 60 s per singer, because 20 s of material can
  hold at most two 10 s windows), and
- the version is a structural consequence of the material: equal bytes give the
  same version regardless of file name or directory, changed bytes give a
  different version, and the pipeline/model identity is part of it.  G4 ("same
  audio, same model/library version, byte-identical output") depends on this, so
  it is asserted rather than trusted.

Also pinned: per-segment embeddings and features are row-aligned with
``index.segments`` (ruling R1), several recordings of one singer merge into one
singer, and material with no usable window is skipped with a warning instead of
crashing or silently shipping an empty singer.

The build reads a ``Singer`` per material and aggregates by ``singer_id``, so
conflicting metadata for one id is refused before any audio is touched.
"""

import warnings
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from voicematch.config import EMBED_DIM, SAMPLE_RATE
from voicematch.embed.encoder import HashEncoder
from voicematch.features.extract import FeatureVector, extract_features
from voicematch.library import (
    CROSS_RECORDING,
    EMPTY_LIBRARY,
    SINGLE_RECORDING,
    UNVERIFIABLE_LEGACY,
    LibraryIndex,
    Segment,
    Singer,
    SingleMaterialSingerWarning,
    build_library,
    pairing_status,
)
from voicematch.library import build as library_build

_GROUPS = ("pitch", "timbre", "resonance", "style")


def _window_audio(audio: np.ndarray, segment) -> np.ndarray:
    """The samples of one segment, sliced the same way the build slices them."""
    return audio[
        int(round(segment.start * SAMPLE_RATE)) : int(round(segment.end * SAMPLE_RATE))
    ]


def _assert_feature_vectors_equal(actual, expected) -> None:
    for group in _GROUPS:
        assert getattr(actual, group) == pytest.approx(
            getattr(expected, group), nan_ok=True
        )


def _stub_audio_pipeline(monkeypatch) -> None:
    """Replace the two expensive stages when only the surrounding logic is tested.

    Window selection and feature extraction are real behaviour covered by the
    two-singer build; the tests that use this helper check the version formula
    and the per-singer grouping, so a deterministic one-window stub keeps them
    honest without decoding 60 s of audio per build.
    """
    monkeypatch.setattr(
        library_build, "select_windows", lambda x, sr, k=5: [(0.0, 1.0)]
    )
    monkeypatch.setattr(
        library_build,
        "extract_features",
        lambda x, sr: FeatureVector(pitch={}, timbre={}, resonance={}, style={}),
    )


# --- shape of the built library ----------------------------------------------


def test_two_singers_yield_five_windows_each(two_singer_index):
    index = two_singer_index

    assert len(index.segments) == 10
    assert Counter(segment.singer_id for segment in index.segments) == {
        "s1": 5,
        "s2": 5,
    }
    assert set(index.singers) == {"s1", "s2"}
    assert set(index.prototypes) == {"s1", "s2"}
    assert isinstance(index.segments, tuple)
    assert index.embeddings.shape == (10, EMBED_DIM)
    assert index.embeddings.dtype == np.float32
    assert len(index.version) == 16
    assert all(char in "0123456789abcdef" for char in index.version)


def test_windows_stay_inside_the_recording_and_do_not_overlap(two_singer_index):
    for singer_id in ("s1", "s2"):
        windows = sorted(
            (segment.start, segment.end)
            for segment in two_singer_index.segments
            if segment.singer_id == singer_id
        )
        assert len(windows) == 5
        for start, end in windows:
            assert 0.0 <= start < end <= 60.0
        for previous, following in zip(windows, windows[1:]):
            assert previous[1] <= following[0]


# --- R1: embeddings and features belong to a specific segment ----------------


def test_embeddings_are_row_aligned_with_segments(two_singer_index, processed_audio):
    encoder = HashEncoder()

    for row, segment in enumerate(two_singer_index.segments):
        window = _window_audio(processed_audio[segment.singer_id], segment)
        np.testing.assert_allclose(
            two_singer_index.embeddings[row],
            encoder.encode(window, SAMPLE_RATE),
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"row {row} is not the embedding of {segment.segment_id}",
        )


def test_features_are_keyed_by_segment_and_match_its_window(
    two_singer_index, processed_audio
):
    index = two_singer_index

    assert set(index.features) == {segment.segment_id for segment in index.segments}

    for singer_id in ("s1", "s2"):
        segment = next(
            segment for segment in index.segments if segment.singer_id == singer_id
        )
        window = _window_audio(processed_audio[singer_id], segment)
        expected = extract_features(window, SAMPLE_RATE)
        _assert_feature_vectors_equal(index.features[segment.segment_id], expected)


def test_build_records_the_encoder_it_used(two_singer_index):
    # R24: the index names the model that produced its vectors; the query path
    # reads this field to refuse an encoder mismatch.
    assert two_singer_index.model_id == HashEncoder.model_id
    assert two_singer_index.model_id == "HashEncoder"


# --- the version is a structural consequence of the material -----------------


def test_version_is_identical_for_identical_material(
    one_singer_index, one_singer_entries, tmp_path
):
    source = one_singer_entries[0][1]
    renamed = tmp_path / "a-completely-different-name.wav"
    renamed.write_bytes(source.read_bytes())

    rebuilt = build_library(
        [(singer, renamed) for singer, _path in one_singer_entries], HashEncoder()
    )

    # Same bytes, different file name and directory: the version must not
    # depend on where the material happens to live on this machine.
    assert rebuilt.version == one_singer_index.version


def test_version_changes_when_material_changes(
    one_singer_index, one_singer_entries, tmp_path, write_wav
):
    changed = tmp_path / "changed.wav"
    write_wav(changed, f0=440.0)

    rebuilt = build_library(
        [(singer, changed) for singer, _path in one_singer_entries], HashEncoder()
    )

    assert rebuilt.version != one_singer_index.version


def test_version_includes_pipeline_identity(monkeypatch, write_wav, tmp_path):
    _stub_audio_pipeline(monkeypatch)
    material = tmp_path / "material.wav"
    write_wav(material, secs=1.0)
    entries = [(Singer("s1", "Singer One", "female", "pop"), material)]

    baseline = build_library(entries, HashEncoder())
    monkeypatch.setattr(library_build, "PIPELINE_VERSION", "pipeline-test")
    changed = build_library(entries, HashEncoder())

    assert changed.version != baseline.version


def test_version_includes_encoder_identity(monkeypatch, write_wav, tmp_path):
    # R24: two encoders put the same material in unrelated vector spaces, so
    # their libraries must not share a version.
    _stub_audio_pipeline(monkeypatch)
    material = tmp_path / "material.wav"
    write_wav(material, secs=1.0)
    entries = [(Singer("s1", "Singer One", "female", "pop"), material)]

    baseline = build_library(entries, HashEncoder())
    monkeypatch.setattr(HashEncoder, "model_id", "model-test")
    changed = build_library(entries, HashEncoder())

    assert changed.version != baseline.version
    assert changed.model_id == "model-test"


# --- the index refuses measurements that cannot belong to its segments --------


def test_index_refuses_measurements_that_are_not_aligned_with_its_segments():
    singer = Singer("s1", "Singer One", "female", "pop")
    segment = Segment("s1-00", "s1", 0.0, 10.0)
    feature = FeatureVector(pitch={}, timbre={}, resonance={}, style={})
    fields = dict(
        version="0" * 16,
        model_id="test-encoder",
        singers={"s1": singer},
        segments=(segment,),
        prototypes={"s1": np.ones(EMBED_DIM, np.float32)},
        features={segment.segment_id: feature},
        embeddings=np.ones((1, EMBED_DIM), np.float32),
    )

    LibraryIndex(**fields)  # an aligned index is accepted

    with pytest.raises(ValueError, match="row i is the embedding"):
        LibraryIndex(**{**fields, "embeddings": np.ones((2, EMBED_DIM), np.float32)})
    with pytest.raises(ValueError, match="features"):
        LibraryIndex(**{**fields, "features": {}})
    with pytest.raises(ValueError, match="prototypes"):
        LibraryIndex(**{**fields, "prototypes": {}})
    with pytest.raises(ValueError, match="not catalogued"):
        LibraryIndex(**{**fields, "singers": {}})


# --- operator-facing behaviour -----------------------------------------------


def test_material_without_usable_windows_is_skipped_and_reported(
    tone_entries, tmp_path, write_wav
):
    silent = tmp_path / "silent.wav"
    write_wav(silent, amplitude=0.0)
    surviving = [
        (singer, path) for singer, path in tone_entries if singer.singer_id == "s1"
    ]

    with pytest.warns(UserWarning) as emitted:
        index = build_library(
            [(Singer("silent", "Silent", "female", "pop"), silent), *surviving],
            HashEncoder(),
        )

    # The operator sees both the file that was skipped and the consequence.
    messages = [str(warning.message) for warning in emitted]
    assert any("no usable windows" in message for message in messages)
    assert any("no usable segments" in message for message in messages)

    # The silent singer is left out instead of failing the whole build or
    # shipping a singer that cannot be scored.
    assert set(index.singers) == {"s1"}
    assert set(index.prototypes) == {"s1"}
    assert {segment.singer_id for segment in index.segments} == {"s1"}
    assert len(index.segments) == 5
    assert index.embeddings.shape == (5, EMBED_DIM)


def test_a_library_with_no_usable_material_is_empty_not_broken(write_wav, tmp_path):
    silent = tmp_path / "silent.wav"
    write_wav(silent, amplitude=0.0, secs=3.0)

    with pytest.warns(UserWarning):
        index = build_library(
            [(Singer("s1", "Singer One", "female", "pop"), silent)], HashEncoder()
        )

    assert index.segments == ()
    assert index.singers == {}
    assert index.prototypes == {}
    assert index.features == {}
    assert index.embeddings.shape == (0, EMBED_DIM)


def test_several_recordings_of_one_singer_are_merged(
    monkeypatch, write_wav, tmp_path
):
    _stub_audio_pipeline(monkeypatch)
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    write_wav(first, f0=220.0, secs=1.0)
    write_wav(second, f0=330.0, secs=1.0)
    singer = Singer("s1", "Singer One", "female", "pop")

    index = build_library([(singer, first), (singer, second)], HashEncoder())

    assert set(index.singers) == {"s1"}
    assert set(index.prototypes) == {"s1"}
    assert len(index.segments) == 2
    assert len({segment.segment_id for segment in index.segments}) == 2
    assert set(index.features) == {segment.segment_id for segment in index.segments}
    assert index.embeddings.shape == (2, EMBED_DIM)


def test_conflicting_metadata_for_one_singer_is_refused(write_wav, tmp_path):
    material = tmp_path / "material.wav"
    write_wav(material, secs=1.0)

    with pytest.raises(ValueError, match="conflicting"):
        build_library(
            [
                (Singer("s1", "First Name", "female", "pop"), material),
                (Singer("s1", "Other Name", "male", "rock"), material),
            ],
            HashEncoder(),
        )


# --- R29: one recording per singer is a calibration trap ---------------------


def test_one_recording_per_singer_warns_about_the_pairing_constraint(
    monkeypatch, write_wav, tmp_path
):
    _stub_audio_pipeline(monkeypatch)
    material = tmp_path / "only.wav"
    write_wav(material, secs=1.0)

    with pytest.warns(SingleMaterialSingerWarning) as emitted:
        index = build_library(
            [(Singer("s1", "Singer One", "female", "pop"), material)], HashEncoder()
        )

    # The build still succeeds: the segments are real, only calibration is at
    # risk.  The warning has to say why, and carry the id machine-readably.
    assert len(index.singers) == 1
    warning = emitted[0].message
    assert warning.singer_id == "s1"
    message = str(warning)
    assert "one usable recording" in message
    assert "cross-recording" in message


def test_two_different_recordings_do_not_warn(monkeypatch, write_wav, tmp_path):
    _stub_audio_pipeline(monkeypatch)
    first = tmp_path / "take-a.wav"
    second = tmp_path / "take-b.wav"
    write_wav(first, f0=220.0, secs=1.0)
    write_wav(second, f0=262.0, secs=1.0)
    singer = Singer("s1", "Singer One", "female", "pop")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        index = build_library([(singer, first), (singer, second)], HashEncoder())

    assert len(index.segments) == 2
    assert not [
        item
        for item in caught
        if issubclass(item.category, SingleMaterialSingerWarning)
    ]


def test_two_identical_files_are_still_one_recording(monkeypatch, write_wav, tmp_path):
    # Same bytes = same content hash = the same recording for pairing.  Counting
    # files instead of hashes would silently bless this library.
    _stub_audio_pipeline(monkeypatch)
    original = tmp_path / "take.wav"
    copy = tmp_path / "take-copy.wav"
    write_wav(original, secs=1.0)
    copy.write_bytes(original.read_bytes())
    singer = Singer("s1", "Singer One", "female", "pop")

    with pytest.warns(SingleMaterialSingerWarning) as emitted:
        build_library([(singer, original), (singer, copy)], HashEncoder())

    assert [item.message.singer_id for item in emitted] == ["s1"]
    assert "one usable recording" in str(emitted[0].message)


def test_a_second_recording_with_no_windows_still_counts_as_one(
    write_wav, tmp_path
):
    # A file that produces no segment does not exist as far as pairing is
    # concerned, so the singer is effectively single-material again.
    usable = tmp_path / "usable.wav"
    silent = tmp_path / "silent.wav"
    write_wav(usable, secs=60.0)
    write_wav(silent, amplitude=0.0, secs=3.0)
    singer = Singer("s1", "Singer One", "female", "pop")

    with pytest.warns(UserWarning) as emitted:
        build_library([(singer, usable), (singer, silent)], HashEncoder())

    categories = [item.category for item in emitted]
    assert SingleMaterialSingerWarning in categories


# --- R29: the pairing state is read from the index, not re-derived -----------


def test_pairing_status_reports_cross_recording_for_two_takes(
    monkeypatch, write_wav, tmp_path
):
    _stub_audio_pipeline(monkeypatch)
    first = tmp_path / "take-a.wav"
    second = tmp_path / "take-b.wav"
    write_wav(first, f0=220.0, secs=1.0)
    write_wav(second, f0=262.0, secs=1.0)
    singer = Singer("s1", "Singer One", "female", "pop")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        index = build_library([(singer, first), (singer, second)], HashEncoder())

    status = pairing_status(index)
    assert status.status == CROSS_RECORDING
    assert status.single_material_singers == ()


def test_pairing_status_names_a_singer_with_one_recording(one_singer_index):
    status = pairing_status(one_singer_index)

    assert status.status == SINGLE_RECORDING
    assert status.single_material_singers == ("s1",)


def test_pairing_status_marks_a_legacy_index_unverifiable(one_singer_index):
    # Libraries built before R29 carry no material_id.  They must not pass as
    # compliant just because the field is empty.
    legacy = replace(
        one_singer_index,
        segments=tuple(
            replace(segment, material_id="") for segment in one_singer_index.segments
        ),
    )

    status = pairing_status(legacy)

    assert status.status == UNVERIFIABLE_LEGACY
    assert status.single_material_singers == ()


def test_pairing_status_of_an_empty_library_is_empty():
    index = LibraryIndex(
        version="0" * 16,
        model_id="test",
        singers={},
        segments=(),
        prototypes={},
        features={},
        embeddings=np.zeros((0, EMBED_DIM), dtype=np.float32),
    )

    status = pairing_status(index)

    assert status.status == EMPTY_LIBRARY
    assert status.single_material_singers == ()
