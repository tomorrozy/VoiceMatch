"""Shared fixtures for the singer-library tests.

Building a library is the expensive part of these tests: it decodes and
preprocesses real audio, runs window selection and extracts features.  Ruling
R14 fixes the fixture duration at 60 s per singer — five non-overlapping 10 s
windows need 60 s of material, so two singers give the ten segments the task
specifies.  The built indexes are session-scoped and shared by the tests that
only read them; a test that needs different material builds it from a copy.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicematch.audio.io import load_audio
from voicematch.audio.preprocess import PREPROCESS_LIBRARY, preprocess
from voicematch.embed.encoder import HashEncoder
from voicematch.library import LibraryIndex, Singer, build_library

SAMPLE_RATE = 24000
FIXTURE_SEC = 60.0


def write_tone(
    path: Path,
    f0: float = 220.0,
    secs: float = FIXTURE_SEC,
    amplitude: float = 0.3,
    sr: int = SAMPLE_RATE,
) -> None:
    """Write ``secs`` of a pure tone; ``amplitude=0`` writes silence."""
    times = np.arange(int(sr * secs)) / sr
    sf.write(path, (amplitude * np.sin(2 * np.pi * f0 * times)).astype(np.float32), sr)


@pytest.fixture
def write_wav():
    """Return the tone writer so a test can create fresh material."""
    return write_tone


@pytest.fixture(scope="session")
def tone_entries(tmp_path_factory) -> list[tuple[Singer, Path]]:
    """Two singers, one 60 s recording each."""
    root = tmp_path_factory.mktemp("voicematch-library-material")
    first = root / "s1.wav"
    second = root / "s2.wav"
    write_tone(first, f0=220.0)
    write_tone(second, f0=330.0)
    return [
        (Singer("s1", "Singer One", "female", "pop"), first),
        (Singer("s2", "Singer Two", "male", "rock"), second),
    ]


@pytest.fixture(scope="session")
def tone_paths(tone_entries) -> dict[str, Path]:
    return {singer.singer_id: path for singer, path in tone_entries}


@pytest.fixture(scope="session")
def two_singer_index(tone_entries) -> LibraryIndex:
    return build_library(tone_entries, HashEncoder())


@pytest.fixture(scope="session")
def processed_audio(tone_paths) -> dict[str, np.ndarray]:
    """The preprocessed audio of each fixture singer, for re-computation."""
    return {
        singer_id: preprocess(load_audio(path), PREPROCESS_LIBRARY)
        for singer_id, path in tone_paths.items()
    }


@pytest.fixture(scope="session")
def one_singer_entries(tmp_path_factory) -> list[tuple[Singer, Path]]:
    """One singer with one 60 s recording, in its own directory."""
    root = tmp_path_factory.mktemp("voicematch-library-one-singer")
    material = root / "s1.wav"
    write_tone(material, f0=220.0)
    return [(Singer("s1", "Singer One", "female", "pop"), material)]


@pytest.fixture(scope="session")
def one_singer_index(one_singer_entries) -> LibraryIndex:
    return build_library(one_singer_entries, HashEncoder())
