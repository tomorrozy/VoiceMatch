"""Tests for the CLI's one-line JSON contract.

``compare`` is the first user-facing entry point, so its output is pinned: one
line of standards-compliant JSON carrying the ranked results and the three
identifiers (``model_id``, ``library_version``, ``calibration_version``).

The "standards-compliant" part is not decoration.  A silent recording's quality
report carries ``snr_db == -inf``, and a ranking score can be ``nan``; neither
``Infinity`` nor ``NaN`` is a valid JSON token.  The tests parse the output with
``parse_constant`` raising, so a leaked token fails loudly instead of being
accepted by ``json.loads``.

The library is written to disk with ``save_library`` and read back by the CLI,
so this also exercises the persisted library against the query path.

Every failure is one line of JSON
---------------------------------
C1: the CLI used to let any failure of the audio path escape as a traceback on
stderr with empty stdout.  The compare path now reports expected refusals as
``status == "rejected"`` (a valid product answer, exit code 0) and unexpected
failures as ``status == "error"`` (exit code 1), always as exactly one line of
strict JSON that names the reason.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicematch.calibrate import NO_SINGERS_IN_POOL
from voicematch.config import EMBED_DIM, SAMPLE_RATE
from voicematch.embed.encoder import HashEncoder, HubertEncoder
from voicematch.library.schema import LibraryIndex
from voicematch.library.store import save_library


def _strict_json(text: str) -> dict:
    """Parse JSON, refusing the non-standard Infinity/NaN tokens."""

    def refuse(token: str) -> None:
        raise AssertionError(f"non-finite JSON token {token!r} in CLI output")

    return json.loads(text, parse_constant=refuse)


def _run_cli(*arguments) -> subprocess.CompletedProcess:
    """Run ``python -m voicematch.cli`` with ``arguments``."""
    return subprocess.run(
        [sys.executable, "-m", "voicematch.cli", *map(str, arguments)],
        capture_output=True,
        text=True,
    )


def _run_and_parse(*arguments) -> tuple[subprocess.CompletedProcess, dict]:
    """Run the CLI and require the one-line-JSON contract (no traceback)."""
    completed = _run_cli(*arguments)
    assert "Traceback" not in completed.stderr, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == 1, completed.stdout
    return completed, _strict_json(lines[0])


def _female_only(index: LibraryIndex) -> LibraryIndex:
    """Return ``index`` restricted to its female singers.

    The cross-gender refusal (C1) needs a library that cannot answer a
    ``--gender male`` query at all, so the query's pool is empty.
    """
    singer_ids = {
        singer_id
        for singer_id, singer in index.singers.items()
        if singer.gender == "female"
    }
    segments = tuple(
        segment for segment in index.segments if segment.singer_id in singer_ids
    )
    row_of = {
        segment.segment_id: row for row, segment in enumerate(index.segments)
    }
    embeddings = (
        np.stack([index.embeddings[row_of[segment.segment_id]] for segment in segments])
        .astype(np.float32)
        if segments
        else np.zeros((0, EMBED_DIM), dtype=np.float32)
    )
    return LibraryIndex(
        version=index.version,
        model_id=index.model_id,
        singers={singer_id: index.singers[singer_id] for singer_id in singer_ids},
        segments=segments,
        prototypes={
            singer_id: index.prototypes[singer_id] for singer_id in singer_ids
        },
        features={
            segment.segment_id: index.features[segment.segment_id]
            for segment in segments
        },
        embeddings=embeddings,
    )


@pytest.fixture
def library_dir(tmp_path, tiny_stack) -> Path:
    """The tiny stack's library persisted the way the CLI reads it."""
    path = tmp_path / "library"
    save_library(tiny_stack.index, path)
    return path


def test_compare_prints_one_json_line_with_results_and_versions(
    tmp_path, tiny_stack, library_dir, write_syllable_query
):
    query = tmp_path / "query.wav"
    write_syllable_query(query)

    completed = _run_cli(
        "compare",
        query,
        "--library",
        library_dir,
        "--gender",
        "female",
    )

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == 1
    payload = _strict_json(lines[0])

    assert payload["model_id"] == tiny_stack.index.model_id
    assert payload["model_id"] == HashEncoder.model_id
    assert payload["library_version"] == tiny_stack.index.version
    # The default encoder is the deterministic stand-in, so the documented
    # command works without the optional 'ml' extra; it is echoed so the
    # consumer can see which encoder produced the scores.
    assert payload["encoder"] == "hash"
    assert isinstance(payload["calibration_version"], str)
    assert set(payload["calibration_version"]) <= set("0123456789abcdef")
    assert payload["status"] == "ok"
    assert payload["quality"]["rejected"] is False

    results = payload["results"]
    by_id = {entry["singer_id"]: entry for entry in results}
    assert set(by_id) == {"s1", "s2"}
    assert by_id["s1"]["name"] == "Singer s1"
    assert 0.0 <= results[0]["score"] <= 1.0
    assert set(results[0]["attribution"]) == {
        "cos",
        "pitch",
        "timbre",
        "resonance",
        "style",
    }


def test_a_silent_query_is_refused_as_strict_json(
    tmp_path, library_dir, write_wav
):
    # Digital silence: the quality gate refuses it, and its snr_db is -inf.
    # The CLI must emit null there, never an Infinity token.
    query = tmp_path / "silence.wav"
    write_wav(query, secs=10.0, amplitude=0.0)

    completed = _run_cli(
        "compare",
        query,
        "--library",
        library_dir,
        "--gender",
        "female",
        "--encoder",
        "hash",
    )

    assert completed.returncode == 0, completed.stderr
    payload = _strict_json(completed.stdout.splitlines()[0])

    assert payload["status"] == "rejected"
    assert payload["results"] == []
    assert payload["quality"]["rejected"] is True
    assert payload["quality"]["snr_db"] is None
    assert "low_snr" in payload["quality"]["reasons"]


def test_compare_ranks_the_pool_of_the_requested_gender(
    tmp_path, tiny_stack, library_dir, write_syllable_query
):
    query = tmp_path / "query.wav"
    write_syllable_query(query, f0=140.0)

    completed = _run_cli(
        "compare",
        query,
        "--library",
        library_dir,
        "--gender",
        "male",
        "--encoder",
        "hash",
    )

    assert completed.returncode == 0, completed.stderr
    payload = _strict_json(completed.stdout.splitlines()[0])
    assert {entry["singer_id"] for entry in payload["results"]} == {"s3", "s4"}


def test_compare_refuses_an_encoder_that_does_not_match_the_library(
    tmp_path, library_dir, write_syllable_query
):
    # R24: the library was built with HashEncoder, so asking for HuBERT must be
    # refused by name.  The check reads the class's identity before
    # construction, so it does not need the optional 'ml' extra.
    query = tmp_path / "query.wav"
    write_syllable_query(query)

    completed = _run_cli(
        "compare",
        query,
        "--library",
        library_dir,
        "--gender",
        "female",
        "--encoder",
        "hubert",
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "--encoder" in completed.stderr
    assert HashEncoder.model_id in completed.stderr
    assert HubertEncoder.model_id in completed.stderr
    assert "refusing" in completed.stderr.lower()


# --- C1: every failure is one line of JSON, never a traceback ----------------


def test_a_too_short_recording_is_rejected_as_one_json_line(tmp_path, library_dir):
    # 0.2 s is shorter than the loudness meter's 0.4 s block, so preprocessing
    # used to raise ValueError before the quality gate could report
    # ``too_short_speech`` (C1).  The gate must now see the audio untouched.
    query = tmp_path / "short.wav"
    times = np.arange(int(0.2 * SAMPLE_RATE)) / SAMPLE_RATE
    sf.write(
        query,
        (0.3 * np.sin(2 * np.pi * 220.0 * times)).astype(np.float32),
        SAMPLE_RATE,
    )

    completed, payload = _run_and_parse(
        "compare", query, "--library", library_dir, "--gender", "female"
    )

    # A refusal is a valid product answer, not a CLI failure: exit code 0.
    assert completed.returncode == 0, completed.stderr
    assert payload["status"] == "rejected"
    assert payload["reason"] == "too_short_speech"
    assert "too_short_speech" in payload["quality"]["reasons"]
    assert payload["results"] == []


def test_a_zero_frame_recording_is_rejected_as_one_json_line(tmp_path, library_dir):
    # A valid WAV header with zero samples: decoding used to warn
    # "Mean of empty slice" and then the loudness meter raised (C1).
    query = tmp_path / "empty.wav"
    sf.write(query, np.zeros(0, dtype=np.float32), SAMPLE_RATE)

    completed, payload = _run_and_parse(
        "compare", query, "--library", library_dir, "--gender", "female"
    )

    assert completed.returncode == 0, completed.stderr
    assert "Mean of empty slice" not in completed.stderr
    assert payload["status"] == "rejected"
    assert payload["reason"] == "too_short_speech"


def test_a_cross_gender_query_is_reported_as_an_error_envelope(
    tmp_path, tiny_stack, write_syllable_query
):
    # The clip is a valid query, but a female-only library cannot answer a male
    # pool: the empty-pool LookupError must become one JSON line naming the
    # documented token, not a traceback with empty stdout (C1).
    query = tmp_path / "query.wav"
    write_syllable_query(query)
    library = tmp_path / "female-library"
    save_library(_female_only(tiny_stack.index), library)

    completed, payload = _run_and_parse(
        "compare", query, "--library", library, "--gender", "male"
    )

    assert completed.returncode != 0
    assert payload["status"] == "error"
    assert payload["reason"] == NO_SINGERS_IN_POOL
    assert payload["results"] == []


def test_an_unreadable_recording_is_reported_as_an_error_envelope(
    tmp_path, library_dir
):
    query = tmp_path / "broken.wav"
    query.write_bytes(b"this is not audio")

    completed, payload = _run_and_parse(
        "compare", query, "--library", library_dir, "--gender", "female"
    )

    assert completed.returncode != 0
    assert payload["status"] == "error"
    assert payload["reason"]
    assert payload["results"] == []


# --- C2: clipping is measured on the decoded signal --------------------------


def test_a_hard_clipped_recording_is_rejected_for_clipping(tmp_path, library_dir):
    # 72 % of the decoded samples sit at full scale.  The gate used to measure
    # the signal *after* loudness normalisation, where the gain is small enough
    # that ``clipped_ratio`` came out as 0.0 and the recording was scored (C2).
    times = np.arange(SAMPLE_RATE * 10) / SAMPLE_RATE
    clipped_burst = np.clip(3.0 * np.sin(2 * np.pi * 220.0 * times), -1.0, 1.0)
    x = (clipped_burst * (np.mod(times, 0.6) < 0.55)).astype(np.float32)
    query = tmp_path / "clipped.wav"
    sf.write(query, x, SAMPLE_RATE)

    completed, payload = _run_and_parse(
        "compare", query, "--library", library_dir, "--gender", "female"
    )

    assert completed.returncode == 0, completed.stderr
    assert payload["status"] == "rejected"
    assert payload["reason"] == "clipped"
    assert "clipped" in payload["quality"]["reasons"]
    assert payload["quality"]["clipped_ratio"] > 0.01
    assert payload["results"] == []
