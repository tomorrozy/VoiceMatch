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
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from voicematch.embed.encoder import MODEL_ID
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

    assert payload["model_id"] == MODEL_ID
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
