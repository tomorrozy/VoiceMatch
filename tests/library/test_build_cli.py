"""Tests for the ``build`` CLI subcommand.

``build`` is the operator's entry point for turning a catalog of real material
into a versioned library.  It must print one line of strict JSON, record the
encoder identity that made the vectors (R24), surface skipped material in the
same document, and fail as a one-line ``catalog_error`` envelope before any
audio is decoded when the catalog is bad.  These tests use the deterministic
``HashEncoder`` (via ``--encoder hash``) only as a test stand-in; it is never
the command's default and never produces a real score.
"""

import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from voicematch.config import SAMPLE_RATE
from voicematch.embed.encoder import HashEncoder
from voicematch.library.store import load_library


def _run_cli(*arguments) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "voicematch.cli", *map(str, arguments)],
        capture_output=True,
        text=True,
    )


def _one_json_line(completed: subprocess.CompletedProcess) -> dict:
    def refuse(token: str) -> None:
        raise AssertionError(f"non-finite JSON token {token!r}")

    assert "Traceback" not in completed.stderr, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == 1, completed.stdout
    return json.loads(lines[0], parse_constant=refuse)


def _write_tone(path, seconds=12.0, f0=220.0) -> None:
    """A 12 s tone: enough for at least one usable 8 s speech window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    sf.write(path, (0.3 * np.sin(2 * np.pi * f0 * t)).astype(np.float32), SAMPLE_RATE)


def _catalog(tmp_path, singers):
    root = tmp_path / "catalog"
    root.mkdir()
    path = root / "catalog.json"
    path.write_text(json.dumps({"singers": singers}), encoding="utf-8")
    return path


def test_build_prints_one_json_line_with_version_and_identifiers(tmp_path):
    material = tmp_path / "material.wav"
    _write_tone(material)
    catalog = _catalog(
        tmp_path,
        [
            {
                "singer_id": "s1",
                "name": "Singer One",
                "gender": "female",
                "style_tag": "pop",
                "materials": ["../material.wav"],
            }
        ],
    )
    out = tmp_path / "library"

    completed = _run_cli(
        "build", "--catalog", catalog, "--out", out, "--encoder", "hash"
    )
    payload = _one_json_line(completed)

    assert completed.returncode == 0, completed.stderr
    assert payload["status"] == "ok"
    assert payload["command"] == "build"
    assert payload["encoder"] == "hash"
    assert payload["model_id"] == HashEncoder.model_id
    assert set(payload["library_version"]) <= set("0123456789abcdef")
    assert len(payload["library_version"]) == 16
    assert payload["singers"] == 1
    assert payload["segments"] >= 1
    assert payload["skipped"] == []

    # The library on disk is the one the version describes and is loadable.
    index = load_library(out)
    assert index.version == payload["library_version"]
    assert index.model_id == HashEncoder.model_id
    assert set(index.singers) == {"s1"}


def test_build_merges_materials_of_one_singer(tmp_path):
    first = tmp_path / "one.wav"
    second = tmp_path / "two.wav"
    _write_tone(first, f0=220.0)
    _write_tone(second, f0=330.0)
    catalog = _catalog(
        tmp_path,
        [
            {
                "singer_id": "s1",
                "name": "Singer One",
                "gender": "female",
                "style_tag": "pop",
                "materials": ["../one.wav", "../two.wav"],
            }
        ],
    )
    out = tmp_path / "library"

    payload = _one_json_line(
        _run_cli("build", "--catalog", catalog, "--out", out, "--encoder", "hash")
    )

    assert payload["status"] == "ok"
    assert payload["singers"] == 1
    assert payload["segments"] >= 2


def test_unusable_material_is_reported_in_skipped_not_failed(tmp_path):
    silent = tmp_path / "silent.wav"
    silent.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(SAMPLE_RATE * 12)) / SAMPLE_RATE
    sf.write(silent, np.zeros_like(t, dtype=np.float32), SAMPLE_RATE)
    catalog = _catalog(
        tmp_path,
        [
            {
                "singer_id": "s1",
                "name": "Singer One",
                "gender": "female",
                "style_tag": "pop",
                "materials": ["../silent.wav"],
            }
        ],
    )
    out = tmp_path / "library"

    completed = _run_cli(
        "build", "--catalog", catalog, "--out", out, "--encoder", "hash"
    )
    payload = _one_json_line(completed)

    # A singer with no usable material is a valid (empty) build, not a failure:
    # the reason is surfaced in 'skipped' and the exit status stays 0.
    assert completed.returncode == 0, completed.stderr
    assert payload["status"] == "ok"
    assert payload["singers"] == 0
    assert payload["segments"] == 0
    assert payload["skipped"]
    assert any("no usable" in message for message in payload["skipped"])


def test_a_bad_catalog_is_a_one_line_catalog_error_envelope(tmp_path):
    catalog = _catalog(
        tmp_path,
        [
            {
                "singer_id": "s1",
                "name": "Singer One",
                "gender": "not-a-gender",
                "style_tag": "pop",
                "materials": ["../missing.wav"],
            }
        ],
    )
    out = tmp_path / "library"

    completed = _run_cli(
        "build", "--catalog", catalog, "--out", out, "--encoder", "hash"
    )
    payload = _one_json_line(completed)

    assert completed.returncode != 0
    assert payload["status"] == "error"
    assert payload["reason"] == "catalog_error"
    assert "gender" in payload["detail"]
    assert not out.exists()  # nothing was built from a bad catalog


def test_build_defaults_to_the_hubert_encoder(tmp_path):
    # A library is a real artefact, so the test stand-in must never be the
    # default.  Check the argparse default without loading the model.
    from voicematch.cli import _build_parser

    args = _build_parser().parse_args(
        ["build", "--catalog", "c.json", "--out", "lib"]
    )
    assert args.encoder == "hubert"


def test_build_help_states_the_encoder_choice():
    completed = _run_cli("build", "--help")

    assert completed.returncode == 0, completed.stderr
    assert "--catalog" in completed.stdout
    assert "--out" in completed.stdout
    assert "hubert" in completed.stdout
    assert "hash" in completed.stdout
