"""Command-line entry point: one query in, one line of JSON out.

``python -m voicematch.cli compare <audio.wav> --library <dir> --gender female``
runs the product query path -- decode, preprocess, quality gate, automatic
window selection, features, frozen embedding, calibrated ranking -- and prints
one line of JSON to stdout.  The output carries the ranked results and the
three identifiers a result must be reproducible against (G4): ``model_id``,
``library_version`` and ``calibration_version``.

``status`` says which question was answered
-------------------------------------------
- ``"ok"``: the recording passed the quality gate and ``results`` is the
  ranking;
- ``"rejected"``: the gate refused the recording (design doc G5) and
  ``quality.reasons`` lists every violated rule; ``results`` is empty.  A
  refusal is not a ranking of nothing;
- ``"no_usable_window"``: the recording passed the gate, but the shared
  automatic rule found no window with enough usable speech to score.  This is
  distinct from a rejected input and from an empty ranking -- which cannot be
  produced, because ranking an empty gender pool is an explicit error rather
  than an empty list.

The encoder must match the library
----------------------------------
The persisted library format does not record which encoder produced its
per-segment embeddings (the version hash covers ``MODEL_ID`` but is not
readable as one), so ``--encoder`` names it and the chosen name is echoed in
the output.  ``hash`` is the default because it is the deterministic stand-in
that needs no optional ``ml`` extra, so the documented command works in a
plain install; a library built with the production backbone must be queried
with ``--encoder hubert``.  The calibration model is fitted from the library's
own stored segment pairs; nothing is trained on the query.

Serialisation
-------------
Everything printed is strict JSON: :func:`_jsonable` maps non-finite floats to
``null`` at this boundary -- a digitally silent recording's ``snr_db`` is
``-inf`` and a score can be ``nan``, neither of which is a valid JSON token --
and ``json.dumps`` is called with ``allow_nan=False`` so a leak is an error
instead of invalid output.  The document is ASCII-escaped, so it is encodable
on any console even when a singer name is not.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from voicematch.audio.io import load_audio
from voicematch.audio.preprocess import PREPROCESS_QUERY, preprocess
from voicematch.audio.quality import QualityReport, assess_quality
from voicematch.audio.windows import select_windows
from voicematch.calibrate.model import CalibrationModel, fit_calibration
from voicematch.calibrate.pairs import build_pairs
from voicematch.calibrate.score import rank_singers
from voicematch.config import SAMPLE_RATE
from voicematch.embed.encoder import (
    MODEL_ID,
    HashEncoder,
    HubertEncoder,
    VocalEncoder,
)
from voicematch.eval.run import calibration_version
from voicematch.features.extract import extract_features
from voicematch.library.schema import LibraryIndex
from voicematch.library.store import load_library

#: Ranking depth the CLI renders (design doc: Top-3).
TOP_K: int = 3


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, run the subcommand and print one line of JSON."""
    args = _build_parser().parse_args(argv)
    payload = args.handler(args)
    print(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree for the CLI."""
    parser = argparse.ArgumentParser(
        prog="voicematch",
        description="Offline voice-to-singer similarity engine.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    compare = subcommands.add_parser(
        "compare",
        help="rank the singers of one gender against one query recording",
    )
    compare.add_argument("audio", type=Path, metavar="AUDIO", help="query recording")
    compare.add_argument(
        "--library",
        type=Path,
        required=True,
        metavar="DIR",
        help="built singer-library directory",
    )
    compare.add_argument(
        "--gender",
        choices=("female", "male"),
        required=True,
        help="candidate pool to rank against",
    )
    compare.add_argument(
        "--encoder",
        choices=("hubert", "hash"),
        default="hash",
        help=(
            "encoder the library was built with; hash (default) is the "
            "stand-in that needs no optional ml extra"
        ),
    )
    compare.set_defaults(handler=_compare)
    return parser


def _compare(args: argparse.Namespace) -> dict[str, Any]:
    """Run one query against a library and build the result document."""
    index = load_library(args.library)
    model = _calibration_for(index)
    encoder = _encoder(args.encoder)
    processed = preprocess(load_audio(args.audio), PREPROCESS_QUERY)
    report = assess_quality(processed, SAMPLE_RATE)

    payload: dict[str, Any] = {
        "model_id": MODEL_ID,
        "library_version": index.version,
        "calibration_version": calibration_version(model),
        "calibrated": model.is_calibrated,
        "encoder": args.encoder,
        "query": {"path": str(args.audio), "gender": args.gender},
        "quality": _quality_payload(report),
        "status": "rejected" if report.rejected else "ok",
        "results": [],
    }
    if report.rejected:
        return payload

    windows = select_windows(processed, SAMPLE_RATE)
    if not windows:
        payload["status"] = "no_usable_window"
        return payload
    payload["results"] = _results(
        processed, windows[0], index, model, encoder, args.gender
    )
    return payload


def _calibration_for(index: LibraryIndex) -> CalibrationModel:
    """Fit the calibration model from the library's own segment pairs.

    This phase stores no calibration model; deriving it from the library's
    own embeddings and features keeps the result reproducible from the library
    and the code revision alone (G4).  A library too small to calibrate gets
    the canonical cosine-only fallback (``is_calibrated`` false), which the
    output reports rather than hiding.
    """
    X, y, groups = build_pairs(index, index.embeddings)
    return fit_calibration(X, y, groups)


def _encoder(name: str) -> VocalEncoder:
    """Return the encoder named on the command line."""
    return HashEncoder() if name == "hash" else HubertEncoder()


def _results(
    processed: np.ndarray,
    window: tuple[float, float],
    index: LibraryIndex,
    model: CalibrationModel,
    encoder: VocalEncoder,
    gender: str,
) -> list[dict[str, Any]]:
    """Rank the gender pool of one query on the first selected window."""
    start, end = window
    segment = processed[_sample_index(start) : _sample_index(end)]
    features = extract_features(segment, SAMPLE_RATE)
    embedding = encoder.encode(segment, SAMPLE_RATE)
    scored = rank_singers(features, embedding, index, model, gender, top_k=TOP_K)
    return [
        {
            "singer_id": entry.singer_id,
            "name": entry.name,
            "score": entry.score,
            "calibrated": entry.calibrated,
            "attribution": dict(entry.attribution),
        }
        for entry in scored
    ]


def _quality_payload(report: QualityReport) -> dict[str, Any]:
    """Render one quality report as a JSON-safe dictionary."""
    return {
        "speech_sec": report.speech_sec,
        "snr_db": report.snr_db,
        "clipped_ratio": report.clipped_ratio,
        "voiced_ratio": report.voiced_ratio,
        "rejected": report.rejected,
        "reasons": list(report.reasons),
    }


def _sample_index(seconds: float) -> int:
    """Convert a window boundary in seconds to a sample index.

    The same rounding as the library builder, so both sides of the pipeline cut
    a window to the same samples.
    """
    return int(round(seconds * SAMPLE_RATE))


def _jsonable(value: Any) -> Any:
    """Return a JSON-safe copy of ``value``; non-finite floats become ``None``.

    This is the serialisation boundary.  A silent recording's ``snr_db`` is
    ``-inf`` and a ranking score can be ``nan``; ``Infinity`` and ``NaN`` are
    not valid JSON tokens, so they are written as ``null`` ("not a
    measurement") instead of being emitted.  An unexpected type raises
    ``TypeError`` rather than being stringified into an unreadable field.
    """
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
