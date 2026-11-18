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
The library records the identity of the encoder that produced its embeddings
(``index.model_id``, ruling R24), and ``--encoder`` selects the query-side
model: ``hash`` is the deterministic stand-in that needs no optional ``ml``
extra (the default, so the documented command works in a plain install) and
``hubert`` is the production backbone.  The CLI **refuses to run** when the
chosen encoder's identity differs from the library's, naming both ids: a query
vector from a different model lives in an unrelated vector space, and the
resulting ranking would look perfectly normal while measuring noise.  The
chosen name is echoed in the output for the same reason.

The calibration model is fitted from the library's own stored segment pairs;
nothing is trained on the query.

Query failures are one line of JSON
----------------------------------
The compare path is wrapped: an expected refusal of the recording is reported as
``status == "rejected"`` and any other failure as ``status == "error"``, both as
the same single line of strict JSON instead of a traceback with empty stdout
(C1).  ``reason`` names the condition with a stable token: the first violated
quality rule for a refusal (or ``no_usable_window``), the calibration module's
``no_singers_in_pool``/``non_finite_prototype`` when the library cannot answer,
``not_supported_in_v1`` for a switch this version deliberately does not
implement, and otherwise the exception's type name.  ``detail`` keeps the
human-readable message.

Exit status is ``0`` for ``ok``, ``rejected`` and ``no_usable_window``: a
refusal is a valid product answer that the payload describes, and a script
reads ``status`` for the distinction.  ``error`` exits with ``1``.  Command-line
usage errors keep argparse's behaviour (message on stderr, no JSON), and so does
the ``--encoder``/library mismatch below, which is a configuration refusal
rather than a property of the recording.

What is assumed about the input
-------------------------------
v1 accepts a-cappella recordings only (ruling R25): it has no accompaniment
detection and no source separation, so every payload states the assumption as
``assumed: "a_cappella"`` rather than letting a score be read as if the input
had been cleaned.  ``--separate`` is kept as the v2 switch and is *refused* with
a structured ``not_supported_in_v1`` error instead of silently doing nothing.

The 60 s input contract is applied here too
-------------------------------------------
``truncated`` says whether the recording was longer than design 5.6's
``MAX_INPUT_SEC`` and was evaluated on its first 60 s.  The same helper as the
evaluation runner does the cutting (I8), truncation is not a rejection, and the
flag lets a consumer tell a cut query from a short one.  The flag is carried
onto the ``error`` document too: a query is cut at the input stage and can then
fail, and the failure must not claim it was never truncated.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from voicematch.audio.io import load_audio
from voicematch.audio.preprocess import PREPROCESS_QUERY, preprocess
from voicematch.audio.quality import QualityReport, assess_quality
from voicematch.audio.windows import select_best_window
from voicematch.calibrate import NO_SINGERS_IN_POOL, NON_FINITE_PROTOTYPE
from voicematch.calibrate.model import CalibrationModel, fit_calibration
from voicematch.calibrate.pairs import build_pairs
from voicematch.calibrate.score import NonFinitePrototypeError, rank_singers
from voicematch.config import SAMPLE_RATE
from voicematch.embed.encoder import HashEncoder, HubertEncoder, VocalEncoder
from voicematch.eval.run import calibration_version, truncate_to_input_contract
from voicematch.features.extract import extract_features
from voicematch.library.schema import LibraryIndex
from voicematch.library.store import load_library

#: Ranking depth the CLI renders (design doc: Top-3).
TOP_K: int = 3

#: Exit status of a successful run and of any run whose document is about a
#: recording rather than a failure.  A refusal is a valid product answer.
_EXIT_OK: int = 0

#: Exit status of an ``error`` document: the query could not be answered.
_EXIT_ERROR: int = 1

#: What the pipeline assumes about the query audio in v1 (ruling R25).  v1
#: accepts clean a-cappella recordings only: it has no accompaniment detection
#: and no source separation, so the field states the assumption instead of
#: hiding it.  A future version that separates would set this to the separation
#: state; a consumer reads it to know what the score was measured under.
ASSUMED: str = "a_cappella"

#: Token for "a feature exists as a switch but this version does not implement
#: it" (``--separate`` in v1).
NOT_SUPPORTED_IN_V1: str = "not_supported_in_v1"


class _UnsupportedInV1(ValueError):
    """A switch exists but v1 deliberately does not implement it (ruling R25)."""


@dataclass
class _QueryAttempt:
    """Facts about a query that a *later* failure must still report.

    Truncation is decided on the audio before the query can fail (the empty
    pool and the non-finite prototype are both reached afterwards), so the flag
    cannot ride on the exception; the handler records it here and the envelope
    reads it back when building the ``error`` document (C1/I8).
    """

    truncated: bool = False


#: Command-line encoder name -> encoder class.  The class carries its own
#: ``model_id`` without being constructed, so the mismatch check below runs
#: before HuBERT would need the optional ``ml`` extra.
_ENCODER_CLASSES: dict[str, type[VocalEncoder]] = {
    "hubert": HubertEncoder,
    "hash": HashEncoder,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, run the subcommand and print one line of JSON.

    Every exception raised by the handler becomes the error document of the
    module docstring, so a broken recording still prints one parseable line.
    ``SystemExit`` (argparse usage errors and the encoder mismatch) is a
    ``BaseException`` and deliberately keeps its stderr behaviour.
    """
    args = _build_parser().parse_args(argv)
    attempt = _QueryAttempt()
    try:
        payload = args.handler(args, attempt)
    except Exception as exc:  # noqa: BLE001 - the envelope is the CLI contract
        payload = _failure_payload(exc, truncated=attempt.truncated)
    print(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return _EXIT_ERROR if payload.get("status") == "error" else _EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree for the CLI."""
    parser = argparse.ArgumentParser(
        prog="voicematch",
        description=(
            "Offline voice-to-singer similarity engine.  v1 assumes a_cappella "
            "query audio: it does not detect or remove accompaniment, so a "
            "take with backing music is scored as if it were clean."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    compare = subcommands.add_parser(
        "compare",
        help="rank the singers of one gender against one query recording",
        description=(
            "Rank the singers of one gender against one query recording.  "
            "v1 assumes a_cappella input: there is no accompaniment detection "
            "and no source separation, and the payload reports "
            f'assumed: "{ASSUMED}".  --separate is kept as the v2 switch '
            "and is not supported in v1."
        ),
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
        choices=tuple(_ENCODER_CLASSES),
        default="hash",
        help=(
            "encoder used for the query; it must match the library's recorded "
            "model id (hash is the default stand-in, hubert is production)"
        ),
    )
    compare.add_argument(
        "--separate",
        action="store_true",
        help=(
            "run vocal separation before scoring (v2; not supported in v1, "
            "where the request is refused with a structured error)"
        ),
    )
    compare.set_defaults(handler=_compare)
    return parser


def _compare(args: argparse.Namespace, attempt: _QueryAttempt) -> dict[str, Any]:
    """Run one query against a library and build the result document.

    The quality gate measures the **decoded** signal, before the LUFS gain, so
    a hard-clipped source is reported as ``clipped`` instead of having its
    clipping hidden by the normalisation, while the limiter inside
    ``preprocess`` cannot be mistaken for source clipping (finding C2).
    """
    if args.separate:
        # Ruling R25: v1 is a-cappella only.  Refuse the switch instead of
        # silently scoring an accompanied take as if it were clean.
        raise _UnsupportedInV1(
            "--separate is not supported in v1: v1 assumes a_cappella query "
            "audio and does not run source separation; remove the take's "
            "accompaniment or wait for the v2 separation switch"
        )
    index = load_library(args.library)
    encoder = _encoder(args.encoder, index)
    model = _calibration_for(index)
    # The 60 s input contract (design 5.6) is applied to the audio with the same
    # helper the evaluation runner uses, and surfaced as ``truncated`` (I8);
    # truncation is not a rejection.
    decoded, truncated = truncate_to_input_contract(load_audio(args.audio))
    # Record the cut before anything below can raise: the error envelope must
    # report the same flag the ordinary payload does (C1/I8).
    attempt.truncated = truncated
    report = assess_quality(decoded, SAMPLE_RATE)

    payload: dict[str, Any] = {
        "model_id": index.model_id,
        "library_version": index.version,
        "calibration_version": calibration_version(model),
        "calibrated": model.is_calibrated,
        "encoder": args.encoder,
        "query": {"path": str(args.audio), "gender": args.gender},
        "assumed": ASSUMED,
        "quality": _quality_payload(report),
        "status": "rejected" if report.rejected else "ok",
        "reason": report.reasons[0] if report.rejected else None,
        "truncated": truncated,
        "results": [],
    }
    if report.rejected:
        return payload

    processed = preprocess(decoded, PREPROCESS_QUERY)
    window = select_best_window(processed, SAMPLE_RATE)
    if window is None:
        payload["status"] = "no_usable_window"
        payload["reason"] = "no_usable_window"
        return payload
    payload["results"] = _results(
        processed, window, index, model, encoder, args.gender
    )
    return payload


def _failure_payload(
    exc: Exception, *, truncated: bool = False
) -> dict[str, Any]:
    """Return the one-line ``error`` document for an exception from the query.

    The reason is a stable token where the code has one -- the empty-pool
    refusal names :data:`NO_SINGERS_IN_POOL` from the ranking module, a broken
    candidate prototype names :data:`NON_FINITE_PROTOTYPE`, so a consumer maps
    on the same constants the library uses instead of on a copy of their text
    (C1/I9) -- and the exception's type name otherwise.  ``detail`` keeps the
    human-readable message, and ``truncated`` is the flag the handler recorded
    on the attempt (a cut query can fail after the cut).  A failure raised
    before the input contract was applied leaves it ``False``.
    """
    if isinstance(exc, LookupError) and NO_SINGERS_IN_POOL in str(exc):
        reason = NO_SINGERS_IN_POOL
    elif isinstance(exc, NonFinitePrototypeError):
        reason = NON_FINITE_PROTOTYPE
    elif isinstance(exc, _UnsupportedInV1):
        reason = NOT_SUPPORTED_IN_V1
    else:
        reason = type(exc).__name__
    return {
        "status": "error",
        "reason": reason,
        "detail": str(exc),
        "assumed": ASSUMED,
        "truncated": truncated,
        "results": [],
    }


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


def _encoder(name: str, index: LibraryIndex) -> VocalEncoder:
    """Construct the named encoder after checking it matches the library.

    The check runs against the chosen *class's* identity before construction,
    so a mismatch is refused even when the production backbone's optional
    dependency is not installed.  Raises ``SystemExit`` with both ids in the
    message: running with a different encoder would compare two unrelated
    vector spaces and report a normal-looking ranking about nothing (R24).
    """
    encoder_class = _ENCODER_CLASSES[name]
    if encoder_class.model_id != index.model_id:
        raise SystemExit(
            f"voicematch: --encoder {name!r} is {encoder_class.model_id!r} but "
            f"library {index.version!r} was built with {index.model_id!r}; "
            "refusing to compare vectors from two different encoders"
        )
    return encoder_class()


def _results(
    processed: np.ndarray,
    window: tuple[float, float],
    index: LibraryIndex,
    model: CalibrationModel,
    encoder: VocalEncoder,
    gender: str,
) -> list[dict[str, Any]]:
    """Rank the gender pool of one query on the best-ranked selected window."""
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
