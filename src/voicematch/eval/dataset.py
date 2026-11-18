"""The evaluation set: annotated queries, and a loader that refuses a broken one.

One evaluation query names a recording, the gender whose candidate pool the
annotator answered against, the candidates they were shown, what they picked,
and how many annotators agreed on it.  ``picked == []`` is a real answer --
"都不像" (none of these), ruling R9 -- which is why the field is a list and the
loader accepts it; a *missing* ``picked`` is a different thing and raises.

G3's capture-channel identifiers are optional
---------------------------------------------
``subject_id`` (which person made the recording) and ``device_id`` (which
capture channel it came from) are what cross-device stability needs: design 6.3
asks for the Top-3 overlap between one subject's devices.  They are optional
(ruling R27) because the phase-0/1 eval sets may not carry them yet, and
:func:`~voicematch.eval.run.run_eval` reports G3 as *not measurable* rather than
inventing a number when they are absent.  A present but empty/mistyped value is
still refused: that is a broken annotation, not a missing one.

Validation is done once, here, and every failure names the query
---------------------------------------------------------------
A broken annotation that loaded successfully would change a denominator
halfway through an evaluation run, and the resulting number would look
measured.  So the loader refuses the whole file instead: missing or mistyped
fields, an unknown gender, a picked singer that was not among the candidates, a
duplicate query id, or a recording path that does not exist.  Each message
names the query id (or, when the id itself is unusable, the record position),
so a large eval set points at its broken record.

A relative recording path is resolved against the eval set's own directory
---------------------------------------------------------------------------
That lets an eval set travel with its clips: ``{"path": "clips/q1.wav"}`` means
``<eval-set dir>/clips/q1.wav``, not "wherever the process happened to run".
An absolute path is used as given.

The type is frozen but deliberately unhashable
----------------------------------------------
``picked`` is a ``set`` (the annotation's meaning is membership), so a
generated ``__hash__`` would raise ``TypeError`` from inside the hash.  Setting
``__hash__ = None`` fails cleanly, naming this type, for the same reason
``LibraryIndex`` does (R19).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: The genders a query's candidate pool can be scoped to (design doc 5.6).
GENDERS: frozenset[str] = frozenset({"female", "male"})


@dataclass(frozen=True)
class EvalQuery:
    """One annotated evaluation query.

    ``candidates`` are the singer ids the annotator was shown, in the order the
    evaluation set recorded them; ``picked`` is the set of those singers they
    judged similar (an empty set means 都不像).  ``annotator_count`` is how many
    annotators voted on this query.  ``path`` is the resolved recording path
    (see the module docstring).

    ``subject_id`` and ``device_id`` are G3's optional capture-channel
    identifiers (ruling R27): which person and which device produced the
    recording.  ``None`` means the eval set does not carry them.
    """

    query_id: str
    path: Path
    gender: str
    candidates: tuple[str, ...]
    picked: set[str]
    annotator_count: int
    subject_id: str | None = None
    device_id: str | None = None

    # Deliberately unhashable: ``picked`` is a set, so the generated hash would
    # fail halfway through with "unhashable type: 'set'".  Failing as this type
    # is the clearer error (same reasoning as R19 for LibraryIndex).
    __hash__ = None


def load_eval_set(path: Path) -> list[EvalQuery]:
    """Read and validate the JSON eval set at ``path``.

    The file must contain a JSON list; each item is one query object.  Every
    record is validated by :func:`_parse_query` and the file is refused whole
    on the first problem, with the query id in the message.  An empty list is
    a valid (empty) eval set -- :func:`voicematch.eval.run.run_eval` then
    reports ``nan``, because nothing was measured, rather than ``0.0``.
    """
    source = Path(path)
    records = _read_records(source)
    queries = [
        _parse_query(record, position, source)
        for position, record in enumerate(records)
    ]
    _refuse_duplicate_ids(queries, source)
    return queries


def _read_records(source: Path) -> list[object]:
    """Parse the file as a JSON list, or explain why it is not one."""
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: not valid JSON ({exc})") from exc
    if not isinstance(payload, list):
        raise ValueError(
            f"{source}: the eval set must be a JSON list of query objects"
        )
    return payload


def _parse_query(record: object, position: int, source: Path) -> EvalQuery:
    """Validate one JSON record into an :class:`EvalQuery`."""
    if not isinstance(record, dict):
        raise ValueError(
            f"{source}: query at position {position} must be a JSON object"
        )
    label = _query_label(record, position)
    query_id = _required_string(record, "query_id", label)
    label = f"query {query_id!r}"
    gender = _required_string(record, "gender", label)
    if gender not in GENDERS:
        raise ValueError(
            f"{label}: gender must be one of {sorted(GENDERS)}, got {gender!r}"
        )
    candidates = _required_strings(record, "candidates", label, allow_empty=False)
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"{label}: 'candidates' contains duplicates")
    picked = set(_required_strings(record, "picked", label, allow_empty=True))
    unknown = picked - set(candidates)
    if unknown:
        raise ValueError(
            f"{label}: picked singers {sorted(unknown)} are not among the "
            "candidates"
        )
    annotator_count = _required_count(record, "annotator_count", label)
    audio_path = _resolve_audio_path(
        _required_string(record, "path", label), label, source
    )
    return EvalQuery(
        query_id=query_id,
        path=audio_path,
        gender=gender,
        candidates=candidates,
        picked=picked,
        annotator_count=annotator_count,
        subject_id=_optional_string(record, "subject_id", label),
        device_id=_optional_string(record, "device_id", label),
    )


def _query_label(record: dict, position: int) -> str:
    """Name a record by its query id when it has one, else by position."""
    query_id = record.get("query_id")
    if isinstance(query_id, str) and query_id:
        return f"query {query_id!r}"
    return f"query at position {position}"


def _required_string(record: dict, key: str, label: str) -> str:
    """Return ``record[key]`` as a non-empty string or raise ``ValueError``."""
    if key not in record:
        raise ValueError(f"{label}: missing required field {key!r}")
    value = record[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: {key!r} must be a non-empty string")
    return value


def _required_strings(
    record: dict, key: str, label: str, *, allow_empty: bool
) -> tuple[str, ...]:
    """Return ``record[key]`` as a tuple of non-empty strings, validated."""
    if key not in record:
        raise ValueError(f"{label}: missing required field {key!r}")
    value = record[key]
    if not isinstance(value, list):
        raise ValueError(f"{label}: {key!r} must be a JSON list of singer ids")
    if not value and not allow_empty:
        raise ValueError(f"{label}: {key!r} must not be empty")
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"{label}: {key!r} must contain only non-empty strings"
            )
    return tuple(value)


def _optional_string(record: dict, key: str, label: str) -> str | None:
    """Return ``record[key]`` as a non-empty string, or ``None`` when absent.

    Absent and JSON ``null`` both mean "the eval set does not carry this
    identifier" (ruling R27); any other type or an empty string is a broken
    annotation and fails with the query id in the message.
    """
    if key not in record:
        return None
    value = record[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{label}: {key!r} must be a non-empty string when present"
        )
    return value


def _required_count(record: dict, key: str, label: str) -> int:
    """Return ``record[key]`` as a positive integer, rejecting booleans."""
    if key not in record:
        raise ValueError(f"{label}: missing required field {key!r}")
    value = record[key]
    # JSON true/false parse to bool, which is an int subclass: the annotation
    # count is a number of people, not a flag.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}: {key!r} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(
            f"{label}: {key!r} must be at least 1, got {value}"
        )
    return value


def _resolve_audio_path(raw_path: str, label: str, source: Path) -> Path:
    """Resolve ``raw_path`` relative to the eval set and require the file."""
    audio_path = Path(raw_path)
    if not audio_path.is_absolute():
        audio_path = source.parent / audio_path
    if not audio_path.is_file():
        raise ValueError(f"{label}: recording does not exist: {audio_path}")
    return audio_path


def _refuse_duplicate_ids(queries: list[EvalQuery], source: Path) -> None:
    """Refuse two queries with the same id: annotations must pair one-to-one."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for query in queries:
        if query.query_id in seen:
            duplicates.add(query.query_id)
        seen.add(query.query_id)
    if duplicates:
        raise ValueError(
            f"{source}: duplicate query ids {sorted(duplicates)}; every "
            "annotation must have a unique id"
        )
