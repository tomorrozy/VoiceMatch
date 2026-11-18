"""Tests for the evaluation-set loader: a broken annotation must fail loudly.

The loader owns the eval schema, so the two answers it must never confuse are
"nobody was picked because 都不像" (``picked == []``, ruling R9) and "the
annotation is missing" (a ``ValueError``).  Every other structural error --
an unknown gender, a missing path, a picked singer that was not among the
candidates, a duplicate query id -- also fails at load time, with the query id
in the message, so a broken annotation cannot silently change a denominator
halfway through a run.
"""

import json

import pytest

from voicematch.eval.dataset import EvalQuery, load_eval_set


def _audio(tmp_path, name: str = "query.wav"):
    """Create an existing file for the loader's path check.

    The loader does not decode audio, so an empty file is enough: it only
    refuses a query whose recording is not there.
    """
    path = tmp_path / name
    path.write_bytes(b"")
    return path


def _record(audio, **overrides):
    """A valid annotation record, with named fields overridable per test."""
    record = {
        "query_id": "q1",
        "path": str(audio),
        "gender": "female",
        "candidates": ["s1", "s2"],
        "picked": ["s1"],
        "annotator_count": 3,
    }
    record.update(overrides)
    return record


def _write(tmp_path, records, name: str = "eval.json"):
    """Write a JSON eval set and return its path."""
    path = tmp_path / name
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_a_valid_query_parses_into_the_declared_shape(tmp_path):
    audio = _audio(tmp_path)
    queries = load_eval_set(_write(tmp_path, [_record(audio)]))

    assert queries == [
        EvalQuery(
            query_id="q1",
            path=audio,
            gender="female",
            candidates=("s1", "s2"),
            picked={"s1"},
            annotator_count=3,
        )
    ]


def test_an_empty_picked_list_is_the_none_like_answer(tmp_path):
    # R9: an empty list is an answer -- "none of the candidates sounded like
    # the query" -- and must load, not be read as a missing annotation.
    audio = _audio(tmp_path)
    queries = load_eval_set(_write(tmp_path, [_record(audio, picked=[])]))

    assert queries[0].picked == set()


def test_a_missing_picked_key_raises_with_the_query_id(tmp_path):
    audio = _audio(tmp_path)
    record = _record(audio)
    del record["picked"]

    with pytest.raises(ValueError) as excinfo:
        load_eval_set(_write(tmp_path, [record]))

    assert "q1" in str(excinfo.value)
    assert "picked" in str(excinfo.value)


def test_an_unknown_gender_fails_with_the_query_id(tmp_path):
    audio = _audio(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        load_eval_set(_write(tmp_path, [_record(audio, gender="other")]))

    message = str(excinfo.value)
    assert "q1" in message
    assert "gender" in message


def test_a_missing_path_key_fails_with_the_query_id(tmp_path):
    audio = _audio(tmp_path)
    record = _record(audio)
    del record["path"]

    with pytest.raises(ValueError) as excinfo:
        load_eval_set(_write(tmp_path, [record]))

    assert "q1" in str(excinfo.value)
    assert "path" in str(excinfo.value)


def test_a_path_that_does_not_exist_fails_with_the_query_id(tmp_path):
    spec = _write(tmp_path, [_record(tmp_path / "absent.wav")])

    with pytest.raises(ValueError) as excinfo:
        load_eval_set(spec)

    message = str(excinfo.value)
    assert "q1" in message
    assert "absent.wav" in message


def test_a_relative_path_is_resolved_against_the_eval_set(tmp_path):
    # An eval set should be able to travel with its clips: a relative path is
    # read from next to the JSON file, not from the process's working directory.
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "q.wav").write_bytes(b"")
    spec = _write(tmp_path, [_record("clips/q.wav")])

    assert load_eval_set(spec)[0].path == clips / "q.wav"


def test_a_picked_singer_must_be_one_of_the_candidates(tmp_path):
    audio = _audio(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        load_eval_set(_write(tmp_path, [_record(audio, picked=["s9"])]))

    message = str(excinfo.value)
    assert "q1" in message
    assert "candidates" in message


def test_candidates_must_be_a_list_of_strings(tmp_path):
    audio = _audio(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        load_eval_set(_write(tmp_path, [_record(audio, candidates="s1")]))

    message = str(excinfo.value)
    assert "q1" in message
    assert "candidates" in message


def test_duplicate_query_ids_are_refused(tmp_path):
    audio = _audio(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        load_eval_set(_write(tmp_path, [_record(audio), _record(audio)]))

    assert "q1" in str(excinfo.value)


def test_the_eval_set_must_be_a_json_list(tmp_path):
    spec = tmp_path / "eval.json"
    spec.write_text(json.dumps({"query_id": "q1"}), encoding="utf-8")

    with pytest.raises(ValueError, match="list"):
        load_eval_set(spec)


def test_an_empty_eval_set_loads_as_an_empty_list(tmp_path):
    assert load_eval_set(_write(tmp_path, [])) == []
