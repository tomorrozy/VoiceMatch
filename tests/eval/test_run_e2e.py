"""End-to-end evaluation tests: every stage meets, and the denominators stay honest.

Two product rules are measured here rather than assumed:

- **G1** -- a Top-K hit rate may only be reported next to the acceptance rate
  it was measured over, and the result must carry the counts behind both.
  ``top1``/``top3`` are computed only over accepted, answerable queries;
  ``acceptance_rate`` is over every query.
- **R9** -- ``picked == []`` is an answer ("都不像"), not a miss.  It is counted
  in ``n_none_like`` and leaves the Top-K denominator, which
  ``metrics.top_k_hit`` already implements; the runner must not count it twice.

The other failure the runner must survive is a single non-finite ranking score
(an upstream prototype with a ``nan``): that query is skipped and counted in
``n_skipped`` instead of aborting the run or being scored as a miss.

``n_answerable`` is reported alongside the counts: it is exactly the Top-K
denominator, so a reader never has to reconstruct it.  For every test the
ledger must add up: ``n_answerable == n_accepted - n_none_like - n_skipped``
(where ``n_none_like`` and ``n_skipped`` count accepted queries).
"""

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from voicematch.calibrate.model import CalibrationModel
from voicematch.calibrate.pairs import FEATURE_NAMES
from voicematch.config import EMBED_DIM, MAX_INPUT_SEC
from voicematch.eval.dataset import load_eval_set
from voicematch.eval.run import calibration_version, run_eval
from voicematch.library.schema import LibraryIndex

#: Every count G1, the non-finite-score ruling and R23 require the report to carry.
_COUNT_KEYS = {
    "n_total",
    "n_accepted",
    "n_rejected",
    "n_none_like",
    "n_skipped",
    "n_truncated",
    "n_answerable",
}


def _record(
    query_id: str,
    audio,
    *,
    gender: str = "female",
    candidates=("s1", "s2"),
    picked=("s1",),
    annotator_count: int = 3,
    **extra,
) -> dict:
    """One eval-set record, with the fields a test wants to vary."""
    record = {
        "query_id": query_id,
        "path": str(audio),
        "gender": gender,
        "candidates": list(candidates),
        "picked": list(picked),
        "annotator_count": annotator_count,
    }
    record.update(extra)
    return record


def _write_eval_set(tmp_path, records) -> object:
    """Write a JSON eval set and return its path."""
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _run(tiny_stack, spec, *, index: LibraryIndex | None = None) -> dict:
    """Load ``spec`` and evaluate it against the tiny stack (or ``index``)."""
    queries = load_eval_set(spec)
    library = tiny_stack.index if index is None else index
    return run_eval(queries, library, tiny_stack.calibration, tiny_stack.encoder)


def _with_non_finite_prototype(index: LibraryIndex, singer_id: str) -> LibraryIndex:
    """Return ``index`` with one singer's prototype set to ``nan``.

    This is the upstream state the non-finite ruling is about: ranking accepts
    the prototype (only its shape is checked) and one scored row comes back
    ``nan``.
    """
    prototypes = {
        key: np.array(value, copy=True) for key, value in index.prototypes.items()
    }
    prototypes[singer_id] = np.full(EMBED_DIM, np.nan, dtype=np.float32)
    return LibraryIndex(
        version=index.version,
        model_id=index.model_id,
        singers=dict(index.singers),
        segments=index.segments,
        prototypes=prototypes,
        features=dict(index.features),
        embeddings=index.embeddings,
    )


def _subset_index(index: LibraryIndex, singer_ids: set[str]) -> LibraryIndex:
    """Return ``index`` restricted to ``singer_ids`` (used to empty a pool)."""
    segments = tuple(
        segment for segment in index.segments if segment.singer_id in singer_ids
    )
    row_of = {
        segment.segment_id: row for row, segment in enumerate(index.segments)
    }
    embeddings = (
        np.stack(
            [index.embeddings[row_of[segment.segment_id]] for segment in segments]
        ).astype(np.float32)
        if segments
        else np.zeros((0, EMBED_DIM), dtype=np.float32)
    )
    return LibraryIndex(
        version=index.version,
        model_id=index.model_id,
        singers={
            singer_id: index.singers[singer_id] for singer_id in singer_ids
        },
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


# --- the brief's test (wrapped only for line length) ------------------------


def test_end_to_end_returns_both_headline_numbers(tmp_path, tiny_stack):
    q = tmp_path / "q.wav"
    t = np.arange(24000 * 12) / 24000
    sf.write(q, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 24000)
    spec = tmp_path / "eval.json"
    spec.write_text(json.dumps([{"query_id": "q1", "path": str(q),
                                 "gender": "female", "candidates": ["s1", "s2"],
                                 "picked": ["s1"], "annotator_count": 3}]))
    queries = load_eval_set(spec)
    out = run_eval(
        queries, tiny_stack.index, tiny_stack.calibration, tiny_stack.encoder
    )
    assert {"top1", "top3", "acceptance_rate"} <= set(out)
    assert out["n_accepted"] <= 1


# --- the fixture really exercises the calibrated path (R21 carry-over) ------


def test_the_fixture_stack_is_calibrated(tiny_stack):
    # Four singers are what makes build_pairs emit pairs at all (R21), so the
    # end-to-end test above runs against a fitted model instead of the
    # canonical cosine-only fallback.
    genders = sorted(singer.gender for singer in tiny_stack.index.singers.values())
    assert genders == ["female", "female", "male", "male"]
    assert tiny_stack.calibration.is_calibrated is True


# --- accepted, answerable queries: both numbers measured --------------------


def test_an_accepted_answerable_query_reports_finite_hit_rates(
    tmp_path, tiny_stack, write_syllable_query
):
    audio = tmp_path / "sung.wav"
    write_syllable_query(audio)
    spec = _write_eval_set(tmp_path, [_record("q1", audio)])

    out = _run(tiny_stack, spec)

    assert _COUNT_KEYS <= set(out)
    assert out["n_total"] == 1
    assert out["n_accepted"] == 1
    assert out["n_rejected"] == 0
    assert out["n_none_like"] == 0
    assert out["n_skipped"] == 0
    assert out["n_answerable"] == 1
    assert out["acceptance_rate"] == 1.0
    for key in ("top1", "top3"):
        assert math.isfinite(out[key])
        assert 0.0 <= out[key] <= 1.0


# --- R9: 都不像 is an answer, not a ranking miss -----------------------------


def test_a_none_like_answer_is_counted_but_not_in_the_top_k_denominator(
    tmp_path, tiny_stack, write_syllable_query
):
    audio = tmp_path / "sung.wav"
    write_syllable_query(audio)
    spec = _write_eval_set(tmp_path, [_record("q1", audio, picked=[])])

    out = _run(tiny_stack, spec)

    # The query was accepted, so it stays in the acceptance-rate denominator...
    assert out["n_accepted"] == 1
    assert out["acceptance_rate"] == 1.0
    # ...but it is an answer with nothing to hit, so nothing was measured.
    assert out["n_none_like"] == 1
    assert out["n_answerable"] == 0
    assert math.isnan(out["top1"])
    assert math.isnan(out["top3"])


# --- rejected queries stay in the acceptance denominator only ---------------


def test_a_rejected_query_is_not_scored_but_is_in_the_acceptance_denominator(
    tmp_path, tiny_stack, write_wav
):
    # Digital silence: the quality gate rejects it, and the report's snr_db is
    # -inf (a value the metrics never read; JSON is Task 14's CLI boundary).
    audio = tmp_path / "silence.wav"
    write_wav(audio, secs=10.0, amplitude=0.0)
    spec = _write_eval_set(tmp_path, [_record("q1", audio)])

    out = _run(tiny_stack, spec)

    assert out["n_total"] == 1
    assert out["n_rejected"] == 1
    assert out["n_accepted"] == 0
    assert out["acceptance_rate"] == 0.0
    assert out["n_answerable"] == 0
    assert math.isnan(out["top1"])
    assert math.isnan(out["top3"])


# --- one bad row is skipped, not fatal (non-finite score ruling) ------------


def test_a_non_finite_ranking_score_skips_the_query(
    tmp_path, tiny_stack, write_syllable_query
):
    audio = tmp_path / "sung.wav"
    write_syllable_query(audio)
    spec = _write_eval_set(tmp_path, [_record("q1", audio)])
    index = _with_non_finite_prototype(tiny_stack.index, "s2")

    out = _run(tiny_stack, spec, index=index)

    assert out["n_accepted"] == 1
    assert out["n_skipped"] == 1
    assert out["n_answerable"] == 0
    assert math.isnan(out["top1"])
    assert math.isnan(out["top3"])


# --- the ledger adds up over a mixed eval set -------------------------------


def test_mixed_eval_set_keeps_the_denominators_apart(
    tmp_path, tiny_stack, write_syllable_query, write_wav
):
    accepted = tmp_path / "accepted.wav"
    write_syllable_query(accepted)
    none_like = tmp_path / "none_like.wav"
    write_syllable_query(none_like, f0=330.0)
    silent = tmp_path / "silent.wav"
    write_wav(silent, secs=10.0, amplitude=0.0)
    spec = _write_eval_set(
        tmp_path,
        [
            _record("a", accepted),
            _record("b", none_like, picked=[]),
            _record("c", silent),
        ],
    )

    out = _run(tiny_stack, spec)

    assert out["n_total"] == 3
    assert out["n_accepted"] == 2
    assert out["n_rejected"] == 1
    assert out["n_none_like"] == 1
    assert out["n_skipped"] == 0
    assert out["n_answerable"] == 1
    assert out["n_answerable"] == (
        out["n_accepted"] - out["n_none_like"] - out["n_skipped"]
    )
    assert out["acceptance_rate"] == pytest.approx(2 / 3)
    assert math.isfinite(out["top3"])


# --- R23: the 60 s input contract is applied to the audio, not the schema ---


def test_a_query_longer_than_the_input_contract_is_truncated_not_rejected(
    tmp_path, tiny_stack, write_syllable_query, write_wav
):
    long_query = tmp_path / "long.wav"
    write_syllable_query(long_query, secs=MAX_INPUT_SEC + 1.0)
    # The boundary is inclusive: exactly 60 s is within the contract.
    exactly_limit = tmp_path / "limit.wav"
    write_wav(exactly_limit, secs=MAX_INPUT_SEC, amplitude=0.0)
    spec = _write_eval_set(
        tmp_path,
        [_record("long", long_query), _record("limit", exactly_limit)],
    )

    out = _run(tiny_stack, spec)

    # Only the >60 s recording is cut, and cutting it is not a rejection: it
    # is still accepted and still answerable.  The silent exactly-60 s query is
    # rejected by the quality gate, not by the length contract.
    assert out["n_truncated"] == 1
    assert out["n_total"] == 2
    assert out["n_accepted"] == 1
    assert out["n_rejected"] == 1
    assert out["n_answerable"] == 1
    assert math.isfinite(out["top3"])


# --- an empty eval set measured nothing -------------------------------------


def test_an_empty_eval_set_measures_nothing(tiny_stack):
    out = run_eval([], tiny_stack.index, tiny_stack.calibration, tiny_stack.encoder)

    assert out["n_total"] == 0
    assert out["n_accepted"] == 0
    assert out["n_rejected"] == 0
    assert out["n_none_like"] == 0
    assert out["n_skipped"] == 0
    assert out["n_truncated"] == 0
    assert out["n_answerable"] == 0
    for key in ("top1", "top3", "acceptance_rate"):
        assert math.isnan(out[key])


# --- the query's own gender selects the pool --------------------------------


def test_ranking_uses_the_querys_own_gender_pool(
    tmp_path, tiny_stack, write_syllable_query
):
    # A male query against a female-only library cannot be answered: ranking
    # must raise the explicit empty-pool LookupError, not silently rank the
    # female pool.  If the runner hard-coded a gender this test would pass a
    # score back instead.
    audio = tmp_path / "query.wav"
    write_syllable_query(audio)
    spec = _write_eval_set(
        tmp_path,
        [_record("q1", audio, gender="male", candidates=("s3", "s4"), picked=["s3"])],
    )
    female_only = _subset_index(tiny_stack.index, {"s1", "s2"})

    with pytest.raises(LookupError, match="no_singers_in_pool"):
        _run(tiny_stack, spec, index=female_only)


# --- I6/G3: cross-device stability is reported, visibly not measurable ------


def test_g3_is_not_measurable_without_the_subject_and_device_identifiers(
    tmp_path, tiny_stack, write_syllable_query
):
    # Ruling R27: a phase-0/1 eval set carries no capture-channel metadata, so
    # G3 cannot be measured at all.  The report must say so instead of leaving
    # a plausible-looking stability number.
    audio = tmp_path / "q.wav"
    write_syllable_query(audio)
    spec = _write_eval_set(tmp_path, [_record("q1", audio)])

    out = _run(tiny_stack, spec)

    assert out["g3_measurable"] is False
    assert math.isnan(out["stability"])


def test_g3_is_not_measurable_when_no_subject_was_recorded_on_two_devices(
    tmp_path, tiny_stack, write_syllable_query
):
    audio = tmp_path / "q.wav"
    write_syllable_query(audio)
    spec = _write_eval_set(
        tmp_path,
        [_record("q1", audio, subject_id="person-1", device_id="phone-a")],
    )

    out = _run(tiny_stack, spec)

    assert out["g3_measurable"] is False
    assert math.isnan(out["stability"])


def test_g3_stability_is_the_mean_top3_overlap_across_a_subjects_devices(
    tmp_path, tiny_stack, write_syllable_query
):
    # Two identical recordings from one subject on two devices give the same
    # Top-3, so their overlap is the pool's maximum.  (``top3_overlap`` divides
    # by the fixed design denominator 3, and the tiny library's pool has two
    # singers, so identical rankings score 2/3.)  A third query from another
    # subject on a single device has no partner and must not enter the mean (it
    # would make the number depend on how many single-device subjects were in
    # the set).
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    other = tmp_path / "other.wav"
    write_syllable_query(first)
    write_syllable_query(second)
    write_syllable_query(other, f0=330.0)
    spec = _write_eval_set(
        tmp_path,
        [
            _record("q1", first, subject_id="person-1", device_id="phone-a"),
            _record("q2", second, subject_id="person-1", device_id="phone-b"),
            _record("q3", other, subject_id="person-2", device_id="phone-a"),
        ],
    )

    out = _run(tiny_stack, spec)

    assert out["n_accepted"] == 3
    assert out["g3_measurable"] is True
    assert out["stability"] == pytest.approx(2 / 3)


# --- calibration_version: the model identity the CLI reports ----------------
_FALLBACK_MODEL = CalibrationModel(
    coef=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    mean=np.zeros(len(FEATURE_NAMES)),
    scale=np.ones(len(FEATURE_NAMES)),
    intercept=0.0,
    auc=float("nan"),
)


def test_calibration_version_is_a_short_stable_hash(tiny_stack):
    first = calibration_version(tiny_stack.calibration)
    second = calibration_version(tiny_stack.calibration)

    assert first == second
    assert len(first) == 16
    assert set(first) <= set("0123456789abcdef")


def test_calibration_version_changes_when_the_model_changes(tiny_stack):
    changed = replace(
        tiny_stack.calibration,
        intercept=tiny_stack.calibration.intercept + 1.0,
    )

    assert calibration_version(changed) != calibration_version(tiny_stack.calibration)


def test_the_cosine_only_fallback_has_its_own_version(tiny_stack):
    assert calibration_version(_FALLBACK_MODEL) != calibration_version(
        tiny_stack.calibration
    )
