"""Tests for the pairwise re-annotation path (design 6.2 fallback)."""

from __future__ import annotations

import pytest

from voicematch.benchmark.agreement import KAPPA_THRESHOLD_DEFAULT
from voicematch.benchmark.pairwise import (
    PairAnnotation,
    adjudicate_pairwise,
    build_pairwise_sheet,
)


def test_threshold_default_is_documented_value():
    assert KAPPA_THRESHOLD_DEFAULT == 0.6


def test_build_pairwise_sheet_all_pairs():
    tasks = build_pairwise_sheet({"q1": ["s1", "s2", "s3"]})
    assert len(tasks) == 3  # C(3,2)
    pair_ids = {task.pair_id for task in tasks}
    assert pair_ids == {"s1|s2", "s1|s3", "s2|s3"}
    assert all(task.query_id == "q1" for task in tasks)


def test_build_pairwise_sheet_deduplicates_and_sorts():
    tasks = build_pairwise_sheet({"q1": ["s2", "s1", "s2"]})
    assert len(tasks) == 1
    assert tasks[0].singer_a == "s1"
    assert tasks[0].singer_b == "s2"


def test_build_pairwise_sheet_rejects_single_singer():
    with pytest.raises(ValueError, match="at least 2 singers"):
        build_pairwise_sheet({"q1": ["s1"]})


def test_pairwise_majority_picks_top_singer():
    tasks = build_pairwise_sheet({"q1": ["s1", "s2", "s3"]})
    # 3 annotators; s1 wins its two pairs 3-0, s2 beats s3 2-1.
    annotations = [
        PairAnnotation("q1", "s1|s2", "a", "a"),  # s1
        PairAnnotation("q1", "s1|s3", "a", "a"),  # s1
        PairAnnotation("q1", "s2|s3", "a", "a"),  # s2
        PairAnnotation("q1", "s1|s2", "b", "a"),
        PairAnnotation("q1", "s1|s3", "b", "a"),
        PairAnnotation("q1", "s2|s3", "b", "a"),
        PairAnnotation("q1", "s1|s2", "c", "a"),
        PairAnnotation("q1", "s1|s3", "c", "a"),
        PairAnnotation("q1", "s2|s3", "c", "b"),
    ]
    result = adjudicate_pairwise(tasks, annotations)
    # s1 wins 2 pairs -> 2 wins; s2 wins 1 -> picked top 2
    assert set(result["q1"].picked) == {"s1", "s2"}
    assert result["q1"].annotator_count == 3


def test_pairwise_unresolved_pair_marks_disputed():
    tasks = build_pairwise_sheet({"q1": ["s1", "s2"]})
    annotations = [
        PairAnnotation("q1", "s1|s2", "a", "a"),
        PairAnnotation("q1", "s1|s2", "b", "b"),
    ]
    result = adjudicate_pairwise(tasks, annotations)
    assert result["q1"].disputed is True
    assert result["q1"].picked == frozenset()


def test_pairwise_null_winner_is_cannot_tell():
    tasks = build_pairwise_sheet({"q1": ["s1", "s2"]})
    annotations = [
        PairAnnotation("q1", "s1|s2", "a", None),
        PairAnnotation("q1", "s1|s2", "b", None),
    ]
    result = adjudicate_pairwise(tasks, annotations)
    assert result["q1"].picked == frozenset()
    assert result["q1"].disputed is True


def test_pairwise_rejects_unknown_pair():
    with pytest.raises(ValueError, match="unknown pair"):
        adjudicate_pairwise(
            build_pairwise_sheet({"q1": ["s1", "s2"]}),
            [PairAnnotation("q1", "s9|s8", "a", "a")],
        )


def test_pairwise_rejects_bad_winner():
    with pytest.raises(ValueError, match="winner must be"):
        adjudicate_pairwise(
            build_pairwise_sheet({"q1": ["s1", "s2"]}),
            [PairAnnotation("q1", "s1|s2", "a", "s1")],
        )


def test_pairwise_caps_picked_at_two():
    tasks = build_pairwise_sheet({"q1": ["s1", "s2", "s3", "s4"]})
    # Each of 4 singers wins at least one pair -> still capped at 2.
    annotations = []
    for i, (pair, winner) in enumerate(
        [("s1|s2", "a"), ("s3|s4", "a"), ("s1|s3", "a"),
         ("s2|s4", "a"), ("s1|s4", "a"), ("s2|s3", "a")]
    ):
        annotations.append(PairAnnotation("q1", pair, "a", winner))
    result = adjudicate_pairwise(tasks, annotations)
    assert len(result["q1"].picked) <= 2
