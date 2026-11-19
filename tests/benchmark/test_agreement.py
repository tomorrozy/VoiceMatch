"""Tests for Cohen's kappa, majority adjudication and agreement reporting."""

from __future__ import annotations

import math

import pytest

from voicematch.benchmark.agreement import (
    Annotation,
    cohen_kappa,
    adjudicate,
    agreement_report,
)


def test_kappa_perfect_agreement_is_one():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0


def test_kappa_with_none_label_counts_as_a_category():
    # Both say "都不像" on q2 -> still perfect agreement.
    assert cohen_kappa(["a", None, "b"], ["a", None, "b"]) == 1.0


def test_kappa_disagreement_is_below_one():
    value = cohen_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"])
    assert value < 0


def test_kappa_constant_labels_are_unmeasurable():
    assert math.isnan(cohen_kappa(["a", "a", "a"], ["a", "a", "a"]))


def test_kappa_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        cohen_kappa(["a", "b"], ["a"])


def test_kappa_empty_raises():
    with pytest.raises(ValueError, match="at least one item"):
        cohen_kappa([], [])


def test_majority_two_of_three_wins():
    annotations = [
        Annotation("q1", "a", ("s1",)),
        Annotation("q1", "b", ("s1",)),
        Annotation("q1", "c", ("s2",)),
    ]
    result = adjudicate(annotations, {"q1": ["s1", "s2", "s3"]})
    assert result["q1"].picked == frozenset({"s1"})
    assert result["q1"].annotator_count == 3
    assert result["q1"].disputed is True  # top1: s1,s1,s2


def test_majority_one_vote_each_is_none_like():
    annotations = [
        Annotation("q1", "a", ("s1",)),
        Annotation("q1", "b", ("s2",)),
        Annotation("q1", "c", ("s3",)),
    ]
    result = adjudicate(annotations, {"q1": ["s1", "s2", "s3"]})
    assert result["q1"].picked == frozenset()  # nobody > 1.5 votes


def test_unanimous_is_not_disputed():
    annotations = [
        Annotation("q1", "a", ("s1",)),
        Annotation("q1", "b", ("s1",)),
        Annotation("q1", "c", ("s1",)),
    ]
    result = adjudicate(annotations, {"q1": ["s1", "s2", "s3"]})
    assert result["q1"].picked == frozenset({"s1"})
    assert result["q1"].disputed is False


def test_none_like_unanimous_is_valid_answer():
    annotations = [
        Annotation("q1", "a", ()),
        Annotation("q1", "b", ()),
        Annotation("q1", "c", ()),
    ]
    result = adjudicate(annotations, {"q1": ["s1", "s2", "s3"]})
    assert result["q1"].picked == frozenset()
    assert result["q1"].disputed is False  # all None -> no disagreement


def test_two_picks_reaching_majority():
    annotations = [
        Annotation("q1", "a", ("s1", "s2")),
        Annotation("q1", "b", ("s1", "s2")),
        Annotation("q1", "c", ("s3",)),
    ]
    result = adjudicate(annotations, {"q1": ["s1", "s2", "s3"]})
    assert result["q1"].picked == frozenset({"s1", "s2"})


def test_rejects_pick_outside_candidates():
    with pytest.raises(ValueError, match="not in the shown candidate set"):
        adjudicate(
            [Annotation("q1", "a", ("s9",))],
            {"q1": ["s1", "s2"]},
        )


def test_rejects_more_than_two_picks():
    with pytest.raises(ValueError, match="at most 2"):
        adjudicate(
            [Annotation("q1", "a", ("s1", "s2", "s3"))],
            {"q1": ["s1", "s2", "s3"]},
        )


def test_rejects_annotation_without_candidate_set():
    with pytest.raises(ValueError, match="has no candidate set"):
        adjudicate([Annotation("q1", "a", ("s1",))], {})


def test_agreement_report_pairwise_and_mean():
    annotations = [
        Annotation("q1", "a", ("s1",)),
        Annotation("q1", "b", ("s1",)),
        Annotation("q2", "a", ("s2",)),
        Annotation("q2", "b", ("s2",)),
    ]
    report = agreement_report(annotations)
    assert report.n_pairs == 1
    assert report.n_measurable_pairs == 1
    assert report.kappa_mean == 1.0
    assert "a|b" in report.pairwise


def test_agreement_report_single_annotator_is_unmeasurable():
    report = agreement_report([Annotation("q1", "a", ("s1",))])
    assert math.isnan(report.kappa_mean)
    assert report.n_pairs == 0
    assert report.is_below(0.6) is False  # can't fail on unmeasurable
