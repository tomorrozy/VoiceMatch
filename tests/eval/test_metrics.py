"""Tests for the evaluation metrics: the numbers that decide whether we are 准.

The three headline numbers (Top-K 命中率, 受理率, Spearman) are only comparable
if "在数据不足时无法测量" never looks like "测过了，全错".  A ``0.0`` says a
measurement was made and everything was wrong, so every metric here returns
``nan`` when it has nothing to measure.  These tests pin that distinction, the
Top-K denominator rule (a query annotated "都不像" has no answer to be right
about, so it is not in the denominator -- ruling R9), and the degenerate inputs
the pipeline can actually produce: a constant ranking, and a quality report
whose ``snr_db`` is ``-inf`` because the recording was digitally silent.
"""

import math
import warnings

import pytest

from voicematch.audio.quality import QualityReport
from voicematch.eval.metrics import (
    acceptance_rate,
    spearman,
    top3_overlap,
    top_k_hit,
)

NAN = float("nan")


def _report(*, rejected: bool, snr_db: float = 20.0) -> QualityReport:
    """A quality report with only the fields a metric may read held meaningful."""
    return QualityReport(
        speech_sec=9.0,
        snr_db=snr_db,
        clipped_ratio=0.0,
        voiced_ratio=0.9,
        rejected=rejected,
        reasons=("low_snr",) if rejected else (),
    )


def _silence_report() -> QualityReport:
    """The report a digital-silence recording really produces.

    ``snr_db`` is ``-inf`` there (``QualityReport``'s docstring: digital silence
    has no defined ratio), and ``-inf`` is not serialisable as strict JSON.  The
    metrics never look at it, so measuring an acceptance rate over such a report
    must neither raise nor need to sanitise it.
    """
    return QualityReport(
        speech_sec=0.0,
        snr_db=float("-inf"),
        clipped_ratio=0.0,
        voiced_ratio=0.0,
        rejected=True,
        reasons=("too_short_speech", "low_snr"),
    )


# --- the brief's three tests, verbatim --------------------------------------


def test_top3_hit_counts_query_when_any_label_in_top3():
    assert top_k_hit([["a", "b", "c"], ["d", "e", "f"]], [{"c"}, {"z"}], k=3) == 0.5


def test_empty_input_is_nan_not_zero():
    assert top_k_hit([], [], k=3) != top_k_hit([], [], k=3)  # NaN


def test_top3_overlap_is_intersection_over_three():
    assert top3_overlap(["a", "b", "c"], ["c", "b", "x"]) == pytest.approx(2 / 3)


# --- top_k_hit: the denominator is what makes the headline number honest ----


def test_top_k_hit_counts_a_hit_when_any_acceptable_label_is_in_the_top_k():
    # The annotation allows 1-2 most-similar artists (design doc 6.2), so a
    # ranking that surfaced either one is a hit.
    assert top_k_hit([["a", "b", "c"]], [{"x", "b"}], k=3) == 1.0
    assert top_k_hit([["a", "b", "c"]], [{"x", "y"}], k=3) == 0.0


def test_top_k_hit_uses_exactly_the_first_k_positions():
    # "c" sits at position 3: in the top-3, not in the top-2.
    assert top_k_hit([["a", "b", "c"]], [{"c"}], k=3) == 1.0
    assert top_k_hit([["a", "b", "c"]], [{"c"}], k=2) == 0.0


def test_top_k_hit_counts_distinct_artists_not_duplicate_positions():
    # A degenerate ranking that repeats an artist must not spend two slots on
    # it: the same artist occupying positions 1 and 2 is one hit, and "b" only
    # enters the top-3 when k reaches it.
    assert top_k_hit([["a", "a", "b"]], [{"b"}], k=2) == 0.0
    assert top_k_hit([["a", "a", "b"]], [{"b"}], k=3) == 1.0


def test_an_empty_ranking_for_an_answerable_query_is_a_miss():
    # Nothing was surfaced, so nothing matched: a measured miss, not "no data".
    # (Unlike top3_overlap, where a missing ranking makes the agreement question
    # unaskable.)
    assert top_k_hit([[]], [{"a"}], k=1) == 0.0


def test_top_k_hit_tolerates_a_ranking_shorter_than_k():
    # A gender pool smaller than k is ordinary (a two-singer pool with k=3); the
    # ranking is simply exhaustive and the hit is still measured.
    assert top_k_hit([["a"]], [{"a"}], k=3) == 1.0


def test_none_like_queries_are_not_in_the_top_k_denominator():
    # R9: a query whose annotation is "都不像" is a library-coverage signal, not
    # a ranking miss.  Counting it as wrong would make the Top-K number depend on
    # how many people said "none of these", which is a different question.
    # Only the answerable query is measured, and it hit -> 1.0, not 0.5.
    assert top_k_hit([["a"], ["b"]], [set(), {"b"}], k=1) == 1.0


def test_top_k_hit_is_nan_when_no_query_has_an_acceptable_artist():
    # There is nothing to be right about, so nothing was measured.
    assert math.isnan(top_k_hit([["a"], ["b"]], [set(), set()], k=1))


def test_top_k_hit_rejects_misaligned_predictions_and_labels():
    # Predicting for two queries while annotating one is a caller bug: pairing
    # by position would silently score the wrong query.
    with pytest.raises(ValueError):
        top_k_hit([["a", "b"], ["c"]], [{"a"}], k=3)


def test_top_k_hit_rejects_k_below_one():
    # k=0 has no top to hit ("the empty top-0 is never a hit") and k<0 slices
    # from the end, which any reader would misread.  A caller bug, not data.
    with pytest.raises(ValueError):
        top_k_hit([["a"]], [{"a"}], k=0)


# --- acceptance_rate: un-rejected over total, immune to non-finite SNR ------


def test_acceptance_rate_is_unrejected_over_total():
    reports = [_report(rejected=False), _report(rejected=False), _report(rejected=True)]
    assert acceptance_rate(reports) == 2 / 3


def test_acceptance_rate_is_one_when_everything_is_accepted():
    assert acceptance_rate([_report(rejected=False)] * 3) == 1.0


def test_acceptance_rate_is_zero_only_when_everything_was_actually_rejected():
    assert acceptance_rate([_report(rejected=True)] * 3) == 0.0


def test_acceptance_rate_is_nan_without_reports():
    assert math.isnan(acceptance_rate([]))


def test_acceptance_rate_does_not_choke_on_a_minus_inf_snr():
    # A silent recording is rejected with snr_db == -inf (QualityReport is
    # frozen; the value is legitimate).  Only ``rejected`` may be read.
    reports = [_silence_report(), _report(rejected=False)]
    assert acceptance_rate(reports) == 0.5


def test_acceptance_rate_does_not_choke_on_a_nan_snr():
    # The quality layer can also report nan snr for non-finite input (a
    # deferred minor of Task 3); neither non-finite value changes the count.
    reports = [_report(rejected=True, snr_db=NAN), _report(rejected=False)]
    assert acceptance_rate(reports) == 0.5


# --- spearman: rank correlation, nan when a ranking is flat -----------------


def test_spearman_is_one_for_a_monotone_relationship():
    # Ranks, not values: a strictly increasing but non-linear mapping is still
    # a perfect agreement.
    assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0]) == 1.0


def test_spearman_is_minus_one_for_a_reversed_relationship():
    assert spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == -1.0


def test_spearman_is_nan_for_a_constant_prediction():
    # Zero variance: scipy's rank correlation is undefined (0/0), not perfect.
    # Returning 1.0 here would claim the score agreed with a ranking it cannot
    # distinguish.  Guarded so it also does not emit ConstantInputWarning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(spearman([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]))


def test_spearman_is_nan_for_constant_annotations():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(spearman([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]))


def test_spearman_is_nan_without_two_points():
    # The design's Spearman is over queries with >= 2 distinct first choices
    # (6.3); with fewer points the correlation is not measurable.
    assert math.isnan(spearman([], []))
    assert math.isnan(spearman([1.5], [2.5]))


def test_spearman_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        spearman([1.0, 2.0], [1.0])


def test_spearman_rejects_non_finite_scores():
    # A nan/inf score is a broken measurement chain, not thin data: letting it
    # yield nan would make it indistinguishable from "not enough annotations".
    with pytest.raises(ValueError):
        spearman([1.0, NAN, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        spearman([1.0, float("inf")], [1.0, 2.0])


# --- top3_overlap: cross-device stability, intersection over three ---------


def test_top3_overlap_is_one_for_the_same_top_three_in_another_order():
    # Two devices that surface the same three artists agree completely.
    assert top3_overlap(["a", "b", "c"], ["c", "a", "b"]) == 1.0


def test_top3_overlap_is_zero_for_disjoint_rankings():
    assert top3_overlap(["a", "b", "c"], ["d", "e", "f"]) == 0.0


def test_top3_overlap_counts_distinct_artists():
    assert top3_overlap(["a", "a", "b"], ["a", "b", "c"]) == pytest.approx(2 / 3)


def test_top3_overlap_compares_only_the_top_three():
    # A caller may hand over a longer ranking (the whole scored pool); the
    # stability criterion is about the top-3, and 4/3 would be an impossible
    # agreement score.
    assert top3_overlap(["a", "b", "c", "d"], ["a", "b", "c", "d"]) == 1.0


def test_top3_overlap_is_nan_when_one_ranking_is_missing():
    # "No ranking" is not "a ranking that disagreed": 0.0 would report measured
    # instability where nothing was measured.
    assert math.isnan(top3_overlap([], []))
    assert math.isnan(top3_overlap(["a", "b", "c"], []))


# --- the eval report is JSON'd by Task 14: no numpy scalars may leak --------


def test_every_metric_returns_a_plain_float():
    # run_eval's dict is handed to json.dumps by the CLI; a numpy scalar or a
    # numpy bool would be a serialisation surprise at the boundary.
    values = {
        "top_k_hit": top_k_hit([["a"]], [{"a"}], k=1),
        "acceptance_rate": acceptance_rate([_report(rejected=False)]),
        "spearman": spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        "top3_overlap": top3_overlap(["a"], ["a"]),
    }
    for name, value in values.items():
        assert type(value) is float, f"{name} returned {type(value).__name__}"
