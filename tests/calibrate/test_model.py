"""Tests for the Bradley-Terry style calibration model.

The two tests from the brief are kept verbatim (plus their ``_toy`` fixture);
the rest pin the rulings this task added.

Ruling R20: normalisation is plain z-scoring.  ``CalibrationModel`` stores the
column mean and scale (standard deviation with a 0 -> 1 guard), and ``score`` is
exactly ``sigmoid(coef . ((x - mean) / scale) + intercept)``.  Nothing else is
applied at scoring time, so the model alone reproduces every score.  The test
with a hand-written model and deliberately non-standard ``scale`` values proves
there is no second, hidden scaling: if a median/MAD step existed, those exact
values would not come out.

Ruling R21: the reported ``auc`` comes from out-of-fold predictions grouped by
the caller's fold ids, which ``build_pairs`` makes singer-disjoint.  The
group-disjoint test below builds two groups whose single useful feature differs
between groups: an in-sample fit separates them perfectly, while holding a
whole group out cannot, so an AUC below 1.0 proves the predictions were out of
fold rather than re-scored training rows.

Degenerate fits are pinned as behaviour, not as exceptions: a single class, a
single group, a constant column, and no training rows at all each return a
usable model and an explicit ``nan`` AUC where the number cannot be measured.
"""

import math
import warnings

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from voicematch.calibrate.model import CalibrationModel, fit_calibration

NAN = float("nan")

FEATURE_KEYS = ("cos", "pitch", "timbre", "resonance", "style")


def _toy(n=200):
    rng = np.random.default_rng(8)
    pos = rng.normal(0.9, 0.05, (n, 5))
    pos[:, 1:] = 0.1
    neg = rng.normal(0.2, 0.05, (n, 5))
    neg[:, 1:] = 0.4
    X = np.vstack([pos, neg])
    y = np.r_[np.ones(n), np.zeros(n)]
    g = np.tile(np.arange(n), 2)  # 每折按歌手切分，正负样本同折
    return pos, neg, X, y, g


def _hand_written_model() -> CalibrationModel:
    """A model with arbitrary mean/scale, to test scoring without a fit."""
    return CalibrationModel(
        coef=np.array([0.5, -1.0, 2.0, 0.0, 1.0]),
        mean=np.array([0.1, -0.2, 0.3, 0.0, 0.5]),
        scale=np.array([2.0, 3.0, 0.5, 1.0, 4.0]),
        intercept=-0.3,
        auc=0.8,
    )


def test_calibration_separates_same_from_different_singer_pairs():
    pos, neg, X, y, g = _toy()
    m = fit_calibration(X, y, g)
    assert m.score(pos).mean() > 0.8 and m.score(neg).mean() < 0.2
    assert m.auc > 0.9  # 按歌手切分的折外 AUC，不是训练集 AUC


def test_attribution_sums_to_one():
    rng = np.random.default_rng(9)
    X = rng.normal(0.5, 0.1, (100, 5))
    y = (X[:, 0] > 0.5).astype(float)
    g = np.repeat(np.arange(50), 2)
    m = fit_calibration(X, y, g)
    a = m.attribution(X[0])
    assert abs(sum(abs(v) for v in a.values()) - 1.0) < 1e-6


def test_score_is_exactly_the_stored_affine_logit_through_a_sigmoid():
    model = _hand_written_model()
    feats = np.array([0.7, 0.2, -1.0, 0.0, 3.0])

    z = (feats - model.mean) / model.scale
    expected = 1.0 / (1.0 + np.exp(-(float(z @ model.coef) + model.intercept)))

    assert model.score(feats) == pytest.approx(expected, rel=1e-12)

    batch = np.vstack([feats, feats + 0.1, feats - 0.2])
    z_batch = (batch - model.mean) / model.scale
    expected_batch = 1.0 / (1.0 + np.exp(-(z_batch @ model.coef + model.intercept)))
    np.testing.assert_allclose(model.score(batch), expected_batch, rtol=1e-12)


def test_score_of_one_pair_vector_is_a_float():
    model = _hand_written_model()

    value = model.score(np.zeros(5))

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_attribution_keys_are_the_five_model_columns_in_order():
    model = _hand_written_model()
    feats = np.array([0.7, 0.2, -1.0, 0.0, 3.0])

    attribution = model.attribution(feats)

    assert tuple(attribution) == FEATURE_KEYS
    assert all(value >= 0.0 for value in attribution.values())
    assert sum(attribution.values()) == pytest.approx(1.0)

    z = (feats - model.mean) / model.scale
    contributions = np.abs(model.coef * z)
    expected = contributions / contributions.sum()
    for key, value in zip(FEATURE_KEYS, expected):
        assert attribution[key] == pytest.approx(value, rel=1e-12)


def test_attribution_uses_absolute_contributions_for_negative_coefficients():
    # The dimension-delta coefficients are negative in a real fit (larger delta
    # means less similar).  Attribution reports magnitude, so a negative coef
    # never produces a negative share.
    model = CalibrationModel(
        coef=np.array([1.0, -2.0, 0.0, 0.0, 0.0]),
        mean=np.zeros(5),
        scale=np.ones(5),
        intercept=0.0,
        auc=NAN,
    )

    attribution = model.attribution(np.array([1.0, 1.0, 0.0, 0.0, 0.0]))

    assert attribution["cos"] == pytest.approx(1.0 / 3.0)
    assert attribution["pitch"] == pytest.approx(2.0 / 3.0)
    assert all(value >= 0.0 for value in attribution.values())


def test_attribution_without_any_contribution_returns_all_zeros():
    # Nothing in the model is weighted, so there is no evidence to distribute.
    # Splitting 1.0 evenly would invent equal evidence for every dimension.
    model = CalibrationModel(
        coef=np.zeros(5),
        mean=np.zeros(5),
        scale=np.ones(5),
        intercept=0.0,
        auc=NAN,
    )

    attribution = model.attribution(np.array([0.4, 0.1, 0.2, 0.3, 0.5]))

    assert tuple(attribution) == FEATURE_KEYS
    assert all(value == 0.0 for value in attribution.values())


def test_stored_normalisation_is_the_column_mean_and_std():
    _, _, X, y, g = _toy(n=40)

    model = fit_calibration(X, y, g)

    np.testing.assert_allclose(model.mean, X.mean(axis=0), rtol=1e-12)
    expected_scale = X.std(axis=0)
    expected_scale[expected_scale == 0.0] = 1.0
    np.testing.assert_allclose(model.scale, expected_scale, rtol=1e-12)


def test_auc_is_group_disjoint_and_not_an_in_sample_score():
    n = 40
    group_zero = np.zeros((n, 5))
    group_one = np.zeros((n, 5))
    y_zero = np.r_[np.ones(n // 2), np.zeros(n // 2)]
    y_one = np.r_[np.ones(n // 2), np.zeros(n // 2)]
    # Group zero is separable on column 0 only; group one on column 1 only.
    # An in-sample linear model sees both columns and separates everything; a
    # model that holds a whole group out can learn only that group's column.
    group_zero[: n // 2, 0] = 1.0
    group_zero[n // 2 :, 0] = -1.0
    group_one[: n // 2, 1] = 1.0
    group_one[n // 2 :, 1] = -1.0
    X = np.vstack([group_zero, group_one])
    y = np.r_[y_zero, y_one]
    groups = np.r_[np.zeros(n), np.ones(n)]

    model = fit_calibration(X, y, groups)

    in_sample_auc = roc_auc_score(y, model.score(X))
    assert in_sample_auc > 0.99
    assert model.auc < 0.75


def test_single_class_fit_returns_a_cosine_fallback_model_with_nan_auc():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20, 5))
    y = np.ones(20)
    groups = np.repeat(np.arange(10), 2)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = fit_calibration(X, y, groups)

    assert math.isnan(model.auc)
    # With no positive/negative contrast there is no perceptual mapping to
    # learn; the documented fallback keeps the frozen embedding's own order.
    np.testing.assert_array_equal(model.coef, np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert model.intercept == 0.0
    assert np.all(np.isfinite(model.score(X)))


def test_fewer_than_two_groups_reports_nan_auc_but_still_fits():
    _, _, X, y, _g = _toy(n=40)
    groups = np.zeros(len(y), dtype=int)

    model = fit_calibration(X, y, groups)

    assert math.isnan(model.auc)
    assert np.all(np.isfinite(model.coef))
    # The full-data fit still separates the two classes.
    assert model.score(X[y == 1]).mean() > model.score(X[y == 0]).mean()


def test_constant_column_is_normalised_with_scale_one():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 5))
    X[:, 2] = 7.0
    y = (X[:, 0] > 0.0).astype(float)
    groups = np.repeat(np.arange(20), 2)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = fit_calibration(X, y, groups)

    assert model.scale[2] == 1.0
    assert np.all(model.scale > 0.0)
    assert np.all(np.isfinite(model.score(X)))


def test_no_training_rows_returns_a_model_that_still_ranks_by_cosine():
    X = np.empty((0, 5))
    y = np.empty(0)
    groups = np.empty(0, dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = fit_calibration(X, y, groups)

    assert math.isnan(model.auc)
    np.testing.assert_array_equal(model.coef, np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(model.mean, np.zeros(5))
    np.testing.assert_array_equal(model.scale, np.ones(5))
    assert model.intercept == 0.0
    assert np.all(np.isfinite(model.score(np.array([0.5, 0.1, 0.2, 0.3, 0.4]))))
    # Monotone in the cosine column, so downstream ranking still has an order.
    assert model.score(np.array([0.9, 0.0, 0.0, 0.0, 0.0])) > model.score(
        np.array([0.1, 0.0, 0.0, 0.0, 0.0])
    )


def test_fit_rejects_malformed_inputs():
    with pytest.raises(ValueError, match="5"):
        fit_calibration(np.zeros((4, 4)), np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError, match="same length"):
        fit_calibration(np.zeros((4, 5)), np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        fit_calibration(
            np.full((4, 5), np.nan), np.array([0.0, 0.0, 1.0, 1.0]), np.zeros(4)
        )
    with pytest.raises(ValueError, match="0 or 1"):
        fit_calibration(np.zeros((4, 5)), np.array([0.0, 1.0, 2.0, 0.0]), np.zeros(4))


def test_score_rejects_a_feature_vector_of_the_wrong_width():
    model = _hand_written_model()

    with pytest.raises(ValueError, match="5"):
        model.score(np.zeros(4))
    with pytest.raises(ValueError, match="5"):
        model.score(np.zeros((2, 4)))
