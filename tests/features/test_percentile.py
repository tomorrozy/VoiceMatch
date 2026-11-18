"""Tests for the same-gender percentile mapping.

The brief supplied the first two tests; the rest pin the contract its prose
requires but its tests do not check: the pool median maps to 50, a descriptor
that could not be measured stays missing instead of becoming an extreme, pool
members with no finite value do not enter the reference distribution, and a
query outside the pool clamps to the ends.

Ruling R4 (overrides the plan text): the plan's
``searchsorted(..., side="right") / n * 100`` maps the pool minimum to
``100/n``, not ``0``, and so contradicts the plan's own expected values.  The
implementation uses ``side="left"`` with ``n - 1`` in the denominator: the
minimum ranks 0, the maximum ranks 100, and the median of three ranks 50.

Ruling R10: ``fit_percentiles`` deliberately has no ``gender`` parameter.  The
same-gender pool is chosen by the caller; one model is fitted per gender.  The
gender-pool test below exercises that calling convention.
"""

import math

from voicematch.features.extract import FeatureVector
from voicematch.features.percentile import fit_percentiles, to_percentiles


def _fv(centroid):
    return FeatureVector(
        pitch={"median_f0_hz": 220.0},
        timbre={"spectral_centroid_hz": centroid},
        resonance={"hnr_db": 20.0},
        style={"vibrato_rate_hz": 5.0},
    )


def test_percentile_is_relative_to_reference_pool():
    pool = [_fv(c) for c in (1000, 2000, 3000)]
    model = fit_percentiles(pool)
    assert to_percentiles(_fv(1000), model)["spectral_centroid_hz"] == 0.0
    assert to_percentiles(_fv(3000), model)["spectral_centroid_hz"] == 100.0
    assert 0 < to_percentiles(_fv(2000), model)["spectral_centroid_hz"] < 100


def test_single_sample_pool_does_not_divide_by_zero():
    model = fit_percentiles([_fv(1000)])
    assert to_percentiles(_fv(1000), model)["spectral_centroid_hz"] == 50.0


# --- contract details the brief leaves to its prose --------------------------


def test_median_of_three_maps_to_fifty():
    model = fit_percentiles([_fv(c) for c in (1000, 2000, 3000)])
    assert to_percentiles(_fv(2000), model)["spectral_centroid_hz"] == 50.0


def test_missing_value_stays_missing():
    # ``nan`` means "not measurable", not "bottom of the pool": coercing it to
    # 0 or 100 would invent a similarity result out of nothing.
    model = fit_percentiles([_fv(c) for c in (1000, 2000, 3000)])
    assert math.isnan(to_percentiles(_fv(float("nan")), model)["spectral_centroid_hz"])


def test_missing_pool_values_do_not_become_reference_points():
    # A pool member whose descriptor was unmeasurable has no rank; the
    # remaining finite members define the reference distribution.
    model = fit_percentiles([_fv(1000), _fv(float("nan")), _fv(2000), _fv(3000)])

    assert len(model.reference["spectral_centroid_hz"]) == 3
    assert to_percentiles(_fv(1000), model)["spectral_centroid_hz"] == 0.0
    assert to_percentiles(_fv(2000), model)["spectral_centroid_hz"] == 50.0
    assert to_percentiles(_fv(3000), model)["spectral_centroid_hz"] == 100.0


def test_queries_outside_the_pool_clamp_to_the_ends():
    # The report promises a 0-100 dimension score; a query below the pool floor
    # is 0, a query above its ceiling is 100.
    model = fit_percentiles([_fv(c) for c in (1000, 2000, 3000)])
    assert to_percentiles(_fv(500), model)["spectral_centroid_hz"] == 0.0
    assert to_percentiles(_fv(5000), model)["spectral_centroid_hz"] == 100.0


def test_each_gender_pool_has_its_own_reference():
    # R10: the caller fits one model per gender; the same 200 Hz centroid is
    # the middle of a low-voiced pool but the bottom of a high-voiced one.
    male_pool = fit_percentiles([_fv(c) for c in (100.0, 200.0, 300.0)])
    female_pool = fit_percentiles([_fv(c) for c in (1000.0, 2000.0, 3000.0)])

    assert to_percentiles(_fv(200.0), male_pool)["spectral_centroid_hz"] == 50.0
    assert to_percentiles(_fv(200.0), female_pool)["spectral_centroid_hz"] == 0.0


def test_percentiles_cover_every_feature_in_the_vector():
    model = fit_percentiles([_fv(c) for c in (1000, 2000, 3000)])
    percentiles = to_percentiles(_fv(2000), model)

    assert set(percentiles) == {
        "median_f0_hz",
        "spectral_centroid_hz",
        "hnr_db",
        "vibrato_rate_hz",
    }
    assert all(0.0 <= value <= 100.0 for value in percentiles.values())


def test_feature_never_seen_in_the_pool_has_no_percentile():
    # Pool members only carry a centroid: the query's other descriptors have
    # no reference distribution, so their rank is unknown (``nan``), not 0.
    pool = [
        FeatureVector(pitch={}, timbre={"spectral_centroid_hz": c}, resonance={}, style={})
        for c in (1000, 2000, 3000)
    ]
    model = fit_percentiles(pool)
    percentiles = to_percentiles(_fv(2000), model)

    assert percentiles["spectral_centroid_hz"] == 50.0
    assert math.isnan(percentiles["median_f0_hz"])
    assert math.isnan(percentiles["hnr_db"])
    assert math.isnan(percentiles["vibrato_rate_hz"])
