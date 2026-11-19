"""Tests for calibration pair construction.

The brief lists this file but supplied no test code (ruling R15), so these
tests pin the interface Tasks 12 and 14 consume.

Ruling R2 is pinned by construction rather than by comment: ``pair_features``
has no prototype arguments at all, and the four dimension deltas are aggregated
through ``DIMENSION_OF``.  The last test proves that aggregation is data-driven:
a descriptor registered in ``DIMENSION_OF`` after import enters the delta
without any edit here.

Ruling R21 is pinned in two places.  ``build_pairs`` may only form negatives
between two singers that share a fold, and every pair's two sides must share
that fold, so ``GroupKFold`` on ``groups`` is genuinely singer-disjoint.  The
accepted cost is pinned too: a library whose folds each hold a single singer
(two singers at all, for instance) yields no pairs, because there is no
same-fold partner to compare against.
"""

import numpy as np
import pytest

import voicematch.calibrate.pairs as pairs_module
from voicematch.calibrate import (
    CalibrationModel,
    build_pairs,
    fit_calibration,
    pair_features,
)
from voicematch.config import EMBED_DIM
from voicematch.features.extract import DIMENSION_OF, FeatureVector
from voicematch.library.schema import LibraryIndex, Segment, Singer

NAN = float("nan")


def _fv(*, pitch=None, timbre=None, resonance=None, style=None) -> FeatureVector:
    """Build a feature vector from explicit groups; omitted groups are empty."""
    return FeatureVector(
        pitch=pitch or {},
        timbre=timbre or {},
        resonance=resonance or {},
        style=style or {},
    )


def _random_fv(rng) -> FeatureVector:
    """One full, finite feature vector keyed exactly like ``DIMENSION_OF``."""
    groups: dict[str, dict[str, float]] = {
        "pitch": {},
        "timbre": {},
        "resonance": {},
        "style": {},
    }
    for name, dimension in DIMENSION_OF.items():
        groups[dimension][name] = float(rng.normal())
    return FeatureVector(**groups)


def _make_index(segment_counts: dict[str, int]) -> LibraryIndex:
    """A synthetic library with the requested number of segments per singer."""
    rng = np.random.default_rng(11)
    singer_ids = sorted(segment_counts)
    singers = {
        singer_id: Singer(singer_id, f"Singer {singer_id}", "female", "pop")
        for singer_id in singer_ids
    }
    segments = []
    features = {}
    for singer_id in singer_ids:
        for number in range(segment_counts[singer_id]):
            segment = Segment(
                f"{singer_id}-{number:02d}",
                singer_id,
                float(number),
                float(number + 1),
            )
            segments.append(segment)
            features[segment.segment_id] = _random_fv(rng)
    embeddings = rng.normal(size=(len(segments), EMBED_DIM)).astype(np.float32)
    prototypes = {
        singer_id: (rng.normal(size=EMBED_DIM) / np.sqrt(EMBED_DIM)).astype(np.float32)
        for singer_id in singer_ids
    }
    return LibraryIndex(
        version="test-pairs",
        model_id="test-pairs-encoder",
        singers=singers,
        segments=tuple(segments),
        prototypes=prototypes,
        features=features,
        embeddings=embeddings,
    )


def _expected_folds(index: LibraryIndex) -> tuple[dict[str, int], int]:
    """The R21 fold map, restated independently of the implementation."""
    singer_ids = sorted(index.singers)
    n_folds = min(5, max(2, len(singer_ids) // 2))
    return {
        singer_id: position % n_folds
        for position, singer_id in enumerate(singer_ids)
    }, n_folds


def _spy_on_pair_features(monkeypatch, index: LibraryIndex):
    """Record which segments each emitted pair compared, in row order.

    ``build_pairs`` returns only numbers, so the segment identity of a row is
    recovered by intercepting ``pair_features``.  The spy delegates to the real
    function, so it does not change the data under test.
    """
    calls: list[tuple[FeatureVector, FeatureVector, float]] = []
    fv_to_segment = {
        id(index.features[segment.segment_id]): segment for segment in index.segments
    }
    real_pair_features = pairs_module.pair_features

    def spy(cos: float, fv_a: FeatureVector, fv_b: FeatureVector) -> np.ndarray:
        calls.append((fv_a, fv_b, cos))
        return real_pair_features(cos, fv_a, fv_b)

    monkeypatch.setattr(pairs_module, "pair_features", spy)
    return calls, fv_to_segment


# ---------------------------------------------------------------------------
# pair_features
# ---------------------------------------------------------------------------


def test_pair_features_is_cosine_then_four_dimension_deltas():
    fv_a = _fv(
        pitch={"median_f0_hz": 200.0, "range_semitones": NAN, "f0_jitter": 0.10},
        timbre={"mfcc1_mean": 1.0, "mfcc2_mean": NAN},
        resonance={"f1_hz": NAN, "f2_hz": 500.0},
        style={"vibrato_rate_hz": 5.0, "slur_ratio": 0.20},
    )
    fv_b = _fv(
        pitch={"median_f0_hz": 210.0, "range_semitones": NAN, "f0_jitter": 0.30},
        timbre={"mfcc1_mean": 3.0, "mfcc2_mean": NAN},
        resonance={"f1_hz": NAN, "f2_hz": 520.0},
        style={"vibrato_rate_hz": 5.5, "slur_ratio": NAN},
    )

    out = pair_features(0.7, fv_a, fv_b)

    assert out.shape == (5,)
    assert np.issubdtype(out.dtype, np.floating)
    # pitch: mean(|200-210|, |0.1-0.3|) = 5.1; range_semitones is nan in both
    # and contributes nothing.  timbre: |1-3| = 2.  resonance: |500-520| = 20.
    # style: |5.0-5.5| = 0.5; slur_ratio is nan in b, so it is excluded too.
    np.testing.assert_allclose(out, [0.7, 5.1, 2.0, 20.0, 0.5], rtol=0, atol=1e-12)


def test_a_dimension_without_comparable_features_reports_zero_not_nan():
    # A vector that measured nothing in resonance or style is not evidence of a
    # difference there; the delta is 0.0 so one missing descriptor cannot make
    # the whole pair unusable.
    fv_a = _fv(
        pitch={"median_f0_hz": 200.0},
        resonance={"f1_hz": NAN},
        style={"vibrato_rate_hz": NAN},
    )
    fv_b = _fv(pitch={"median_f0_hz": 205.0})

    out = pair_features(0.5, fv_a, fv_b)

    assert np.all(np.isfinite(out))
    assert out[1] == pytest.approx(5.0)
    assert out[2] == 0.0  # no timbre descriptor in either vector
    assert out[3] == 0.0  # resonance descriptors are unavailable
    assert out[4] == 0.0  # style descriptors are unavailable


def test_pair_features_is_symmetric():
    fv_a = _fv(pitch={"median_f0_hz": 100.0}, timbre={"mfcc1_mean": 1.0})
    fv_b = _fv(pitch={"median_f0_hz": 130.0}, timbre={"mfcc1_mean": 4.0})

    forward = pair_features(0.25, fv_a, fv_b)
    backward = pair_features(0.25, fv_b, fv_a)

    np.testing.assert_array_equal(forward, backward)


def test_aggregation_follows_dimension_of_instead_of_a_hard_coded_list(monkeypatch):
    # Registering a new descriptor must add it to the delta with no change in
    # this module: that is the reason the aggregation reads DIMENSION_OF.
    monkeypatch.setitem(DIMENSION_OF, "extra_descriptor", "pitch")
    fv_a = _fv(pitch={"extra_descriptor": 1.0})
    fv_b = _fv(pitch={"extra_descriptor": 3.0})

    out = pair_features(0.0, fv_a, fv_b)

    assert out[1] == pytest.approx(2.0)


def test_all_nan_vectors_still_produce_finite_deltas():
    fv_a = _fv(
        pitch={"median_f0_hz": NAN},
        timbre={"mfcc1_mean": NAN},
        resonance={"f1_hz": NAN},
        style={"vibrato_rate_hz": NAN},
    )
    fv_b = _fv(
        pitch={"median_f0_hz": NAN},
        timbre={"mfcc1_mean": NAN},
        resonance={"f1_hz": NAN},
        style={"vibrato_rate_hz": NAN},
    )

    out = pair_features(0.0, fv_a, fv_b)

    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out, np.zeros(5))


# ---------------------------------------------------------------------------
# build_pairs: labels, folds and row alignment
# ---------------------------------------------------------------------------


def test_labels_and_fold_groups_are_aligned_with_every_pair(monkeypatch):
    index = _make_index({"s0": 2, "s1": 2, "s2": 2, "s3": 2})
    fold_of, n_folds = _expected_folds(index)
    assert n_folds == 2
    calls, fv_to_segment = _spy_on_pair_features(monkeypatch, index)

    X, y, groups = build_pairs(index, index.embeddings)

    assert X.shape == (len(y), 5)
    assert len(y) == len(groups) == len(calls)
    assert set(np.unique(y)) == {0, 1}
    for row, (fv_a, fv_b, cos) in enumerate(calls):
        segment_a = fv_to_segment[id(fv_a)]
        segment_b = fv_to_segment[id(fv_b)]
        same_singer = segment_a.singer_id == segment_b.singer_id
        assert bool(y[row]) is same_singer
        # R21: both singers of a pair live in the same fold ...
        assert fold_of[segment_a.singer_id] == fold_of[segment_b.singer_id]
        # ... and the pair is grouped by that fold.
        assert groups[row] == fold_of[segment_a.singer_id]
        assert X[row, 0] == pytest.approx(cos, rel=0, abs=1e-12)
    assert (y == 1).any() and (y == 0).any()


def test_negatives_only_join_singers_from_the_same_fold(monkeypatch):
    index = _make_index({f"s{number}": 2 for number in range(6)})
    fold_of, n_folds = _expected_folds(index)
    assert n_folds == 3
    calls, fv_to_segment = _spy_on_pair_features(monkeypatch, index)

    X, y, groups = build_pairs(index, index.embeddings)

    negatives = [row for row in range(len(y)) if y[row] == 0]
    assert negatives  # the fixture must actually exercise negative pairs
    for row in negatives:
        segment_a = fv_to_segment[id(calls[row][0])]
        segment_b = fv_to_segment[id(calls[row][1])]
        assert segment_a.singer_id != segment_b.singer_id
        assert fold_of[segment_a.singer_id] == fold_of[segment_b.singer_id]
        assert groups[row] == fold_of[segment_a.singer_id]


def test_positives_are_two_different_segments_of_one_singer(monkeypatch):
    index = _make_index({"s0": 3, "s1": 3, "s2": 3, "s3": 3})
    calls, fv_to_segment = _spy_on_pair_features(monkeypatch, index)

    X, y, groups = build_pairs(index, index.embeddings)

    positives = [row for row in range(len(y)) if y[row] == 1]
    assert positives
    for row in positives:
        segment_a = fv_to_segment[id(calls[row][0])]
        segment_b = fv_to_segment[id(calls[row][1])]
        assert segment_a.singer_id == segment_b.singer_id
        assert segment_a.segment_id != segment_b.segment_id


def test_pairs_are_sampled_once_even_when_both_singers_generate_them(monkeypatch):
    # A negative pair is a property of two segments, not of an anchor singer.
    # Both singers' candidate lists contain it; the emitted data must contain
    # it exactly once, otherwise the fit counts an identical row twice.
    index = _make_index({"s0": 4, "s1": 4, "s2": 4, "s3": 4})
    calls, fv_to_segment = _spy_on_pair_features(monkeypatch, index)

    build_pairs(index, index.embeddings)

    seen: set[tuple[str, str]] = set()
    for fv_a, fv_b, _cos in calls:
        a = fv_to_segment[id(fv_a)].segment_id
        b = fv_to_segment[id(fv_b)].segment_id
        key = tuple(sorted((a, b)))
        assert key not in seen
        seen.add(key)


def test_positives_and_negatives_are_balanced_per_singer(monkeypatch):
    # One singer has eight segments (28 positive pairs), its only same-fold
    # partner has one segment (8 possible negatives).  Without balancing the
    # fit would see 28 positives against 8 negatives; the per-singer sample
    # caps both classes at the smaller of the two lists.
    index = _make_index({"s0": 8, "s1": 1, "s2": 1, "s3": 1})
    fold_of, n_folds = _expected_folds(index)
    assert n_folds == 2
    assert fold_of["s0"] == fold_of["s2"] == 0

    X, y, groups = build_pairs(index, index.embeddings)

    assert int((y == 1).sum()) == 8
    assert int((y == 0).sum()) == 8
    assert len(groups) == len(y)


def test_a_fold_with_one_singer_contributes_no_pairs(monkeypatch):
    index = _make_index({"s0": 2, "s1": 2, "s2": 2})
    fold_of, n_folds = _expected_folds(index)
    assert n_folds == 2
    assert fold_of == {"s0": 0, "s1": 1, "s2": 0}
    calls, fv_to_segment = _spy_on_pair_features(monkeypatch, index)

    X, y, groups = build_pairs(index, index.embeddings)

    assert len(y) > 0
    assert set(np.unique(groups)) == {0}
    singers_seen = {
        fv_to_segment[id(fv)].singer_id
        for fv_a, fv_b, _cos in calls
        for fv in (fv_a, fv_b)
    }
    assert singers_seen == {"s0", "s2"}


@pytest.mark.parametrize(
    "segment_counts",
    [
        {"solo": 3},
        {"s0": 2, "s1": 2},
    ],
    ids=["one-singer", "two-singers"],
)
def test_fewer_than_two_singers_per_fold_yields_no_pairs(segment_counts):
    # Accepted cost of R21: two singers land in different folds, so there is no
    # same-fold partner and no negative pair to learn from.  The empty result
    # is explicit (correct shapes) rather than an exception, because a small
    # library is a legitimate, if uncalibratable, state.
    index = _make_index(segment_counts)

    X, y, groups = build_pairs(index, index.embeddings)

    assert X.shape == (0, 5)
    assert y.size == 0
    assert groups.size == 0

    model = fit_calibration(X, y, groups)
    assert isinstance(model, CalibrationModel)


# ---------------------------------------------------------------------------
# build_pairs: cosine computation and input validation
# ---------------------------------------------------------------------------


def test_cosine_is_normalised_not_a_raw_dot_product(monkeypatch):
    index = _make_index({"s0": 2, "s1": 2, "s2": 2, "s3": 2})
    scales = np.arange(1, len(index.segments) + 1, dtype=np.float64)
    embeddings = (
        index.embeddings.astype(np.float64) * scales[:, None]
    ).astype(np.float32)
    row_of = {segment.segment_id: row for row, segment in enumerate(index.segments)}
    calls, fv_to_segment = _spy_on_pair_features(monkeypatch, index)

    X, _y, _groups = build_pairs(index, embeddings)

    values = embeddings.astype(np.float64)
    norms = np.linalg.norm(values, axis=1)
    for row, (fv_a, fv_b, _cos) in enumerate(calls):
        i = row_of[fv_to_segment[id(fv_a)].segment_id]
        j = row_of[fv_to_segment[id(fv_b)].segment_id]
        expected = float(np.dot(values[i], values[j]) / (norms[i] * norms[j]))
        assert X[row, 0] == pytest.approx(expected, rel=1e-6, abs=1e-9)
        raw_dot = float(np.dot(values[i], values[j]))
        if raw_dot != pytest.approx(expected, rel=1e-6):
            # The fixture must distinguish cosine from a raw dot product.
            assert X[row, 0] != pytest.approx(raw_dot, rel=1e-6)


def test_zero_norm_embedding_yields_zero_cosine_not_nan(monkeypatch):
    index = _make_index({"s0": 2, "s1": 2, "s2": 2, "s3": 2})
    embeddings = index.embeddings.copy()
    embeddings[0] = 0.0
    calls, fv_to_segment = _spy_on_pair_features(monkeypatch, index)

    X, _y, _groups = build_pairs(index, embeddings)

    assert np.all(np.isfinite(X))
    touched = [
        row
        for row, (fv_a, fv_b, _cos) in enumerate(calls)
        if fv_to_segment[id(fv_a)].segment_id == index.segments[0].segment_id
        or fv_to_segment[id(fv_b)].segment_id == index.segments[0].segment_id
    ]
    assert touched
    for row in touched:
        assert X[row, 0] == 0.0


def test_build_pairs_is_deterministic():
    index = _make_index({f"s{number}": 3 for number in range(10)})

    first = build_pairs(index, index.embeddings)
    second = build_pairs(index, index.embeddings)
    third = build_pairs(index, index.embeddings.copy())

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(first, third):
        np.testing.assert_array_equal(left, right)


def test_build_pairs_rejects_misaligned_or_non_finite_embeddings():
    index = _make_index({"s0": 2, "s1": 2})

    with pytest.raises(ValueError, match="row"):
        build_pairs(index, index.embeddings[:-1])
    with pytest.raises(ValueError, match="2-D"):
        build_pairs(index, index.embeddings[:, 0])
    with pytest.raises(ValueError, match="finite"):
        build_pairs(index, np.full_like(index.embeddings, np.nan))


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


def test_package_reexports_the_public_names():
    import voicematch.calibrate as package
    from voicematch.calibrate.model import CalibrationModel as ModelType
    from voicematch.calibrate.model import fit_calibration as fit
    from voicematch.calibrate.pairs import build_pairs as build
    from voicematch.calibrate.pairs import pair_features as features

    assert package.CalibrationModel is ModelType
    assert package.fit_calibration is fit
    assert package.build_pairs is build
    assert package.pair_features is features
