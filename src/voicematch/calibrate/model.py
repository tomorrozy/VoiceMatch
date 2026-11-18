"""The calibration layer: turning a pair of feature vectors into a probability.

The design (spec 5.4) is a Bradley-Terry style model trained on in-library
pairs: the frozen embedding says "who is close", and this layer learns how much
cosine and each acoustic dimension delta should count when a human decides two
voices are similar.  The fitted object is a plain logistic regression, so every
score and every attribution is reproducible from five stored numbers.

Normalisation is a plain z-score (ruling R20)
---------------------------------------------
``fit_calibration`` stores the column mean and the column standard deviation
(with a 0 -> 1 guard for a constant column), and ``score`` is exactly

    ``sigmoid(coef · ((x - mean) / scale) + intercept)``

Nothing else is applied at scoring time.  The scale *is* the normalisation: a
separate median scaling before the z-score would cancel most of itself and
leave a stored value that cannot reproduce the score, so there is only one.

The reported AUC is singer-disjoint (ruling R21)
------------------------------------------------
``auc`` comes from out-of-fold predictions grouped by the caller's fold ids.
``build_pairs`` puts both singers of every pair in the same fold, so a fold
holds complete pairs and never half of one: both singers of every pair live in
the same fold, so the out-of-fold AUC is singer-disjoint.  The scaler is re-fit
inside each fold, so the number measures the whole procedure on unseen singers,
not just the logistic layer fitted on all singers.  A group-balanced AUC of
``nan`` means "not measurable", never "perfect" or "zero" -- with fewer than
two groups, or with a single class, there is no contrast to rank.

Degenerate fits return a model instead of raising
-------------------------------------------------
A library can be too small or too lopsided to calibrate, and that is a normal
state on the way to a full library, not a caller bug:

- **No training rows at all** (a 2-singer library has one singer per fold and
  therefore no pairs): the model carries ``mean = 0``, ``scale = 1``, a unit
  coefficient on the cosine column and intercept 0.  That is the design's own
  baseline -- rank by the frozen embedding -- so downstream ranking still has a
  strict order instead of tying every singer; ``auc`` is ``nan``.
- **One class only**: there is no positive/negative contrast to learn from, so
  the same cosine-only fallback is returned and ``auc`` is ``nan``.  Filling in
  an arbitrary intercept would pretend to a calibrated probability that no data
  supports.
- **Fewer than two groups**: the coefficients are still fitted on all rows, but
  there is no held-out singer to measure, so ``auc`` is ``nan``.
- **A constant column**: its standard deviation is 0, and dividing by it would
  emit ``inf``/``nan`` for every score; the scale is pinned to 1.0 instead, so
  the column contributes ``coef * 0 = 0`` and the rest of the model is usable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from voicematch.calibrate.pairs import FEATURE_NAMES

#: Logistic-regression settings from the task brief.  ``C=1.0`` is the default
#: regularisation and ``max_iter=1000`` is generous enough for the small,
#: z-scored matrices calibration produces.
_LOGISTIC_C: float = 1.0
_LOGISTIC_MAX_ITER: int = 1000

#: Maximum number of out-of-fold groups used for the AUC.  Five is the brief's
#: number; when the library has fewer folds all of them are used.
_MAX_AUC_FOLDS: int = 5

#: Coefficients of the documented no-signal fallback: rank on the frozen
#: embedding's cosine alone (``[cos, pitch, timbre, resonance, style]``).  It is
#: not a learned value -- it is the design's baseline made explicit when there
#: is nothing to learn.
_FALLBACK_COEF: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class CalibrationModel:
    """A fitted pair scorer, plus everything needed to reproduce it.

    ``score`` and ``attribution`` read only the five fields, so a model can be
    saved and reloaded as five numbers/arrays without the training data.
    """

    coef: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    auc: float

    def score(self, feats: np.ndarray) -> np.ndarray | float:
        """Return the calibrated same-singer probability for feature rows.

        ``feats`` is ``(5,)`` (one pair) or ``(n, 5)``.  A single vector comes
        back as a ``float`` so callers can put it straight into a result;
        several rows come back as ``(n,)`` float64.  The computation is exactly
        ``sigmoid(coef · ((x - mean) / scale) + intercept)`` -- nothing is
        cached in a fitted object beyond these arrays (R20).
        """
        values, single = _feature_rows(feats)
        logits = ((values - self.mean) / self.scale) @ self.coef + self.intercept
        probabilities = _sigmoid(logits)
        if single:
            return float(probabilities[0])
        return probabilities

    def attribution(self, feats: np.ndarray) -> dict[str, float]:
        """Return each column's absolute share of one pair's logit.

        The share is ``|coef_i * z_i| / sum_j |coef_j * z_j|`` with the keys
        exactly ``cos/pitch/timbre/resonance/style``; the shares sum to 1.  The
        contribution is absolute because a negative coefficient on a dimension
        delta means "a larger difference lowers similarity", and the report
        shows how much a dimension *moved* the decision, not its sign.

        A feature vector that contributes nothing (a zero model, or a row equal
        to the mean) returns all zeros instead of splitting 1.0 evenly: with no
        evidence there is no basis for claiming every dimension mattered
        equally.
        """
        values = _one_feature_row(feats)
        standardised = (values - self.mean) / self.scale
        contributions = np.abs(self.coef * standardised)
        total = float(contributions.sum())
        if not np.isfinite(total) or total <= 0.0:
            return {name: 0.0 for name in FEATURE_NAMES}
        return {
            name: float(contribution / total)
            for name, contribution in zip(FEATURE_NAMES, contributions)
        }


def fit_calibration(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> CalibrationModel:
    """Fit the calibration model on pair rows, labels and fold ids.

    ``X`` is ``(n_pairs, 5)`` from :func:`~voicematch.calibrate.pairs.build_pairs`
    (or an equivalent matrix), ``y`` holds ``1`` for same-singer pairs and ``0``
    for different-singer pairs, and ``groups`` holds the singer fold id of each
    pair.  The final coefficients are fitted on all rows; ``auc`` is measured on
    out-of-fold predictions grouped by ``groups`` (see the module docstring for
    why that is singer-disjoint).

    Malformed input -- the wrong number of columns, mismatched lengths,
    non-finite values, or labels other than 0/1 -- raises ``ValueError``,
    because each one is a caller bug that a silent result would hide.
    Degenerate but legitimate fits (no rows, one class, one group, a constant
    column) return a model; see the module docstring for each behaviour.
    """
    values = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    fold_ids = np.asarray(groups)

    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"X must have shape (n_pairs, {len(FEATURE_NAMES)}), got "
            f"{values.shape}"
        )
    if (
        labels.ndim != 1
        or fold_ids.ndim != 1
        or len(labels) != len(values)
        or len(fold_ids) != len(values)
    ):
        raise ValueError(
            "X, y and groups must have the same length; got "
            f"{len(values)}, {labels.size}, {fold_ids.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "X must be finite; a nan feature would poison the calibration fit"
        )
    if labels.size and not np.all(np.isin(labels, (0.0, 1.0))):
        raise ValueError("y must contain only 0 or 1 (1 = same singer)")

    if len(values) == 0:
        return CalibrationModel(
            coef=_FALLBACK_COEF.copy(),
            mean=np.zeros(len(FEATURE_NAMES), dtype=np.float64),
            scale=np.ones(len(FEATURE_NAMES), dtype=np.float64),
            intercept=0.0,
            auc=float("nan"),
        )

    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    standardised = (values - mean) / scale

    if np.unique(labels).size < 2:
        coef = _FALLBACK_COEF.copy()
        intercept = 0.0
    else:
        estimator = LogisticRegression(
            C=_LOGISTIC_C, max_iter=_LOGISTIC_MAX_ITER
        ).fit(standardised, labels)
        coef = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
        intercept = float(np.asarray(estimator.intercept_).reshape(-1)[0])

    auc = _grouped_out_of_fold_auc(values, labels, fold_ids)
    return CalibrationModel(
        coef=coef,
        mean=mean,
        scale=scale,
        intercept=intercept,
        auc=auc,
    )


def _grouped_out_of_fold_auc(
    values: np.ndarray, labels: np.ndarray, fold_ids: np.ndarray
) -> float:
    """Out-of-fold AUC with whole folds held out, or ``nan`` if unmeasurable.

    The scaler is re-fit on each training split, so no statistic of the held-out
    singer leaks into the estimate.  Returns ``nan`` when there is no contrast
    to rank (fewer than two folds, a single class overall, or a training split
    with a single class).
    """
    if np.unique(fold_ids).size < 2 or np.unique(labels).size < 2:
        return float("nan")
    n_splits = min(_MAX_AUC_FOLDS, int(np.unique(fold_ids).size))
    predictions = np.full(len(labels), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for train_rows, test_rows in splitter.split(values, labels, fold_ids):
        if np.unique(labels[train_rows]).size < 2:
            return float("nan")
        fold_mean = values[train_rows].mean(axis=0)
        fold_scale = values[train_rows].std(axis=0)
        fold_scale = np.where(fold_scale > 0.0, fold_scale, 1.0)
        estimator = LogisticRegression(
            C=_LOGISTIC_C, max_iter=_LOGISTIC_MAX_ITER
        ).fit(
            (values[train_rows] - fold_mean) / fold_scale,
            labels[train_rows],
        )
        predictions[test_rows] = estimator.predict_proba(
            (values[test_rows] - fold_mean) / fold_scale
        )[:, 1]
    if not np.all(np.isfinite(predictions)):
        return float("nan")
    return float(roc_auc_score(labels, predictions))


def _feature_rows(feats: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``(rows, single)`` for one vector or a matrix of feature rows."""
    values = np.asarray(feats, dtype=np.float64)
    if values.ndim == 1:
        if values.shape[0] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected one feature vector of width {len(FEATURE_NAMES)}, "
                f"got shape {values.shape}"
            )
        return values.reshape(1, -1), True
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"expected feature rows of width {len(FEATURE_NAMES)}, got shape "
            f"{values.shape}"
        )
    return values, False


def _one_feature_row(feats: np.ndarray) -> np.ndarray:
    """Return one feature vector, accepting a 1-row matrix as well."""
    values = np.asarray(feats, dtype=np.float64)
    if values.ndim == 1:
        if values.shape[0] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected one feature vector of width {len(FEATURE_NAMES)}, "
                f"got shape {values.shape}"
            )
        return values
    if values.ndim == 2 and values.shape[0] == 1:
        if values.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected one feature vector of width {len(FEATURE_NAMES)}, "
                f"got shape {values.shape}"
            )
        return values[0]
    raise ValueError(
        "attribution takes one feature vector of width "
        f"{len(FEATURE_NAMES)}, got shape {values.shape}"
    )


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function without overflow warnings."""
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output
