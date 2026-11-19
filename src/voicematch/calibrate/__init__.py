"""The human-calibrated similarity layer over the frozen embedding.

The frozen encoder supplies similarity; this subpackage learns how much that
similarity -- and each of the four acoustic dimension deltas -- should count
when the question is "would a human call these two voices similar".

``build_pairs`` turns a ``LibraryIndex`` into singer-disjoint training pairs,
``pair_features`` reduces one pair to ``[cos, Δpitch, Δtimbre, Δresonance,
Δstyle]``, and ``fit_calibration`` fits the logistic model into a
``CalibrationModel`` whose ``score`` returns a calibrated probability and whose
``attribution`` says which columns moved it.  ``rank_singers`` is what the query
path calls: it applies that model to one query and returns the ranked, explained
top-k singers of the requested gender.
"""

from voicematch.calibrate.model import CalibrationModel, fit_calibration
from voicematch.calibrate.pairs import build_pairs, pair_features
from voicematch.calibrate.score import (
    NO_SINGERS_IN_POOL,
    NON_FINITE_PROTOTYPE,
    NonFinitePrototypeError,
    ScoredSinger,
    rank_singers,
)

__all__ = [
    "NO_SINGERS_IN_POOL",
    "NON_FINITE_PROTOTYPE",
    "CalibrationModel",
    "NonFinitePrototypeError",
    "ScoredSinger",
    "build_pairs",
    "fit_calibration",
    "pair_features",
    "rank_singers",
]
