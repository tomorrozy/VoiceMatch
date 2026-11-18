"""Evaluation: the metrics that decide whether the product is "准" at all.

``metrics`` holds the four numbers of design doc 6.3 -- Top-K hit rate,
acceptance rate, Spearman against human preference, and cross-device Top-3
overlap -- together with the rule that makes them honest: a metric that has
nothing to measure returns ``nan``, never ``0.0``.

``dataset`` loads and validates the annotated eval set (``picked == []`` is the
"都不像" answer, R9), and ``run`` is the end-to-end runner that walks every
query through the product pipeline and reports both headline numbers with the
denominators behind them (G1).  The denominator rules live in the metrics
themselves, so the runner cannot disagree with them.

``run.calibration_version`` names a fitted calibration model by a short hash of
its five fields; the CLI reports it next to ``model_id`` and the library
version so a result is traceable to the model that produced it.
"""

from voicematch.eval.dataset import EvalQuery, load_eval_set
from voicematch.eval.metrics import (
    acceptance_rate,
    spearman,
    top3_overlap,
    top_k_hit,
)
from voicematch.eval.run import calibration_version, run_eval

__all__ = [
    "EvalQuery",
    "acceptance_rate",
    "calibration_version",
    "load_eval_set",
    "run_eval",
    "spearman",
    "top3_overlap",
    "top_k_hit",
]
