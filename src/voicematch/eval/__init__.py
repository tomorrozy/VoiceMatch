"""Evaluation: the metrics that decide whether the product is "准" at all.

``metrics`` holds the four numbers of design doc 6.3 -- Top-K hit rate,
acceptance rate, Spearman against human preference, and cross-device Top-3
overlap -- together with the rule that makes them honest: a metric that has
nothing to measure returns ``nan``, never ``0.0``.

The eval set loader and the end-to-end runner (Task 14) build on this package;
the denominator rules (a "都不像" query is not a ranking miss) live in the
metrics themselves so the runner cannot disagree with them.
"""

from voicematch.eval.metrics import (
    acceptance_rate,
    spearman,
    top3_overlap,
    top_k_hit,
)

__all__ = [
    "acceptance_rate",
    "spearman",
    "top3_overlap",
    "top_k_hit",
]
