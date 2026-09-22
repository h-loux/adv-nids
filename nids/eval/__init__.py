"""Evaluation package: metrics, validity checks, transfer matrix."""
from nids.eval.metrics import compute_metrics, write_result
from nids.eval.validity import (
    check_validity, check_behaviour_preserved,
    perturbation_magnitude, feature_diff,
)

__all__ = [
    "compute_metrics", "write_result",
    "check_validity", "check_behaviour_preserved",
    "perturbation_magnitude", "feature_diff",
]
