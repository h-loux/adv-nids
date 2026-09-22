"""PGD — Projected Gradient Descent.

Iterative attack in SCALED feature space. Takes n_steps small steps,
re-evaluating the gradient and re-projecting at every step.

After the final iteration, we do iterative raw-space / scaled-space
projection to ensure all constraints (eps-ball, bounds, integers, one-hot)
are simultaneously satisfied.
"""
from __future__ import annotations
import numpy as np

from nids.config import BENIGN
from nids.attacks.projection import project
from nids.attacks.fgsm import _gather_constraints


def pgd(model, x_raw: np.ndarray, eps: float, mask: np.ndarray = None,
        n_steps: int = 40, step_size: float | None = None,
        n_project: int = 5) -> np.ndarray:
    """PGD attack with projection inside every step.

    Args:
        model: VictimModel
        x_raw: (n, d) clean raw feature vectors
        eps:   max L-inf perturbation per feature (in SCALED space)
        mask:  (d,) boolean — mutable features
        n_steps: PGD iterations
        step_size: per-step size (default: eps/n_steps)
        n_project: final projection iterations

    Returns:
        x_adv: (n, d) adversarial raw feature vectors
    """
    if mask is None:
        mask = model.mutable_mask
    bounds, int_feats, one_hot = _gather_constraints(model)

    x_scaled = model.to_scaled(x_raw)
    x_adv_scaled = x_scaled.copy()
    if step_size is None:
        step_size = eps / n_steps

    for _ in range(n_steps):
        grad = model.gradient_scaled(x_adv_scaled, target=BENIGN)
        sign = np.sign(grad) * mask[None, :]
        x_adv_scaled = x_adv_scaled + step_size * sign
        x_adv_scaled = np.clip(x_adv_scaled,
                               x_scaled - eps * mask[None, :],
                               x_scaled + eps * mask[None, :])

    x_adv_raw = model.to_raw(x_adv_scaled)
    # Final iterative projection between scaled-space eps and raw-space constraints
    for _ in range(n_project):
        x_adv_raw = project(x_adv_raw, x_raw, mutable_mask=mask,
                            bounds=bounds, int_features=int_feats,
                            one_hot_groups=one_hot)
        xs = model.to_scaled(x_adv_raw)
        delta_s = np.clip(xs - x_scaled, -eps, eps) * mask[None, :]
        x_adv_raw = model.to_raw(x_scaled + delta_s)

    return x_adv_raw
