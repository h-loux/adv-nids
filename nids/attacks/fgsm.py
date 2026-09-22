"""FGSM — Fast Gradient Sign Method.

Single-step attack in SCALED feature space:
  x_adv_scaled = x_scaled + eps * sign(grad)
  x_adv = inverse_transform(x_adv_scaled)

The gradient is d P(benign) / d x_scaled.
We target BENIGN (0), so the perturbation pushes P(benign) UP.

After the single step, we do iterative projection (scaled-space eps-ball
↔ raw-space bounds/integers) for 5 iterations to ensure all constraints
are satisfied simultaneously.
"""
from __future__ import annotations
import numpy as np

from nids.config import BENIGN
from nids.attacks.projection import project


def _gather_constraints(model):
    """Extract bounds, int_features, one_hot_groups from model.meta."""
    meta = getattr(model, "meta", {})
    return (meta.get("bounds"), meta.get("int_features"),
            meta.get("one_hot_groups"))


def fgsm(model, x_raw: np.ndarray, eps: float, mask: np.ndarray = None,
         n_project: int = 5) -> np.ndarray:
    """FGSM attack.

    Args:
        model: VictimModel (has .meta with bounds, .feature_names, .mutable_mask)
        x_raw: (n, d) raw feature vectors
        eps:   max L-inf perturbation per feature (in SCALED space)
        mask:  (d,) boolean — True for mutable features (overrides model.mutable_mask)
        n_project: projection iterations to satisfy all constraints

    Returns:
        x_adv: (n, d) adversarial raw feature vectors
    """
    if mask is None:
        mask = model.mutable_mask
    bounds, int_feats, one_hot = _gather_constraints(model)

    x_scaled = model.to_scaled(x_raw)
    grad = model.gradient_scaled(x_scaled, target=BENIGN)  # (n, d)
    sign = np.sign(grad) * mask[None, :]
    x_adv_scaled = x_scaled + eps * sign

    # Project to eps-ball in scaled space
    x_adv_scaled = np.clip(x_adv_scaled,
                           x_scaled - eps * mask[None, :],
                           x_scaled + eps * mask[None, :])

    # Convert to raw then iteratively project to satisfy BOTH scaled-space eps
    # and raw-space (bounds, integers, one-hot).
    x_adv_raw = model.to_raw(x_adv_scaled)
    for _ in range(n_project):
        # Raw-space projection: bounds, int, one-hot
        x_adv_raw = project(x_adv_raw, x_raw, mutable_mask=mask,
                            bounds=bounds, int_features=int_feats,
                            one_hot_groups=one_hot)
        # Scaled-space re-projection: eps-ball
        x_adv_scaled = model.to_scaled(x_adv_raw)
        delta_s = x_adv_scaled - x_scaled
        delta_s = np.clip(delta_s, -eps, eps) * mask[None, :]
        x_adv_scaled = x_scaled + delta_s
        x_adv_raw = model.to_raw(x_adv_scaled)

    return x_adv_raw
