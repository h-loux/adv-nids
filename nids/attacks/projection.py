"""Shared constraint-enforcement used by every attack.

After computing an adversarial perturbation, we project it back into
the feasible region:
  1. Only mutable features are touched
  2. L-infinity norm per feature ≤ eps
  3. All features stay within their declared [min, max] bounds
  4. Integer features stay integers
  5. One-hot groups still sum to 1
"""
from __future__ import annotations
import numpy as np


def project(x_adv: np.ndarray, x_clean: np.ndarray, eps: float | None = None,
            mutable_mask: np.ndarray | None = None, bounds: dict | None = None,
            int_features: list[int] | None = None,
            one_hot_groups: dict | None = None) -> np.ndarray:
    """Project x_adv back into the feasible perturbation region.

    Args:
        x_adv:      (n, d) or (d,) adversarial sample(s)
        x_clean:    (n, d) or (d,) clean sample(s)
        eps:        max L-inf perturbation per feature
        mutable_mask: (d,) boolean — True if feature is attacker-controllable
        bounds:     {feature_name: {"min": float, "max": float}} (optional)
        int_features: list of feature indices that must be integers (optional)
        one_hot_groups: {"feature_prefix": [column indices or feature names]} (optional)

    Returns:
        x_proj: projected adversarial sample, same shape as x_adv
    """
    x_adv = x_adv.copy()
    x_clean = np.atleast_2d(x_clean) if x_adv.ndim == 1 else x_clean

    if x_adv.ndim == 1:
        x_adv = x_adv.reshape(1, -1)

    # 1. Zero out perturbations on immutable features
    pert = x_adv - x_clean
    if mutable_mask is not None:
        pert *= mutable_mask[None, :]
    x_adv = x_clean + pert

    # 2. Clip to eps-ball (L-infinity)
    if eps is not None:
        x_adv = np.clip(x_adv,
                        x_clean - eps * (mutable_mask[None, :] if mutable_mask is not None else 1.0),
                        x_clean + eps * (mutable_mask[None, :] if mutable_mask is not None else 1.0))

    # 3. Clip to feature bounds (if provided)
    if bounds is not None:
        feature_names = list(bounds.keys())
        for i, name in enumerate(feature_names):
            if i < x_adv.shape[1]:
                b = bounds[name]
                x_adv[:, i] = np.clip(x_adv[:, i], b["min"], b["max"])

    # 4. Integer features stay integers
    if int_features:
        for i in int_features:
            if i < x_adv.shape[1]:
                x_adv[:, i] = np.round(x_adv[:, i])

    # 5. One-hot groups: ensure each group sums to 1.
    # UNSW metadata records one-hot members by feature name, while existing
    # callers/tests may supply integer positions. Accept both representations.
    if one_hot_groups:
        feature_index = (
            {name: i for i, name in enumerate(bounds.keys())}
            if bounds is not None else {}
        )
        for _group_name, col_indices in one_hot_groups.items():
            cols = []
            for col in col_indices:
                if isinstance(col, (int, np.integer)) and not isinstance(col, (bool, np.bool_)):
                    index = int(col)
                elif isinstance(col, str):
                    index = feature_index.get(col)
                else:
                    index = None
                if index is not None and 0 <= index < x_adv.shape[1] and index not in cols:
                    cols.append(index)
            if not cols:
                continue
            group_vals = x_adv[:, cols].copy()
            # For each sample, keep the argmax column as 1, rest as 0
            argmax = np.argmax(group_vals, axis=1)
            x_adv[:, cols] = 0
            x_adv[np.arange(len(x_adv)), np.asarray(cols)[argmax]] = 1

    return x_adv
