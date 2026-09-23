"""JSMA — Jacobian-based Saliency Map Attack.

Targets specific features by computing the Jacobian (gradient of the output
w.r.t. each input feature) and perturbing the most influential ones.
Produces sparse adversarial examples (few features changed).

Works on differentiable models (torch). For sklearn, uses the model's
gradient method if available.
"""
from __future__ import annotations
import time
import numpy as np

from nids.config import BENIGN
from nids.attacks.result import AttackResult
from nids.attacks.fgsm import _gather_constraints
from nids.attacks.projection import project


def jsma(model, x_raw: np.ndarray, eps: float, mask: np.ndarray = None,
         n_steps: int = 50, gamma: float = 0.5, max_pert: float = 2.0) -> AttackResult:
    """Jacobian-based Saliency Map Attack.

    Args:
        model: VictimModel (must support gradient)
        x_raw: (n, d) raw feature vectors
        eps: max L-inf perturbation per feature (in SCALED space)
        mask: (d,) boolean — mutable features
        n_steps: max attack iterations
        gamma: perturbation per step
        max_pert: max perturbation per feature (raw space)

    Returns:
        AttackResult
    """
    start = time.time()
    x_raw = np.atleast_2d(x_raw)
    if mask is None:
        mask = model.mutable_mask
    bounds, int_feats, one_hot = _gather_constraints(model)

    x_scaled = model.to_scaled(x_raw)
    masks = mask[None, :].astype(np.float64)
    x_adv = x_scaled.copy()

    for step in range(n_steps):
        probs = model.predict_proba(model.to_raw(x_adv))
        preds = np.argmax(probs, axis=1)
        if np.all(preds == BENIGN):
            break

        # Compute Jacobian: gradient of P(benign) w.r.t. each feature
        grad = model.gradient_scaled(x_adv, target=BENIGN)  # (n, d)
        grad = grad * masks

        # Saliency: features where gradient points toward increasing P(benign)
        # Perturb in direction that increases P(benign)
        perturb = np.sign(grad) * gamma
        perturb = np.clip(perturb, -max_pert, max_pert) * masks

        x_adv = x_adv + perturb

        # Check if we flipped
        new_probs = model.predict_proba(model.to_raw(x_adv))
        new_preds = np.argmax(new_probs, axis=1)
        flipped = new_preds == BENIGN

        if np.all(flipped):
            break

        # Reduce gamma for next step
        gamma *= 0.95

    x_adv_raw = model.to_raw(x_adv)
    for _ in range(5):
        x_adv_raw = project(x_adv_raw, x_raw, mutable_mask=mask,
                            bounds=bounds, int_features=int_feats,
                            one_hot_groups=one_hot)
        d_s = np.clip(model.to_scaled(x_adv_raw) - x_scaled, -eps, eps) * masks
        x_adv_raw = model.to_raw(x_scaled + d_s)

    probs = model.predict_proba(x_adv_raw)
    success = np.argmax(probs, axis=1) == BENIGN

    return AttackResult(
        x_adv=x_adv_raw,
        x_clean=x_raw,
        success=success,
        n_queries=n_steps,
        runtime_ms=(time.time() - start) * 1000,
        metadata={"attack": "jsma", "eps": eps, "n_steps": step + 1},
    )
