"""DeepFool — minimal perturbation attack via line search.

Works by iteratively linearizing the decision boundary around the current
point and taking the minimal step to cross it. Stronger than FGSM at finding
the decision boundary, weaker than CW but much faster.

Works on both differentiable (torch) and non-differentiable (sklearn) models.
For sklearn, uses gradient when available.
"""
from __future__ import annotations
import time
import numpy as np

from nids.config import BENIGN
from nids.attacks.result import AttackResult
from nids.attacks.fgsm import _gather_constraints
from nids.attacks.projection import project


def deepfool(model, x_raw: np.ndarray, eps: float, mask: np.ndarray = None,
             n_steps: int = 50, step_scale: float = 1.0) -> AttackResult:
    """DeepFool attack.

    Args:
        model: VictimModel
        x_raw: (n, d) raw feature vectors
        eps: max L-inf perturbation per feature (in SCALED space)
        mask: (d,) boolean — mutable features
        n_steps: iterations
        step_scale: multiplier for step size

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

    for _ in range(n_steps):
        probs = model.predict_proba(model.to_raw(x_adv))
        preds = np.argmax(probs, axis=1)
        if np.all(preds == BENIGN):
            break

        grad = model.gradient_scaled(x_adv, target=BENIGN)  # (n, d) or (n, 2, d)
        if grad.ndim == 3:
            # (n, n_classes, d) -> use difference between target and predicted
            grad = grad[:, BENIGN, :] - grad[:, preds, np.arange(len(preds))]

        # Minimal perturbation to cross decision boundary
        # w = gradient direction, f = loss value
        f = probs[:, BENIGN] - probs[np.arange(len(probs)), preds]
        w = grad * masks

        # step = -f / ||w||^2 * w
        w_norm_sq = np.sum(w**2, axis=1, keepdims=True) + 1e-12
        step = -f[:, None] / w_norm_sq * w
        step = np.clip(step, -eps * masks, eps * masks) * step_scale

        x_adv_new = x_adv + step
        # Check if prediction flipped
        new_probs = model.predict_proba(model.to_raw(x_adv_new))
        new_preds = np.argmax(new_probs, axis=1)
        flipped = new_preds == BENIGN
        x_adv[flipped] = x_adv_new[flipped]
        x_adv[~flipped] = x_adv[~flipped] + step[~flipped] * 0.1

        x_adv = np.clip(x_adv, x_scaled - eps * masks, x_scaled + eps * masks)

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
        metadata={"attack": "deepfool", "eps": eps, "n_steps": n_steps},
    )
