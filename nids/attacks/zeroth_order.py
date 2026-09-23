"""Black-box attacks: zero-query (transfer) and query-based (NES).

Zeroth-order (no gradient access):
- NES: gradient estimate via Monte Carlo sampling + PGD on estimates
- Transfer: attack on a substitute model, hope it transfers to the victim

Both return AttackResult for consistency with white-box attacks.
"""
from __future__ import annotations
import time
import numpy as np

from nids.config import BENIGN
from nids.attacks.result import AttackResult
from nids.attacks.pgd import _gather_constraints
from nids.attacks.projection import project


def nes_attack(model, x_raw: np.ndarray, eps: float, mask: np.ndarray = None,
               n_queries: int = 200, n_directions: int = 32,
               sigma: float = 0.01, n_steps: int = 20,
               step_size: float | None = None) -> AttackResult:
    """Zeroth-order attack via NES (Natural Evolution Strategies) gradient estimation.

    Estimates the gradient of the loss with respect to the input by querying
    the model with random perturbations. No gradient access required.

    Args:
        model: VictimModel (access via predict_proba only)
        x_raw: (n, d) raw feature vectors
        eps: max L-inf perturbation per feature (in SCALED space)
        mask: (d,) boolean — mutable features
        n_queries: total query budget
        n_directions: random directions per gradient estimate
        sigma: perturbation scale for finite differences
        n_steps: PGD iterations (each uses n_directions queries)
        step_size: per-step size (default: eps/n_steps)

    Returns:
        AttackResult
    """
    start = time.time()
    x_raw = np.atleast_2d(x_raw)
    if mask is None:
        mask = model.mutable_mask
    bounds, int_feats, one_hot = _gather_constraints(model)

    x_scaled = model.to_scaled(x_raw)
    x_adv_scaled = x_scaled.copy()
    if step_size is None:
        step_size = eps / n_steps

    masks = mask[None, :].astype(np.float64)
    total_queries = 0

    for step in range(n_steps):
        if total_queries >= n_queries:
            break

        # NES gradient estimation
        n_batch = min(n_directions, n_queries - total_queries)
        directions = np.random.randn(n_batch, x_scaled.shape[1]) * masks
        directions /= np.sqrt(np.sum(directions**2, axis=1, keepdims=True) + 1e-12)

        # Query model with +/- perturbations
        x_pos = x_adv_scaled + sigma * directions
        x_neg = x_adv_scaled - sigma * directions

        probs_pos = model.predict_proba(model.to_raw(x_pos))[:, BENIGN]
        probs_neg = model.predict_proba(model.to_raw(x_neg))[:, BENIGN]
        total_queries += 2 * n_batch

        # Gradient estimate: E[ Q(x + sigma*u) - Q(x - sigma*u) ] * u / (2*sigma)
        grad_est = np.mean((probs_pos - probs_neg)[:, None] * directions, axis=0) / (2 * sigma)
        grad_est = grad_est * masks

        # PGD step
        sign = np.sign(grad_est) * masks
        x_adv_scaled = x_adv_scaled + step_size * sign
        x_adv_scaled = np.clip(x_adv_scaled,
                               x_scaled - eps * masks,
                               x_scaled + eps * masks)

    # Final projection to raw space
    x_adv_raw = model.to_raw(x_adv_scaled)
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
        n_queries=total_queries,
        runtime_ms=(time.time() - start) * 1000,
        metadata={"attack": "nes", "eps": eps, "n_queries": total_queries},
    )


def transfer_attack(substitute_model, victim_model, x_raw, eps, mask=None) -> AttackResult:
    """Transfer-based black-box attack.

    Attack a substitute model (white-box) and test the result on the victim
    (zero queries to the victim). If the adversarial example transfers, the
    victim is fooled without ever querying it.

    Args:
        substitute_model: model with gradient access (AttackerModel)
        victim_model: VictimModel to attack (zero-query)
        x_raw: (n, d) raw feature vectors
        eps: max L-inf perturbation (in SCALED space of the substitute)
        mask: (d,) boolean — mutable features

    Returns:
        AttackResult (with victim model predictions)
    """
    start = time.time()
    from nids.attacks.pgd import pgd
    result = pgd(substitute_model, x_raw, eps, mask=mask)

    # Test on victim (zero queries to the attack itself)
    x_adv = result.x_adv
    probs_victim = victim_model.predict_proba(x_adv)
    success = np.argmax(probs_victim, axis=1) == BENIGN

    return AttackResult(
        x_adv=x_adv,
        x_clean=x_raw,
        success=success,
        n_queries=result.n_queries,  # only queries to substitute, not victim
        runtime_ms=(time.time() - start) * 1000,
        metadata={"attack": "transfer", "eps": eps,
                  "substitute": substitute_model.meta.get("model_name", "unknown")},
    )
