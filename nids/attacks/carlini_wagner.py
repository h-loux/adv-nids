"""Carlini-Wagner L2 attack — optimization-based adversarial attack.

Finds the minimum L2 perturbation that fools the model by solving:

    minimize  ||delta||_2 + c * f(x_adv)

where f(x_adv) is 0 if the model is fooled and grows otherwise.
The constant c is found via binary search.

Works in SCALED feature space for numerical stability.
"""
from __future__ import annotations
import time
import numpy as np

from nids.config import BENIGN
from nids.attacks.result import AttackResult
from nids.attacks.projection import project


def carlini_wagner(model, x_raw: np.ndarray, eps: float, mask: np.ndarray = None,
                   n_steps: int = 200, c_lr: float = 0.01,
                   c_init: float = 1e-3, n_binary_search: int = 5,
                   n_project: int = 5) -> AttackResult:
    """Carlini-Wagner L2 attack.

    Args:
        model: VictimModel
        x_raw: (n, d) raw feature vectors
        eps:   max L-inf perturbation per feature (used for projection)
        mask:  (d,) boolean — mutable features
        n_steps: Adam iterations per binary search step
        c_lr:  learning rate for Adam optimizer
        c_init: initial c constant
        n_binary_search: number of binary search steps on c
        n_project: projection iterations (raw-space constraints ↔ scaled-space eps)

    Returns:
        AttackResult with x_adv, success, n_queries, runtime_ms
    """
    start = time.time()
    if mask is None:
        mask = model.mutable_mask
    meta = getattr(model, "meta", {})

    x_raw = np.atleast_2d(x_raw)
    x_scaled = model.to_scaled(x_raw)
    n_samples, n_feats = x_scaled.shape

    # Binary search bounds for c
    c_low = np.zeros(n_samples)
    c_high = np.ones(n_samples) * 1e10
    c = np.ones(n_samples) * c_init

    best_x_adv = x_raw.copy()
    best_l2 = np.full(n_samples, np.inf)
    best_success = np.zeros(n_samples, dtype=bool)

    bounds = meta.get("bounds")
    int_feats = meta.get("int_features")
    one_hot = meta.get("one_hot_groups")

    for _ in range(n_binary_search):
        # Optimize delta = x_adv - x_clean in scaled space
        delta = np.zeros_like(x_scaled)
        opt = torch.optim.Adam([torch.nn.Parameter(torch.as_tensor(delta))],
                               lr=c_lr) if _is_torch_model(model) else None

        if opt is not None:
            param = opt.param_groups[0]["params"][0]
            for step in range(n_steps):
                opt.zero_grad()
                x_adv_s = x_scaled + param.data * mask[None, :].astype(np.float32)
                logits = _forward_torch(model, x_adv_s)
                # CW loss: maximize loss for true class, minimize for target
                loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fn(logits, torch.as_tensor(
                    np.full(n_samples, BENIGN, dtype=np.int64))).detach().numpy()
                cw_loss = np.maximum(0, loss)  # f(x) = max(0, loss)

                l2_term = np.sum((param.data.numpy() * mask)**2, axis=1)
                total_loss = l2_term + c * cw_loss
                (-total_loss).sum().backward()  # maximize
                # Wait, wrong direction — we need backward
                opt.zero_grad()
                l2_tensor = torch.sum((param * torch.as_tensor(mask[None, :], dtype=torch.float32))**2, dim=1)
                loss_tensor = torch.nn.functional.cross_entropy(
                    _forward_torch(model, x_scaled + param * torch.as_tensor(mask[None, :], dtype=torch.float32)),
                    torch.as_tensor(np.full(n_samples, BENIGN, dtype=np.int64)),
                    reduction="none",
                )
                f_tensor = torch.clamp(loss_tensor, min=0)
                total = l2_tensor + c * f_tensor
                loss = (-total).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            x_adv_scaled = (x_scaled + param.data.numpy() * mask[None, :]).astype(np.float64)
        else:
            # sklearn model: use gradient if available, else finite differences
            x_adv_scaled = _cw_sklearn(model, x_scaled, mask, c, n_steps, c_lr)

        x_adv_raw = model.to_raw(x_adv_scaled)

        # Iterative projection: raw-space constraints ↔ scaled-space eps-ball
        for _ in range(n_project):
            x_adv_raw = project(x_adv_raw, x_raw, mutable_mask=mask,
                                bounds=bounds, int_features=int_feats,
                                one_hot_groups=one_hot)
            d_s = model.to_scaled(x_adv_raw) - x_scaled
            d_s = np.clip(d_s, -eps, eps) * mask[None, :]
            x_adv_raw = model.to_raw(x_scaled + d_s)

        # Evaluate
        probs = model.predict_proba(x_adv_raw)
        preds = np.argmax(probs, axis=1)
        success = (preds == BENIGN)
        l2 = np.sqrt(np.sum((x_adv_raw - x_raw)**2, axis=1))

        improved = (l2 < best_l2) & success
        best_x_adv[improved] = x_adv_raw[improved]
        best_l2[improved] = l2[improved]
        best_success = best_success | success

        # Update c: increase where attack failed, decrease where it succeeded
        success_mask = success.astype(float)
        c_low = np.where(success, c_low, c)
        c_high = np.where(success, c, c_high)
        c = np.where(success, (c + c_low) / 2, (c + c_high) / 2)
        c = np.clip(c, 1e-5, 1e10)

    runtime_ms = (time.time() - start) * 1000
    n_queries = n_binary_search * n_steps * (2 if not _is_torch_model(model) else 1)

    return AttackResult(
        x_adv=best_x_adv,
        x_clean=x_raw,
        success=best_success,
        n_queries=n_queries,
        runtime_ms=runtime_ms,
        metadata={"attack": "carlini_wagner", "eps": eps},
    )


def _is_torch_model(model) -> bool:
    return getattr(model, "is_torch", False)


def _forward_torch(model, x_scaled: np.ndarray) -> "torch.Tensor":
    import torch
    x_t = torch.as_tensor(x_scaled, dtype=torch.float32)
    if model.is_torch:
        model.base.eval()
        logits = model.base(x_t)
        if hasattr(model, "to_logits") and callable(model.to_logits):
            logits = model.to_logits(x_t)
        return logits
    else:
        # sklearn model: no autograd, fall back to numpy
        probs = model.base.predict_proba(x_scaled)
        return torch.as_tensor(probs, dtype=torch.float32)


def _cw_sklearn(model, x_scaled, mask, c, n_steps, lr):
    """CW optimization for sklearn models using finite-difference gradients."""
    from scipy.optimize import minimize
    n = x_scaled.shape[0]
    x_adv = x_scaled.copy()
    for _ in range(n_steps):
        for i in range(n):
            def obj(delta):
                d = np.zeros_like(x_scaled[i])
                d[mask] = delta
                x_t = (x_scaled[i] + d).reshape(1, -1)
                probs = model.predict_proba(model.to_raw(x_t))
                logit_diff = probs[0, BENIGN]  # minimize this (push toward benign)
                l2 = np.sum(d**2)
                return l2 + c[i] * max(0, -logit_diff + 0.01)  # f(x)

            delta0 = np.zeros(int(mask.sum()))
            res = minimize(obj, delta0, method="L-BFGS-B", options={"maxiter": 1})
            d = np.zeros_like(x_scaled[i])
            d[mask] = res.x
            x_adv[i] = x_adv[i] + d
    return x_adv
