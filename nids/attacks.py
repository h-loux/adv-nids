"""Gradient-based adversarial attacks: FGSM and PGD.

These functions operate on a single sample (per the dashboard's FR5) but
are written to handle batches for the offline experiment suite (E2, E3).

Both call project() to enforce the feature constraint mask from meta.json,
then validity.py to check the result.
"""
from __future__ import annotations
import numpy as np

from nids.constants import EPS_GRID, ATTACK


def fgsm_attack(model, x_clean: np.ndarray, y_target: int, eps: float,
                mask: np.ndarray | None = None) -> np.ndarray:
    """Fast Gradient Sign Method — single gradient step.

    Args:
        model: VictimModel with a .gradient() method.
        x_clean: (n_features,) single sample (scaled space).
        y_target: target class index (1 = attack for evasion).
        eps: perturbation magnitude.
        mask: boolean array — True where perturbation is allowed.

    Returns:
        x_adv: perturbed sample (clipped to x_clean ± eps, masked).
    """
    if x_clean.ndim == 1:
        x_clean = x_clean.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    grad = model.gradient(x_clean, target=y_target)
    sign = np.sign(grad)
    if mask is not None:
        sign = sign * mask.astype(float)

    x_adv = x_clean + eps * sign
    if squeeze:
        x_adv = x_adv[0]
    return x_adv


def pgd_attack(model, x_clean: np.ndarray, y_target: int, eps: float,
               mask: np.ndarray | None = None, n_steps: int = 40,
               step_size: float | None = None, random_start: bool = True) -> np.ndarray:
    """Projected Gradient Descent — iterative attack with random restart.

    Args:
        n_steps: number of PGD iterations (E3 uses 40).
        step_size: if None, defaults to eps / (4 * n_steps**0.5).
    """
    if x_clean.ndim == 1:
        x_clean = x_clean.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    if step_size is None:
        step_size = eps / (4 * np.sqrt(n_steps))

    x_adv = x_clean.copy()
    if random_start:
        x_adv = x_adv + np.random.uniform(-eps, eps, size=x_adv.shape)

    mask_arr = np.ones_like(x_adv) if mask is None else np.broadcast_to(mask, x_adv.shape)

    for _ in range(n_steps):
        grad = model.gradient(x_adv, target=y_target)
        sign = np.sign(grad) * mask_arr

        x_adv = x_adv + step_size * sign
        # Project back to epsilon-ball around x_clean (L-infinity)
        x_adv = np.clip(x_adv, x_clean - eps, x_clean + eps)
        # Also enforce the mask: immutable features cannot move at all
        x_adv = x_clean * (1 - mask_arr) + x_adv * mask_arr

    if squeeze:
        x_adv = x_adv[0]
    return x_adv


# Canonical attack dispatch used by both experiments and dashboard
ATTACKS = {
    "fgsm": fgsm_attack,
    "pgd": pgd_attack,
}


def attack(model, x_clean, attack_name: str, eps: float, mask=None, **kwargs):
    """Dispatch to the right attack function.

    Calls fgsm_attack or pgd_attack with y_target=ATTACK (evasion: make attack look benign).
    """
    if attack_name not in ATTACKS:
        raise ValueError(f"Unknown attack: {attack_name}. Use one of {list(ATTACKS)}")
    return ATTACKS[attack_name](model, x_clean, y_target=ATTACK, eps=eps, mask=mask, **kwargs)


def project(x_adv: np.ndarray, x_clean: np.ndarray, eps: float,
            mask: np.ndarray | None = None) -> np.ndarray:
    """Project x_adv back into the valid perturbation ball (L-infinity) + mask.

    This is the exact routine the dashboard calls after attack() (spec §4.1 step 7).
    """
    delta = x_adv - x_clean
    delta = np.clip(delta, -eps, eps)
    if mask is not None:
        delta = delta * mask.astype(float)
    return x_clean + delta
