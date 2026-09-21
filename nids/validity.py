"""Validity and behaviour-preservation checks for adversarial samples.

After an attack runs, these functions determine:
  - valid: is the perturbation within the allowed feature bounds? (FR6)
  - behaviour_preserved: does the attacked sample still look like an attack to the attacker's intent? (FR6)
"""
from __future__ import annotations
import numpy as np

from nids.constants import BENIGN, ATTACK, LABEL_NAMES


def check_validity(x_adv: np.ndarray, x_clean: np.ndarray, meta: dict,
                   eps: float) -> tuple[bool, list[str]]:
    """Check that x_adv satisfies all feature constraints.

    Returns: (is_valid, list_of_violations)
    """
    violations: list[str] = []
    feature_names = meta["feature_names"]
    bounds = meta["bounds"]
    mutable_mask = np.array(meta["mutable_mask"], dtype=bool)

    # 1. Immutable features must not change at all
    delta = x_adv - x_clean
    if np.any(delta[~mutable_mask] != 0):
        changed_immutable = [feature_names[i] for i in np.where(~mutable_mask)[0]
                             if delta[i] != 0]
        if changed_immutable:
            violations.append(f"Immutable features changed: {changed_immutable}")

    # 2. Perturbation magnitude must not exceed eps (L-infinity already enforced by project,
    #    but check anyway for safety)
    if np.any(np.abs(delta) > eps + 1e-6):
        violations.append("Perturbation exceeds eps bound")

    # 3. Each feature must stay within its declared [min, max] range
    for i, name in enumerate(feature_names):
        b = bounds[name]
        if x_adv[i] < b["min"] - 1e-6 or x_adv[i] > b["max"] + 1e-6:
            violations.append(f"Feature '{name}' out of bounds [{b['min']}, {b['max']}]")

    return (len(violations) == 0, violations)


def check_behaviour_preserved(pred_before: int, pred_after: int,
                              true_label: int) -> tuple[bool, str]:
    """Check that an evasion attack preserved the malicious *intent*.

    For an evasion attack, the attacker wants the sample to still be an attack
    in ground-truth terms (so it would be caught without the attack) but be
    misclassified as benign. We check:
      - The sample was originally an attack (true_label == ATTACK)
      - The model correctly classified it as attack before (pred_before == ATTACK)
      - The model misclassifies it as benign after (pred_after == BENIGN)

    This confirms the attack both *could* have worked (benign-looking output)
    and the intent is still malicious (ground truth is attack).
    """
    if true_label == ATTACK:
        if pred_before == ATTACK and pred_after == BENIGN:
            return True, "Evasion: attack misclassified as benign"
        if pred_before == ATTACK and pred_after == ATTACK:
            return False, "Attack still detected as attack (evasion failed)"
        return False, f"Unexpected: pred_before={LABEL_NAMES[pred_before]}, pred_after={LABEL_NAMES[pred_after]}"
    # For benign samples, an attack that flips it to attack-class = false positive
    return False, "Sample is benign — evasion scenario does not apply"


def perturbation_magnitude(x_adv: np.ndarray, x_clean: np.ndarray) -> tuple[float, float]:
    """Return (L1, L-infinity) perturbation norms."""
    delta = x_adv - x_clean
    return float(np.sum(np.abs(delta))), float(np.max(np.abs(delta)))


def feature_diff(x_adv: np.ndarray, x_clean: np.ndarray, feature_names: list[str],
                 top_n: int = 10) -> list[dict]:
    """Return the top-N changed features as a diff (for FR7 feature diff view)."""
    delta = np.abs(x_adv - x_clean)
    # sort by absolute change descending
    order = np.argsort(delta)[::-1][:top_n]
    return [
        {
            "feature": feature_names[i],
            "before": float(x_clean[i]),
            "after": float(x_adv[i]),
            "delta": float(delta[i]),
        }
        for i in order if delta[i] > 1e-10
    ]
