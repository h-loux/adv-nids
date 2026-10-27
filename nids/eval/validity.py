"""Layer 4b — Validity and behaviour-preservation checks for adversarial samples.

After an attack runs, these functions determine:
  - valid: is the perturbation within the allowed feature bounds?
  - behaviour_preserved: does the attack still represent the original intent?

Works on both 1D (single sample, shape (d,)) and 2D (batch, shape (n, d)).
"""
from __future__ import annotations
import numpy as np

from nids.config import BENIGN, ATTACK, LABEL_NAMES


def _flatten(x):
    """Squeeze a (1, d) array to (d,) for per-sample checks."""
    return x.reshape(-1) if x.ndim > 1 else x


def check_validity(x_adv, x_clean, meta, eps, model=None):
    """Check that x_adv satisfies all feature constraints.

    If model (VictimModel) is provided, the eps bound is checked
    in SCALED space (matching the attack's working space). Otherwise,
    checks in whatever space x_adv/x_clean are in.

    Returns: (is_valid, list_of_violations)
    """
    x_clean_1d = _flatten(x_clean)
    x_adv_1d = _flatten(x_adv)
    delta = x_adv_1d - x_clean_1d

    feature_names = meta["feature_names"]
    bounds = meta["bounds"]
    mutable_mask = np.array(meta["mutable_mask"], dtype=bool)

    violations = []

    # 1. Immutable features must not change (allow floating-point transform noise).
    changed = [
        feature_names[i] for i in np.where(~mutable_mask)[0]
        if abs(delta[i]) > 1e-8
    ]
    if changed:
        violations.append(f"Immutable features changed: {changed}")

    # 2. Perturbation magnitude must not exceed eps
    if model is not None:
        # Check in scaled space
        x_clean_s = model.to_scaled(x_clean_1d.reshape(1, -1)).reshape(-1)
        x_adv_s = model.to_scaled(x_adv_1d.reshape(1, -1)).reshape(-1)
        delta_s = x_adv_s - x_clean_s
        max_pert = float(np.max(np.abs(delta_s) * mutable_mask))
    else:
        delta_s = delta
        max_pert = float(np.max(np.abs(delta * mutable_mask)))

    if max_pert > eps + 1e-6:
        violations.append(f"Perturbation exceeds eps ({max_pert:.4f} > {eps})")

    # 3. Each feature must stay within its declared [min, max] range
    for i, name in enumerate(feature_names):
        b = bounds[name]
        v = x_adv_1d[i]
        if v < b["min"] - 1e-6 or v > b["max"] + 1e-6:
            violations.append(f"'{name}' out of bounds [{b['min']:.4f}, {b['max']:.4f}]")

    # 4. Packet/count features inferred as integers must remain integral.
    for index in meta.get("int_features", []):
        i = int(index)
        if 0 <= i < len(x_adv_1d) and abs(x_adv_1d[i] - round(x_adv_1d[i])) > 1e-6:
            violations.append(f"Integer feature '{feature_names[i]}' is not integral")

    # 5. One-hot groups may be all-zero for a category unseen in training; otherwise
    # values must be binary and the sum must be one.
    name_to_index = {name: i for i, name in enumerate(feature_names)}
    for group_name, members in meta.get("one_hot_groups", {}).items():
        indices = []
        for member in members:
            if isinstance(member, (int, np.integer)) and not isinstance(member, (bool, np.bool_)):
                col = int(member)
            elif isinstance(member, str):
                col = name_to_index.get(member)
            else:
                col = None
            if col is not None and 0 <= col < len(x_adv_1d) and col not in indices:
                indices.append(col)
        if not indices:
            continue
        values = x_adv_1d[indices]
        binary = np.isclose(values, 0.0, atol=1e-6) | np.isclose(values, 1.0, atol=1e-6)
        total = float(np.sum(np.rint(values)))
        if not np.all(binary) or not (np.isclose(total, 0.0, atol=1e-6) or np.isclose(total, 1.0, atol=1e-6)):
            violations.append(f"One-hot group '{group_name}' is not a binary encoding")

    return (len(violations) == 0, violations)


def check_behaviour_preserved(pred_before, pred_after, true_label):
    """Check that an evasion attack preserved the malicious intent.

    For evasion: true_label == ATTACK, pred_before == ATTACK, pred_after == BENIGN.
    """
    if true_label == ATTACK:
        if pred_before == ATTACK and pred_after == BENIGN:
            return True, "Evasion: attack misclassified as benign"
        if pred_before == ATTACK and pred_after == ATTACK:
            return False, "Attack still detected"
        return False, f"Unexpected: before={LABEL_NAMES[pred_before]}, after={LABEL_NAMES[pred_after]}"
    return False, "Sample is benign"


def perturbation_magnitude(x_adv, x_clean):
    """Return (L1, L-infinity) perturbation norms."""
    delta = _flatten(x_adv) - _flatten(x_clean)
    return float(np.sum(np.abs(delta))), float(np.max(np.abs(delta)))


def feature_diff(x_adv, x_clean, feature_names, top_n=10):
    """Return the top-N changed features as a diff (for visualization)."""
    delta = np.abs(_flatten(x_adv) - _flatten(x_clean))
    order = np.argsort(delta)[::-1][:top_n]
    x_clean_f = _flatten(x_clean)
    x_adv_f = _flatten(x_adv)
    return [
        {
            "feature": feature_names[i],
            "before": float(x_clean_f[i]),
            "after": float(x_adv_f[i]),
            "delta": float(delta[i]),
        }
        for i in order if delta[i] > 1e-10
    ]
