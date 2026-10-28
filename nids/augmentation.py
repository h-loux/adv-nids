"""Training-time adversarial augmentation for network-flow classifiers.

Only uses rows from the clean training split. Perturbations are generated
toward the benign class with the project's existing FGSM or PGD implementation,
then filtered through the configured feature-validity checks. The inherited
ground-truth label remains ATTACK; this is for defensive retraining, not relabeling.
"""
from __future__ import annotations

import numpy as np

from nids.attacks.fgsm import fgsm
from nids.attacks.pgd import pgd
from nids.config import ATTACK
from nids.eval.validity import check_validity


def generate_adversarial_training_augmentations(
    model,
    train_arrays: dict[str, np.ndarray],
    meta: dict,
    *,
    method: str = "FGSM",
    eps: float = 0.1,
    max_samples: int = 16,
    seed: int = 7,
    pgd_steps: int = 5,
) -> tuple[dict[str, np.ndarray], dict]:
    """Generate validated, label-preserving augmentations from training rows.

    Source rows are restricted to malicious training examples the baseline
    currently classifies as malicious. Their order is seeded and shuffled, and
    the scan stops once max_samples eligible source rows have been found (or
    all malicious training rows have been checked). This bounds unnecessary
    full-dataset prediction work on large datasets. FGSM/PGD attempts to move
    selected rows toward the benign decision region. Changed samples that pass
    configured bounds, immutable-feature, integer-feature, one-hot, and
    scaled-epsilon checks are retained whether or not the model's prediction
    flips. Their true label stays ATTACK.

    This function never reads the held-out test split.
    """
    method = str(method).strip().upper()
    if method not in {"FGSM", "PGD"}:
        raise ValueError("Generation method must be FGSM or PGD.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("Epsilon must be a finite positive number.")
    if not isinstance(max_samples, (int, np.integer)) or isinstance(max_samples, bool) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer.")
    if not isinstance(pgd_steps, (int, np.integer)) or pgd_steps < 1:
        raise ValueError("pgd_steps must be a positive integer.")

    X = np.asarray(train_arrays.get("X_raw"), dtype=np.float64)
    y = np.asarray(train_arrays.get("y"), dtype=np.int64)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("Training arrays must contain 2D X_raw and matching 1D y.")
    if X.shape[1] != len(meta.get("feature_names", [])):
        raise ValueError("Training feature count does not match dataset metadata.")
    if not np.isfinite(X).all():
        raise ValueError("Training features contain NaN or infinite values.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Training labels must use 0=benign and 1=attack.")

    mutable_mask = np.asarray(
        getattr(model, "mutable_mask", meta.get("mutable_mask", [])), dtype=bool
    )
    if mutable_mask.shape != (X.shape[1],):
        raise ValueError("Model mutable-feature mask does not match training features.")

    # Randomize the malicious-row order and stop as soon as enough eligible
    # sources are found. The previous implementation predicted every malicious
    # training row before applying max_samples, which could dominate runtime on
    # CICIDS2017 even when the caller requested only a handful of augmentations.
    rng = np.random.default_rng(seed)
    attack_indices = np.flatnonzero(y == ATTACK)
    rng.shuffle(attack_indices)
    eligible_parts = []
    eligible_count = 0
    attack_rows_scanned = 0
    prediction_batch_size = 1024
    for start in range(0, len(attack_indices), prediction_batch_size):
        row_indices = attack_indices[start:start + prediction_batch_size]
        attack_rows_scanned += len(row_indices)
        row_predictions = np.argmax(model.predict_proba(X[row_indices]), axis=1)
        correctly_detected = row_indices[row_predictions == ATTACK]
        if len(correctly_detected):
            eligible_parts.append(correctly_detected)
            eligible_count += len(correctly_detected)
        if eligible_count >= int(max_samples):
            break

    eligible = (
        np.concatenate(eligible_parts).astype(np.int64, copy=False)
        if eligible_parts else np.empty(0, dtype=np.int64)
    )
    if len(eligible) == 0:
        raise ValueError(
            "No correctly detected malicious rows were found in the scanned training rows "
            "for this baseline. Try increasing max_samples or inspect the clean baseline."
        )

    # Keep stable source-index order after selecting a seeded random candidate pool.
    eligible = np.sort(eligible)
    if len(eligible) > int(max_samples):
        eligible = np.sort(rng.choice(eligible, size=int(max_samples), replace=False))

    kept_X = []
    kept_indices = []
    evasion_successes = 0
    invalid_rejected = 0
    unchanged_rejected = 0
    violation_examples: list[str] = []

    for source_idx in eligible:
        source = X[int(source_idx):int(source_idx) + 1]
        if method == "FGSM":
            result = fgsm(model, source, eps=float(eps), mask=mutable_mask)
        else:
            result = pgd(
                model, source, eps=float(eps), mask=mutable_mask,
                n_steps=int(pgd_steps),
            )

        candidate = np.asarray(result.x_adv, dtype=np.float64).reshape(-1)
        if candidate.shape != (X.shape[1],) or not np.isfinite(candidate).all():
            invalid_rejected += 1
            violation_examples.append("Attack returned a malformed or non-finite sample")
            continue
        source_row = source.reshape(-1)
        if (
            np.allclose(candidate, source_row, rtol=0.0, atol=1e-10)
            # The trainer deliberately converts flow features to float32. A
            # smaller float64-only change would disappear at training time and
            # later be rejected as an unchanged augmentation.
            or np.array_equal(candidate.astype(np.float32), source_row.astype(np.float32))
        ):
            unchanged_rejected += 1
            continue

        is_valid, violations = check_validity(
            candidate, source.reshape(-1), meta, float(eps), model=model
        )
        if not is_valid:
            invalid_rejected += 1
            violation_examples.extend(violations[:2])
            continue

        post_prediction = int(np.argmax(model.predict_proba(candidate.reshape(1, -1)), axis=1)[0])
        if post_prediction == 0:
            evasion_successes += 1
        kept_X.append(candidate)
        kept_indices.append(int(source_idx))

    if not kept_X:
        detail = f" Example validation issue: {violation_examples[0]}." if violation_examples else ""
        raise ValueError(
            "No changed perturbations passed the configured validity checks. "
            "Try a different epsilon or attack method, or inspect feature constraints."
            + detail
        )

    source_indices = np.asarray(kept_indices, dtype=np.int64)
    augmented = {
        "X_raw": np.vstack(kept_X).astype(np.float64, copy=False),
        "y": y[source_indices].astype(np.int64, copy=True),
        "source_indices": source_indices,
    }
    stats = {
        "method": method,
        "epsilon_scaled": float(eps),
        "seed": int(seed),
        "pgd_steps": int(pgd_steps) if method == "PGD" else None,
        "eligible_source_rows": int(len(eligible)),
        "eligible_candidates_found_before_cap": int(eligible_count),
        "attack_training_rows_total": int(len(attack_indices)),
        "attack_training_rows_scanned": int(attack_rows_scanned),
        "candidate_scan_complete": bool(attack_rows_scanned >= len(attack_indices)),
        "generated_valid_rows": int(len(kept_X)),
        "valid_candidates_predicted_benign": int(evasion_successes),
        "invalid_rejected": int(invalid_rejected),
        "unchanged_rejected": int(unchanged_rejected),
        "label_policy": "inherit source label; generated malicious rows remain label 1 (attack)",
        "data_scope": "clean training split only; held-out test split not used",
        "validity_scope": (
            "Configured numeric bounds, epsilon, immutable, integer and one-hot checks; "
            "these checks do not prove real-world flow behavior is unchanged."
        ),
    }
    return augmented, stats
