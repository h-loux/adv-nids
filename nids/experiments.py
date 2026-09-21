"""Offline experiment runner — executes E1-E10 from the Execution Plan.

This is the *research* side of the pipeline. The dashboard (dashboard/app.py)
only reads the artifacts this produces; it never calls run_experiment().

Usage:
    python -m nids.experiments --dataset cicids2017 --model all --e1 --e2 --e3 --e5 --e8
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from nids.constants import (
    ARTIFACTS_DIR, CHECKPOINTS_DIR, EPS_GRID, FIGURES_DIR,
    LABEL_NAMES, PROCESSED_DIR, RESULTS_DIR, SUPPORTED_DATASETS, SUPPORTED_MODELS, ATTACK, BENIGN,
)
from nids.attacks import attack, project
from nids.data import get_meta, get_test_arrays, load_dataset
from nids.models import load_model, make_model, save_registry
from nids.validity import check_validity, check_behaviour_preserved, perturbation_magnitude


def ensure_data(dataset: str):
    """Precondition: dataset must be built. Skips download if already processed."""
    if (PROCESSED_DIR / f"{dataset}_full.npz").exists():
        return get_meta(dataset)
    return load_dataset(dataset)


# ---------------------------------------------------------------------------
# E1 — Train & evaluate clean victim models
# ---------------------------------------------------------------------------

def run_e1(dataset: str, model_names=None, random_state: int = 42):
    """Train all victim models, save checkpoints, compute E1 metrics."""
    model_names = model_names or SUPPORTED_MODELS
    meta = get_meta(dataset)
    n_features = meta["n_features"]
    mutable_mask = np.array(meta["mutable_mask"], dtype=bool)

    # Load data
    full = np.load(PROCESSED_DIR / f"{dataset}_full.npz")
    # Reconstruct train/test split by index (same seed as data build)
    rng = np.random.RandomState(random_state)
    n_total = len(full["y"])
    n_test = min(5000, n_total // 5)
    perm = rng.permutation(n_total)
    train_idx = perm[n_test:]
    test_idx = perm[:n_test]

    X_train = full["X_scaled"][train_idx]
    y_train = full["y"][train_idx]
    X_test = full["X_scaled"][test_idx]
    y_test = full["y"][test_idx]

    results = {}
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    for name in model_names:
        print(f"  E1: training {name} on {dataset}...")
        t0 = time.time()
        vm = make_model(name, dataset)

        if vm.is_torch:
            import torch
            torch.manual_seed(random_state)
            vm.base = _train_torch(vm.base, X_train, y_train, epochs=50, lr=1e-3)
        else:
            vm.base.fit(X_train, y_train)

        # Save checkpoint
        ckpt_path = CHECKPOINTS_DIR / f"{dataset}_{name}_clean.joblib" \
            if not vm.is_torch else CHECKPOINTS_DIR / f"{dataset}_{name}_clean.pt"
        if vm.is_torch:
            torch.save(vm.base.state_dict(), ckpt_path)
        else:
            import joblib
            joblib.dump(vm.base, ckpt_path)

        # Evaluate
        preds = vm.predict(X_test)
        probs = vm.predict_proba(X_test)[:, 1]
        metrics = _compute_metrics(y_test, preds, probs)
        metrics["train_time_sec"] = time.time() - t0
        results[name] = metrics
        print(f"    -> F1={metrics['f1']:.4f}  recall={metrics['recall']:.4f}  "
              f"FPR={metrics['fpr']:.4f}  AUROC={metrics['auroc']:.4f}  ({metrics['train_time_sec']:.1f}s)")

    # Save results + refresh registry
    RESULT_E1 = RESULTS_DIR / "E1"
    RESULT_E1.mkdir(parents=True, exist_ok=True)
    with open(RESULT_E1 / f"{dataset}_clean_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    save_registry()
    return results


# ---------------------------------------------------------------------------
# E2 — FGSM attack sweep over EPS_GRID (single sample + batch)
# ---------------------------------------------------------------------------

def run_e2(dataset: str, model_names=None, random_state: int = 42):
    """Run FGSM at every eps in EPS_GRID against every model; compute ASR."""
    np.random.seed(random_state)
    model_names = model_names or SUPPORTED_MODELS
    meta = get_meta(dataset)
    mutable_mask = np.array(meta["mutable_mask"], dtype=bool)

    arr = get_test_arrays(dataset)
    X_test, y_test = arr["X_scaled"], arr["y"]

    results = {}
    for name in model_names:
        vm = load_model(dataset, name, variant="clean")
        asr_per_eps = {}
        for eps in EPS_GRID:
            asr, avg_delta = _eval_attack(vm, X_test, y_test, "fgsm", eps, mutable_mask, n_samples=1000)
            asr_per_eps[str(eps)] = {"asr": asr, "avg_perturbation_linf": avg_delta}
        results[name] = asr_per_eps
        print(f"  E2: {name} FGSM — " + ", ".join(f"eps={e}: ASR={asr_per_eps[str(e)]['asr']:.3f}" for e in EPS_GRID))

    RESULT_E2 = RESULTS_DIR / "E2"
    RESULT_E2.mkdir(parents=True, exist_ok=True)
    with open(RESULT_E2 / f"{dataset}_fgsm_asr.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ---------------------------------------------------------------------------
# E3 — PGD attack sweep
# ---------------------------------------------------------------------------

def run_e3(dataset: str, model_names=None, random_state: int = 42):
    """Run PGD at every eps in EPS_GRID; compute ASR and validity rate."""
    np.random.seed(random_state)
    model_names = model_names or SUPPORTED_MODELS
    meta = get_meta(dataset)
    mutable_mask = np.array(meta["mutable_mask"], dtype=bool)
    bounds = meta["bounds"]

    arr = get_test_arrays(dataset)
    X_test, y_test = arr["X_scaled"], arr["y"]

    results = {}
    for name in model_names:
        vm = load_model(dataset, name, variant="clean")
        asr_per_eps = {}
        for eps in EPS_GRID:
            asr, avg_delta, valid_rate = _eval_attack_pgd(vm, X_test, y_test, eps,
                                                           mutable_mask, bounds, n_samples=1000)
            asr_per_eps[str(eps)] = {"asr": asr, "avg_perturbation_linf": avg_delta,
                                     "validity_rate": valid_rate}
        results[name] = asr_per_eps
        print(f"  E3: {name} PGD — " + ", ".join(f"eps={e}: ASR={asr_per_eps[str(e)]['asr']:.3f}" for e in EPS_GRID))

    RESULT_E3 = RESULTS_DIR / "E3"
    RESULT_E3.mkdir(parents=True, exist_ok=True)
    with open(RESULT_E3 / f"{dataset}_pgd_asr.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ---------------------------------------------------------------------------
# E5 — Cross-model transfer evaluation
# ---------------------------------------------------------------------------

def run_e5(dataset: str, model_names=None, random_state: int = 42):
    """Generate adversarial samples with each model, test against all others → transfer matrix."""
    np.random.seed(random_state)
    model_names = model_names or SUPPORTED_MODELS
    meta = get_meta(dataset)
    mutable_mask = np.array(meta["mutable_mask"], dtype=bool)
    bounds = meta["bounds"]

    arr = get_test_arrays(dataset)
    X_test, y_test = arr["X_scaled"], arr["y"]

    # Use a subset of attack-class samples
    attack_idx = np.where(y_test == ATTACK)[0]
    n_gen = min(500, len(attack_idx))
    gen_idx = np.random.choice(attack_idx, n_gen, replace=False)

    # Generate adversarials with each model as generator
    adv_samples: dict[str, np.ndarray] = {}
    for gen_name in model_names:
        vm = load_model(dataset, gen_name, variant="clean")
        eps = 0.1
        adv = attack(vm, X_test[gen_idx], "pgd", eps, mask=mutable_mask, n_steps=25)
        adv = np.array([project(a, c, eps, mutable_mask) for a, c in zip(adv, X_test[gen_idx])])
        adv_samples[gen_name] = adv

    # Evaluate on every model
    transfer_matrix = {}
    for gen_name in model_names:
        row = {}
        for vic_name in model_names:
            vm = load_model(dataset, vic_name, variant="clean")
            preds = vm.predict(adv_samples[gen_name])
            if gen_name == vic_name:
                # self-transfer (attack success rate on generator)
                row[vic_name] = float((preds == BENIGN).mean())
            else:
                # cross-model transfer
                row[vic_name] = float((preds == BENIGN).mean())
        transfer_matrix[gen_name] = row
        print(f"  E5: transfer from {gen_name} -> " + ", ".join(f"{k}: {v:.3f}" for k, v in row.items()))

    RESULT_E5 = RESULTS_DIR / "E5"
    RESULT_E5.mkdir(parents=True, exist_ok=True)
    with open(RESULT_E5 / f"{dataset}_transfer_matrix.json", "w") as f:
        json.dump(transfer_matrix, f, indent=2)
    return transfer_matrix


# ---------------------------------------------------------------------------
# E8 — Adversarial training (robust models)
# ---------------------------------------------------------------------------

def run_e8(dataset: str, model_names=("mlp", "cnn1d"), eps: float = 0.1,
           epochs: int = 20, random_state: int = 42):
    """Train adversarially-robust versions of MLP and CNN1D via FGSM adversarial training."""
    np.random.seed(random_state)
    import torch
    torch.manual_seed(random_state)

    meta = get_meta(dataset)
    mutable_mask = np.array(meta["mutable_mask"], dtype=bool)

    full = np.load(PROCESSED_DIR / f"{dataset}_full.npz")
    rng = np.random.RandomState(random_state)
    n_total = len(full["y"])
    n_test = min(5000, n_total // 5)
    perm = rng.permutation(n_total)
    train_idx = perm[n_test:]

    X_train = full["X_scaled"][train_idx]
    y_train = full["y"][train_idx]
    X_test = full["X_scaled"][perm[:n_test]]
    y_test = full["y"][perm[:n_test]]

    results = {}
    for name in model_names:
        print(f"  E8: adversarially training {name} on {dataset} (eps={eps})...")
        vm = make_model(name, dataset)
        if not vm.is_torch:
            # For sklearn models, do adversarial training by augmenting training set
            _adv_train_sklearn(vm, X_train, y_train, eps, mutable_mask, rounds=epochs)
        else:
            vm.base = _train_torch_adv(vm.base, X_train, y_train, eps, mutable_mask,
                                       epochs=epochs, lr=1e-3)

        # Save robust checkpoint
        if vm.is_torch:
            torch.save(vm.base.state_dict(), CHECKPOINTS_DIR / f"{dataset}_{name}_e8.pt")
        else:
            import joblib
            joblib.dump(vm.base, CHECKPOINTS_DIR / f"{dataset}_{name}_e8.joblib")

        # Evaluate clean + attacked robust
        probs_clean = vm.predict_proba(X_test)[:, 1]
        preds_clean = vm.predict(X_test)
        clean_metrics = _compute_metrics(y_test, preds_clean, probs_clean)

        # Attack the robust model
        attack_idx = np.where(y_test == ATTACK)[0]
        adv = attack(vm, X_test[attack_idx], "fgsm", eps, mask=mutable_mask)
        adv = np.array([project(a, c, eps, mutable_mask) for a, c in zip(adv, X_test[attack_idx])])
        preds_adv = vm.predict(adv)
        robust_asr = float((preds_adv == BENIGN).mean())

        results[name] = {
            "clean_metrics": clean_metrics,
            "asr_under_attack": robust_asr,
        }
        print(f"    robust {name}: clean F1={clean_metrics['f1']:.4f}, ASR={robust_asr:.4f}")

    RESULT_E8 = RESULTS_DIR / "E8"
    RESULT_E8.mkdir(parents=True, exist_ok=True)
    with open(RESULT_E8 / f"{dataset}_robust.json", "w") as f:
        json.dump(results, f, indent=2)
    save_registry()
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    from sklearn.metrics import (
        f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[BENIGN, ATTACK])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = 0.5
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "auroc": float(auroc),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def _eval_attack(vm, X, y, attack_name, eps, mask, n_samples=1000):
    """Run attack_name on n_samples attack-class flows; return (ASR, avg_linf_perturbation)."""
    attack_idx = np.where(y == ATTACK)[0][:n_samples]
    x_targets = X[attack_idx]
    x_adv = attack(vm, x_targets, attack_name, eps, mask=mask)
    x_adv = np.array([project(a, c, eps, mask) for a, c in zip(x_adv, x_targets)])
    preds = vm.predict(x_adv)
    asr = float((preds == BENIGN).mean())
    avg_delta = float(np.mean(np.max(np.abs(x_adv - x_targets), axis=1)))
    return asr, avg_delta


def _eval_attack_pgd(vm, X, y, eps, mask, bounds, n_samples=1000):
    from nids.validity import check_validity
    attack_idx = np.where(y == ATTACK)[0][:n_samples]
    x_targets = X[attack_idx]
    x_adv = attack(vm, x_targets, "pgd", eps, mask=mask, n_steps=25)
    x_adv = np.array([project(a, c, eps, mask) for a, c in zip(x_adv, x_targets)])
    preds = vm.predict(x_adv)
    asr = float((preds == BENIGN).mean())
    avg_delta = float(np.mean(np.max(np.abs(x_adv - x_targets), axis=1)))
    # validity rate
    valid_count = sum(1 for a, c in zip(x_adv, x_targets) if check_validity(a, c, {"feature_names": list(bounds.keys()), "bounds": bounds, "mutable_mask": mask.tolist()}, eps)[0])
    valid_rate = valid_count / len(x_adv)
    return asr, avg_delta, valid_rate


# ---------------------------------------------------------------------------
# Torch training helpers
# ---------------------------------------------------------------------------

def _train_torch(model, X, y, epochs=50, lr=1e-3, batch_size=256):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    return model


def _train_torch_adv(model, X, y, eps, mask, epochs=20, lr=1e-3, batch_size=256):
    """Adversarial training: augment each batch with FGSM samples."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    mask_t = torch.as_tensor(mask, dtype=torch.float32)
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            # Clean loss
            out = model(xb)
            loss_clean = criterion(out, yb)
            # FGSM adversarial loss
            xb_adv = xb.clone().requires_grad_(True)
            out_adv = model(xb_adv)
            loss_adv = criterion(out_adv, yb)
            grad = torch.autograd.grad(loss_adv, xb_adv, retain_graph=True)[0]
            x_adv = xb + eps * torch.sign(grad) * mask_t
            x_adv = torch.clamp(x_adv, xb - eps, xb + eps) * mask_t + xb * (1 - mask_t)
            out_adv2 = model(x_adv)
            loss_adv2 = criterion(out_adv2, yb)
            loss = 0.5 * (loss_clean + loss_adv2)
            loss.backward()
            optimizer.step()
    return model


def _adv_train_sklearn(vm, X, y, eps, mask, rounds=20):
    """Adversarial training for sklearn models: augment with FGSM samples."""
    from sklearn.base import clone
    X_aug = [X]
    y_aug = [y]
    for _ in range(rounds):
        # re-fit model on current augmented set
        X_cur = np.vstack(X_aug)
        y_cur = np.concatenate(y_aug)
        vm.base.fit(X_cur, y_cur)
        # Generate adversarial samples from attack-class
        attack_idx = np.where(y_cur == ATTACK)[0]
        if len(attack_idx) == 0:
            break
        x_target = X_cur[attack_idx]
        x_adv = attack(vm, x_target, "fgsm", eps, mask=mask)
        x_adv = np.array([project(a, c, eps, mask) for a, c in zip(x_adv, x_target)])
        X_aug.append(x_adv)
        y_aug.append(y_cur[attack_idx])
    # final fit
    X_cur = np.vstack(X_aug)
    y_cur = np.concatenate(y_aug)
    vm.base = clone(vm.base).fit(X_cur, y_cur)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NIDS adversarial experiments E1-E10")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--model", default="all",
                        help="comma-separated model names or 'all'")
    parser.add_argument("--e1", action="store_true", help="Train & evaluate clean models")
    parser.add_argument("--e2", action="store_true", help="FGSM attack sweep")
    parser.add_argument("--e3", action="store_true", help="PGD attack sweep")
    parser.add_argument("--e5", action="store_true", help="Cross-model transfer matrix")
    parser.add_argument("--e8", action="store_true", help="Adversarial training (robust models)")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    models = SUPPORTED_MODELS if args.model == "all" else args.model.split(",")
    for m in models:
        assert m in SUPPORTED_MODELS, f"Unknown model {m}. Choose from {SUPPORTED_MODELS}"

    print(f"Running experiments for dataset={args.dataset}, models={models}")

    if args.e1:
        print("\n=== E1: Clean model training & evaluation ===")
        ensure_data(args.dataset)
        run_e1(args.dataset, models, args.random_state)

    if args.e2:
        print("\n=== E2: FGSM attack sweep ===")
        ensure_data(args.dataset)
        run_e2(args.dataset, models, args.random_state)

    if args.e3:
        print("\n=== E3: PGD attack sweep ===")
        ensure_data(args.dataset)
        run_e3(args.dataset, models, args.random_state)

    if args.e5:
        print("\n=== E5: Cross-model transfer ===")
        ensure_data(args.dataset)
        run_e5(args.dataset, models, args.random_state)

    if args.e8:
        print("\n=== E8: Adversarial training ===")
        ensure_data(args.dataset)
        run_e8(args.dataset, models, random_state=args.random_state)

    save_registry()
    print(f"\nAll done. Registry refreshed at {ARTIFACTS_DIR / 'model_registry.json'}")


if __name__ == "__main__":
    main()
