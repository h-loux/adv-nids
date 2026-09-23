"""Experiments runner.

Executes the full evaluation battery across datasets and writes JSON results:
  E1: Clean model metrics (F1, recall, FPR, AUROC, AUPRC) per model
  E2: FGSM attack success rate per model × epsilon
  E3: PGD attack success rate per model × epsilon
  E4: Adversarial training defense vs. FGSM/PGD transfer

Usage:
  python -m nids.experiments --e1 --e2 --e3
  python -m nids.experiments --dataset cicids2017 --models lr rf mlp --eps 0.3 0.5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from nids.config import BENIGN, EPS_GRID, LABEL_NAMES, ALL_MODELS as SUPPORTED_MODELS
from nids.data import get_meta, get_test_arrays
from nids.models import make_model
from nids.attacks import fgsm, pgd, cw
from nids.attacks.result import AttackResult
from nids.eval.metrics import compute_metrics
from nids.eval.validity import (
    check_validity,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def train_model_for_dataset(dataset, model_name, seed=42):
    """Train a VictimModel for the given dataset and model."""
    meta = get_meta(dataset)
    arr = get_test_arrays(dataset)
    X, y = arr["X_raw"], arr["y"]
    vm = make_model(model_name, X.shape[1])

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    np.random.seed(seed)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    vm.set_scaler(scaler)
    vm.meta = meta  # for validity checks and attack constraints

    if vm.is_torch:
        import torch
        torch.manual_seed(seed)
        X_s = scaler.transform(X_tr)
        ds = torch.utils.data.TensorDataset(
            torch.as_tensor(X_s, dtype=torch.float32),
            torch.as_tensor(y_tr, dtype=torch.long))
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
        opt = torch.optim.Adam(vm.base.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        vm.base.train()
        for ep in range(100):
            for xb, yb in loader:
                opt.zero_grad()
                crit(vm.base(xb), yb).backward()
                opt.step()
    else:
        vm.base.fit(scaler.transform(X_tr), y_tr)

    return vm, arr, meta


def run_clean(dataset, models):
    """E1: Clean metrics per model."""
    results = {}
    arr = get_test_arrays(dataset)
    X, y = arr["X_raw"], arr["y"]
    for m in models:
        vm, _, _ = train_model_for_dataset(dataset, m)
        probs = vm.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        metrics = compute_metrics(y, preds, probs)
        results[m] = metrics
        print(f"  {m}: F1={metrics['f1']:.4f} AUROC={metrics['auroc']:.4f}")
    out = RESULTS_DIR / "E1" / f"{dataset}_clean_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"  -> {out}")
    return results


def run_attack(dataset, models, attack_fn, eps_list, attack_name):
    """E2/E3: Attack success rate per model × epsilon."""
    results = {}
    arr = get_test_arrays(dataset)
    X, y = arr["X_raw"], arr["y"]
    meta = get_meta(dataset)
    mutable_mask = np.array(meta["mutable_mask"], dtype=bool)

    # Only attack successful (correctly-detected) attack flows
    for m in models:
        vm, _, _ = train_model_for_dataset(dataset, m)
        results[m] = []
        for eps in eps_list:
            # Find attack flows that are classified as attack
            probs = vm.predict_proba(X)
            preds = np.argmax(probs, axis=1)
            attack_flows = np.where((y == 1) & (preds == 1))[0]
            if len(attack_flows) == 0:
                results[m].append({"eps": eps, "asr": 0.0, "n": 0})
                continue
            successes = 0
            valid_count = 0
            for idx in attack_flows:
                x = X[idx:idx+1]
                result = attack_fn(vm, x, eps, mask=mutable_mask)
                x_adv_raw = result.x_adv
                p_adv = vm.predict_proba(x_adv_raw)[0]
                if np.argmax(p_adv) == BENIGN:
                    successes += 1
                is_valid, _ = check_validity(x_adv_raw, x, meta, eps, model=vm)
                if is_valid:
                    valid_count += 1
            asr = successes / len(attack_flows)
            results[m].append({
                "eps": eps, "asr": float(asr), "n": int(len(attack_flows)),
                "valid": float(valid_count / len(attack_flows)),
            })
            print(f"  {m} eps={eps}: ASR={asr:.4f} ({successes}/{len(attack_flows)})")
    dir_map = {"fgsm": "E2", "pgd": "E3", "cw": "E6"}
    exp_dir = dir_map.get(attack_name, "E2")
    out = RESULTS_DIR / exp_dir / f"{dataset}_{attack_name}_asr.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"  -> {out}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run adversarial NIDS experiments")
    parser.add_argument("--dataset", default="cicids2017")
    parser.add_argument("--models", nargs="+", default=["lr", "rf", "xgb", "mlp", "cnn1d"],
                        choices=SUPPORTED_MODELS)
    parser.add_argument("--eps", type=float, nargs="+", default=EPS_GRID)
    parser.add_argument("--e1", action="store_true", help="Clean metrics")
    parser.add_argument("--e2", action="store_true", help="FGSM ASR")
    parser.add_argument("--e3", action="store_true", help="PGD ASR")
    parser.add_argument("--e6", action="store_true", help="Carlini-Wagner ASR")
    args = parser.parse_args()

    if not any([args.e1, args.e2, args.e3, args.e6]):
        args.e1 = args.e2 = args.e3 = True

    if args.e1:
        print(f"\n=== E1: Clean metrics ({args.dataset}) ===")
        run_clean(args.dataset, args.models)
    if args.e2:
        print(f"\n=== E2: FGSM ASR ({args.dataset}) ===")
        run_attack(args.dataset, args.models, fgsm, args.eps, "fgsm")
    if args.e3:
        print(f"\n=== E3: PGD ASR ({args.dataset}) ===")
        run_attack(args.dataset, args.models, pgd, args.eps, "pgd")
    if args.e6:
        print(f"\n=== E6: CW ASR ({args.dataset}) ===")
        run_attack(args.dataset, args.models, cw, args.eps, "cw")

    return 0


if __name__ == "__main__":
    sys.exit(main())
