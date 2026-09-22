"""Demo: end-to-end adversarial NIDS evasion (terminal).

Flow:
  1. Load dataset  — pick a network flow (a malicious packet)
  2. Train model   — victim classifier that detects attacks
  3. Predict       — model says "attack"
  4. Attack        — FGSM/PGD perturbs features, model says "benign"
  5. Visualize     — show which features changed and why

Run:  python -m nids.demo --dataset cicids2017 --eps 0.5
      python -m nids.demo --dataset dummy --method pgd --eps 0.3
"""
import argparse
import numpy as np

from nids.config import BENIGN, ATTACK, EPS_GRID, LABEL_NAMES, ALL_MODELS
from nids.data import get_meta, get_test_arrays
from nids.models import make_model
from nids.attacks import fgsm, pgd
from nids.eval.validity import check_validity


def train_victim(vm, X, y, seed=42):
    """Train a VictimModel (sklearn or torch) on the dataset."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    np.random.seed(seed)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_train)
    vm.set_scaler(scaler)

    if vm.is_torch:
        import torch
        torch.manual_seed(seed)
        X_s = scaler.transform(X_train)
        ds = torch.utils.data.TensorDataset(
            torch.as_tensor(X_s, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.long))
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
        vm.base.fit(scaler.transform(X_train), y_train)


def main():
    parser = argparse.ArgumentParser(description="Adversarial NIDS evasion demo")
    parser.add_argument("--dataset", default="cicids2017")
    parser.add_argument("--model", default="lr",
                        choices=list(ALL_MODELS))
    parser.add_argument("--eps", type=float, default=0.5, choices=EPS_GRID,
                        help="Max L-inf perturbation per feature (scaled space)")
    parser.add_argument("--method", default="fgsm", choices=["fgsm", "pgd"])
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print(" Adversarial NIDS Evasion Demo")
    print("=" * 60)

    # 1. Load data
    print(f"\n[1] Loading dataset: {args.dataset}")
    meta = get_meta(args.dataset)
    arr = get_test_arrays(args.dataset)
    X, y = arr["X_raw"], arr["y"]
    fam = arr["y_family"]
    feature_names = meta["feature_names"]
    mutable_mask = np.array(meta["mutable_mask"], dtype=bool)

    print(f"    {len(y)} test flows, {X.shape[1]} features")
    print(f"    Benign: {(y==BENIGN).sum()}, Attack: {(y==ATTACK).sum()}")

    # Pick a sample
    if args.index is not None:
        idx = args.index
    else:
        idx = int(np.where(y == ATTACK)[0][0])
    x_clean = X[idx:idx+1]
    print(f"\n[2] Target flow #{idx}: {LABEL_NAMES[y[idx]]} ({fam[idx]})")

    # 2. Train model
    print(f"\n[3] Training {args.model} victim model...")
    vm = make_model(args.model, X.shape[1])
    vm.meta = meta  # for validity/projection
    train_victim(vm, X, y)
    print("    Trained")

    # 3. Predict (before)
    print(f"\n[4] Classification (before attack):")
    probs_before = vm.predict_proba(x_clean)[0]
    pred_before = int(np.argmax(probs_before))
    print(f"    P(benign)  = {probs_before[0]:.4f}")
    print(f"    P(attack)  = {probs_before[1]:.4f}")
    print(f"    Predicted:  {LABEL_NAMES[pred_before]} (true: {LABEL_NAMES[y[idx]]})")

    # 4. Attack
    print(f"\n[5] Launching {args.method.upper()} attack (eps={args.eps}):")
    attack_fn = fgsm if args.method == "fgsm" else pgd
    x_adv = attack_fn(vm, x_clean, args.eps, mask=mutable_mask)

    l2 = float(np.linalg.norm(x_adv - x_clean))
    linf = float(np.max(np.abs(x_adv - x_clean) * mutable_mask))
    print(f"    L2 perturbation: {l2:.4f}")
    print(f"    L-inf perturbation (raw): {linf:.4f}")

    grad = vm.gradient(x_clean, target=BENIGN).reshape(-1)
    top_grad = np.argsort(np.abs(grad) * mutable_mask)[::-1][:5]
    print(f"    Top 5 gradient targets:")
    for i in top_grad:
        sign = "+" if grad[i] > 0 else "-"
        print(f"      {feature_names[i]:<20} grad={grad[i]:+10.6f}  -> {sign}perturb")

    # 5. Predict (after)
    probs_after = vm.predict_proba(x_adv)[0]
    pred_after = int(np.argmax(probs_after))
    print(f"\n[6] Classification (after attack):")
    print(f"    P(benign)  = {probs_after[0]:.4f}  (was {probs_before[0]:.4f})")
    print(f"    P(attack)  = {probs_after[1]:.4f}  (was {probs_before[1]:.4f})")
    print(f"    Predicted:  {LABEL_NAMES[pred_after]}")

    evasion = bool(pred_after == BENIGN and y[idx] == ATTACK and pred_before == ATTACK)
    print(f"\n[7] Result: {'EVASION SUCCEEDED' if evasion else 'Attack blocked'}")

    # 6. Top changes
    print(f"\n[8] Top 5 perturbed features:")
    delta = np.abs(x_adv - x_clean) * mutable_mask
    top = np.argsort(delta[0])[::-1][:5]
    print(f"    {'Feature':<20} {'Before':>12} {'After':>12} {'Delta':>12}")
    print(f"    {'-'*56}")
    for i in top:
        d = x_adv[0, i] - x_clean[0, i]
        print(f"    {feature_names[i]:<20} {x_clean[0,i]:>12.6f} {x_adv[0,i]:>12.6f} {d:>12.6f}")

    # Validity
    valid, viols = check_validity(x_adv, x_clean, meta, args.eps, model=vm)
    print(f"\n[9] Validity: {'valid' if valid else 'INVALID'}")
    for v in viols:
        print(f"    - {v}")

    if evasion:
        print(f"\n    The model was fooled: a malicious flow now looks benign.")
        print(f"    Epsilon={args.eps} in scaled space flips the prediction.")

    return 0 if evasion else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
