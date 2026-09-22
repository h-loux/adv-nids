"""Pre-train all victim models on the dataset and save as artifacts.

This replaces runtime training in the dashboard. Run once:

  python -m scripts.pretrain --dataset dummy --models lr rf xgb mlp cnn1d

After this, artifacts/<model>.joblib contains a dict with:
  {model, scaler, feature_names, mutable_mask, metrics}
"""
import argparse
import json
from pathlib import Path

import numpy as np

from nids.config import (
    ALL_MODELS, ATTACK, BENIGN, LABEL_NAMES, MODEL_SEEDS, PROCESSED_DIR, RESULTS_DIR, ARTIFACTS_DIR,
)
from nids.data import get_meta, get_test_arrays, get_train_arrays
from nids.eval.metrics import compute_metrics
from nids.models.cnn1d import CNN1D


def train_sklearn(name, X_train_scaled, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None

    if name == "lr":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif name == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=12,
                                      random_state=42, n_jobs=-1)
    elif name == "xgb" and XGBClassifier is not None:
        return XGBClassifier(n_estimators=200, max_depth=6,
                            random_state=42, eval_metric="logloss")
    elif name == "xgb":
        # Fallback to GradientBoosting if XGBoost not installed
        return GradientBoostingClassifier(n_estimators=200, max_depth=6,
                                          random_state=42)
    raise ValueError(f"Unknown sklearn model: {name}")


def train_pytorch(name, n_features, X_train_scaled, y_train, seed=42):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    if name == "mlp":
        model = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )
    elif name == "cnn1d":
        model = CNN1D(n_features)
    else:
        raise ValueError(f"Unknown torch model: {name}")

    X_t = torch.as_tensor(X_train_scaled, dtype=torch.float32)
    y_t = torch.as_tensor(y_train, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(50):
        for i in range(0, len(X_t), 256):
            xb, yb = X_t[i:i+256], y_t[i:i+256]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dummy")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS)
    args = parser.parse_args()

    dataset = args.dataset
    print(f"Pre-training models on '{dataset}' dataset...")

    train = get_train_arrays(dataset)
    test = get_test_arrays(dataset)
    meta = get_meta(dataset)

    X_train_raw = train["X_raw"]
    X_train_scaled = train["X_scaled"]
    y_train = train["y"]
    X_test_scaled = test["X_scaled"]
    X_test_raw = test["X_raw"]
    y_test = test["y"]
    feature_names = meta["feature_names"]
    mutable_mask = np.array(meta["mutable_mask"], dtype=bool)

    # The data was already scaled during preprocessing. Re-fit scaler on
    # X_train_raw so we can convert raw <-> scaled at runtime.
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    X_tr_raw, _, _, _ = train_test_split(X_train_raw, y_train,
                                          test_size=0.2, random_state=42,
                                          stratify=y_train)
    scaler = StandardScaler().fit(X_tr_raw)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    e1_results = {}
    full_meta = dict(meta)
    full_meta["scaler_mean"] = scaler.mean_.tolist()
    full_meta["scaler_scale"] = scaler.scale_.tolist()

    for model_name in args.models:
        print(f"\n  Training {model_name}...", end=" ")
        is_torch = model_name in ("mlp", "cnn1d")

        if is_torch:
            base = train_pytorch(model_name, len(feature_names),
                                X_train_scaled, y_train)
        else:
            base = train_sklearn(model_name, X_train_scaled, y_train)
            base.fit(X_train_scaled, y_train)

        # Predictions + metrics
        if is_torch:
            import torch
            model = base.eval()
            with torch.no_grad():
                logits = model(torch.as_tensor(X_test_scaled, dtype=torch.float32))
                probs = torch.softmax(logits, dim=1).numpy()
        else:
            probs = base.predict_proba(X_test_scaled)

        metrics = compute_metrics(y_test, np.argmax(probs, axis=1), probs)
        e1_results[model_name] = metrics
        print(f"F1={metrics['f1']:.4f} AUROC={metrics['auroc']:.4f}")

        # Save artifact with scaler
        artifact = {
            "model": base,
            "is_torch": is_torch,
            "scaler": scaler,
            "feature_names": feature_names,
            "mutable_mask": mutable_mask.tolist(),
            "model_name": model_name,
            "dataset": dataset,
            "n_features": len(feature_names),
            "metrics": metrics,
            "meta": full_meta,
        }
        out = ARTIFACTS_DIR / f"{dataset}_{model_name}.joblib"
        import joblib
        joblib.dump(artifact, out)
        print(f"    -> {out}")

    # Save E1 results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    e1_path = RESULTS_DIR / "E1" / f"{dataset}_clean_metrics.json"
    e1_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({k: v for k, v in e1_results.items()},
              open(e1_path, "w"), indent=2)
    print(f"\n  E1 results -> {e1_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
