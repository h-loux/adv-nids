"""Shared model training and artifact persistence."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from nids.config import ARTIFACTS_DIR, MODEL_SEEDS
from nids.eval.metrics import compute_metrics
from nids.models import make_model


def fit_model(
    model_name: str,
    train_arrays: dict[str, np.ndarray],
    meta: dict,
    seed: int = MODEL_SEEDS[0],
    epochs: int = 50,
    batch_size: int = 256,
    model_params: dict | None = None,
):
    """Fit a victim model using only the supplied training split."""
    X_train_raw = np.asarray(train_arrays["X_raw"], dtype=np.float32)
    y_train = np.asarray(train_arrays["y"], dtype=np.int64)
    if X_train_raw.ndim != 2 or len(X_train_raw) != len(y_train):
        raise ValueError("Training features and labels must have matching 2D/1D shapes")
    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("Training split must contain both benign and attack labels")
    if X_train_raw.shape[1] != len(meta["feature_names"]):
        raise ValueError("Training feature count does not match dataset metadata")
    if not np.isfinite(X_train_raw).all():
        raise ValueError("Training features contain NaN or infinite values")

    scaler = StandardScaler().fit(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)

    random.seed(seed)
    np.random.seed(seed)
    if model_name in {"mlp", "cnn1d"}:
        import torch

        torch.manual_seed(seed)

    victim = make_model(model_name, X_train_scaled.shape[1], seed=seed)
    if model_params:
        victim.base.set_params(**model_params)
    victim.set_scaler(scaler)
    victim.feature_names = list(meta["feature_names"])
    victim.mutable_mask = np.asarray(meta["mutable_mask"], dtype=bool)
    victim.meta = meta

    if victim.is_torch:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        victim.base.to(device)
        dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(X_train_scaled, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.long),
        )
        generator = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=generator,
            pin_memory=device.type == "cuda",
        )
        optimizer = torch.optim.Adam(victim.base.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        victim.base.train()
        for _ in range(epochs):
            for features, labels in loader:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                loss = criterion(victim.base(features), labels)
                loss.backward()
                optimizer.step()
        victim.base.eval()
        victim.base.to("cpu")
    else:
        victim.base.fit(X_train_scaled, y_train)

    return victim


def train_and_save_model(
    dataset: str,
    model_name: str,
    train_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    meta: dict,
    seed: int = MODEL_SEEDS[0],
    epochs: int = 50,
    batch_size: int = 256,
    model_params: dict | None = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> tuple[dict, dict]:
    """Fit on train data, evaluate once on test data, and persist a dashboard artifact."""
    victim = fit_model(
        model_name, train_arrays, meta, seed=seed, epochs=epochs,
        batch_size=batch_size, model_params=model_params,
    )

    X_test_raw = np.asarray(test_arrays["X_raw"], dtype=np.float32)
    y_test = np.asarray(test_arrays["y"], dtype=np.int64)
    if X_test_raw.ndim != 2 or len(X_test_raw) != len(y_test):
        raise ValueError("Test features and labels must have matching 2D/1D shapes")
    if X_test_raw.shape[1] != len(victim.feature_names):
        raise ValueError("Test feature count does not match the trained model")
    if set(np.unique(y_test)) - {0, 1}:
        raise ValueError("Test labels must use 0=benign and 1=attack")
    if not np.isfinite(X_test_raw).all():
        raise ValueError("Test features contain NaN or infinite values")

    probabilities = victim.predict_proba(X_test_raw)
    predictions = np.argmax(probabilities, axis=1)
    metrics = compute_metrics(y_test, predictions, probabilities)
    victim.metrics = metrics

    artifact = {
        "model": victim.base,
        "is_torch": victim.is_torch,
        "scaler": victim.scaler_,
        "feature_names": victim.feature_names,
        "mutable_mask": victim.mutable_mask.tolist(),
        "model_name": model_name,
        "dataset": dataset,
        "n_features": victim.n_features,
        "metrics": metrics,
        "meta": meta,
        "seed": seed,
        "model_params": model_params or {},
        "epochs": epochs if model_name in {"mlp", "cnn1d"} else None,
        "batch_size": batch_size if model_name in {"mlp", "cnn1d"} else None,
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / f"{dataset}_{model_name}.joblib"
    joblib.dump(artifact, output_path)
    return artifact, metrics
