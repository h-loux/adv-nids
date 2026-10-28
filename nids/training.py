"""Shared model training and artifact persistence."""
from __future__ import annotations

import hashlib
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
    augmented_train_arrays: dict[str, np.ndarray] | None = None,
):
    """Fit on clean training data, optionally adding sourced labeled augmentations."""
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

    X_fit_raw = X_train_raw
    y_fit = y_train
    source_indices = np.empty(0, dtype=np.int64)
    if augmented_train_arrays is not None:
        if not {"X_raw", "y", "source_indices"} <= augmented_train_arrays.keys():
            raise ValueError(
                "Augmented training data must contain X_raw, y, and source_indices"
            )
        X_aug_raw = np.asarray(augmented_train_arrays["X_raw"], dtype=np.float32)
        y_aug_raw = np.asarray(augmented_train_arrays["y"])
        source_indices = np.asarray(augmented_train_arrays["source_indices"])
        if X_aug_raw.ndim != 2 or len(X_aug_raw) != len(y_aug_raw):
            raise ValueError("Augmented features and labels must have matching 2D/1D shapes")
        if y_aug_raw.ndim != 1:
            raise ValueError("Augmented labels must be a one-dimensional array")
        if X_aug_raw.shape[1] != X_train_raw.shape[1]:
            raise ValueError("Augmented feature count does not match clean training data")
        if len(y_aug_raw) == 0:
            raise ValueError("At least one augmented training row is required")
        if not np.isfinite(X_aug_raw).all():
            raise ValueError("Augmented features contain NaN or infinite values")
        if (
            not np.issubdtype(y_aug_raw.dtype, np.number)
            or not np.isfinite(y_aug_raw).all()
            or not np.isin(y_aug_raw, [0, 1]).all()
        ):
            raise ValueError("Augmented labels must use integer values 0=benign and 1=attack")
        y_aug = y_aug_raw.astype(np.int64, copy=False)
        if set(np.unique(y_aug)) - {0, 1}:
            raise ValueError("Augmented labels must use 0=benign and 1=attack")
        if source_indices.ndim != 1 or len(source_indices) != len(y_aug):
            raise ValueError("source_indices must contain one clean-training index per row")
        if not np.issubdtype(source_indices.dtype, np.integer):
            raise ValueError("source_indices must contain integer row indices")
        if np.any(source_indices < 0) or np.any(source_indices >= len(y_train)):
            raise ValueError("source_indices must refer to rows in the clean training split")
        source_indices = source_indices.astype(np.int64, copy=False)
        source_rows = X_train_raw[source_indices]
        if not np.array_equal(y_aug, y_train[source_indices]):
            raise ValueError("Augmented labels must match their indexed clean-training labels")
        unchanged = np.all(X_aug_raw == source_rows, axis=1)
        if np.any(unchanged):
            raise ValueError("Augmented rows must differ from their indexed clean source rows")
        X_fit_raw = np.concatenate((X_train_raw, X_aug_raw), axis=0)
        y_fit = np.concatenate((y_train, y_aug), axis=0)

    # Keep the baseline preprocessing reference stable: fit only on clean train rows.
    scaler = StandardScaler().fit(X_train_raw)
    X_fit_scaled = scaler.transform(X_fit_raw)

    random.seed(seed)
    np.random.seed(seed)
    if model_name in {"mlp", "cnn1d"}:
        import torch

        torch.manual_seed(seed)

    victim = make_model(model_name, X_fit_scaled.shape[1], seed=seed)
    if model_params:
        victim.base.set_params(**model_params)
    # Keep XGBoost execution on CPU even when inherited/model-supplied parameters
    # came from a GPU-trained artifact. The dashboard and attack pipeline feed
    # NumPy/CPU arrays, so a CUDA booster would trigger device-mismatch fallback.
    if model_name == "xgb":
        victim.base.set_params(device="cpu")
    elif model_name == "ensemble":
        victim.base.set_params(xgb__device="cpu")
    victim.set_scaler(scaler)
    victim.feature_names = list(meta["feature_names"])
    victim.mutable_mask = np.asarray(meta["mutable_mask"], dtype=bool)
    victim.meta = meta

    if victim.is_torch:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        victim.base.to(device)
        dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(X_fit_scaled, dtype=torch.float32),
            torch.as_tensor(y_fit, dtype=torch.long),
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
        victim.base.fit(X_fit_scaled, y_fit)

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


def train_augmented_and_save_model(
    dataset: str,
    model_name: str,
    train_arrays: dict[str, np.ndarray],
    augmented_train_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    meta: dict,
    seed: int = MODEL_SEEDS[0],
    epochs: int = 50,
    batch_size: int = 256,
    model_params: dict | None = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
    augmentation_sha256: str | None = None,
) -> tuple[dict, dict]:
    """Train and save a separate model using clean data plus supplied augmentations.

    Augmented rows must include source indices into the clean training split. This
    function consumes existing rows; it does not generate perturbations or attacks.
    """
    victim = fit_model(
        model_name,
        train_arrays,
        meta,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        model_params=model_params,
        augmented_train_arrays=augmented_train_arrays,
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
        "training_strategy": "clean_plus_supplied_augmentations",
        "n_clean_train": int(len(train_arrays["y"])),
        "n_augmented_train": int(len(augmented_train_arrays["y"])),
        "augmentation_sha256": augmentation_sha256,
        "augmentation_source_indices_sha256": hashlib.sha256(
            np.asarray(augmented_train_arrays["source_indices"], dtype=np.int64).tobytes(),
        ).hexdigest(),
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / f"{dataset}_{model_name}_augmented.joblib"
    joblib.dump(artifact, output_path)
    return artifact, metrics
