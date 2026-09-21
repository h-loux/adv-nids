"""Victim model definitions for E1.

Supports: lr (logistic regression), rf (random forest), xgb (xgboost),
mlp (sklearn MLP), cnn1d (PyTorch 1D-CNN).

Models are dataset-agnostic: they read meta.json for feature count and produce
binary predictions + confidence via predict_proba.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from nids.constants import (
    ARTIFACTS_DIR, CHECKPOINTS_DIR,
)
from nids.data import get_meta

ATTACK_IDX = 1  # index of "attack" class in predict_proba output


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str          # "lr", "rf", "xgb", "mlp", "cnn1d"
    dataset: str       # "cicids2017" | "unsw_nb15"
    variant: str       # "clean" | "robust"  (robust from E8)
    path: str          # checkpoint file path

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.name}__{self.variant}"


def _checkpoint_root(dataset: str, model_name: str, variant: str = "clean") -> Path:
    suffix = "_e8" if variant == "robust" else "_clean"
    if model_name in ("lr", "rf", "xgb", "mlp"):
        ext = ".joblib"
    else:
        ext = ".pt"
    return CHECKPOINTS_DIR / f"{dataset}_{model_name}{suffix}{ext}"


def build_registry() -> dict:
    """Walk checkpoints/ and build (dataset, model, variant) -> path."""
    reg: dict = {}
    if not CHECKPOINTS_DIR.exists():
        return reg
    for p in CHECKPOINTS_DIR.glob("*"):
        # filename: {dataset}_{model}_{variant}.{ext} where variant is "clean" or "e8"
        stem = p.stem.replace(".joblib", "").replace(".pt", "")
        dataset = "cicids2017" if stem.startswith("cicids2017") else "unsw_nb15"
        remainder = stem[len(dataset) + 1:]  # skip dataset + underscore

        if remainder.endswith("_e8"):
            variant = "robust"
            model_name = remainder[:-3].rstrip("_")
        elif remainder.endswith("_clean"):
            variant = "clean"
            model_name = remainder[:-6].rstrip("_")
        else:
            variant = "clean"
            model_name = remainder

        if dataset not in reg:
            reg[dataset] = {}
        reg[dataset][model_name] = {**reg[dataset].get(model_name, {}), variant: str(p)}
    return reg


def save_registry(reg: dict | None = None) -> Path:
    """Write the registry JSON (used by the dashboard's model_registry.json too)."""
    reg = reg or build_registry()
    out = ARTIFACTS_DIR / "model_registry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(reg, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(model_name: str, dataset: str):
    """Instantiate an untrained model by name. Returns a VictimModel wrapper."""
    meta = get_meta(dataset)
    n_features = meta["n_features"]

    if model_name == "lr":
        from sklearn.linear_model import LogisticRegression
        base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        base = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=42, n_jobs=-1)
    elif model_name == "xgb":
        import xgboost as xgb
        base = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 eval_metric="logloss", random_state=42, n_jobs=-1)
    elif model_name == "mlp":
        from sklearn.neural_network import MLPClassifier
        base = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                             random_state=42, early_stopping=True)
    elif model_name == "cnn1d":
        base = _CNN1D(num_features=n_features)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return VictimModel(name=model_name, dataset=dataset, base_model=base)


def load_model(dataset: str, model_name: str, variant: str = "clean") -> "VictimModel":
    """Load a checkpointed VictimModel from disk."""
    import joblib
    path = _checkpoint_root(dataset, model_name, variant)
    if not path.exists():
        # check registry fallback
        reg = build_registry()
        if dataset in reg and model_name in reg[dataset] and variant in reg[dataset][model_name]:
            path = Path(reg[dataset][model_name][variant])
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint for {dataset}/{model_name}/{variant}")

    vm = make_model(model_name, dataset)
    vm.load(path)
    return vm


# ---------------------------------------------------------------------------
# VictimModel wrapper — uniform interface across sklearn + torch
# ---------------------------------------------------------------------------

class VictimModel:
    """Wraps sklearn or torch models behind a uniform predict interface."""

    def __init__(self, name: str, dataset: str, base_model):
        self.name = name
        self.dataset = dataset
        self.base = base_model
        self.is_torch = name == "cnn1d"
        self._cached_logits_grad = None  # for attack gradient access

    def load(self, path: Union[str, Path]):
        path = Path(path)
        if self.is_torch:
            import torch
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            self.base.load_state_dict(ckpt)
            self.base.eval()
        else:
            import joblib
            self.base = joblib.load(path)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, 2) probability array [P(benign), P(attack)]."""
        if self.is_torch:
            import torch
            with torch.no_grad():
                t = torch.as_tensor(X, dtype=torch.float32)
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                probs = self.base(t).cpu().numpy()
                if probs.shape[1] == 1:
                    probs = np.hstack([1 - probs, probs])
            return probs
        else:
            probs = self.base.predict_proba(X)
            if probs.shape[1] == 1:
                probs = np.hstack([1 - probs, probs])
            return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the attack-relevant score: P(attack)."""
        if self.is_torch:
            import torch
            with torch.no_grad():
                t = torch.as_tensor(X, dtype=torch.float32)
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                out = self.base(t)
                if out.shape[1] == 1:
                    probs = torch.sigmoid(out)
                    return probs.squeeze(-1).cpu().numpy().astype(float)
                return out.softmax(dim=1)[:, ATTACK_IDX].cpu().numpy()
        else:
            return self.predict_proba(X)[:, 1]

    def gradient(self, X: np.ndarray, target: int = 1):
        """Compute d(score)/d(X) for gradient-based attacks.

        For torch models: exact gradient via autograd.
        For sklearn models: NES (Natural Evolution Strategies) gradient estimate.
        """
        if self.is_torch:
            import torch
            self.base.eval()
            t = torch.as_tensor(X, dtype=torch.float32)
            t.requires_grad_(True)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            out = self.base(t)
            if out.shape[1] == 1:
                loss = out.sum()
            else:
                loss = out[:, target].sum()
            loss.backward()
            grad = t.grad.detach().cpu().numpy()
            return grad.squeeze()
        else:
            # NES gradient estimate for sklearn models (black-box approach)
            return self._nes_gradient(X, target)

    def _nes_gradient(self, X: np.ndarray, target: int = 1,
                      n_samples: int = 50, sigma: float = 1e-3) -> np.ndarray:
        """Estimate gradient via NES (Natural Evolution Strategies).

        Uses symmetric finite differences with random Gaussian perturbations.
        Reference: "Natural Evolution Strategies for Gradient-Free Adversaries"
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        n, d = X.shape
        grad_est = np.zeros_like(X)

        for _ in range(n_samples):
            noise = np.random.randn(n, d)
            X_plus = X + sigma * noise
            X_minus = X - sigma * noise

            score_plus = self._attack_score(X_plus, target)
            score_minus = self._attack_score(X_minus, target)

            grad_est += ((score_plus - score_minus) / (2 * sigma))[:, None] * noise

        grad_est /= n_samples
        if squeeze:
            grad_est = grad_est[0]
        return grad_est

    def _attack_score(self, X: np.ndarray, target: int) -> np.ndarray:
        """Score function for NES: higher = more likely to be class `target`."""
        probs = self.predict_proba(X)
        return probs[:, target]

    @property
    def input_shape(self) -> int:
        return self.base.input_shape if self.is_torch else None


# ---------------------------------------------------------------------------
# CNN1D architecture (PyTorch)
# ---------------------------------------------------------------------------

class _CNN1D:
    """1D-CNN over the feature vector — treats features as a sequence.

    Implemented as a lazy-load wrapper so torch is only imported when CNN1D is used.
    Used by both E1 (clean) and E8 (robust-adversarial training).
    """

    def __new__(cls, num_features: int, num_classes: int = 2):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _CNN1DImpl(nn.Module):
            def __init__(self_, num_features, num_classes=2):
                super().__init__()
                self_.input_shape = num_features
                self_.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
                self_.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self_.pool = nn.AdaptiveAvgPool1d(1)
                self_.fc = nn.Linear(64, num_classes)
                self_.dropout = nn.Dropout(0.3)

            def forward(self_, x):
                x = x.unsqueeze(1)
                x = F.relu(self_.conv1(x))
                x = F.relu(self_.conv2(x))
                x = self_.pool(x).squeeze(-1)
                x = self_.dropout(x)
                return self_.fc(x)

        return _CNN1DImpl(num_features, num_classes)
