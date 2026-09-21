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
    ARTIFACTS_DIR, CHECKPOINTS_DIR, LABEL_NAMES, META_DIR,
)
from nids.data import get_meta


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
    # sklearn / xgboost
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
        # parse filename: {dataset}_{model}{_e8?}.{ext}
        stem = p.stem.replace(".joblib", "").replace(".pt", "")
        parts = stem.split("_")
        # dataset is first two tokens joined (cicids2017 / unsw_nb15)
        dataset = parts[0] + ("_" + parts[1] if parts[1] in ("nb15",) else "")
        variant = "robust" if "e8" in p.name else "clean"
        model_name = "_".join(parts[1:]).replace("e8", "").rstrip("_")
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    import xgboost as xgb
    import torch

    meta = get_meta(dataset)
    n_features = meta["n_features"]

    if model_name == "lr":
        base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif model_name == "rf":
        base = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=42, n_jobs=-1)
    elif model_name == "xgb":
        base = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 eval_metric="logloss", random_state=42, n_jobs=-1)
    elif model_name == "mlp":
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
            ckpt = torch.load(path, map_location="cpu", state_dict_key=None)
            if "state_dict" in ckpt:
                self.base.load_state_dict(ckpt["state_dict"])
            elif "model_state_dict" in ckpt:
                self.base.load_state_dict(ckpt["model_state_dict"])
            else:
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
                # ensure 2-column
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
        """Return the attack-relevant score: P(attack) (or logit for torch)."""
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
        """Compute d(score)/d(X) for gradient-based attacks. Only for torch models."""
        if not self.is_torch:
            raise NotImplementedError("Gradient only available for CNN1D")
        import torch
        self.base.eval()
        t = torch.as_tensor(X, dtype=torch.float32, requires_grad=True)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        out = self.base(t)
        # Attack target: push probability of ATTACK class up (for evasion)
        if out.shape[1] == 1:
            loss = out.sum()
        else:
            loss = out[:, target].sum()
        loss.backward()
        grad = t.grad.detach().cpu().numpy()
        return grad.squeeze()

    @property
    def input_shape(self) -> int:
        return self.base.input_shape if self.is_torch else None


ATTACK_IDX = 1  # index of "attack" class in predict_proba output


# ---------------------------------------------------------------------------
# CNN1D architecture (PyTorch)
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CNN1D(nn.Module):
    """1D-CNN over the feature vector — treats features as a sequence.

    Used by both E1 (clean) and E8 (robust-adversarial training).
    """

    def __init__(self, num_features: int, num_classes: int = 2):
        super().__init__()
        self.input_shape = num_features
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (batch, num_features) -> (batch, 1, num_features)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)
