"""Layer 2 — Model definitions and VictimModel wrapper.

Models:
  lr     — Logistic Regression (sklearn)
  rf     — Random Forest (sklearn)
  xgb    — XGBoost (sklearn API)
  mlp    — Multi-Layer Perceptron (PyTorch)
  cnn1d  — 1D CNN (PyTorch)

VictimModel: wraps any of the above, provides a unified API:
  predict_proba_scaled(x_scaled) -> (n, 2) [P(benign), P(attack)]
  predict_proba(x_raw) -> same (auto-scales if scaler set)
  gradient_scaled(x_scaled, target) -> (n, d) gradient in scaled space
  predict(x_raw) -> (n,) int labels
  is_torch -> bool

All gradient/predict operations for attacks run in SCALED feature space
(same space the model uses). This ensures eps is consistent between
attack and validity check.
"""
from __future__ import annotations
import numpy as np
import joblib
from pathlib import Path

from nids.config import CHECKPOINTS_DIR, LABEL_NAMES, MODEL_SEEDS


# ---------------------------------------------------------------------------
# Factory + checkpoint I/O
# ---------------------------------------------------------------------------
def make_model(name: str, n_features: int, seed: int = MODEL_SEEDS[0]):
    """Create a fresh VictimModel instance by name."""
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        return VictimModel(
            LogisticRegression(max_iter=2000, random_state=seed, C=0.5),
            is_torch=False, n_features=n_features)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return VictimModel(
            RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
            is_torch=False, n_features=n_features)
    if name == "xgb":
        import xgboost as xgb
        return VictimModel(
            xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                              random_state=seed, eval_metric="logloss",
                              device="cpu"),
            is_torch=False, n_features=n_features)
    if name == "ensemble":
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        import xgboost as xgb

        ensemble = VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=2000, random_state=seed, C=0.5)),
                ("rf", RandomForestClassifier(
                    n_estimators=200, random_state=seed, n_jobs=-1,
                )),
                ("xgb", xgb.XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    random_state=seed, eval_metric="logloss", device="cpu",
                )),
            ],
            voting="soft",
            n_jobs=1,
        )
        return VictimModel(ensemble, is_torch=False, n_features=n_features)
    if name == "mlp":
        from nids.models.mlp import MLP
        return VictimModel(MLP(n_features), is_torch=True, n_features=n_features)
    if name == "cnn1d":
        from nids.models.cnn1d import CNN1D
        return VictimModel(CNN1D(n_features), is_torch=True, n_features=n_features)
    raise ValueError(f"Unknown model: {name}")


def load_model(dataset: str, name: str, seed: int = MODEL_SEEDS[0]) -> VictimModel:
    """Load a saved VictimModel checkpoint from disk."""
    ckpt_dir = CHECKPOINTS_DIR / dataset
    for ext in [".joblib", ".pt"]:
        path = ckpt_dir / f"{name}_s{seed}{ext}"
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(f"No checkpoint for {dataset}/{name}_s{seed}")


def save_model(vm: "VictimModel", dataset: str, name: str, seed: int) -> None:
    """Save a VictimModel to disk."""
    ckpt_dir = CHECKPOINTS_DIR / dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{name}_s{seed}.joblib"
    joblib.dump(vm, path)


# ---------------------------------------------------------------------------
# VictimModel wrapper
# ---------------------------------------------------------------------------
class VictimModel:
    """Unified wrapper for sklearn + torch models.

    Working space is SCALED features (what the model actually sees).
    predict_proba_scaled / gradient_scaled operate directly in scaled space.
    predict_proba / gradient accept raw features and auto-scale.

    For torch models there is no scaler — scaled == raw.
    """

    def __init__(self, base, is_torch: bool, n_features: int):
        self.base = base
        self.is_torch = is_torch
        self.n_features = n_features
        self.scaler_ = None
        self.feature_names = None
        self.mutable_mask = None
        self.meta = {}
        self.metrics = {}

    # -- Scaler --
    def set_scaler(self, scaler):
        self.scaler_ = scaler

    def to_scaled(self, x_raw: np.ndarray) -> np.ndarray:
        if self.scaler_ is not None:
            return self.scaler_.transform(x_raw)
        return x_raw

    def to_raw(self, x_scaled: np.ndarray) -> np.ndarray:
        if self.scaler_ is not None:
            return self.scaler_.inverse_transform(x_scaled)
        return x_scaled

    # -- Prediction (scaled space, the model's native space) --
    def predict_proba_scaled(self, x_scaled: np.ndarray) -> np.ndarray:
        """(n, d) scaled features -> (n, 2) [P(benign), P(attack)]."""
        if self.is_torch:
            import torch
            with torch.no_grad():
                t = torch.as_tensor(x_scaled, dtype=torch.float32)
                logits = self.base(t)
                logits = logits.detach().cpu().numpy()
            if logits.shape[1] == 2:
                from scipy.special import softmax
                return softmax(logits, axis=1)
            else:
                p_attack = 1.0 / (1.0 + np.exp(-logits.ravel()))
                return np.column_stack([1 - p_attack, p_attack])
        else:
            return self.base.predict_proba(x_scaled)

    def predict_proba(self, x_raw: np.ndarray) -> np.ndarray:
        """(n, d) raw features -> (n, 2). Auto-scales for sklearn."""
        return self.predict_proba_scaled(self.to_scaled(x_raw))

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        x_scaled = self.to_scaled(x_raw)
        if self.is_torch:
            import torch
            with torch.no_grad():
                t = torch.as_tensor(x_scaled, dtype=torch.float32)
                return self.base(t).argmax(dim=1).cpu().numpy()
        else:
            return self.base.predict(x_scaled)

    # -- Gradient (scaled space) --
    def gradient_scaled(self, x_scaled: np.ndarray, target: int = 0) -> np.ndarray:
        """Gradient of P(target) w.r.t. SCALED features. (n, d) array.

        This is the primary gradient used by attacks.
        """
        n = x_scaled.shape[0]
        d = x_scaled.shape[1]
        eps_fd = 1e-4

        if self.is_torch:
            import torch
            grads = []
            for i in range(n):
                # as_tensor() does not accept requires_grad; create a leaf tensor
                # and request only the input gradient (model weights stay untouched).
                t = torch.as_tensor(
                    x_scaled[i:i+1], dtype=torch.float32,
                ).detach().clone().requires_grad_(True)
                logits = self.base(t)
                if logits.shape[1] == 2:
                    prob = torch.softmax(logits, dim=1)[0, target]
                else:
                    sig = torch.sigmoid(logits)[0]
                    prob = sig if target == 1 else 1 - sig
                (g_tensor,) = torch.autograd.grad(prob, t)
                g = g_tensor[0].detach().cpu().numpy().astype(np.float64)
                grads.append(g)
            return np.array(grads)
        else:
            # Finite differences in scaled space (for sklearn)
            grads = []
            for i in range(n):
                g = np.zeros(d)
                for j in range(d):
                    x_p = x_scaled[i:i+1].copy()
                    x_m = x_scaled[i:i+1].copy()
                    x_p[0, j] += eps_fd
                    x_m[0, j] -= eps_fd
                    p_p = self.predict_proba_scaled(x_p)[0, target]
                    p_m = self.predict_proba_scaled(x_m)[0, target]
                    g[j] = (p_p - p_m) / (2 * eps_fd)
                grads.append(g)
            return np.array(grads)

    def gradient(self, x_raw: np.ndarray, target: int = 0) -> np.ndarray:
        """Gradient of P(target) w.r.t. RAW features.

        Convenience: scales x_raw, calls gradient_scaled, chain-rules back.
        """
        x_scaled = self.to_scaled(x_raw)
        grad_s = self.gradient_scaled(x_scaled, target)
        if self.scaler_ is not None:
            sigma = self.scaler_.scale_[None, :]
            return grad_s / sigma  # chain rule: dP/dx_raw = dP/dx_scaled * (1/sigma)
        return grad_s

    # -- Serialization --
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
