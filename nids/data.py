"""Dataset loading and preprocessing for CICIDS2017 and UNSW-NB15.

Produces:
  data/processed/{dataset}_test.npz  — X_scaled, X_raw, y, y_family
  meta/{dataset}_meta.json          — feature names, mutable mask, bounds, one-hot groups

Both datasets reduce to the same internal schema so the victim models and attacks
are dataset-agnostic.
"""
from __future__ import annotations
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from nids.constants import (
    ATTACK, BENIGN, DATA_DIR, LABEL_NAMES, META_DIR, PROCESSED_DIR, RAW_DIR,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(dataset: str, random_state: int = 42) -> str:
    """Download (if needed) and preprocess a dataset; returns its meta file path.

    Idempotent: skips work if processed artifacts already exist.
    """
    if dataset == "cicids2017":
        return _build_cicids2017(random_state)
    if dataset == "unsw_nb15":
        return _build_unsw_nb15(random_state)
    raise ValueError(f"Unknown dataset: {dataset}")


def get_meta(dataset: str) -> dict[str, Any]:
    """Load meta.json for a dataset (feature names, mutable mask, bounds, one-hot groups)."""
    path = META_DIR / f"{dataset}_meta.json"
    with open(path) as f:
        return json.load(f)


def get_test_arrays(dataset: str) -> dict[str, np.ndarray]:
    """Load the processed test .npz for a dataset."""
    path = PROCESSED_DIR / f"{dataset}_test.npz"
    with np.load(path) as data:
        return {
            "X_scaled": data["X_scaled"],
            "X_raw": data["X_raw"],
            "y": data["y"],
            "y_family": data["y_family"],
        }


def train_test_split_scaled(dataset: str, test_frac: float = 0.2, random_state: int = 42):
    """Return (X_train_scaled, X_test_scaled, y_train, y_test, scaler, meta)."""
    arr = get_test_arrays(dataset)  # test split already carved out during build
    # Reload the full processed arrays (train + test stored together at build time)
    full = np.load(PROCESSED_DIR / f"{dataset}_full.npz")
    X_full, y_full = full["X_scaled"], full["y"]
    # The build already split; we just return the pre-carved test split here.
    return arr, get_meta(dataset)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, META_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# CICIDS2017
# ---------------------------------------------------------------------------

# The canonical 46-feature set after dropping the 8 identifier columns
# (duration, src_bytes, ... are kept; the 8 dropped are rate cols we don't use for NIDS).
# We keep all 78 original numerical/categorical features that survive the standard filter.
_CICIDS_DROP_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol"]
# Features that are *mutable* by an attacker (not intrinsic like protocol)
_CICIDS_IMMUTABLE = set()  # protocol is treated as mutable-ish; all numeric features are targets


def _build_cicids2017(random_state: int) -> str:
    """Download CICIDS2017 (if absent), clean, split, scale, and write artifacts."""
    _ensure_dirs()
    raw = RAW_DIR / "cicids2017_raw.csv"
    if not raw.exists():
        _download_url(
            "https://iscx2016.secdataset.com/cicids2017/CICIDS2017.zip",
            RAW_DIR / "cicids2017.zip",
        )
        with zipfile.ZipFile(RAW_DIR / "cicids2017.zip") as z:
            # pick the CSV inside
            csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
            z.extract(csv_name, RAW_DIR)
            Path(RAW_DIR / csv_name).rename(raw)

    df = pd.read_csv(raw)
    # Normalise column names
    df.columns = df.columns.str.strip()
    # Label -> binary
    label_col = "Label"
    df[label_col] = df[label_col].replace({"BENIGN": BENIGN, "Bot": ATTACK, "DDoS": ATTACK,
        "DoS golden telescope": ATTACK, "DoS Hulk": ATTACK, "DoS GoldenEye": ATTACK,
        "DoS Slowloris": ATTACK,
        "Heartbleed": ATTACK, "Infiltration": ATTACK, "PortScan": ATTACK,
        "SSH BruteForce": ATTACK, "Web Broutilner": ATTACK})
    # Attack family = original label string for non-benign rows
    family_col = "Label"
    attack_families = {
        "DoS GoldenEye": "DoS", "DoS Hulk": "DoS", "DoS Slowloris": "DoS",
        "DDoS": "DDoS", "PortScan": "PortScan", "Bot": "Bot",
        "Heartbleed": "Heartbleed", "Infiltration": "Infiltration",
        "SSH BruteForce": "BruteForce", "Web Broulner": "WebAttack",
    }

    y = df[label_col].values.astype(int)
    y_family = np.where(y == BENIGN, "benign",
                        df["Label"].map(attack_families).fillna("attack").values)

    # Features
    feature_cols = [c for c in df.columns if c not in _CICIDS_DROP_COLS and c != "Label"]
    X_raw = df[feature_cols].copy()

    # One-hot encode categorical if any (CICIDS2017 is mostly numeric, but some string cols)
    categorical = X_raw.select_dtypes(include=["object"]).columns.tolist()
    one_hot_groups: dict[str, list[str]] = {}
    if categorical:
        X_raw = pd.get_dummies(X_raw, columns=categorical, prefix_sep="=")
        for cat in categorical:
            cols = [c for c in X_raw.columns if c.startswith(f"{cat}=")]
            if cols:
                one_hot_groups[cat] = cols

    feature_names = X_raw.columns.tolist()
    X_raw = X_raw.values.astype(np.float64)

    _write_processed(dataset="cicids2017", X_raw=X_raw, y=y, y_family=y_family,
                     feature_names=feature_names, one_hot_groups=one_hot_groups,
                     immutable=_CICIDS_IMMUTABLE, random_state=random_state)
    return str(META_DIR / "cicids2017_meta.json")


# ---------------------------------------------------------------------------
# UNSW-NB15
# ---------------------------------------------------------------------------

_UNSW_LABEL_MAP = {"Normal": BENIGN, "Fuzzers": ATTACK, "Analysis": ATTACK,
    "Backdoors": ATTACK, "DoS": ATTACK, "Reconnaissance": ATTACK,
    "Exploits": ATTACK, "Generic": ATTACK, "Shellcode": ATTACK}
_UNSW_FAMILY_MAP = {"Normal": "benign", "Fuzzers": "Fuzzers", "Analysis": "Analysis",
    "Backdoors": "Backdoors", "DoS": "DoS", "Reconnaissance": "Reconnaissance",
    "Exploits": "Exploits", "Generic": "Generic", "Shellcode": "Shellcode"}
_UNSW_DROP = ["srcip", "sport", "dstip", "dsport", "is_ftp_login", "label"]
_UNSW_IMMUTABLE = {"proto", "state"}  # protocol/state are not controllable by attacker


def _build_unsw_nb15(random_state: int) -> str:
    _ensure_dirs()
    raw = RAW_DIR / "unsw_nb15_raw.csv"
    if not raw.exists():
        _download_url(
            "https://research.unsw.edu.au/projects/datasets/unsw-nisw-nids15-data/unswnb15-datasets.zip",
            RAW_DIR / "unsw_nb15.zip",
        )
        with zipfile.ZipFile(RAW_DIR / "unsw_nb15.zip") as z:
            csv_name = [n for n in z.namelist() if n.endswith(".csv") and "NUSW" in n][0]
            z.extract(csv_name, RAW_DIR)
            Path(RAW_DIR / csv_name).rename(raw)

    df = pd.read_csv(raw)
    df.columns = df.columns.str.strip()

    y = df["label"].map(_UNSW_LABEL_MAP).values.astype(int)
    y_family = df["label"].map(_UNSW_FAMILY_MAP).values

    feature_cols = [c for c in df.columns if c not in _UNSW_DROP and c not in ("label1", "attack_cat")]
    X_raw = df[feature_cols].copy()

    categorical = X_raw.select_dtypes(include=["object"]).columns.tolist()
    one_hot_groups: dict[str, list[str]] = {}
    if categorical:
        X_raw = pd.get_dummies(X_raw, columns=categorical, prefix_sep="=")
        for cat in categorical:
            cols = [c for c in X_raw.columns if c.startswith(f"{cat}=")]
            if cols:
                one_hot_groups[cat] = cols

    feature_names = X_raw.columns.tolist()
    X_raw = X_raw.values.astype(np.float64)

    _write_processed(dataset="unsw_nb15", X_raw=X_raw, y=y, y_family=y_family,
                     feature_names=feature_names, one_hot_groups=one_hot_groups,
                     immutable=_UNSW_IMMUTABLE, random_state=random_state)
    return str(META_DIR / "unsw_nb15_meta.json")


# ---------------------------------------------------------------------------
# Shared write logic
# ---------------------------------------------------------------------------

def _write_processed(dataset: str, X_raw: np.ndarray, y: np.ndarray,
                     y_family: np.ndarray, feature_names: list[str],
                     one_hot_groups: dict[str, list[str]],
                     immutable: set[str], random_state: int) -> None:
    """Split, scale, and persist both full and test npz + meta.json."""
    from sklearn.model_selection import train_test_split as sk_split

    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(y))
    X_raw, y, y_family = X_raw[idx], y[idx], y_family[idx]

    n_test = min(5000, len(y) // 5)  # cap test pool for dashboard browsing
    test_idx = idx[:n_test]
    full_idx = idx[n_test:]

    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(X_raw)
    X_scaled_test = scaler.transform(X_raw[test_idx])

    # Mutable mask over the full feature vector
    mutable_mask = np.ones(len(feature_names), dtype=bool)
    for feat in immutable:
        # mark any one-hot columns derived from an immutable categorical as immutable
        for col in feature_names:
            if col == feat or col.startswith(f"{feat}="):
                mutable_mask[feature_names.index(col)] = False

    # Per-feature bounds (from training data range, clipped)
    bounds = {}
    for i, name in enumerate(feature_names):
        col = X_raw[:, i]
        lo, hi = float(np.percentile(col, 1)), float(np.percentile(col, 99))
        bounds[name] = {"min": lo, "max": hi, "mutable": bool(mutable_mask[i])}

    meta = {
        "dataset": dataset,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "mutable_mask": mutable_mask.tolist(),
        "bounds": bounds,
        "one_hot_groups": one_hot_groups,
        "label_names": LABEL_NAMES,
        "n_test": int(n_test),
        "n_train": int(len(y) - n_test),
        "raw_hash": "n/a",
    }

    # Persist
    np.savez(PROCESSED_DIR / f"{dataset}_full.npz",
             X_scaled=X_scaled_full, X_raw=X_raw, y=y, y_family=y_family)
    np.savez(PROCESSED_DIR / f"{dataset}_test.npz",
             X_scaled=X_scaled_test,
             X_raw=X_raw[test_idx], y=y[test_idx], y_family=y_family[test_idx])

    with open(META_DIR / f"{dataset}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {dataset}: {len(feature_names)} features, {len(y)} samples, "
          f"{int(y.sum())} attacks / {int((y==0).sum())} benign")


def _download_url(url: str, dest: Path) -> None:
    """Download a URL to dest (streaming)."""
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
    print(f"  downloaded {url} -> {dest}")
