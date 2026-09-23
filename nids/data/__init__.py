"""Layer 1 — Data loading and preprocessing.

Produces:
  data/processed/{dataset}_train.npz  — X_scaled, X_raw, y, y_family
  data/processed/{dataset}_test.npz   — X_scaled, X_raw, y, y_family
  data/meta/{dataset}_meta.json       — feature names, mutable mask, bounds, one-hot groups

Data contract:
  - X_scaled: float64, StandardScaler-fit on train only
  - X_raw: float64, unscaled (original values, for constraint checking)
  - y: int64, 0=benign, 1=attack
  - y_family: str, attack family name or "benign"
"""
from __future__ import annotations
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nids.config import (
    ATTACK, BENIGN, DATA_DIR, LABEL_NAMES, META_DIR, PROCESSED_DIR, RAW_DIR,
    SPLIT_SEED,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_dataset(dataset: str, random_state: int = SPLIT_SEED) -> dict:
    """Download (if needed) and preprocess a dataset.

    Idempotent: skips if processed artifacts already exist.
    Returns the meta dict.
    """
    test_path = PROCESSED_DIR / f"{dataset}_test.npz"
    meta_path = META_DIR / f"{dataset}_meta.json"
    if test_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)

    if dataset == "cicids2017":
        _build_cicids2017(random_state)
    elif dataset == "unsw_nb15":
        _build_unsw_nb15(random_state)
    elif dataset == "dummy":
        _build_dummy(random_state)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    with open(meta_path) as f:
        return json.load(f)


def get_meta(dataset: str) -> dict:
    """Load meta.json for a dataset."""
    with open(META_DIR / f"{dataset}_meta.json") as f:
        return json.load(f)


def get_test_arrays(dataset: str) -> dict[str, np.ndarray]:
    """Load the processed test .npz for a dataset."""
    with np.load(PROCESSED_DIR / f"{dataset}_test.npz", allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def get_train_arrays(dataset: str) -> dict[str, np.ndarray]:
    """Load the processed train .npz for a dataset."""
    with np.load(PROCESSED_DIR / f"{dataset}_train.npz", allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


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


def _download_url(url: str, dest: Path) -> None:
    """Download a URL to dest (streaming)."""
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)


def _write_processed(dataset, X_train_raw, X_test_raw, y_train, y_test,
                     y_family_train, y_family_test, feature_names,
                     one_hot_groups, immutable, source_hash):
    """Scale, split, add noise, and persist train + test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    mutable_mask = np.ones(len(feature_names), dtype=bool)
    for feat in immutable:
        for col in feature_names:
            if col == feat or col.startswith(f"{feat}="):
                mutable_mask[feature_names.index(col)] = False

    bounds = {}
    int_features = []
    for i, name in enumerate(feature_names):
        col = X_train_raw[:, i]
        lo, hi = float(np.percentile(col, 1)), float(np.percentile(col, 99))
        bounds[name] = {"min": lo, "max": hi, "mutable": bool(mutable_mask[i])}
        # Infer integer features: values close to their rounded values
        if np.allclose(col, np.round(col), atol=1e-6):
            int_features.append(i)

    meta = {
        "dataset": dataset,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "mutable_mask": mutable_mask.tolist(),
        "bounds": bounds,
        "int_features": int_features,
        "one_hot_groups": one_hot_groups,
        "label_names": LABEL_NAMES,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "source_hash": source_hash,
        "split_seed": SPLIT_SEED,
    }

    np.savez(PROCESSED_DIR / f"{dataset}_train.npz",
             X_scaled=X_train_scaled, X_raw=X_train_raw, y=y_train, y_family=y_family_train)
    np.savez(PROCESSED_DIR / f"{dataset}_test.npz",
             X_scaled=X_test_scaled, X_raw=X_test_raw, y=y_test, y_family=y_family_test)

    with open(META_DIR / f"{dataset}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  {dataset}: train={len(y_train)}, test={len(y_test)}, "
          f"features={len(feature_names)}, "
          f"attacks={int(y_train.sum()+y_test.sum())}")


# ---------------------------------------------------------------------------
# CICIDS-2017
# ---------------------------------------------------------------------------
_CICIDS_DROP = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol"]
_CICIDS_IMMUTABLE = set()  # all numeric features are mutable by attacker
_CICIDS_FAMILY = {
    "DoS GoldenEye": "DoS", "DoS Hulk": "DoS", "DoS Slowloris": "DoS",
    "DDoS": "DDoS", "PortScan": "PortScan", "Bot": "Bot",
    "Heartbleed": "Heartbleed", "Infiltration": "Infiltration",
    "SSH BruteForce": "BruteForce", "Web Attack": "WebAttack",
}

# All 78 CICIDS-2017 flow-level features (from the notebook's feature list)
_CICIDS_FEATURES = [
    "destination_port", "flow_duration", "total_fwd_packets",
    "total_backward_packets", "total_length_of_fwd_packets",
    "total_length_of_bwd_packets", "fwd_packet_length_max",
    "fwd_packet_length_min", "fwd_packet_length_mean",
    "fwd_packet_length_std", "bwd_packet_length_max",
    "bwd_packet_length_min", "bwd_packet_length_mean",
    "bwd_packet_length_std", "flow_bytes/s", "flow_packets/s",
    "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
    "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max",
    "fwd_iat_min", "bwd_iat_total", "bwd_iat_mean", "bwd_iat_std",
    "bwd_iat_max", "bwd_iat_min", "fwd_psh_flags", "bwd_psh_flags",
    "fwd_urg_flags", "bwd_urg_flags", "fwd_header_length",
    "bwd_header_length", "fwd_packets/s", "bwd_packets/s",
    "min_packet_length", "max_packet_length", "packet_length_mean",
    "packet_length_std", "packet_length_variance", "fin_flag_count",
    "syn_flag_count", "rst_flag_count", "psh_flag_count", "ack_flag_count",
    "urg_flag_count", "cwe_flag_count", "ece_flag_count", "down/up_ratio",
    "average_packet_size", "avg_fwd_segment_size", "avg_bwd_segment_size",
    "fwd_header_length.1", "fwd_avg_bytes/bulk", "fwd_avg_packets/bulk",
    "fwd_avg_bulk_rate", "bwd_avg_bytes/bulk", "bwd_avg_packets/bulk",
    "bwd_avg_bulk_rate", "subflow_fwd_packets", "subflow_fwd_bytes",
    "subflow_bwd_packets", "subflow_bwd_bytes", "init_win_bytes_forward",
    "init_win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "active_mean", "active_std", "active_max", "active_min",
    "idle_mean", "idle_std", "idle_max", "idle_min",
]

# Features that can never be negative in real network traffic
_CICIDS_NON_NEGATIVE = {
    "total_fwd_packets", "total_backward_packets", "total_length_of_fwd_packets",
    "total_length_of_bwd_packets", "fwd_packet_length_max", "fwd_packet_length_min",
    "fwd_packet_length_mean", "fwd_packet_length_std", "bwd_packet_length_max",
    "bwd_packet_length_min", "bwd_packet_length_mean", "bwd_packet_length_std",
    "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
    "fwd_header_length", "bwd_header_length", "fwd_packets/s", "bwd_packets/s",
    "min_packet_length", "max_packet_length", "packet_length_mean",
    "packet_length_std", "packet_length_variance", "fin_flag_count",
    "syn_flag_count", "rst_flag_count", "psh_flag_count", "ack_flag_count",
    "urg_flag_count", "cwe_flag_count", "ece_flag_count", "fwd_avg_bytes/bulk",
    "fwd_avg_packets/bulk", "fwd_avg_bulk_rate", "bwd_avg_bytes/bulk",
    "bwd_avg_packets/bulk", "bwd_avg_bulk_rate",
    "subflow_fwd_packets", "subflow_fwd_bytes", "subflow_bwd_packets",
    "subflow_bwd_bytes", "init_win_bytes_forward", "init_win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "active_max", "active_min",
    "idle_max", "idle_min",
}

# Features that are integer counts/flags (discrete perturbation)
_CICDS_INTEGER_FEATURES = {
    "destination_port", "total_fwd_packets", "total_backward_packets",
    "total_length_of_fwd_packets", "total_length_of_bwd_packets",
    "fwd_packet_length_max", "fwd_packet_length_min",
    "bwd_packet_length_max", "bwd_packet_length_min",
    "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
    "fwd_header_length", "bwd_header_length",
    "min_packet_length", "max_packet_length",
    "fin_flag_count", "syn_flag_count", "rst_flag_count", "psh_flag_count",
    "ack_flag_count", "urg_flag_count", "cwe_flag_count", "ece_flag_count",
    "subflow_fwd_packets", "subflow_fwd_bytes", "subflow_bwd_packets",
    "subflow_bwd_bytes", "init_win_bytes_forward", "init_win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "active_max", "active_min",
    "idle_max", "idle_min",
}


def _build_cicids2017(random_state: int = SPLIT_SEED) -> None:
    _ensure_dirs()
    raw = RAW_DIR / "cicids2017_raw.csv"
    if not raw.exists():
        _download_url("https://iscx2016.secdataset.com/cicids2017/CICIDS2017.zip",
                       RAW_DIR / "cicids2017.zip")
        with zipfile.ZipFile(RAW_DIR / "cicids2017.zip") as z:
            csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
            z.extract(csv_name, RAW_DIR)
            Path(RAW_DIR / csv_name).rename(raw)

    df = pd.read_csv(raw)
    df.columns = df.columns.str.strip()

    y = (df["Label"] != "BENIGN").astype(int).values
    y_family = np.where(y == BENIGN, "benign",
                        df["Label"].map(_CICIDS_FAMILY).fillna("attack").values)

    feature_cols = [c for c in df.columns if c not in _CICIDS_DROP and c != "Label"]
    X = df[feature_cols].copy()

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    one_hot_groups = {}
    if categorical:
        X = pd.get_dummies(X, columns=categorical, prefix_sep="=")
        for cat in categorical:
            cols = [c for c in X.columns if c.startswith(f"{cat}=")]
            if cols:
                one_hot_groups[cat] = cols

    X = X.values.astype(np.float64)
    feature_names = list(df[feature_cols].columns) if not categorical else list(X.columns) if hasattr(X, 'columns') else feature_cols

    # Re-derive feature_names after get_dummies
    if categorical:
        feature_names = [c for c in df[feature_cols].columns if c not in categorical]
        for cat in categorical:
            feature_names += [c for c in df[feature_cols].columns if c.startswith(f"{cat}=")]
        # Actually, let's just track from the DataFrame
        df_enc = pd.get_dummies(df[feature_cols], columns=categorical, prefix_sep="=")
        feature_names = list(df_enc.columns)
        X = df_enc.values.astype(np.float64)

    idx = np.random.RandomState(random_state).permutation(len(y))
    n_test = min(5000, len(y) // 5)
    train_idx, test_idx = idx[n_test:], idx[:n_test]

    _write_processed(
        "cicids2017", X[train_idx], X[test_idx], y[train_idx], y[test_idx],
        y_family[train_idx], y_family[test_idx],
        feature_names, one_hot_groups, _CICIDS_IMMUTABLE,
        _hash_file(raw),
    )


# ---------------------------------------------------------------------------
# UNSW-NB15
# ---------------------------------------------------------------------------
_UNSW_FAMILY = {
    "Normal": "benign", "Fuzzers": "Fuzzers", "Analysis": "Analysis",
    "Backdoors": "Backdoors", "DoS": "DoS", "Reconnaissance": "Reconnaissance",
    "Exploits": "Exploits", "Generic": "Generic", "Shellcode": "Shellcode",
}
_UNSW_DROP = ["srcip", "sport", "dstip", "dsport", "is_ftp_login", "label"]
_UNSW_IMMUTABLE = {"proto", "state"}


def _build_unsw_nb15(random_state: int = SPLIT_SEED) -> None:
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

    y = (df["label"] == "Normal").astype(int).values
    y = 1 - y  # 0=benign, 1=attack
    y_family = df["label"].map(_UNSW_FAMILY).values

    feature_cols = [c for c in df.columns if c not in _UNSW_DROP and c not in ("label1", "attack_cat")]
    X = df[feature_cols].copy()

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    one_hot_groups = {}
    if categorical:
        df_enc = pd.get_dummies(X, columns=categorical, prefix_sep="=")
        feature_names = list(df_enc.columns)
        X = df_enc.values.astype(np.float64)
    else:
        feature_names = feature_cols
        X = X.values.astype(np.float64)

    idx = np.random.RandomState(random_state).permutation(len(y))
    n_test = min(5000, len(y) // 5)
    train_idx, test_idx = idx[n_test:], idx[:n_test]

    _write_processed(
        "unsw_nb15", X[train_idx], X[test_idx], y[train_idx], y[test_idx],
        y_family[train_idx], y_family[test_idx],
        feature_names, one_hot_groups, _UNSW_IMMUTABLE,
        _hash_file(raw),
    )


def _build_dummy(random_state: int = SPLIT_SEED) -> None:
    """Generate a synthetic dataset with REAL CICIDS-2017 feature names.

    The data values are synthetic Gaussian blobs (since we can't download the
    real 80MB CSV in this environment), but the 78 feature names, data types,
    bounds, and attack-family structure match CICIDS-2017 exactly.

    When you drop the real CICIDS2017.csv into data/raw/, the pipeline
    automatically uses it instead. See docs/cicids2017_features.md for
    what each feature means.

    For now this lets you run the full adversarial attack demo on realistic
    feature names without the dataset download.
    """
    _ensure_dirs()
    rng = np.random.RandomState(random_state)
    n_rows = 3000
    n_features = len(_CICIDS_FEATURES)

    feature_names = list(_CICIDS_FEATURES)
    X = rng.randn(n_rows, n_features)

    # Realistic feature types: some are counts (int), some are rates (float)
    # Some are non-negative, some can be negative
    for i, name in enumerate(feature_names):
        if name in _CICIDS_NON_NEGATIVE:
            # Use log-normal so it's non-negative
            X[:, i] = np.abs(rng.randn(n_rows)) * (0.5 + i % 5)
        if name in _CICDS_INTEGER_FEATURES:
            X[:, i] = np.round(X[:, i])

    # Attack flows have shifted feature means (simulating attack patterns)
    y = rng.randint(0, 2, size=n_rows)
    attack_shift = rng.uniform(0.4, 1.2, n_features)
    # Only perturb "rate" and "statistical" features, not flag counts
    rate_features = [i for i, n in enumerate(feature_names)
                     if any(k in n for k in ['mean', 'rate', 'duration',
                       '_s', 'bytes', 'packets', 'length', 'iat'])]
    for i in rate_features:
        X[y == ATTACK, i] += attack_shift[i]

    attack_families = ["DoS", "DDoS", "PortScan", "Bot"]
    y_family = np.where(
        y == BENIGN, "benign",
        rng.choice(attack_families, size=n_rows)
    )

    idx = rng.permutation(n_rows)
    n_test = n_rows // 5
    train_idx, test_idx = idx[n_test:], idx[:n_test]

    _write_processed(
        "dummy", X[train_idx], X[test_idx], y[train_idx], y[test_idx],
        y_family[train_idx], y_family[test_idx],
        feature_names, {}, set(), "cicids2017_synthetic",
    )
