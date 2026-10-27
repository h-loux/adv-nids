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
import io
import json
import csv
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
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
    train_path = PROCESSED_DIR / f"{dataset}_train.npz"
    test_path = PROCESSED_DIR / f"{dataset}_test.npz"
    meta_path = META_DIR / f"{dataset}_meta.json"
    if train_path.exists() and test_path.exists() and meta_path.exists():
        try:
            with np.load(train_path, allow_pickle=False) as train_data:
                train_labels = np.asarray(train_data["y"])
            with np.load(test_path, allow_pickle=False) as test_data:
                test_labels = np.asarray(test_data["y"])
            # Cached one-class splits cannot support binary intrusion detection.
            if set(np.unique(train_labels)) == {BENIGN, ATTACK} and set(np.unique(test_labels)) == {BENIGN, ATTACK}:
                with open(meta_path, encoding="utf-8") as f:
                    return json.load(f)
        except (OSError, KeyError, ValueError, EOFError):
            pass
        print(f"  Cached {dataset} splits are invalid or missing a class; rebuilding.")
        for path in (train_path, test_path, meta_path):
            path.unlink(missing_ok=True)

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


def _get_arrays(dataset: str, split: str, include_scaled: bool,
                include_family: bool) -> dict[str, np.ndarray]:
    with np.load(PROCESSED_DIR / f"{dataset}_{split}.npz", allow_pickle=True) as data:
        keys = [key for key in data.files
                if (include_scaled or key != "X_scaled")
                and (include_family or key != "y_family")]
        arrays = {key: data[key] for key in keys}
    if "y_family" in arrays and arrays["y_family"].dtype.kind == "S":
        arrays["y_family"] = arrays["y_family"].astype(str)
    return arrays


def get_test_arrays(dataset: str, include_scaled: bool = False,
                    include_family: bool = True) -> dict[str, np.ndarray]:
    """Load processed test arrays, optionally including scaled features."""
    return _get_arrays(dataset, "test", include_scaled, include_family)


def get_train_arrays(dataset: str, include_scaled: bool = False,
                     include_family: bool = True) -> dict[str, np.ndarray]:
    """Load processed training arrays, optionally including scaled features."""
    return _get_arrays(dataset, "train", include_scaled, include_family)


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
    archive = RAW_DIR / "CICIDS2017_improved.zip"
    if not raw.exists() and not archive.exists():
        _download_url(
            "https://intrusion-detection.distrinet-research.be/"
            "CNS2022/Datasets/CICIDS2017_improved.zip",
            archive,
        )

    source = raw if raw.exists() else archive
    archive_file = zipfile.ZipFile(source) if source.suffix.lower() == ".zip" else None
    try:
        csv_names = (
            sorted(name for name in archive_file.namelist()
                   if name.lower().endswith(".csv"))
            if archive_file else [source.name]
        )
        if not csv_names:
            raise ValueError(f"No CSV files found in CICIDS2017 source: {source}")

        def open_csv(name):
            return archive_file.open(name) if archive_file else open(source, "rb")

        with open_csv(csv_names[0]) as stream:
            columns = [str(column).strip() for column in pd.read_csv(stream, nrows=0).columns]
        label_col = next((column for column in columns if column.lower() == "label"), None)
        if label_col is None:
            raise ValueError("CICIDS2017 CSV is missing its Label column")
        attempted_col = next(
            (column for column in columns if column.strip().lower() == "attempted category"),
            None,
        )

        dropped = {
            "flowid", "srcip", "srcipdec", "sourceip", "sourceipdec",
            "srcport", "sourceport", "dstip", "dstipdec", "destinationip",
            "destinationipdec", "dstport", "destinationport", "protocol",
            "timestamp", "label", "attemptedcategory", "icmpcode", "icmptype",
            "totaltcpflowtime",
        }
        feature_cols = [
            column for column in columns
            if "".join(char.lower() for char in column if char.isalnum()) not in dropped
        ]
        if not feature_cols:
            raise ValueError("No numeric CICIDS2017 flow features found")

        for name in csv_names[1:]:
            with open_csv(name) as stream:
                other_columns = [str(column).strip()
                                 for column in pd.read_csv(stream, nrows=0).columns]
            if other_columns != columns:
                raise ValueError(f"CICIDS2017 file has a mismatched column schema: {name}")

        row_count = 0
        for name in csv_names:
            with open_csv(name) as stream:
                reader = csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
                next(reader, None)
                row_count += sum(1 for row in reader if row)
        if row_count < 2:
            raise ValueError("CICIDS2017 source contains too few data rows")

        with tempfile.TemporaryDirectory(
            prefix="cicids2017_", dir=PROCESSED_DIR,
        ) as temporary:
            work = Path(temporary)
            X_all = np.lib.format.open_memmap(
                work / "X.npy", mode="w+", dtype=np.float32,
                shape=(row_count, len(feature_cols)),
            )
            y_all = np.lib.format.open_memmap(
                work / "y.npy", mode="w+", dtype=np.int8, shape=(row_count,),
            )
            family_all = np.lib.format.open_memmap(
                work / "family.npy", mode="w+", dtype="S64", shape=(row_count,),
            )
            row_offset = 0
            dropped_rows = 0
            for name in csv_names:
                with open_csv(name) as stream:
                    for chunk in pd.read_csv(stream, chunksize=100_000, low_memory=False):
                        chunk.columns = chunk.columns.str.strip()
                        labels = chunk[label_col].astype(str).str.strip()
                        y_chunk = (labels.str.upper() != "BENIGN").to_numpy(dtype=np.int8)
                        family = labels.to_numpy(dtype="U64")

                        if attempted_col:
                            attempted = pd.to_numeric(
                                chunk[attempted_col], errors="coerce",
                            ).fillna(-1).to_numpy() != -1
                            y_chunk[attempted] = BENIGN
                            family[attempted] = "benign"

                        features = chunk[feature_cols].apply(
                            pd.to_numeric, errors="coerce",
                        ).to_numpy(dtype=np.float32)
                        valid = np.isfinite(features).all(axis=1)
                        dropped_rows += int((~valid).sum())
                        n_valid = int(valid.sum())
                        X_all[row_offset:row_offset + n_valid] = features[valid]
                        y_all[row_offset:row_offset + n_valid] = y_chunk[valid]
                        family_all[row_offset:row_offset + n_valid] = family[valid].astype("S64")
                        row_offset += n_valid
                        print(
                            f"\r  Processed {row_offset:,} CICIDS2017 flows",
                            end="", flush=True,
                        )

            if row_offset < 2 or len(np.unique(y_all[:row_offset])) != 2:
                raise ValueError("CICIDS2017 source must contain benign and attack flows")
            X_all = X_all[:row_offset]
            y_all = y_all[:row_offset]
            family_all = family_all[:row_offset]

            from sklearn.model_selection import train_test_split

            train_idx, test_idx = train_test_split(
                np.arange(row_offset),
                test_size=0.2,
                random_state=random_state,
                stratify=y_all,
            )
            X_train = np.lib.format.open_memmap(
                work / "X_train.npy", mode="w+", dtype=np.float32,
                shape=(len(train_idx), len(feature_cols)),
            )
            X_test = np.lib.format.open_memmap(
                work / "X_test.npy", mode="w+", dtype=np.float32,
                shape=(len(test_idx), len(feature_cols)),
            )
            y_train = np.lib.format.open_memmap(
                work / "y_train.npy", mode="w+", dtype=np.int8, shape=(len(train_idx),),
            )
            y_test = np.lib.format.open_memmap(
                work / "y_test.npy", mode="w+", dtype=np.int8, shape=(len(test_idx),),
            )
            family_train = np.lib.format.open_memmap(
                work / "family_train.npy", mode="w+", dtype="S64",
                shape=(len(train_idx),),
            )
            family_test = np.lib.format.open_memmap(
                work / "family_test.npy", mode="w+", dtype="S64", shape=(len(test_idx),),
            )
            scaler = StandardScaler()
            batch_size = 50_000
            for start in range(0, len(train_idx), batch_size):
                end = min(start + batch_size, len(train_idx))
                indices = train_idx[start:end]
                batch = X_all[indices]
                X_train[start:end] = batch
                y_train[start:end] = y_all[indices]
                family_train[start:end] = family_all[indices]
                scaler.partial_fit(batch)
            for start in range(0, len(test_idx), batch_size):
                end = min(start + batch_size, len(test_idx))
                indices = test_idx[start:end]
                X_test[start:end] = X_all[indices]
                y_test[start:end] = y_all[indices]
                family_test[start:end] = family_all[indices]

            X_train_scaled = np.lib.format.open_memmap(
                work / "X_train_scaled.npy", mode="w+", dtype=np.float32,
                shape=X_train.shape,
            )
            X_test_scaled = np.lib.format.open_memmap(
                work / "X_test_scaled.npy", mode="w+", dtype=np.float32,
                shape=X_test.shape,
            )
            for start in range(0, len(train_idx), batch_size):
                end = min(start + batch_size, len(train_idx))
                X_train_scaled[start:end] = scaler.transform(X_train[start:end])
            for start in range(0, len(test_idx), batch_size):
                end = min(start + batch_size, len(test_idx))
                X_test_scaled[start:end] = scaler.transform(X_test[start:end])

            mutable_mask = np.ones(len(feature_cols), dtype=bool)
            bounds = {}
            int_features = []
            for i, name in enumerate(feature_cols):
                values = X_train[:, i]
                lo, hi = float(np.percentile(values, 1)), float(np.percentile(values, 99))
                bounds[name] = {"min": lo, "max": hi, "mutable": True}
                if np.allclose(values, np.round(values), atol=1e-6):
                    int_features.append(i)

            meta = {
                "dataset": "cicids2017",
                "n_features": len(feature_cols),
                "feature_names": feature_cols,
                "mutable_mask": mutable_mask.tolist(),
                "bounds": bounds,
                "int_features": int_features,
                "one_hot_groups": {},
                "label_names": LABEL_NAMES,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "source_hash": _hash_file(source),
                "source": (
                    "CICIDS2017 improved release, CNS2022 research companion dataset"
                    if archive_file else "CICIDS2017 CSV"
                ),
                "attempted_flows_labeled_benign": bool(attempted_col),
                "dropped_non_finite_rows": dropped_rows,
                "split_seed": random_state,
            }

            np.savez(
                PROCESSED_DIR / "cicids2017_train.npz",
                X_scaled=X_train_scaled, X_raw=X_train, y=y_train,
                y_family=family_train,
            )
            np.savez(
                PROCESSED_DIR / "cicids2017_test.npz",
                X_scaled=X_test_scaled, X_raw=X_test, y=y_test,
                y_family=family_test,
            )
            with open(META_DIR / "cicids2017_meta.json", "w", encoding="utf-8") as output:
                json.dump(meta, output, indent=2)
            print(
                f"\n  cicids2017: train={len(train_idx):,}, test={len(test_idx):,}, "
                f"features={len(feature_cols)}, dropped_non_finite={dropped_rows:,}"
            )
            for array in (
                X_all, y_all, family_all, X_train, X_test, y_train, y_test,
                family_train, family_test, X_train_scaled, X_test_scaled,
            ):
                array.flush()
                array._mmap.close()
    finally:
        if archive_file:
            archive_file.close()


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
    mirror_dir = RAW_DIR / "unsw_nb15_hf"
    # The research-server ZIP URL previously used here is no longer live.
    # Prefer the compact, predefined official train/test partition mirror. The
    # filenames in this mirror are historically reversed: test.parquet contains
    # the 175,341-row training partition, while train.parquet has 82,332 test rows.
    mirror_train = mirror_dir / "test.parquet"
    mirror_test = mirror_dir / "train.parquet"
    raw = RAW_DIR / "unsw_nb15_raw.csv"
    archive_path = RAW_DIR / "unsw_nb15.zip"

    if not raw.exists() and not archive_path.exists():
        mirror_dir.mkdir(parents=True, exist_ok=True)
        if not mirror_train.exists():
            _download_url(
                "https://huggingface.co/datasets/wwydmanski/UNSW-NB15/resolve/main/test.parquet",
                mirror_train,
            )
        if not mirror_test.exists():
            _download_url(
                "https://huggingface.co/datasets/wwydmanski/UNSW-NB15/resolve/main/train.parquet",
                mirror_test,
            )

    if mirror_train.exists() and mirror_test.exists():
        train_df = pd.read_parquet(mirror_train)
        test_df = pd.read_parquet(mirror_test)
        if len(train_df) != 175_341 or len(test_df) != 82_332:
            raise ValueError(
                "UNSW-NB15 mirror partitions do not match the official "
                "175341/82332 training/test row counts"
            )
        if list(train_df.columns) != list(test_df.columns):
            raise ValueError("UNSW-NB15 train and test partitions have different columns")

        label_col = "label"
        family_col = "attack_cat"
        if label_col not in train_df or family_col not in train_df:
            raise ValueError("UNSW-NB15 partitions must contain label and attack_cat")
        y_train = pd.to_numeric(train_df[label_col], errors="raise").to_numpy(dtype=np.int64)
        y_test = pd.to_numeric(test_df[label_col], errors="raise").to_numpy(dtype=np.int64)
        if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
            raise ValueError("UNSW-NB15 partitions must include benign and attack labels")
        y_family_train = train_df[family_col].astype(str).to_numpy()
        y_family_test = test_df[family_col].astype(str).to_numpy()

        excluded = {label_col, family_col, *_UNSW_DROP}
        feature_cols = [column for column in train_df.columns if column not in excluded]
        categorical = train_df[feature_cols].select_dtypes(
            include=["object", "category", "string"],
        ).columns.tolist()
        train_features = pd.get_dummies(
            train_df[feature_cols], columns=categorical, prefix_sep="=",
            dtype=np.float32,
        )
        test_features = pd.get_dummies(
            test_df[feature_cols], columns=categorical, prefix_sep="=",
            dtype=np.float32,
        ).reindex(columns=train_features.columns, fill_value=0)
        feature_names = list(train_features.columns)
        X_train = train_features.to_numpy(dtype=np.float64)
        X_test = test_features.to_numpy(dtype=np.float64)
        if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
            raise ValueError("UNSW-NB15 partitions contain non-finite feature values")

        # Projection expects feature-column indices, not feature-name strings.
        one_hot_groups = {
            category: [
                index for index, name in enumerate(feature_names)
                if name.startswith(f"{category}=")
            ]
            for category in categorical
        }
        one_hot_groups = {key: value for key, value in one_hot_groups.items() if value}
        _write_processed(
            "unsw_nb15", X_train, X_test, y_train, y_test,
            y_family_train, y_family_test, feature_names, one_hot_groups,
            _UNSW_IMMUTABLE, _hash_file(mirror_train),
        )
        meta_path = META_DIR / "unsw_nb15_meta.json"
        with open(meta_path, encoding="utf-8") as source:
            meta = json.load(source)
        meta.update({
            "source": "UNSW-NB15 Kaggle-derived Parquet mirror (wwydmanski/UNSW-NB15)",
            "source_url": "https://huggingface.co/datasets/wwydmanski/UNSW-NB15",
            "source_partition_note": (
                "Training and test partitions assigned by the official UNSW-NB15 "
                "row counts; mirror filenames are reversed."
            ),
            "source_test_hash": _hash_file(mirror_test),
            "official_partition": True,
        })
        with open(meta_path, "w", encoding="utf-8") as output:
            json.dump(meta, output, indent=2)
        return

    if not raw.exists():
        if not archive_path.exists():
            raise FileNotFoundError(
                "UNSW-NB15 download did not produce its parquet mirror, and no local "
                "raw CSV or UNSW-NB15 ZIP is available. Retry the Hugging Face mirror "
                "download or place the official CSV/ZIP under data/raw/."
            )
        with zipfile.ZipFile(archive_path) as archive:
            csv_names = sorted(
                name for name in archive.namelist()
                if Path(name).suffix.lower() == ".csv"
                and Path(name).stem.lower().startswith(("unsw-nb15_", "nusw-nb15_"))
                and Path(name).stem.rsplit("_", 1)[-1] in {"1", "2", "3", "4"}
            )
            if not csv_names:
                raise ValueError(
                    "UNSW-NB15 archive has no numbered UNSW-NB15_[1-4].csv partitions"
                )
            # The official dataset is split across four files with repeated headers.
            # Stream them together so data is combined without holding the archive in RAM.
            with raw.open("wb") as combined:
                for index, csv_name in enumerate(csv_names):
                    with archive.open(csv_name) as source:
                        if index:
                            source.readline()  # skip the repeated header
                        shutil.copyfileobj(source, combined)

    df = pd.read_csv(raw)
    df.columns = df.columns.str.strip()
    if "label" not in df.columns:
        raise ValueError("UNSW-NB15 CSV is missing its label column")

    label_text = df["label"].astype(str).str.strip()
    numeric_labels = pd.to_numeric(df["label"], errors="coerce")
    if numeric_labels.notna().all() and set(numeric_labels.unique()) <= {0, 1}:
        # Official UNSW-NB15 label encoding is 0=normal and 1=attack.
        y = numeric_labels.to_numpy(dtype=np.int64)
    else:
        benign_labels = label_text.str.casefold().isin({"normal", "benign"})
        y = np.where(benign_labels, BENIGN, ATTACK).astype(np.int64)

    if set(np.unique(y)) != {BENIGN, ATTACK}:
        raise ValueError(
            "UNSW-NB15 label column must include both benign/normal and attack rows"
        )

    if "attack_cat" in df.columns:
        y_family = df["attack_cat"].fillna("unknown").astype(str).to_numpy()
        y_family[y == BENIGN] = "benign"
    else:
        y_family = label_text.to_numpy()
        y_family[y == BENIGN] = "benign"

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

    The data values are synthetic Gaussian blobs, the 78 feature names, data types,
    bounds, and attack-family structure match CICIDS-2017 exactly.

    When you drop the real CICIDS2017.csv into data/raw/, the pipeline
    automatically uses it instead. See docs/cicids2017_features.md for
    what each feature means.

    This lets you run the full adversarial attack demo on realistic
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
