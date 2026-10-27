"""Regression tests for gradient, CW optimization, and UNSW-NB15 loading."""
import io
import zipfile

import numpy as np
import pytest

from nids import data
from nids.attacks import cw
from nids.models import make_model


def _small_meta(n_features):
    names = [f"f{i}" for i in range(n_features)]
    return {
        "feature_names": names,
        "mutable_mask": [True] * n_features,
        "bounds": {name: {"min": -10.0, "max": 10.0, "mutable": True} for name in names},
        "int_features": [],
        "one_hot_groups": {},
    }


def test_torch_gradient_scaled_uses_supported_input_gradient_api():
    pytest.importorskip("torch")
    model = make_model("mlp", n_features=4, seed=0)
    X = np.asarray([[0.2, -0.5, 0.1, 1.0]], dtype=np.float32)

    gradient = model.gradient_scaled(X, target=0)

    assert gradient.shape == X.shape
    assert np.isfinite(gradient).all()


def test_cw_torch_path_runs_a_small_optimization():
    pytest.importorskip("torch")
    model = make_model("mlp", n_features=4, seed=0)
    model.mutable_mask = np.ones(4, dtype=bool)
    model.meta = _small_meta(4)
    X = np.asarray([[0.7, -0.2, 0.5, 0.1]], dtype=np.float32)

    result = cw(
        model, X, eps=0.2, mask=model.mutable_mask,
        n_steps=2, n_binary_search=1, n_project=1,
    )

    assert result.x_adv.shape == X.shape
    assert np.isfinite(result.x_adv).all()
    assert np.asarray(result.success).shape == (1,)


def _write_unsw_archive(archive_path):
    header = (
        "srcip,sport,dstip,dsport,proto,state,dur,spkts,dpkts,attack_cat,label\n"
    )
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        # The official source is split across numbered files; each repeats a header.
        for part in (1, 2, 3, 4):
            lines = [header.rstrip("\n")]
            for i in range(25):
                idx = (part - 1) * 25 + i
                label = 0 if idx % 2 == 0 else 1
                category = "Normal" if label == 0 else ("Generic" if idx % 4 else "Fuzzers")
                lines.append(
                    f"10.0.0.{idx + 1},{1000 + idx},10.0.1.{idx + 1},"
                    f"{2000 + idx},tcp,FIN,{0.1 + idx / 1000},"
                    f"{3 + idx % 5},{1 + idx % 3},{category},{label}"
                )
            archive.writestr(f"UNSW-NB15_{part}.csv", "\n".join(lines) + "\n")


def test_unsw_loader_merges_numbered_csvs_and_keeps_both_labels(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    meta_dir = tmp_path / "meta"
    for path in (raw, processed, meta_dir):
        path.mkdir(parents=True)

    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "RAW_DIR", raw)
    monkeypatch.setattr(data, "PROCESSED_DIR", processed)
    monkeypatch.setattr(data, "META_DIR", meta_dir)
    _write_unsw_archive(raw / "unsw_nb15.zip")

    meta = data.load_dataset("unsw_nb15")
    train = data.get_train_arrays("unsw_nb15")
    test = data.get_test_arrays("unsw_nb15")

    assert meta["n_train"] == len(train["y"])
    assert meta["n_test"] == len(test["y"])
    assert set(np.unique(train["y"])) == {0, 1}
    assert set(np.unique(test["y"])) == {0, 1}
    assert np.all(train["y_family"][train["y"] == 0] == "benign")
    assert np.all(test["y_family"][test["y"] == 0] == "benign")
    # 100 input rows from four partitions, with exactly one header in the merged file.
    with (raw / "unsw_nb15_raw.csv").open(encoding="utf-8") as merged:
        assert sum(1 for _ in merged) == 101


def test_unsw_loader_rebuilds_stale_one_class_cache(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    meta_dir = tmp_path / "meta"
    for path in (raw, processed, meta_dir):
        path.mkdir(parents=True)

    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "RAW_DIR", raw)
    monkeypatch.setattr(data, "PROCESSED_DIR", processed)
    monkeypatch.setattr(data, "META_DIR", meta_dir)
    _write_unsw_archive(raw / "unsw_nb15.zip")
    data.load_dataset("unsw_nb15")

    np.savez(processed / "unsw_nb15_train.npz", y=np.ones(20, dtype=np.int64))
    np.savez(processed / "unsw_nb15_test.npz", y=np.ones(20, dtype=np.int64))
    (meta_dir / "unsw_nb15_meta.json").write_text('{"dataset":"unsw_nb15"}', encoding="utf-8")

    data.load_dataset("unsw_nb15")
    train = data.get_train_arrays("unsw_nb15")
    test = data.get_test_arrays("unsw_nb15")

    assert set(np.unique(train["y"])) == {0, 1}
    assert set(np.unique(test["y"])) == {0, 1}


def test_unsw_loader_downloads_predefined_parquet_split(tmp_path, monkeypatch):
    """Fresh setup uses the public mirror when neither local CSV nor ZIP exists."""
    import pandas as pd

    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    meta_dir = tmp_path / "meta"
    for path in (raw, processed, meta_dir):
        path.mkdir(parents=True)

    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "RAW_DIR", raw)
    monkeypatch.setattr(data, "PROCESSED_DIR", processed)
    monkeypatch.setattr(data, "META_DIR", meta_dir)

    downloaded = []

    def fake_download(url, destination):
        downloaded.append((url, destination.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake parquet placeholder for mocked reader")

    monkeypatch.setattr(data, "_download_url", fake_download)

    def frame(rows):
        i = np.arange(rows)
        label = (i % 2).astype(np.int64)
        return pd.DataFrame({
            "srcip": [f"10.0.0.{int(x % 250) + 1}" for x in i],
            "sport": 1000 + i % 50000,
            "dstip": [f"10.0.1.{int(x % 250) + 1}" for x in i],
            "dsport": 2000 + i % 40000,
            "proto": np.where(i % 2, "tcp", "udp"),
            "state": np.where(i % 3, "FIN", "CON"),
            "dur": (i % 1000) / 10.0,
            "spkts": 1 + (i % 8),
            "dpkts": 1 + (i % 5),
            "attack_cat": np.where(label == 0, "Normal", "Generic"),
            "label": label,
        })

    expected_train = frame(175_341)
    expected_test = frame(82_332)

    def fake_read_parquet(path, *args, **kwargs):
        return expected_train.copy() if str(path).endswith("test.parquet") else expected_test.copy()

    monkeypatch.setattr(data.pd, "read_parquet", fake_read_parquet)

    meta = data.load_dataset("unsw_nb15")
    train = data.get_train_arrays("unsw_nb15")
    test = data.get_test_arrays("unsw_nb15")

    assert downloaded == [
        (
            "https://huggingface.co/datasets/wwydmanski/UNSW-NB15/resolve/main/test.parquet",
            "test.parquet",
        ),
        (
            "https://huggingface.co/datasets/wwydmanski/UNSW-NB15/resolve/main/train.parquet",
            "train.parquet",
        ),
    ]
    assert meta["n_train"] == 175_341
    assert meta["n_test"] == 82_332
    assert set(np.unique(train["y"])) == {0, 1}
    assert set(np.unique(test["y"])) == {0, 1}
    assert train["X_raw"].shape[1] == meta["n_features"]
    assert meta["one_hot_groups"], "categorical columns should have one-hot groups"
    assert all(
        isinstance(index, int)
        for indices in meta["one_hot_groups"].values()
        for index in indices
    ), "projection metadata must store one-hot column indices, not feature names"
    assert all(
        isinstance(bounds["min"], (int, float))
        and isinstance(bounds["max"], (int, float))
        for bounds in meta["bounds"].values()
    )
