import json
import zipfile

import numpy as np
from sklearn.ensemble import VotingClassifier

from nids import data
from nids.models import make_model
from nids.training import fit_model, train_and_save_model


def _arrays(seed=1):
    rng = np.random.RandomState(seed)
    y_train = np.tile([0, 1], 40)
    X_train = rng.normal(size=(len(y_train), 4))
    X_train[:, 0] += y_train * 2
    y_test = np.tile([0, 1], 10)
    X_test = rng.normal(size=(len(y_test), 4))
    X_test[:, 0] += y_test * 2
    return (
        {"X_raw": X_train, "y": y_train},
        {"X_raw": X_test, "y": y_test},
        {
            "feature_names": ["a", "b", "c", "d"],
            "mutable_mask": [True] * 4,
        },
    )


def test_fit_model_fits_scaler_on_training_rows_only():
    train, _, meta = _arrays()
    victim = fit_model("lr", train, meta, seed=0)

    np.testing.assert_allclose(victim.scaler_.mean_, train["X_raw"].mean(axis=0))
    assert victim.feature_names == meta["feature_names"]
    assert victim.predict_proba(train["X_raw"][:3]).shape == (3, 2)


def test_fit_model_applies_selected_estimator_parameters():
    train, _, meta = _arrays()
    victim = fit_model("lr", train, meta, seed=0, model_params={"C": 2.0})

    assert victim.base.C == 2.0


def test_train_and_save_model_persists_test_metrics(tmp_path):
    train, test, meta = _arrays()

    artifact, metrics = train_and_save_model(
        "sample", "lr", train, test, meta, seed=0, artifacts_dir=tmp_path,
    )

    assert artifact["metrics"] == metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert (tmp_path / "sample_lr.joblib").is_file()


def test_ensemble_uses_soft_voting():
    ensemble = make_model("ensemble", n_features=4)

    assert isinstance(ensemble.base, VotingClassifier)
    assert ensemble.base.voting == "soft"
    assert [name for name, _ in ensemble.base.estimators] == ["lr", "rf", "xgb"]
    assert ensemble.is_torch is False


def test_load_dataset_rebuilds_when_training_split_is_missing(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    metadata = tmp_path / "meta"
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(data, "PROCESSED_DIR", processed)
    monkeypatch.setattr(data, "META_DIR", metadata)
    processed.mkdir()
    metadata.mkdir()
    np.savez(processed / "dummy_test.npz", y=np.array([0, 1]))
    (metadata / "dummy_meta.json").write_text(json.dumps({"dataset": "dummy"}))

    loaded_meta = data.load_dataset("dummy")

    assert loaded_meta["dataset"] == "dummy"
    assert (processed / "dummy_train.npz").is_file()
    assert (processed / "dummy_test.npz").is_file()


def test_cicids_loader_builds_full_stratified_split_from_archive(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    metadata = tmp_path / "meta"
    raw = tmp_path / "raw"
    for directory in (processed, metadata, raw):
        directory.mkdir(parents=True)
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "RAW_DIR", raw)
    monkeypatch.setattr(data, "PROCESSED_DIR", processed)
    monkeypatch.setattr(data, "META_DIR", metadata)

    rows = [
        "Src IP dec,Src Port,Dst IP dec,Dst Port,Protocol,Timestamp,"
        "Flow Duration,Label,Attempted Category",
    ]
    for index in range(20):
        label = "BENIGN" if index < 10 else "DoS Hulk"
        attempted = 1 if index == 19 else -1
        rows.append(f"1,2,3,4,6,12:34.5,{index},{label},{attempted}")
    archive = raw / "CICIDS2017_improved.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("Monday.csv", "\n".join(rows))

    meta = data.load_dataset("cicids2017")
    train = data.get_train_arrays("cicids2017", include_family=False)
    test = data.get_test_arrays("cicids2017", include_family=True)

    assert meta["n_train"] == 16
    assert meta["n_test"] == 4
    assert meta["attempted_flows_labeled_benign"] is True
    assert train["X_raw"].dtype == np.float32
    assert len(train["y"]) == 16
    assert len(test["y_family"]) == 4
    assert set(np.unique(train["y"])) == {0, 1}
    assert set(np.unique(test["y"])) == {0, 1}
    assert np.count_nonzero(train["y"] == 0) + np.count_nonzero(test["y"] == 0) == 11
    assert "Flow Duration" in meta["feature_names"]
