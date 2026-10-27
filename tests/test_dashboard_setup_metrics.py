from dashboard import app


def test_clean_diagnostics_reuses_saved_confusion_counts(monkeypatch):
    saved_metrics = {
        "tn": 26999,
        "fp": 10001,
        "fn": 1016,
        "tp": 44316,
    }

    class SavedModel:
        metrics = saved_metrics

        def predict_proba(self, _):
            raise AssertionError("Saved metrics should avoid a second full-test prediction")

    def fail_if_computed(*_args):
        raise AssertionError("Saved metrics should be used directly")

    monkeypatch.setattr(app, "_compute_clean_metrics", fail_if_computed)

    assert app._clean_diagnostics_metrics(SavedModel(), "unsw_nb15", "xgb") is saved_metrics



def test_uploaded_unsw_csv_encodes_categoricals_and_keeps_numeric_binary_labels():
    import pandas as pd
    import numpy as np

    rows = []
    for i in range(100):
        label = i % 2
        rows.append({
            "srcip": f"10.0.0.{i + 1}",
            "sport": 1000 + i,
            "dstip": f"10.0.1.{i + 1}",
            "dsport": 2000 + i,
            "proto": ("tcp", "udp", "icmp")[i % 3],
            "state": ("FIN", "CON", "INT")[i % 3],
            "service": ("-", "http", "dns", "ftp")[i % 4],
            "dur": float(i) / 10,
            "spkts": 1 + (i % 7),
            "attack_cat": "Normal" if label == 0 else "Generic",
            "label": label,
        })

    X, y, families, feature_names, one_hot_groups = app._prepare_uploaded_csv(
        pd.DataFrame(rows)
    )

    assert X.shape[0] == 100
    assert np.isfinite(X).all()
    assert set(np.unique(y)) == {0, 1}
    assert np.all(families[y == 0] == "benign")
    assert np.all(families[y == 1] == "Generic")
    assert "srcip" not in feature_names
    assert "dstip" not in feature_names
    assert "sport" not in feature_names
    assert any(name.startswith("proto=") for name in feature_names)
    assert any(name.startswith("state=") for name in feature_names)
    assert any(name.startswith("service=") for name in feature_names)
    assert {"proto", "state", "service"} <= set(one_hot_groups)


def test_uploaded_cicids_text_labels_are_binarized():
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        "Flow ID": [f"flow-{i}" for i in range(100)],
        "Destination IP": [f"192.0.2.{i}" for i in range(100)],
        "Flow Duration": [float(i + 1) for i in range(100)],
        "Total Fwd Packets": [i % 10 + 1 for i in range(100)],
        "Protocol": ["TCP" if i % 2 else "UDP" for i in range(100)],
        "Label": ["BENIGN" if i % 2 == 0 else "DoS Hulk" for i in range(100)],
    })

    X, y, families, feature_names, one_hot_groups = app._prepare_uploaded_csv(df)

    assert X.shape[0] == 100
    assert set(np.unique(y)) == {0, 1}
    assert np.isfinite(X).all()
    assert "Flow ID" not in feature_names
    assert "Destination IP" not in feature_names
    assert any(name.startswith("Protocol=") for name in feature_names)
    assert np.all(families[y == 0] == "benign")
