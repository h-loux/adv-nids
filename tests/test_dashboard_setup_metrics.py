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
