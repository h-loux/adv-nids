from dashboard import app, ui


def test_clean_model_landscape_renders_actual_metrics_and_error_counts(monkeypatch):
    rendered = []
    monkeypatch.setattr(ui.st, "html", rendered.append)

    ui.render_clean_model_landscape(
        {
            "lr": {
                "f1": 0.84,
                "recall": 0.91,
                "fp": 120,
                "fn": 18,
            },
            "rf": {
                "f1": 0.96,
                "recall": 0.93,
                "fp": 10,
                "fn": 8,
            },
        },
        animate=False,
    )

    assert len(rendered) == 1
    chart = rendered[0]
    assert "RF</strong><span>#1 by F1" in chart
    assert "84.00%" in chart
    assert "96.00%" in chart
    assert "120" in chart
    assert "18" in chart
    assert "Scale: 0–120 rows across these models" in chart


def test_clean_model_comparison_renders_paired_rates_and_counts(monkeypatch):
    rendered = []
    monkeypatch.setattr(ui.st, "html", rendered.append)
    baseline = {
        "f1": 0.80,
        "precision": 0.75,
        "recall": 0.86,
        "fpr": 0.12,
        "fnr": 0.14,
        "fp": 40,
        "fn": 28,
    }
    augmented = {
        "f1": 0.84,
        "precision": 0.82,
        "recall": 0.86,
        "fpr": 0.08,
        "fnr": 0.14,
        "fp": 24,
        "fn": 28,
    }

    ui.render_clean_model_comparison(baseline, augmented)

    comparison = rendered[-1]
    assert 'aria-label="Visual clean-test comparison"' in comparison
    assert ">+4.00 percentage points</strong>" in comparison
    assert "+0.0000 pp" in comparison
    assert "Base <b>40</b>" in comparison
    assert "Aug. <b>24</b>" in comparison
    assert "not an adversarial-robustness measurement" in comparison


def test_small_rates_are_not_rounded_to_zero():
    assert ui._format_rate(0.000018) == "0.0018%"
    assert ui._format_percentage_points(-0.0009) == "-0.0009 pp"


def test_results_overview_uses_best_model_and_actual_held_out_counts(monkeypatch):
    rendered = []
    monkeypatch.setattr(ui.st, "html", rendered.append)
    model_results = {
        "lr": {"f1": 0.80, "tn": 70, "fp": 10, "fn": 15, "tp": 105},
        "rf": {"f1": 0.94, "tn": 78, "fp": 2, "fn": 6, "tp": 114},
    }

    ui.render_results_overview("test_network", model_results)

    overview = rendered[0]
    assert "TEST NETWORK" in overview
    assert "94.00%" in overview
    assert "RF" in overview
    assert ">200</strong>" in overview
    assert "role=\"img\"" in overview


def test_clean_metric_chart_passes_actual_model_scores_to_streamlit(monkeypatch):
    rendered_charts = []
    monkeypatch.setattr(
        ui.st,
        "bar_chart",
        lambda data, **kwargs: rendered_charts.append((data, kwargs)),
    )
    monkeypatch.setattr(ui.st, "container", lambda **kwargs: _ChartContainer(kwargs))
    monkeypatch.setattr(ui.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.st, "caption", lambda *args, **kwargs: None)
    model_results = {
        "lr": {"f1": 0.8, "precision": 0.75, "recall": 0.86, "auroc": 0.93},
        "rf": {"f1": 0.94, "precision": 0.96, "recall": 0.92, "auroc": 0.98},
    }

    ui.render_clean_metric_chart(model_results)

    chart_data, chart_options = rendered_charts[0]
    assert chart_data.loc["LR", "F1"] == 80
    assert chart_data.loc["RF", "Precision"] == 96
    assert chart_data.loc["RF", "Recall"] == 92
    assert chart_data.loc["RF", "AUROC"] == 98
    assert chart_options == {"height": 360, "stack": False}


class _ChartContainer:
    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


import pytest


@pytest.mark.parametrize("method", ["FGSM", "PGD", "CW", "DeepFool", "JSMA", "NES"])
def test_interactive_attack_kwargs_are_bounded(method):
    params = app._interactive_attack_kwargs(method, 5)

    assert params.get("n_steps", 0) <= 3
    assert params.get("n_project", 0) <= 2
    assert params.get("n_binary_search", 0) <= 2
    assert params.get("n_queries", 0) <= 24
    if method == "CW":
        assert params == {"n_steps": 3, "n_binary_search": 2, "n_project": 2}
    elif method == "PGD":
        assert params == {"n_steps": 3, "n_project": 2}
    elif method == "NES":
        assert params == {"n_queries": 24, "n_directions": 4, "n_steps": 3}
    elif method == "JSMA":
        assert params == {"n_steps": 3, "gamma": 0.1}
    elif method == "DeepFool":
        assert params == {"n_steps": 3}
    elif method == "FGSM":
        assert params == {}


def test_interactive_attack_kwargs_clamp_malformed_high_budget():
    params = app._interactive_attack_kwargs("CW", 10_000)

    assert params == {"n_steps": 3, "n_binary_search": 2, "n_project": 2}


def test_interactive_attack_kwargs_reject_unknown_method():
    with pytest.raises(ValueError, match="Unsupported interactive method"):
        app._interactive_attack_kwargs("unknown", 3)


def test_attack_outcome_prioritizes_invalid_perturbation(monkeypatch):
    """An invalid perturbation must never be presented as a valid evasion."""
    from dashboard import app

    captured = {"warning": [], "error": [], "success": []}
    monkeypatch.setattr(app.st, "warning", lambda message, **kwargs: captured["warning"].append(message))
    monkeypatch.setattr(app.st, "error", lambda message, **kwargs: captured["error"].append(message))
    monkeypatch.setattr(app.st, "success", lambda message, **kwargs: captured["success"].append(message))

    # Exercise the status rendering helper so this can be tested without a live Streamlit session.
    assert app._attack_outcome(True, valid=False)[0] == "invalid_after_flip"
    assert app._attack_outcome(False, valid=False)[0] == "invalid"
    assert app._attack_outcome(True, valid=True)[0] == "valid_evasion"
    assert app._attack_outcome(False, valid=True)[0] == "not_evasion"


def test_attack_outcome_marks_clean_baseline_outlier_ineligible():
    from dashboard import app

    outcome, detail = app._attack_outcome(True, valid=False, baseline_valid=False)

    assert outcome == "ineligible_baseline"
    assert "original clean flow" in detail
