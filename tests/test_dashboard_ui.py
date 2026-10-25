from dashboard import ui


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
