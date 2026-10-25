"""Shared UI primitives and design-system styling for the dashboard."""

from html import escape
from pathlib import Path

import streamlit as st

STYLES_PATH = Path(__file__).resolve().parent / "styles.css"


def apply_design_system() -> None:
    """Apply the dashboard's shared, authored CSS."""
    st.markdown(
        f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def render_page_header(eyebrow: str, title: str, description: str) -> None:
    """Render a consistent page heading with escaped dynamic text."""
    st.markdown(
        f"""
        <header class="page-heading">
          <div class="heading-copy">
            <p class="eyebrow"><span class="eyebrow-mark"></span>{escape(eyebrow)}</p>
            <h1>{escape(title)}</h1>
            <p class="page-description">{escape(description)}</p>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: list[tuple[str, str, str]]) -> None:
    """Render compact, responsive metric cards."""
    cards = "".join(
        f'<div class="metric-card" role="group" aria-label="{escape(label)}">'
        f'<p class="metric-label">{escape(label)}</p>'
        f'<p class="metric-value">{escape(value)}</p>'
        f'<p class="metric-context">{escape(context)}</p></div>'
        for label, value, context in metrics
    )
    st.markdown(
        f'<div class="metric-grid">{cards}</div>',
        unsafe_allow_html=True,
    )


def render_prediction_card(
    prediction: str,
    true_label: str,
    attack_confidence: float,
    family: str,
) -> None:
    """Show the clean prediction and its confidence without trusting data as HTML."""
    prediction_class = "badge-attack" if prediction == "attack" else "badge-benign"
    true_class = "badge-attack" if true_label == "attack" else "badge-benign"
    confidence_color = "#df624e" if attack_confidence >= 0.5 else "#4059d9"
    st.markdown(
        f"""
        <section class="pred-card" aria-label="Clean model prediction">
          <div class="pred-row">
            <span class="pred-label">Prediction</span>
            <span class="pred-badge {prediction_class}">{escape(prediction.title())}</span>
          </div>
          <div class="pred-row">
            <span class="pred-label">P(attack)</span>
            <div class="confidence-track" role="img"
                 aria-label="Attack confidence {attack_confidence:.1%}">
              <div class="confidence-fill"
                   style="width:{attack_confidence * 100:.1f}%;background:{confidence_color}"></div>
            </div>
            <span class="confidence-value" style="color:{confidence_color}">
              {attack_confidence:.4f}
            </span>
          </div>
          <div class="pred-row">
            <span class="pred-label">True label</span>
            <span class="pred-badge {true_class}">{escape(true_label.title())}</span>
          </div>
          <div class="pred-row">
            <span class="pred-label">Attack family</span>
            <span>{escape(family)}</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_profile(
    dataset: str,
    n_features: int,
    n_train: int,
    n_test: int,
    n_benign: int,
    n_attack: int,
) -> None:
    """Present counts from the selected dataset metadata and held-out split."""
    kind = "Synthetic pipeline fixture" if dataset == "dummy" else "Prepared network-flow data"
    total = max(n_benign + n_attack, 1)
    benign_share = n_benign / total * 100
    st.markdown(
        f"""
        <section class="profile-panel" aria-label="Selected dataset profile">
          <p class="profile-kicker">{escape(kind)}</p>
          <h2 class="profile-title">{escape(dataset.replace("_", " ").upper())}</h2>
          <div class="profile-stat"><span>Feature dimensions</span><strong>{n_features:,}</strong></div>
          <div class="profile-stat"><span>Training rows</span><strong>{n_train:,}</strong></div>
          <div class="profile-stat"><span>Held-out rows</span><strong>{n_test:,}</strong></div>
          <div class="profile-stat"><span>Benign / attack</span>
            <strong>{n_benign:,} / {n_attack:,}</strong></div>
          <div class="dataset-balance" role="img"
               aria-label="Held-out class distribution: {benign_share:.1f}% benign and {100 - benign_share:.1f}% attack">
            <span class="dataset-balance-benign" style="width:{benign_share:.3f}%"></span>
            <span class="dataset-balance-attack" style="width:{100 - benign_share:.3f}%"></span>
          </div>
          <div class="dataset-legend">
            <span><i class="legend-benign"></i>Benign</span>
            <span><i class="legend-attack"></i>Attack</span>
          </div>
          <p class="profile-note">Evaluation remains on the dataset's existing held-out test split.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_setup_hero(animate: bool) -> None:
    """Render the safe model-preparation introduction."""
    motion_class = " setup-enter" if animate else ""
    st.html(
        f"""
        <section class="setup-hero{motion_class}">
          <div class="setup-hero-copy">
            <p class="setup-kicker"><span></span>MODEL LAB&nbsp; / &nbsp;01</p>
            <h1>Know your baseline.<br><em>Then ask better questions.</em></h1>
            <p>Prepare a labeled traffic dataset, select a detector, and inspect its clean-set behavior.</p>
          </div>
          <div class="setup-orbit" aria-hidden="true">
            <span class="orbit orbit-one"></span>
            <span class="orbit orbit-two"></span>
            <span class="orbit-core"></span>
          </div>
          <div class="setup-hero-index" aria-hidden="true"></div>
        </section>
        """
    )


def render_clean_scoreboard(metrics: dict) -> None:
    """Show real clean-set model metrics in a focused typographic layout."""
    f1 = float(metrics.get("f1", 0))
    items = [
        ("Accuracy", f"{metrics.get('accuracy', 0):.4%}", "All held-out records"),
        ("Recall", f"{metrics.get('recall', 0):.4%}", "Attack detection rate"),
        ("AUROC", f"{metrics.get('auroc', 0):.4%}", "Class separation"),
    ]
    secondary = "".join(
        f'<div class="score-item"><span>{escape(label)}</span>'
        f'<strong>{escape(value)}</strong><small>{escape(note)}</small></div>'
        for label, value, note in items
    )
    st.html(
        f"""
        <section class="clean-scoreboard setup-reveal" aria-label="Clean model performance">
          <div class="score-lead">
            <p class="score-kicker">CLEAN-SET BASELINE</p>
            <span class="score-lead-value">{f1:.4%}</span>
            <strong>F1 score</strong>
            <small>Precision / recall balance</small>
          </div>
          <div class="score-secondary">{secondary}</div>
        </section>
        """
    )


def render_clean_model_landscape(
    model_results: dict[str, dict],
    animate: bool = True,
) -> None:
    """Show every model's actual clean-set scores and confusion counts."""
    ranked = sorted(
        model_results.items(),
        key=lambda item: float(item[1]["f1"]),
        reverse=True,
    )
    quality_rows = []
    false_positive_rows = []
    false_negative_rows = []
    for rank, (model, metrics) in enumerate(ranked, start=1):
        label = escape(model.replace("_", " ").upper())
        quality_rows.append(
            f"""
            <div class="landscape-row">
              <div class="landscape-model"><strong>{label}</strong><span>#{rank} by F1</span></div>
              {_landscape_rate_bar("F1", metrics["f1"], "f1")}
              {_landscape_rate_bar("Recall", metrics["recall"], "recall")}
            </div>
            """
        )
        for key, destination, color in (
            ("fp", false_positive_rows, "false-positive"),
            ("fn", false_negative_rows, "false-negative"),
        ):
            count = metrics.get(key)
            if _is_nonnegative_count(count):
                destination.append((label, count, color))

    error_sections = []
    for title, count_rows, description in (
        ("False alerts", false_positive_rows, "Benign rows predicted as attack"),
        ("Missed attacks", false_negative_rows, "Attack rows predicted as benign"),
    ):
        if count_rows:
            maximum = max(count for _, count, _ in count_rows) or 1
            rows = "".join(
                f"""
                <div class="error-count-row">
                  <span>{label}</span>
                  <div class="error-count-track" role="img"
                       aria-label="{title} for {label}: {count:,} rows">
                    <i class="{color}" style="width:{count / maximum * 100:.3f}%"></i>
                  </div>
                  <strong>{count:,}</strong>
                </div>
                """
                for label, count, color in count_rows
            )
            error_sections.append(
                f"""
                <section class="error-count-chart">
                  <header><strong>{title}</strong><span>{description}</span></header>
                  {rows}
                  <small>Scale: 0–{maximum:,} rows across these models</small>
                </section>
                """
            )

    error_content = "".join(error_sections) or (
        '<p class="landscape-empty">Confusion counts are not available for these results.</p>'
    )
    motion_class = " landscape-enter" if animate else ""
    st.html(
        f"""
        <section class="model-landscape{motion_class}" aria-label="All models on the clean test set">
          <article class="landscape-quality">
            <header class="landscape-heading">
              <div><p class="comparison-kicker">CLEAN-SET SCORECARD</p>
                <h2>Quality, at a glance</h2>
                <p>F1 balances precision and recall. Recall is the share of attack rows detected. Higher is better for both.</p>
              </div>
              <span class="landscape-axis">0–100%</span>
            </header>
            <div class="landscape-column-labels"><span>Model · rank by F1</span><span>F1 score</span><span>Recall</span></div>
            {''.join(quality_rows)}
          </article>
          <article class="landscape-errors">
            <header class="landscape-heading">
              <div><p class="comparison-kicker">WHAT THE SCORES MEAN</p>
                <h2>Where mistakes happen</h2>
                <p>Counts are actual rows from the clean held-out test set; each measure has its own zero-based scale.</p>
              </div>
            </header>
            {error_content}
          </article>
        </section>
        """
    )


def _landscape_rate_bar(label: str, value: float, color: str) -> str:
    value = float(value)
    return f"""
      <div class="landscape-score">
        <span>{escape(label)}</span>
        <div class="landscape-track" role="img"
             aria-label="{escape(label)}: {value:.2%} on a 0 to 100 percent scale">
          <i class="{color}" style="width:{value * 100:.3f}%"></i>
        </div>
        <strong>{value:.2%}</strong>
      </div>
    """


def _is_nonnegative_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def render_clean_model_comparison(baseline: dict, augmented: dict) -> None:
    """Compare clean-test model measurements without implying adversarial robustness."""
    rows = [
        ("F1", "f1", "rate"),
        ("Precision", "precision", "rate"),
        ("Recall", "recall", "rate"),
        ("False-positive rate", "fpr", "rate"),
        ("False-negative rate", "fnr", "rate"),
        ("False negatives", "fn", "count"),
    ]
    table_rows = "".join(
        "<tr>"
        f"<th scope='row'>{escape(label)}</th>"
        f"<td>{_format_rate(float(baseline[name]))}</td>"
        f"<td>{_format_rate(float(augmented[name]))}</td>"
        "</tr>"
        if value_type == "rate"
        else (
            "<tr>"
            f"<th scope='row'>{escape(label)}</th>"
            f"<td>{int(baseline[name]):,}</td>"
            f"<td>{int(augmented[name]):,}</td>"
            "</tr>"
        )
        for label, name, value_type in rows
    )
    st.html(
        f"""
        <section class="clean-comparison">
          <p class="comparison-kicker">SAME HELD-OUT CLEAN SPLIT</p>
          <table>
            <thead><tr><th scope="col">Measure</th><th scope="col">Baseline</th>
              <th scope="col">Clean + supplied augmentations</th></tr></thead>
            <tbody>{table_rows}</tbody>
          </table>
          <p>These measurements describe clean-set performance only; no adversarial evaluation was run.</p>
        </section>
        """
    )
    render_clean_comparison_visuals(baseline, augmented)


def render_clean_comparison_visuals(baseline: dict, augmented: dict) -> None:
    """Pair clean-test rates and error counts for a supplied-augmentation run."""
    rate_rows = []
    for label, name in (
        ("F1", "f1"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("False-positive rate", "fpr"),
        ("False-negative rate", "fnr"),
    ):
        baseline_value = float(baseline[name])
        augmented_value = float(augmented[name])
        delta = (augmented_value - baseline_value) * 100
        rate_rows.append(
            f"""
            <article class="comparison-metric">
              <div class="comparison-metric-heading">
                <strong>{escape(label)}</strong><span>{_format_percentage_points(delta)}</span>
              </div>
              {_comparison_rate_bar("Base", baseline_value, "baseline")}
              {_comparison_rate_bar("Aug.", augmented_value, "augmented")}
            </article>
            """
        )

    count_rows = []
    for label, name in (("False positives", "fp"), ("False negatives", "fn")):
        if all(_is_nonnegative_count(metrics.get(name)) for metrics in (baseline, augmented)):
            baseline_count = baseline[name]
            augmented_count = augmented[name]
            maximum = max(baseline_count, augmented_count, 1)
            count_rows.append(
                f"""
                <article class="comparison-count">
                  <strong>{escape(label)}</strong>
                  <div class="comparison-count-values">
                    <span>Base <b>{baseline_count:,}</b></span>
                    <span>Aug. <b>{augmented_count:,}</b></span>
                  </div>
                  <div class="comparison-count-track" role="img"
                       aria-label="{escape(label)}: baseline {baseline_count:,}, augmented {augmented_count:,}">
                    <div><i class="baseline" style="width:{baseline_count / maximum * 100:.3f}%"></i></div>
                    <div><i class="augmented" style="width:{augmented_count / maximum * 100:.3f}%"></i></div>
                  </div>
                </article>
                """
            )

    f1_delta = (float(augmented["f1"]) - float(baseline["f1"])) * 100
    fn_delta = int(augmented["fn"]) - int(baseline["fn"])
    if fn_delta < 0:
        fn_change = f"{abs(fn_delta):,} fewer attack records missed"
    elif fn_delta > 0:
        fn_change = f"{fn_delta:,} more attack records missed"
    else:
        fn_change = "No change in missed attack records"
    st.html(
        f"""
        <section class="clean-comparison-visual" aria-label="Visual clean-test comparison">
          <header class="comparison-visual-heading">
            <div><p class="comparison-kicker">WHAT CHANGED?</p>
              <h3>Baseline → clean + supplied augmentations</h3>
              <p>F1 changed by <strong>{f1_delta:+.2f} percentage points</strong>;
                {escape(fn_change)}.</p>
            </div>
            <span class="comparison-scale-note">Same held-out clean set</span>
          </header>
          <div class="comparison-legend">
            <span><i class="comparison-swatch baseline"></i>Baseline</span>
            <span><i class="comparison-swatch augmented"></i>Augmented model</span>
          </div>
          <div class="comparison-rate-grid">{''.join(rate_rows)}</div>
          <div class="comparison-count-grid">{''.join(count_rows)}</div>
          <p class="comparison-scale-note">Rate bars share a 0–100% scale. Count bars scale from zero to the larger count for that measure; exact values are shown. This is a clean-set comparison, not an adversarial-robustness measurement.</p>
        </section>
        """
    )


def _comparison_rate_bar(label: str, value: float, color: str) -> str:
    return f"""
      <div class="comparison-bar-row">
        <span>{escape(label)}</span>
        <div class="comparison-track" role="img"
             aria-label="{escape(label)}: {_format_rate(value)} on a 0 to 100 percent scale">
          <i class="{color}" style="width:{value * 100:.3f}%"></i>
        </div>
        <b>{_format_rate(value)}</b>
      </div>
    """


def _format_rate(value: float) -> str:
    precision = 2 if abs(value) >= 0.001 else 4
    return f"{value:.{precision}%}"


def _format_percentage_points(value: float) -> str:
    precision = 2 if abs(value) >= 0.01 else 4
    return f"{value:+.{precision}f} pp"


def render_clean_confusion_matrix(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    highlight: str | None = None,
) -> None:
    """Render an accessible confusion matrix with the actual clean-set counts."""
    benign_row_class = " focus-row" if highlight == "Benign traffic" else ""
    attack_row_class = " focus-row" if highlight == "Attack traffic" else ""
    st.html(
        f"""
        <section class="confusion-module setup-reveal" aria-label="Confusion matrix for clean test set">
          <div class="confusion-head"><span>ACTUAL ↓ &nbsp; PREDICTED →</span><span>BENIGN</span><span>ATTACK</span></div>
          <div class="confusion-row{benign_row_class}"><strong>BENIGN</strong>
            <span class="confusion-cell correct"><b>{tn:,}</b><small>True negative</small></span>
            <span class="confusion-cell alert"><b>{fp:,}</b><small>False positive</small></span>
          </div>
          <div class="confusion-row{attack_row_class}"><strong>ATTACK</strong>
            <span class="confusion-cell alert"><b>{fn:,}</b><small>False negative</small></span>
            <span class="confusion-cell correct"><b>{tp:,}</b><small>True positive</small></span>
          </div>
        </section>
        """
    )


def render_clean_outcome_explorer(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    cohort: str,
) -> None:
    """Explain held-out outcomes for one actual class using its confusion counts."""
    if cohort == "Attack traffic":
        total = fn + tp
        correct_count, error_count = tp, fn
        correct_label, error_label = "Detected as attack", "Missed as benign"
        correct_class, error_class = "outcome-good", "outcome-error"
        success_description, error_description = "detected", "missed"
    else:
        total = tn + fp
        correct_count, error_count = tn, fp
        correct_label, error_label = "Correctly identified as benign", "False alerts"
        correct_class, error_class = "outcome-good", "outcome-error"
        success_description, error_description = "correctly identified", "flagged as attack"

    correct_share = correct_count / total * 100 if total else 0.0
    error_share = error_count / total * 100 if total else 0.0
    if total:
        explanation = (
            f"Of {total:,} actual {cohort.lower()} records, "
            f"{correct_share:.1f}% were {success_description} and "
            f"{error_share:.1f}% were {error_description}."
        )
        share_labels = f"{correct_share:.1f}% correctly handled; {error_share:.1f}% errors"
    else:
        explanation = f"No actual {cohort.lower()} records were present in this held-out split."
        share_labels = "No class-specific rate available"

    st.html(
        f"""
        <section class="outcome-explorer setup-reveal" aria-label="{escape(cohort)} outcome breakdown">
          <div class="outcome-heading">
            <div>
              <p class="outcome-kicker">COHORT BREAKDOWN</p>
              <h3>{escape(cohort)}</h3>
            </div>
            <div class="outcome-total"><strong>{total:,}</strong><span>actual records</span></div>
          </div>
          <p class="outcome-explanation">{escape(explanation)}</p>
          <div class="outcome-track" role="img" aria-label="{escape(share_labels)}">
            <span class="{correct_class}" style="width:{correct_share:.3f}%"></span>
            <span class="{error_class}" style="width:{error_share:.3f}%"></span>
          </div>
          <div class="outcome-legend">
            <div><span class="outcome-swatch outcome-good"></span>
              <span>{escape(correct_label)}</span>
              <strong>{correct_count:,}</strong>
            </div>
            <div><span class="outcome-swatch outcome-error"></span>
              <span>{escape(error_label)}</span>
              <strong>{error_count:,}</strong>
            </div>
          </div>
          <p class="outcome-note">Shares are calculated within the selected actual class, not across all records.</p>
        </section>
        """
    )


def render_clean_diagnostics(metrics: dict, key: str) -> None:
    """Pair the held-out confusion matrix with an interactive class-level explanation."""
    tn, fp, fn, tp = (int(metrics[name]) for name in ("tn", "fp", "fn", "tp"))
    cohort = st.radio(
        "Inspect actual class",
        ("Attack traffic", "Benign traffic"),
        horizontal=True,
        key=key,
    )
    matrix_column, outcome_column = st.columns([0.9, 1.1], gap="large")
    with matrix_column:
        st.markdown("### Prediction matrix")
        st.markdown(
            '<p class="section-intro">Actual class by clean-set prediction.</p>',
            unsafe_allow_html=True,
        )
        render_clean_confusion_matrix(tn, fp, fn, tp, highlight=cohort)
    with outcome_column:
        render_clean_outcome_explorer(tn, fp, fn, tp, cohort)
