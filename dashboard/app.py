"""Adversarial NIDS Evasion Console — Streamlit dashboard.

Loads pre-trained models from artifacts/ and lets you attack individual
network flows with FGSM, PGD, or Carlini-Wagner.

Run:  streamlit run dashboard/app.py
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.ui import (
    apply_design_system,
    render_metric_cards,
    render_page_header,
    render_prediction_card,
    render_dataset_profile,
    render_setup_hero,
    render_clean_scoreboard,
    render_clean_model_comparison,
    render_clean_diagnostics,
    render_clean_model_landscape,
    render_results_overview,
    render_clean_metric_chart,
)
from nids.config import EPS_GRID, LABEL_NAMES, BENIGN, ATTACK, ALL_MODELS
from nids.data import get_meta, get_test_arrays, get_train_arrays, load_dataset
from nids.models import VictimModel
from nids.training import train_and_save_model, train_augmented_and_save_model
from nids.attacks import fgsm, pgd, cw, deepfool, jsma, nes_attack
from nids.eval.validity import (
    check_validity, check_behaviour_preserved, perturbation_magnitude, feature_diff,
)
from nids.eval.metrics import compute_metrics

st.set_page_config(
    page_title="Adversarial NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_design_system()

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@st.cache_data
def process_uploaded_csv(uploaded_file, dataset_name: str) -> str:
    """Convert an uploaded CSV into the processed .npz + meta.json format."""
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Auto-detect label column
    label_col = None
    for candidate in ["label", "Label", "class", "Class", "target", "Target"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        st.error("Could not find a label column. Expected: 'label', 'Label', 'class', or 'target'.")
        return ""

    # Auto-detect attack family column (CICIDS-style)
    family_col = label_col

    # Drop non-feature columns
    drop_cols = {label_col, family_col}
    # Also drop ID-like columns
    for c in df.columns:
        if any(k in c.lower() for k in ["flow id", "src ip", "dst ip", "src port", "dst port"]):
            drop_cols.add(c)

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Convert feature columns to numeric, coercing errors to NaN
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN in features or label
    df = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)

    if len(df) == 0:
        st.error("All rows were dropped during cleaning (non-numeric values in features). "
                 "Try a CSV with numeric feature columns only.")
        return ""

    X = df[feature_cols].values.astype(np.float64)
    # Replace inf values (from division by zero in flow features like bytes_per_packet)
    np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    label_series = df[label_col]
    y = np.where(label_series.astype(str).str.upper() != "BENIGN", ATTACK, BENIGN)
    y_family = label_series.astype(str).values

    # Split
    from nids.config import SPLIT_SEED
    idx = np.random.RandomState(SPLIT_SEED).permutation(len(y))
    n_test = min(5000, len(y) // 5)
    train_idx, test_idx = idx[n_test:], idx[:n_test]

    scaler = StandardScaler().fit(X[train_idx])
    X_train_scaled = scaler.transform(X[train_idx])
    X_test_scaled = scaler.transform(X[test_idx])

    mutable_mask = [True] * len(feature_cols)
    bounds = {}
    for i, name in enumerate(feature_cols):
        col = X[train_idx, i]
        lo, hi = float(np.percentile(col, 1)), float(np.percentile(col, 99))
        bounds[name] = {"min": lo, "max": hi, "mutable": True}
    int_features = [i for i, name in enumerate(feature_cols) if np.allclose(X[:, i], np.round(X[:, i]), atol=1e-6)]

    from nids.config import LABEL_NAMES
    meta = {
        "dataset": dataset_name,
        "n_features": len(feature_cols),
        "feature_names": list(feature_cols),
        "mutable_mask": mutable_mask,
        "bounds": bounds,
        "int_features": int_features,
        "one_hot_groups": {},
        "label_names": LABEL_NAMES,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }

    proc = DATA_DIR / "processed"
    meta_dir = DATA_DIR / "meta"
    proc.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    np.savez(proc / f"{dataset_name}_train.npz",
             X_scaled=X_train_scaled, X_raw=X[train_idx], y=y[train_idx],
             y_family=y_family[train_idx])
    np.savez(proc / f"{dataset_name}_test.npz",
             X_scaled=X_test_scaled, X_raw=X[test_idx], y=y[test_idx],
             y_family=y_family[test_idx])
    import json as _json
    with open(meta_dir / f"{dataset_name}_meta.json", "w") as f:
        _json.dump(meta, f, indent=2)

    st.success(f"Dataset '{dataset_name}' loaded: {len(y)} flows, {len(feature_cols)} features")
    return dataset_name


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(dataset: str, model_name: str):
    """Load a pre-trained VictimModel from artifacts/."""
    import joblib
    path = ARTIFACTS_DIR / f"{dataset}_{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No pre-trained model at {path}. Run:\n"
            f"  python -m scripts.pretrain --dataset {dataset} --models {model_name}"
        )
    artifact = joblib.load(path)
    vm = VictimModel(
        artifact["model"],
        is_torch=artifact["is_torch"],
        n_features=artifact["n_features"],
    )
    vm.set_scaler(artifact["scaler"])
    vm.feature_names = artifact["feature_names"]
    vm.mutable_mask = np.array(artifact["mutable_mask"], dtype=bool)
    vm.meta = artifact.get("meta", {})
    vm.metrics = artifact.get("metrics", {})
    vm.model_params = artifact.get("model_params", {})
    vm.seed = artifact.get("seed", 0)
    vm.epochs = artifact.get("epochs")
    vm.batch_size = artifact.get("batch_size", 256)
    return vm


@st.cache_resource
def load_data(dataset):
    return get_test_arrays(dataset), get_meta(dataset)


def _has_confusion_counts(metrics) -> bool:
    """Return whether saved metrics can directly render the clean confusion matrix."""
    return isinstance(metrics, dict) and all(
        isinstance(metrics.get(name), (int, np.integer))
        and not isinstance(metrics.get(name), (bool, np.bool_))
        and metrics[name] >= 0
        for name in ("tn", "fp", "fn", "tp")
    )


@st.cache_data
def _compute_clean_metrics(dataset: str, model_name: str, artifact_modified_ns: int) -> dict:
    """Compute clean diagnostics once for older artifacts without saved metrics."""
    vm = load_model(dataset, model_name)
    arrays, _ = load_data(dataset)
    probabilities = vm.predict_proba(arrays["X_raw"])
    predictions = np.argmax(probabilities, axis=1)
    return compute_metrics(arrays["y"], predictions, probabilities)


def _clean_diagnostics_metrics(vm, dataset: str, model_name: str) -> dict:
    """Prefer the held-out metrics saved with the model over rerunning inference."""
    metrics = vm.metrics
    if _has_confusion_counts(metrics):
        return metrics

    artifact_path = ARTIFACTS_DIR / f"{dataset}_{model_name}.joblib"
    return _compute_clean_metrics(dataset, model_name, artifact_path.stat().st_mtime_ns)


def _read_augmented_training_archive(contents: bytes) -> tuple[dict, str]:
    """Read the documented non-pickle NPZ format and return its digest."""
    archive_digest = hashlib.sha256(contents).hexdigest()
    try:
        with np.load(io.BytesIO(contents), allow_pickle=False) as archive:
            required = {"X_raw", "y", "source_indices"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(
                    "NPZ is missing required arrays: " + ", ".join(sorted(missing))
                )
            arrays = {name: archive[name] for name in required}
    except (OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Could not read the augmented-data NPZ: {exc}") from exc
    return arrays, archive_digest


def _clean_comparison_path(dataset: str, model_name: str) -> Path:
    return RESULTS_DIR / "E9" / f"{dataset}_{model_name}_augmented_comparison.json"


def _has_comparison_metrics(metrics) -> bool:
    rate_names = ("f1", "precision", "recall", "fpr", "fnr")
    return (
        isinstance(metrics, dict)
        and all(
            isinstance(metrics.get(name), (int, float))
            and not isinstance(metrics[name], bool)
            and np.isfinite(metrics[name])
            and 0 <= metrics[name] <= 1
            for name in rate_names
        )
        and isinstance(metrics.get("fn"), int)
        and not isinstance(metrics.get("fn"), bool)
        and metrics["fn"] >= 0
    )


def render_augmented_training(dataset: str, model_name: str, baseline) -> None:
    """Offer a separate clean-evaluation run using user-supplied training rows."""
    result_path = _clean_comparison_path(dataset, model_name)
    with st.expander(
        "Train a separate model from supplied training augmentations",
        expanded=True,
    ):
        st.markdown(
            "Use this workflow to compare the baseline with a model trained on "
            "augmentations you already prepared. Upload an NPZ containing "
            "`X_raw` (rows × features), `y` (labels), and `source_indices` "
            "(source row in the selected clean training split). The labels must "
            "match their source rows. This page does not generate augmentations; "
            "review their validity and provenance before training."
        )
        augmented_file = st.file_uploader(
            "Precomputed training augmentations (.npz)",
            type=["npz"],
            key=f"augmented_training_file_{dataset}_{model_name}",
            help="Required arrays: X_raw (N x features), y (N,), source_indices (N,).",
        )
        if st.button(
            "Train separate comparison model",
            disabled=augmented_file is None,
            key=f"train_augmented_{dataset}_{model_name}",
            type="secondary",
        ):
            try:
                augmented_arrays, archive_digest = _read_augmented_training_archive(
                    augmented_file.getvalue()
                )
                load_dataset(dataset)
                train_arrays = get_train_arrays(dataset)
                test_arrays = get_test_arrays(dataset)
                dataset_meta = get_meta(dataset)
                baseline_meta = getattr(baseline, "meta", {})
                baseline_source_hash = baseline_meta.get("source_hash")
                current_source_hash = dataset_meta.get("source_hash")
                if (
                    baseline_source_hash
                    and current_source_hash
                    and baseline_source_hash != current_source_hash
                ):
                    raise ValueError(
                        "The loaded baseline was trained from a different dataset source. "
                        "Refresh the baseline before comparing."
                    )
                if list(baseline.feature_names or []) != list(dataset_meta["feature_names"]):
                    raise ValueError(
                        "The loaded baseline feature order does not match the selected dataset."
                    )
                expected_scaler = StandardScaler().fit(train_arrays["X_raw"])
                baseline_scaler = getattr(baseline, "scaler_", None)
                if (
                    baseline_scaler is None
                    or not np.allclose(baseline_scaler.mean_, expected_scaler.mean_)
                    or not np.allclose(baseline_scaler.scale_, expected_scaler.scale_)
                ):
                    raise ValueError(
                        "The loaded baseline preprocessing does not match the current "
                        "clean training split. Refresh the baseline before comparing."
                    )
                baseline_seed = int(getattr(baseline, "seed", 0))
                baseline_epochs = getattr(baseline, "epochs", None)
                training_epochs = int(baseline_epochs or 50)
                training_batch_size = int(getattr(baseline, "batch_size", 256))
                baseline_params = getattr(baseline, "model_params", {}) or None

                with st.status("Training a separate augmented-data model...", expanded=True) as status:
                    status.write("Validating source indices and inherited training labels...")
                    status.write("Fitting preprocessing on clean training rows only...")
                    augmented_artifact, augmented_metrics = train_augmented_and_save_model(
                        dataset,
                        model_name,
                        train_arrays,
                        augmented_arrays,
                        test_arrays,
                        dataset_meta,
                        seed=baseline_seed,
                        epochs=training_epochs,
                        batch_size=training_batch_size,
                        model_params=baseline_params,
                        augmentation_sha256=archive_digest,
                    )
                    baseline_probs = baseline.predict_proba(test_arrays["X_raw"])
                    baseline_metrics = compute_metrics(
                        test_arrays["y"],
                        np.argmax(baseline_probs, axis=1),
                        baseline_probs,
                    )
                    baseline_scaler = getattr(baseline, "scaler_", None)
                    defended_scaler = augmented_artifact["scaler"]
                    preprocessing_matches = (
                        baseline_scaler is None and defended_scaler is None
                    ) or (
                        baseline_scaler is not None
                        and defended_scaler is not None
                        and np.allclose(baseline_scaler.mean_, defended_scaler.mean_)
                        and np.allclose(baseline_scaler.scale_, defended_scaler.scale_)
                    )
                    result = {
                        "experiment": "clean_performance_after_supplied_training_augmentation",
                        "dataset": dataset,
                        "model": model_name,
                        "baseline_metrics": baseline_metrics,
                        "augmented_metrics": augmented_metrics,
                        "training": {
                            "strategy": "clean_plus_supplied_augmentations",
                            "n_clean_train": int(len(train_arrays["y"])),
                            "n_augmented_train": int(len(augmented_arrays["y"])),
                            "seed": baseline_seed,
                            "epochs": training_epochs if baseline.is_torch else None,
                            "batch_size": training_batch_size if baseline.is_torch else None,
                            "model_params": baseline_params or {},
                            "augmentation_file_sha256": archive_digest,
                            "dataset_source_hash": current_source_hash,
                            "split_seed": dataset_meta.get("split_seed"),
                            "preprocessing_matches_baseline": bool(preprocessing_matches),
                        },
                        "scope": (
                            "Clean held-out evaluation only. This workflow does not "
                            "generate or evaluate adversarial examples."
                        ),
                    }
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    with result_path.open("w", encoding="utf-8") as output:
                        json.dump(result, output, indent=2, default=str)
                    status.update(
                        label="Separate model trained and clean-set evaluation saved",
                        state="complete",
                    )
                st.success(
                    f"Saved the separate model to "
                    f"`artifacts/{dataset}_{model_name}_augmented.joblib`."
                )
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                ImportError,
                RuntimeError,
                zipfile.BadZipFile,
            ) as exc:
                st.error(f"Augmented-data training failed: {exc}")

        if result_path.is_file():
            try:
                with result_path.open(encoding="utf-8") as result_file:
                    result = json.load(result_file)
                st.markdown("### Clean held-out comparison")
                render_clean_model_comparison(
                    result["baseline_metrics"],
                    result["augmented_metrics"],
                )
                if not result["training"]["preprocessing_matches_baseline"]:
                    st.warning(
                        "The preprocessing values differ from the loaded baseline "
                        "artifact; interpret this comparison with that caveat."
                    )
                st.caption(
                    f"Training rows: {result['training']['n_clean_train']:,} clean + "
                    f"{result['training']['n_augmented_train']:,} supplied augmentations. "
                    "No adversarial robustness evaluation is included."
                )
                with st.expander("Inspect comparison metadata"):
                    st.json(result)
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                st.error(f"Could not read the saved clean comparison: {exc}")


# ---------------------------------------------------------------------------
# Tab: Setup
# ---------------------------------------------------------------------------
def render_setup():
    first_visit = not st.session_state.get("_setup_intro_seen", False)
    render_setup_hero(animate=first_visit)
    st.session_state["_setup_intro_seen"] = True

    # Discover available datasets
    proc_dir = DATA_DIR / "processed"
    available = ["dummy"]
    if proc_dir.exists():
        for f in proc_dir.glob("*_test.npz"):
            name = f.name.replace("_test.npz", "")
            if name != "dummy":
                available.append(name)

    # Prefer CICIDS2017 when a trained real-data model is available.
    default_dataset = (
        "cicids2017"
        if "cicids2017" in available
        and any(ARTIFACTS_DIR.glob("cicids2017_*.joblib"))
        else "dummy"
    )
    default_idx = available.index(default_dataset)
    uploaded_ds = st.session_state.get("uploaded_dataset")
    if uploaded_ds and uploaded_ds in available:
        available = list(dict.fromkeys([uploaded_ds] + available))
        default_idx = available.index(uploaded_ds)

    st.markdown(
        '<div class="setup-section-heading setup-reveal">'
        '<span>01</span><div><strong>Configure the clean baseline</strong>'
        '<small>Data source and detector settings</small></div></div>',
        unsafe_allow_html=True,
    )
    data_column, model_column = st.columns([1.08, 0.92], gap="large")
    with data_column:
        st.markdown(
            '<div class="setup-control-panel setup-reveal">'
            '<div class="setup-control-heading"><span class="control-index">A</span>'
            '<div><strong>Traffic source</strong><small>Select prepared data or import a labeled CSV.</small></div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.container():
            uploaded = st.file_uploader(
                "Import a labeled CSV",
                type=["csv"],
                help="Expected a label, class, or target column and numeric feature columns."
            )
            if uploaded is not None:
                ds_name = uploaded.name.replace(".csv", "")
                new_ds = process_uploaded_csv(uploaded, ds_name)
                if new_ds:
                    st.session_state["uploaded_dataset"] = new_ds

            dataset = st.selectbox(
                "Prepared dataset",
                options=available,
                index=default_idx,
                help="Uploaded CSV datasets appear here after processing.",
            )
            st.session_state["dataset"] = dataset

    with model_column:
        st.markdown(
            '<div class="setup-control-panel setup-reveal">'
            '<div class="setup-control-heading"><span class="control-index coral">B</span>'
            '<div><strong>Detection model</strong><small>Choose the classifier for clean-set evaluation.</small></div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.container():
            model_name = st.selectbox(
                "Victim model",
                ALL_MODELS,
                index=0,
                format_func=lambda name: {
                    "lr": "Logistic Regression",
                    "rf": "Random Forest",
                    "xgb": "XGBoost",
                    "mlp": "MLP",
                    "cnn1d": "1D CNN",
                    "ensemble": "Soft-voting ensemble (LR + RF + XGBoost)",
                }.get(name, name),
                help="The ensemble averages class probabilities from LR, RF, and XGBoost.",
            )
            epochs = st.number_input(
                "Deep-model epochs",
                min_value=1,
                max_value=500,
                value=50,
                help="Used by MLP and 1D CNN. Sklearn models ignore this setting.",
            )

    try:
        dataset_arr = get_test_arrays(dataset)
        dataset_meta = get_meta(dataset)
        y = dataset_arr["y"]
        render_dataset_profile(
            dataset,
            int(dataset_meta["n_features"]),
            int(dataset_meta["n_train"]),
            len(y),
            int((y == BENIGN).sum()),
            int((y == ATTACK).sum()),
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        st.warning(f"Dataset summary is unavailable: {exc}")

    st.markdown(
        '<div class="train-ribbon setup-reveal"><div>'
        '<span>02 / FIT THE DETECTOR</span><strong>Train on the training split.<br>'
        'Measure only on held-out data.</strong>'
        '<small>Model artifacts are saved locally. Clean-set diagnostics stay separate from the training process.</small>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    train_model = st.button(
        "Train / refresh selected model",
        type="primary",
        use_container_width=True,
    )

    if train_model:
        try:
            with st.status("Training model...", expanded=True) as status:
                load_dataset(dataset)
                st.write("Loading the train and held-out test splits...")
                train_arrays = get_train_arrays(dataset)
                test_arrays = get_test_arrays(dataset)
                meta = get_meta(dataset)
                st.write(f"Fitting on {len(train_arrays['y']):,} training flows...")
                _, trained_metrics = train_and_save_model(
                    dataset,
                    model_name,
                    train_arrays,
                    test_arrays,
                    meta,
                    epochs=int(epochs),
                )
                status.update(label="Training and test evaluation complete", state="complete")
            load_model.clear()
            st.success(
                f"Saved `{dataset}_{model_name}`. "
                f"Test F1: {trained_metrics['f1']:.3f} · "
                f"AUROC: {trained_metrics['auroc']:.3f}"
            )
        except (FileNotFoundError, ImportError, ValueError, OSError, RuntimeError) as exc:
            st.error(f"Training failed: {exc}")

    st.session_state["model_trained"] = False
    try:
        vm = load_model(dataset, model_name)
        st.session_state["victim_model"] = vm
        st.session_state["model_trained"] = True
        st.session_state["model_name"] = model_name

        m = vm.metrics
        if m:
            st.markdown(
                '<div class="setup-section-heading setup-reveal">'
                '<span>03</span><div><strong>Clean-set diagnostics</strong>'
                '<small>Observed performance on the existing held-out split</small></div></div>',
                unsafe_allow_html=True,
            )
            render_clean_scoreboard(m)
    except FileNotFoundError as e:
        st.warning(str(e))

    # Confusion matrix + feature importance (when model is loaded)
    if st.session_state.get("model_trained"):
        vm = st.session_state["victim_model"]
        try:
            metrics = _clean_diagnostics_metrics(vm, dataset, model_name)
            render_clean_diagnostics(metrics, key="setup_clean_cohort")

            fns = vm.feature_names or []
            if hasattr(vm.base, "feature_importances_"):
                imps = vm.base.feature_importances_
                importance_label = "Tree-based importance"
            elif hasattr(vm.base, "coef_"):
                imps = np.abs(vm.base.coef_[0])
                importance_label = "Absolute model coefficients"
            else:
                imps = None
                importance_label = ""

            if imps is not None:
                fi_df = pd.DataFrame({
                    "feature": fns[:len(imps)],
                    "importance": imps,
                }).sort_values("importance", ascending=True).tail(15)
                st.markdown("### Most influential features")
                st.markdown(
                    f'<p class="section-intro">{importance_label} · top 15.</p>',
                    unsafe_allow_html=True,
                )
                st.bar_chart(fi_df.set_index("feature"))
        except (OSError, KeyError, ValueError, AttributeError, RuntimeError) as exc:
            st.warning(f"Model diagnostics could not be generated: {exc}")

    if st.session_state.get("model_trained"):
        render_augmented_training(
            dataset,
            model_name,
            st.session_state["victim_model"],
        )

    with st.expander("Why this baseline matters"):
        st.markdown(
            "A strong clean-data score does not guarantee robustness. "
            "This baseline records performance on the held-out split before "
            "testing whether small, constrained feature changes alter a prediction."
        )


# ---------------------------------------------------------------------------
# Tab: Attack Console
# ---------------------------------------------------------------------------
def render_attack():
    render_page_header(
        "02 / Robustness probe",
        "A small change.\nA different verdict?",
        "Choose one held-out connection and compare the detector's judgment before and after a bounded adversarial perturbation.",
    )

    if not st.session_state.get("model_trained"):
        st.info("Load a model in **Setup** first.")
        return

    vm = st.session_state["victim_model"]
    dataset = st.session_state["dataset"]
    arr, meta = load_data(dataset)
    X = arr["X_raw"]
    y = arr["y"]
    families = arr["y_family"]
    feature_names = vm.feature_names
    mutable_mask = vm.mutable_mask

    st.markdown("**Step 1 — Pick a flow**: Select a network flow to inspect.")
    label_filter = st.radio("Show", ("All flows", "Attack flows only"),
                            horizontal=True,
                            help="Attack flows are the evasion targets.")
    idxs = np.arange(len(y))
    if label_filter == "Attack flows only":
        idxs = idxs[y == ATTACK]

    if len(idxs) == 0:
        st.warning("No flows match this filter in the selected test split.")
        return

    idx = st.selectbox(
        "Select flow",
        options=list(idxs),
        format_func=lambda i: f"#{i:04d} · {LABEL_NAMES[y[i]]} · {families[i]}",
        key="flow_idx",
    )
    x_clean = X[idx:idx+1]
    true_label = int(y[idx])
    true_family = str(families[idx])

    probs_before = vm.predict_proba(x_clean)[0]
    pred_before = int(np.argmax(probs_before))
    conf_before = float(probs_before[1])  # P(attack)
    st.markdown("**Step 2 — Before attack**: The model classifies this flow.")
    render_prediction_card(
        LABEL_NAMES[pred_before],
        LABEL_NAMES[true_label],
        conf_before,
        true_family,
    )

    st.markdown("**Step 3 — Configure attack**: Select an attack and perturbation budget.")
    col_a, col_b = st.columns(2)
    with col_a:
        method = st.radio("Attack method", ("FGSM", "PGD", "CW", "DeepFool", "JSMA", "NES"),
                          horizontal=True,
                          help="**FGSM** = single-step gradient sign. **PGD** = iterative projected gradient. "
                               "**CW** = optimization-based. **DeepFool** = boundary distance estimation. "
                               "**JSMA** = saliency-based. **NES** = black-box zeroth-order.")
        attack_fn = {"FGSM": fgsm, "PGD": pgd, "CW": cw, "DeepFool": deepfool,
                     "JSMA": jsma, "NES": nes_attack}[method]
    with col_b:
        eps = st.select_slider("Epsilon (ε)", options=EPS_GRID, value=0.2,
                               help="Maximum allowed perturbation per feature.")

    # Reset attacking flag on each render to avoid stuck state from prior crashes
    st.session_state.pop("_attacking", None)

    if st.button("Run attack ▶", type="primary", use_container_width=True,
                 disabled=st.session_state.get("_attacking", False),
                 key="run_attack_btn"):
        st.session_state["_attacking"] = True
        try:
            with st.status(f"Running {method} attack (ε={eps})...",
                           expanded=True) as status:
                st.markdown('<div class="loader-container"><div class="loader"></div>'
                            '<p>Perturbing network-flow features...</p></div>',
                            unsafe_allow_html=True)
                result = attack_fn(vm, x_clean, eps, mask=mutable_mask)
                x_adv = result.x_adv  # AttackResult -> raw ndarray
                queries = result.n_queries
                probs_after = vm.predict_proba(x_adv)[0]
                pred_after = int(np.argmax(probs_after))

                valid, violations = check_validity(x_adv, x_clean, meta, eps, model=vm)
                _, behaviour_msg = check_behaviour_preserved(pred_before, pred_after, true_label)
                l1, linf = perturbation_magnitude(x_adv, x_clean)
                diff = feature_diff(x_adv, x_clean, feature_names, top_n=10)
                evasion = bool(pred_after == BENIGN and true_label == ATTACK and pred_before == ATTACK)
                grad = vm.gradient(x_clean, target=BENIGN)
                grad = grad.reshape(-1) if grad.ndim > 1 else grad

                st.session_state["result"] = dict(
                    evasion=evasion, valid=valid, violations=violations,
                    msg=behaviour_msg, label_before=LABEL_NAMES[pred_before],
                    label_after=LABEL_NAMES[pred_after],
                    probs_before={"benign": float(probs_before[0]), "attack": float(probs_before[1])},
                    probs_after={"benign": float(probs_after[0]), "attack": float(probs_after[1])},
                    l1=l1, linf=linf, diff=diff, grad=grad,
                    x_clean=x_clean, x_adv=x_adv, true_label=true_label,
                    mutable_mask=mutable_mask, feature_names=feature_names,
                    eps=eps, method=method, queries=queries,
                )
                status.update(label=f"✓ {method} attack complete", state="complete",
                              expanded=False)
        finally:
            st.session_state["_attacking"] = False

    r = st.session_state.get("result")
    if not r:
        st.caption("Pick a flow and click 'Run attack' to see the result.")
        return

    # --- Results ---
    st.markdown("---")
    st.subheader("Result — what happened?")

    if r["evasion"]:
        st.error("**Evasion succeeded** — the model now classifies this attack as benign.\n"
                 f"P(attack): {r['probs_before']['attack']:.4f} → {r['probs_after']['attack']:.4f}")
    elif not r["valid"]:
        st.warning(f"Invalid — {len(r['violations'])} constraint violations")
    else:
        st.success(f"Blocked — model still says {r['label_after']}")

    st.markdown("**After attack — model's new prediction**")
    col_e, col_f, col_g = st.columns(3)
    col_e.metric("Before", f"P(attack) = {r['probs_before']['attack']:.4f}")
    col_f.metric("After", f"P(attack) = {r['probs_after']['attack']:.4f}")
    col_g.metric("Status", r["msg"])

    st.markdown("### Perturbation cost")
    st.markdown(
        '<p class="section-intro">L1 summarizes total feature movement; L∞ is the largest '
        'single-feature change. Query count records model evaluations used by the method.</p>',
        unsafe_allow_html=True,
    )
    render_metric_cards([
        ("L1 norm", f"{r['l1']:.4f}", "Total absolute change"),
        ("L∞ norm", f"{r['linf']:.4f}", "Largest feature change"),
        ("Queries", f"{r.get('queries', 0):,}", "Model evaluations"),
    ])

    # Validity + behaviour checks
    st.markdown("### Constraint checks")
    st.markdown(
        '<p class="section-intro">Validity checks enforce feature bounds, immutability, and '
        'the epsilon budget; behavior checks compare the prediction before and after.</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    with cols[0]:
        icon = "✅" if r["valid"] else "❌"
        st.markdown(f"{icon} **Validity**: {'pass' if r['valid'] else 'FAIL'}")
        if r["violations"]:
            for v in r["violations"]:
                st.caption(f"  - {v}")
    with cols[1]:
        ok = "preserved" in r["msg"].lower()
        icon = "✅" if ok else "❌"
        st.markdown(f"{icon} **Behaviour**: {r['msg']}")

    # Feature table — the key visualization
    st.markdown("---")
    st.markdown(f"**Which features did the attack change? (top {len(r['diff'])})**")
    st.caption("The attack pushes on the features the model relies on most. "
               "Positive delta (cyan) = increased; Negative delta (red) = decreased.")

    diff_df = pd.DataFrame(r["diff"])
    if len(diff_df) == 0 or "delta" not in diff_df.columns:
        st.caption("No features were modified by the attack (or perturbation was below threshold).")
    else:
        def color_delta(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return "color: #70dcff"
                elif val < 0:
                    return "color: #ff9e92"
            return ""
        styled = diff_df.style.map(color_delta, subset=["delta"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # Gradient magnitudes
    grad_masked = np.abs(r["grad"]) * r["mutable_mask"].astype(float)
    if grad_masked.any():
        top_idx = np.argsort(grad_masked)[::-1][:15]
        grad_df = pd.DataFrame({
            "feature": [feature_names[i] for i in top_idx],
            "gradient": grad_masked[top_idx],
        }).set_index("feature")
        with st.expander("Gradient magnitudes — what the attack targets"):
            st.caption("These are the features the model's decision depends on most. "
                       "The attack focuses its perturbation budget here.")
            st.bar_chart(grad_df)


# ---------------------------------------------------------------------------
# Tab: Results Gallery
# ---------------------------------------------------------------------------
def render_gallery():
    render_page_header(
        "03 / Clean-set evidence",
        "Understand the results.",
        "Compare real held-out scores, see exactly what each model gets wrong, and find any saved clean-set comparison from supplied training augmentations.",
    )
    dataset = st.session_state.get("dataset", "dummy")

    e1_path = RESULTS_DIR / "E1" / f"{dataset}_clean_metrics.json"
    if e1_path.exists():
        st.markdown("### Clean model evaluation")
        try:
            with e1_path.open(encoding="utf-8") as result_file:
                data = json.load(result_file)
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read the clean-metrics result: {exc}")
        else:
            required_scores = ("accuracy", "f1", "recall", "auroc")
            model_results = {
                model: metrics
                for model, metrics in data.items()
                if isinstance(data, dict)
                and isinstance(model, str)
                and isinstance(metrics, dict)
                and all(
                    isinstance(metrics.get(name), (int, float))
                    and not isinstance(metrics.get(name), bool)
                    and np.isfinite(metrics[name])
                    and 0 <= metrics[name] <= 1
                    for name in required_scores
                )
            } if isinstance(data, dict) else {}

            if model_results:
                st.markdown("#### Compare the models")
                render_results_overview(dataset, model_results)
                render_clean_metric_chart(model_results)
                animation_key = f"_results_landscape_seen_{dataset}"
                render_clean_model_landscape(
                    model_results,
                    animate=not st.session_state.get(animation_key, False),
                )
                st.session_state[animation_key] = True
                model_options = sorted(model_results)
                highest_f1_model = max(
                    model_options,
                    key=lambda model: float(model_results[model]["f1"]),
                )
                selected_model = st.selectbox(
                    "Inspect one model's detailed outcome",
                    options=model_options,
                    index=model_options.index(highest_f1_model),
                    key=f"gallery_clean_model_{dataset}",
                )
                selected_metrics = model_results[selected_model]
                render_clean_scoreboard(selected_metrics)
                if all(
                    isinstance(selected_metrics.get(name), int)
                    and not isinstance(selected_metrics.get(name), bool)
                    and selected_metrics[name] >= 0
                    for name in ("tn", "fp", "fn", "tp")
                ):
                    render_clean_diagnostics(
                        selected_metrics,
                        key=f"gallery_clean_cohort_{dataset}",
                    )
                else:
                    st.info("This result contains no complete confusion counts for a class-level breakdown.")
            else:
                st.warning("This file does not match the expected model-to-clean-metrics structure.")

            with st.expander("Inspect raw E1 result data"):
                st.json(data)
    else:
        st.info(
            f"No clean-metric result is available for **{dataset}** yet. "
            "Train and evaluate a model to create the E1 artifact."
        )

    st.markdown("### Training with supplied augmentations")
    augmented_result_files = sorted(
        (RESULTS_DIR / "E9").glob(f"{dataset}_*_augmented_comparison.json")
    )
    if augmented_result_files:
        result_by_model = {}
        for result_file in augmented_result_files:
            try:
                with result_file.open(encoding="utf-8") as file:
                    result = json.load(file)
                model = result.get("model")
                if (
                    isinstance(model, str)
                    and _has_comparison_metrics(result.get("baseline_metrics"))
                    and _has_comparison_metrics(result.get("augmented_metrics"))
                    and isinstance(result.get("training"), dict)
                ):
                    result_by_model[model] = result
                else:
                    st.warning(f"Unsupported comparison schema in {result_file.name}.")
            except (OSError, json.JSONDecodeError) as exc:
                st.error(f"Could not read {result_file.name}: {exc}")

        if result_by_model:
            comparison_model = st.selectbox(
                "Model comparison",
                options=sorted(result_by_model),
                key=f"augmented_result_model_{dataset}",
            )
            comparison = result_by_model[comparison_model]
            render_clean_model_comparison(
                comparison["baseline_metrics"],
                comparison["augmented_metrics"],
            )
            training = comparison["training"]
            st.caption(
                f"{training['n_clean_train']:,} clean training rows + "
                f"{training['n_augmented_train']:,} supplied augmentations. "
                "This is a clean-test comparison, not an adversarial robustness result."
            )
            with st.expander("Inspect comparison metadata"):
                st.json(comparison)
        else:
            st.warning("Comparison files exist, but none contain a supported clean-set result.")
    else:
        st.html(
            """
            <section class="augmentation-empty-state">
              <p class="comparison-kicker">NO SAVED COMPARISON YET</p>
              <h3>Run the supplied-data training workflow to see it here.</h3>
              <ol>
                <li>Open <strong>Model Setup</strong> and load or train the baseline for this dataset.</li>
                <li>Expand <strong>Train a separate model from supplied training augmentations</strong>.</li>
                <li>Upload an NPZ with <code>X_raw</code>, <code>y</code>, and <code>source_indices</code>, then train.</li>
              </ol>
              <p>The comparison uses the same clean held-out test set. It does not generate augmentations or measure adversarial robustness.</p>
            </section>
            """
        )

    st.markdown("### Generate batch results")
    st.markdown(
        '<p class="section-intro">Run experiments locally to populate attack-success '
        'curves and other result artifacts.</p>',
        unsafe_allow_html=True,
    )
    st.code("python -m nids.experiments --e2 --e3", language="bash")
    with st.expander("Research questions"):
        st.markdown(
            "- **RQ1** — Which model architectures are most vulnerable?\n"
            "- **RQ2** — Which features are most susceptible? Inspect feature changes and gradients.\n"
            "- **RQ3** — Can adversarial training improve robustness? Compare the supported experiments."
        )


# ---------------------------------------------------------------------------
# Tab: About
# ---------------------------------------------------------------------------
def render_about():
    render_page_header(
        "04 / Research notes",
        "Robustness is a property to test.",
        "A reproducible research console for studying adversarial evasion in network intrusion detection.",
    )
    st.markdown("""
### Research question
Can a malicious network flow be perturbed within realistic constraints so that an
intrusion-detection model classifies it as benign?

### Architecture
1. **config** — frozen seeds, epsilon grid, label convention
2. **data** — CICIDS2017 / UNSW-NB15 network-flow features
3. **models** — LR, RF, XGBoost, MLP, CNN1D, and their soft-voting LR/RF/XGBoost ensemble
4. **attacks** — FGSM, PGD, CW, DeepFool, JSMA (white-box), NES (black-box), projection
5. **eval** — metrics (F1, AUROC, FPR), validity checks, transfer matrix
6. **dashboard** — this UI

### Key terms
- **Epsilon (ε)**: max perturbation per feature. Controls attack strength.
- **FGSM**: nudge each feature by ε × sign(gradient) — one shot.
- **PGD**: iterate FGSM with projection — stronger than FGSM.
- **CW**: Carlini-Wagner optimization — finds minimal perturbation via Adam + binary search — strongest white-box attack.
- **DeepFool**: Line-search to decision boundary — finds perturbation direction to cross the boundary.
- **JSMA**: Jacobian-based salience map — targets specific features for sparse perturbation.
- **NES**: Black-box zeroth-order — estimates gradients via random sampling, no model access needed.
- **Validity**: perturbed sample stays within realistic bounds.
- **Evasion**: attack succeeded — malicious packet classified as benign.
- **L1/L∞ norms**: how much features changed (total vs. max single change).
""")
    st.markdown("### Train models")
    st.code(
        "python -m scripts.pretrain --dataset cicids2017 "
        "--models lr rf xgb mlp cnn1d ensemble",
        language="bash",
    )
    st.caption(
        "Provide and prepare the full labeled dataset before training. "
        "The built-in dummy data is synthetic and for pipeline checks only. "
        "CW attack is ready — select 'CW' in the Attack Console."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False

    with st.sidebar:
        st.markdown(
            '<div class="brand-lockup">'
            '<p class="eyebrow">RESEARCH CONSOLE</p>'
            '<strong>Adversarial NIDS</strong>'
            '<p>Network detection robustness</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            '<p class="sidebar-nav-label">Navigate</p>',
            unsafe_allow_html=True,
        )
        workspace = st.radio(
            "Workspace",
            ("Model setup", "Attack console", "Results gallery", "About"),
            key="active_workspace",
            label_visibility="collapsed",
            width="stretch",
        )

    if workspace == "Model setup":
        render_setup()
    elif workspace == "Attack console":
        render_attack()
    elif workspace == "Results gallery":
        render_gallery()
    else:
        render_about()


if __name__ == "__main__":
    main()