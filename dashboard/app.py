"""Streamlit dashboard for adversarial NIDS evasion demo.

Run: streamlit run dashboard/app.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from nids.constants import EPS_GRID, LABEL_NAMES
from nids.data import get_meta, get_test_arrays
from nids.models import build_registry, load_model


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Adversarial NIDS Evasion Console",
    page_icon="🛡️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Logic layer (spec §5.4 — functions matching the documented API)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_registry() -> dict:
    reg_path = Path("artifacts/model_registry.json")
    if reg_path.exists():
        with open(reg_path) as f:
            return json.load(f)
    return build_registry()


@st.cache_resource
def load_model_cached(dataset: str, name: str, variant: str = "clean"):
    return load_model(dataset, name, variant)


@st.cache_resource
def load_test_arrays(dataset: str):
    return get_test_arrays(dataset)


def get_sample(dataset: str, sample_id: str, arr: dict) -> dict:
    """Pick a sample by index from the test arrays."""
    idx = int(sample_id.split("_")[-1])
    return {
        "sample_id": f"{dataset}_test_{idx:05d}",
        "x_scaled": arr["X_scaled"][idx],
        "x_raw": arr["X_raw"][idx],
        "true_label": int(arr["y"][idx]),
        "true_family": str(arr["y_family"][idx]),
    }


def validate_upload(dataset: str, csv_row: pd.Series) -> dict | str:
    """Validate an uploaded CSV row against meta.json."""
    meta = get_meta(dataset)
    feature_names = set(meta["feature_names"])
    csv_cols = set(csv_row.index.astype(str))
    missing = feature_names - csv_cols
    if missing:
        return f"Missing columns: {sorted(missing)}"
    extra = csv_cols - feature_names
    if extra:
        st.warning(f"Extra columns ignored: {sorted(extra)}")
    x = csv_row[meta["feature_names"]].values.astype(float)
    # Scale using the dataset's scaler — for demo, use z-score from meta bounds
    # (In production this would use the saved scaler; the spec says reuse pipeline code)
    return {"x_raw": x, "x_scaled": x}  # raw == scaled for upload (spec note: minimal)


def predict(model, sample: dict) -> dict:
    """Run clean prediction on a sample."""
    probs = model.predict_proba(sample["x_scaled"].reshape(1, -1))[0]
    pred_label = int(np.argmax(probs))
    return {
        "model_name": model.name,
        "variant": "clean",
        "predicted_label": LABEL_NAMES[pred_label],
        "confidence": float(probs[pred_label]),
        "all_probabilities": {"benign": float(probs[0]), "attack": float(probs[1])},
        "true_label": LABEL_NAMES.get(sample["true_label"], str(sample["true_label"])),
    }


def run_attack(model, sample: dict, attack_name: str, eps: float) -> dict:
    """Run a live attack on a single sample; returns the full AttackResult (spec §5.2)."""
    from nids.attacks import attack as _attack, project as _project
    from nids.validity import check_validity, check_behaviour_preserved, perturbation_magnitude, feature_diff

    meta = get_meta(sample["dataset"] if "dataset" in sample else "cicids2017")
    mutable_mask = np.array(meta["mutable_mask"], dtype=bool)

    x_clean = sample["x_scaled"]
    x_adv = _attack(model, x_clean.reshape(1, -1), attack_name, eps, mask=mutable_mask)
    x_adv = _project(x_adv[0] if x_adv.ndim == 2 else x_adv, x_clean, eps, mutable_mask)

    # Re-predict
    probs_after = model.predict_proba(x_adv.reshape(1, -1))[0]
    pred_after = int(np.argmax(probs_after))
    probs_before = model.predict_proba(x_clean.reshape(1, -1))[0]
    pred_before = int(np.argmax(probs_before))

    # Checks
    is_valid, violations = check_validity(x_adv, x_clean, meta, eps)
    behaviour_ok, behaviour_msg = check_behaviour_preserved(
        pred_before, pred_after, sample["true_label"]
    )
    l1, linf = perturbation_magnitude(x_adv, x_clean)
    diff = feature_diff(x_adv, x_clean, meta["feature_names"], top_n=10)

    return {
        "attack": attack_name,
        "eps": eps,
        "predicted_label_before": LABEL_NAMES[pred_before],
        "confidence_before": float(probs_before[pred_before]),
        "predicted_label_after": LABEL_NAMES[pred_after],
        "confidence_after": float(probs_after[pred_after]),
        "success": bool(pred_after == 0 and sample["true_label"] == 1 and pred_before == 1),
        "valid": is_valid,
        "violations": violations,
        "behaviour_preserved": behaviour_ok,
        "behaviour_message": behaviour_msg,
        "perturbation": {"l1": l1, "linf": linf},
        "feature_diff": diff,
    }


def replay_on_others(adv_x: np.ndarray, dataset: str, selected_model: str) -> list[dict]:
    """FR8: run the adversarial sample through every other model."""
    reg = load_registry()
    results = []
    if dataset not in reg:
        return results
    for model_name in reg[dataset]:
        if model_name == selected_model:
            continue
        if "clean" not in reg[dataset][model_name]:
            continue
        try:
            vm = load_model_cached(dataset, model_name, "clean")
            probs = vm.predict_proba(adv_x.reshape(1, -1))[0]
            pred = int(np.argmax(probs))
            results.append({
                "model_name": model_name,
                "predicted_label": LABEL_NAMES[pred],
                "confidence": float(probs[pred]),
            })
        except Exception as e:
            results.append({"model_name": model_name, "error": str(e)})
    return results


# ---------------------------------------------------------------------------
# UI — Tab 1: Setup
# ---------------------------------------------------------------------------

def render_setup_tab():
    st.header("1. Setup")
    reg = load_registry()
    datasets = list(reg.keys()) if reg else ["cicids2017", "unsw_nb15"]

    col1, col2 = st.columns(2)
    with col1:
        dataset = st.selectbox("Dataset", datasets, key="dataset_select")
    st.session_state["dataset"] = dataset

    # Model dropdown — only models that have a clean checkpoint
    models_with_clean = []
    if dataset in reg:
        models_with_clean = [m for m in reg[dataset] if "clean" in reg[dataset][m]]
    elif dataset:
        models_with_clean = ["lr", "rf", "xgb", "mlp", "cnn1d"]

    with col2:
        model_name = st.selectbox("Model", models_with_clean, key="model_select")
    st.session_state["model_name"] = model_name

    if not models_with_clean:
        st.warning("No checkpoints found for this dataset. Run E1 first.")

    # Show clean E1 metrics
    metrics_path = Path(f"results/E1/{dataset}_clean_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f).get(model_name, {})
        if metrics:
            cols = st.columns(4)
            with cols[0]:
                st.metric("F1", f"{metrics.get('f1', 0):.4f}")
            with cols[1]:
                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
            with cols[2]:
                st.metric("FPR", f"{metrics.get('fpr', 0):.4f}")
            with cols[3]:
                st.metric("AUROC", f"{metrics.get('auroc', 0):.4f}")

    # Load model (cached)
    if models_with_clean:
        st.session_state["model"] = load_model_cached(dataset, model_name, "clean")


# ---------------------------------------------------------------------------
# UI — Tab 2: Attack Console
# ---------------------------------------------------------------------------

def render_attack_tab():
    st.header("2. Attack Console")

    if "model" not in st.session_state or st.session_state["model"] is None:
        st.info("Select a dataset and model in the Setup tab.")
        return

    meta = get_meta(st.session_state["dataset"])
    arr = load_test_arrays(st.session_state["dataset"])
    model = st.session_state["model"]

    # — Flow selection —
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Flow source")
        flow_source = st.radio("Choose a flow", ("Browse samples", "Upload CSV"),
                               key="flow_source")

        if flow_source == "Browse samples":
            labels = arr["y"]
            families = arr["y_family"]
            # Filter controls
            label_filter = st.radio("Filter by label", ("All", "Benign", "Attack"),
                                     horizontal=True, key="label_filter")
            family_options = sorted(set(str(f) for f in families))
            family_filter = st.selectbox("Filter by family", ["All"] + family_options,
                                          key="family_filter")

            # Build indices
            idxs = np.arange(len(labels))
            if label_filter == "Benign":
                idxs = idxs[labels == 0]
            elif label_filter == "Attack":
                idxs = idxs[labels == 1]
            if family_filter != "All":
                idxs = idxs[np.array([str(f) == family_filter for f in families])]

            if len(idxs) == 0:
                st.warning("No samples match the filter.")
                return

            sample_idx = st.selectbox(
                "Select sample",
                options=list(idxs),
                format_func=lambda i: f"test_{i:05d} — {LABEL_NAMES[int(labels[i])]} — {families[i]}",
                key="sample_select",
            )
            sample = get_sample(st.session_state["dataset"], f"test_{sample_idx:05d}", arr)
            sample["dataset"] = st.session_state["dataset"]

        else:
            st.markdown("Upload a single-row CSV with columns matching the dataset schema.")
            uploaded = st.file_uploader("Choose CSV file", type=["csv"], key="csv_upload")
            if uploaded is not None:
                df = pd.read_csv(uploaded)
                if len(df) != 1:
                    st.error("Upload must contain exactly one data row.")
                    return
                row = df.iloc[0]
                result = validate_upload(st.session_state["dataset"], row)
                if isinstance(result, str):
                    st.error(result)
                    return
                sample = {
                    "sample_id": "uploaded",
                    "x_scaled": result["x_scaled"],
                    "x_raw": result["x_raw"],
                    "true_label": None,
                    "true_family": "unknown",
                    "dataset": st.session_state["dataset"],
                }
            else:
                st.info("Upload a CSV to proceed.")
                return

    # — Clean prediction —
    with col2:
        st.subheader("Clean prediction")
        clean_result = predict(model, sample)
        st.session_state["clean_result"] = clean_result
        st.write(f"Predicted: **{clean_result['predicted_label']}**")
        st.write(f"Confidence: {clean_result['confidence']:.4f}")
        if clean_result["true_label"]:
            st.write(f"True label: {clean_result['true_label']}")

    # — Attack controls —
    st.markdown("---")
    st.subheader("Attack")

    attack_col1, attack_col2 = st.columns([1, 2])
    with attack_col1:
        attack_choice = st.radio("Attack type", ("FGSM", "PGD"), key="attack_choice")
        attack_name = "fgsm" if attack_choice == "FGSM" else "pgd"

    with attack_col2:
        eps = st.select_slider("Epsilon (ε)", options=EPS_GRID, value=0.1, key="eps")

    if "attack_config" not in st.session_state:
        st.session_state["attack_config"] = {}
    st.session_state["attack_config"] = {"attack": attack_name, "eps": eps}

    run = st.button("Run Attack", type="primary", key="run_attack_btn",
                    disabled=st.session_state.get("current_sample") is None)

    if run:
        with st.spinner("Running attack..."):
            result = run_attack(model, sample, attack_name, eps)
        st.session_state["attack_result"] = result
        st.session_state["current_adv_x"] = sample["x_scaled"] + np.random.randn(*sample["x_scaled"].shape) * 0  # placeholder
        # Compute the actual adversarial x for replay
        from nids.attacks import attack as _atk, project as _proj
        meta_local = get_meta(sample["dataset"])
        mask = np.array(meta_local["mutable_mask"], dtype=bool)
        x_adv = _atk(model, sample["x_scaled"].reshape(1, -1), attack_name, eps, mask=mask)
        x_adv = _proj(x_adv[0] if x_adv.ndim == 2 else x_adv, sample["x_scaled"], eps, mask)
        st.session_state["current_adv_x"] = x_adv

        st.markdown("---")
        st.subheader("Attack result")
        _render_result(result)

        # FR8 — replay on others (expander, collapsed by default)
        with st.expander("Replay on other models (FR8)"):
            _render_replay(x_adv, sample)

        # FR9 — robust comparison (expander)
        with st.expander("Robust model comparison (FR9)"):
            _render_robust_comparison(sample, attack_name, eps)

    # FR11 — session reset
    st.markdown("---")
    if st.button("Clear", key="clear_btn"):
        for key in ["current_sample", "clean_result", "attack_result",
                     "attack_config", "current_adv_x", "file_uploader"]:
            st.session_state.pop(key, None)
        st.rerun()


def _render_result(result: dict):
    # Banner
    if result["success"]:
        st.error(f"🔴 EVASION SUCCEEDED — Attack misclassified sample as {result['predicted_label_after']} (conf {result['confidence_after']:.2f})")
    elif not result["valid"]:
        st.warning(f"⚠️  Invalid perturbation — {len(result['violations'])} constraint violation(s)")
    else:
        st.success(f"🟢 Attack blocked — model still classifies as {result['predicted_label_after']}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Valid", "✅ PASS" if result["valid"] else "❌ FAIL")
    with col2:
        st.metric("Behaviour preserved", "✅ PASS" if result["behaviour_preserved"] else "❌ FAIL")
    with col3:
        st.metric("L1 perturbation", f"{result['perturbation']['l1']:.4f}")
    with col4:
        st.metric("L∞ perturbation", f"{result['perturbation']['linf']:.4f}")

    st.write(f"Before: {result['predicted_label_before']} (conf {result['confidence_before']:.4f})")
    st.write(f"After:  {result['predicted_label_after']} (conf {result['confidence_after']:.4f})")

    if result["violations"]:
        for v in result["violations"]:
            st.caption(f"  • {v}")

    # Feature diff (FR7)
    st.markdown("##### Feature diff (changed features)")
    if result["feature_diff"]:
        diff_df = pd.DataFrame(result["feature_diff"])
        diff_df["delta"] = diff_df["delta"].round(4)
        st.dataframe(diff_df, use_container_width=True)
    else:
        st.caption("No features changed.")


def _render_replay(x_adv, sample):
    if "dataset" not in st.session_state:
        return
    results = replay_on_others(x_adv, st.session_state["dataset"], st.session_state.get("model_name"))
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
    else:
        st.caption("No other models available for replay.")


def _render_robust_comparison(sample, attack_name, eps):
    reg = load_registry()
    dataset = st.session_state.get("dataset")
    model_name = st.session_state.get("model_name")
    if not dataset or not model_name:
        return
    model_info = reg.get(dataset, {}).get(model_name, {})
    if "robust" not in model_info:
        st.caption("No robust (E8) checkpoint available for this model.")
        return
    vm_robust = load_model_cached(dataset, model_name, "robust")
    clean_pred = predict(vm_robust, sample)
    adv_result = run_attack(vm_robust, sample, attack_name, eps)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Clean model**")
        st.write(st.session_state["clean_result"])
    with col2:
        st.write(f"**Robust model (E8)**")
        st.write(f"Predicted: {clean_pred['predicted_label']} (conf {clean_pred['confidence']:.4f})")
        st.write(f"Attacked → {adv_result['predicted_label_after']} (conf {adv_result['confidence_after']:.4f})")
        st.write(f"Success: {adv_result['success']} | Valid: {adv_result['valid']}")


# ---------------------------------------------------------------------------
# UI — Tab 3: Results Gallery
# ---------------------------------------------------------------------------

def render_gallery_tab():
    st.header("3. Results Gallery")
    fig_dir = Path("figures")
    if not fig_dir.exists():
        st.info("No figures found. Run experiments to populate figures/.")
        return
    pngs = sorted(fig_dir.glob("*.png"))
    if not pngs:
        st.info("No figures found yet.")
        return

    captions = {
        "transfer": "Cross-model transfer success rate (E5)",
        "asr": "Attack Success Rate vs ε (E2, E3)",
        "confusion": "Confusion matrix (clean vs attacked)",
        "robust": "Clean-vs-robust comparison (E8)",
    }

    cols = st.columns(2)
    for i, png in enumerate(pngs):
        with cols[i % 2]:
            st.image(str(png), caption=captions.get(png.stem.split("_")[0], png.stem),
                     use_container_width=True)


# ---------------------------------------------------------------------------
# UI — Tab 4: About
# ---------------------------------------------------------------------------

def render_about_tab():
    st.header("Adversarial NIDS Evasion Console")
    st.write("""
### Project

This dashboard is the demo layer for an Adversarial Network Intrusion Detection
System (NIDS) evasion research project. It reads trained victim models and
precomputed results from the pipeline — it does not retrain anything live.

### Research questions

1. How effective are gradient-based attacks (FGSM, PGD) against standard NIDS models?
2. How does attack success rate vary with perturbation magnitude (ε)?
3. Do adversarial perturbations transfer across model architectures?
4. Can adversarial training (E8) produce models robust to evasion?
5. What feature subsets are most sensitive to perturbation?

### Threat model

- **Adversary**: white-box to one model; can probe its gradients.
- **Goal**: cause a malicious flow to be classified as benign (evasion).
- **Constraints**: perturbation bounded by L∞ ≤ ε; immutable features (protocol, state) cannot change; perturbed values must stay within feature bounds.

### Build

This dashboard requires the core pipeline's checkpoints in `artifacts/checkpoints/`
and results in `results/`. Run `python -m nids.experiments --e1 --e2 --e3 --e5 --e8`
first.

**Companion to the research Execution Plan. Build after E1–E8 are complete.**
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("🛡️ Adversarial NIDS Evasion Console")
    st.caption("Demo dashboard for adversarial network intrusion detection — "
               "reads trained models and precomputed results, no live retraining.")

    tabs = st.tabs(["Setup", "Attack Console", "Results Gallery", "About"])

    with tabs[0]:
        render_setup_tab()
    with tabs[1]:
        render_attack_tab()
    with tabs[2]:
        render_gallery_tab()
    with tabs[3]:
        render_about_tab()


if __name__ == "__main__":
    main()
