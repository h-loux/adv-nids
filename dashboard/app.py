"""Adversarial NIDS Evasion Console — Streamlit dashboard.

Loads pre-trained models from artifacts/ and lets you attack individual
network flows with FGSM, PGD, or Carlini-Wagner.

Run:  streamlit run dashboard/app.py
"""
import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from nids.config import EPS_GRID, LABEL_NAMES, BENIGN, ATTACK, ALL_MODELS
from nids.data import get_meta, get_test_arrays
from nids.models import VictimModel
from nids.attacks import fgsm, pgd, cw
from nids.eval.validity import (
    check_validity, check_behaviour_preserved, perturbation_magnitude, feature_diff,
)

st.set_page_config(page_title="Adversarial NIDS", page_icon="🏰", layout="wide")

# ---------------------------------------------------------------------------
# CSS — clean dark theme with cyan accents + Font Awesome fortress
# ---------------------------------------------------------------------------
st.markdown("""<script src="https://kit.fontawesome.com/a076d0569d.js" crossorigin="anonymous"></script>
<style>
#MainMenu,footer,header{visibility:hidden}
.stApp{background:#0f1117;color:#e4e4e4;font-family:'Segoe UI',sans-serif}
h1,h2,h3,h4{color:#ffffff;font-weight:700}
h2{border-bottom:2px solid #334155;padding-bottom:8px}
h3{color:#cbd5e1}
.stButton>button{background:#0ea5e9;color:white;border:none;border-radius:6px;padding:8px 20px;font-weight:500}
.stButton>button:hover{background:#0284c7}
.stProgress>div>div{border-radius:4px;background:#0ea5e9}
</style>""", unsafe_allow_html=True)

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

    X = df[feature_cols].values.astype(np.float64)
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
    return vm


@st.cache_resource
def load_data(dataset):
    return get_test_arrays(dataset), get_meta(dataset)


# ---------------------------------------------------------------------------
# Tab: Setup
# ---------------------------------------------------------------------------
def render_setup():
    st.subheader("Setup")
    st.caption("Select a dataset and model to begin.")

    # Discover available datasets
    proc_dir = DATA_DIR / "processed"
    available = ["dummy"]
    if proc_dir.exists():
        for f in proc_dir.glob("*_test.npz"):
            name = f.name.replace("_test.npz", "")
            if name != "dummy":
                available.append(name)

    # CSV upload
    uploaded = st.file_uploader(
        "Upload a CSV dataset (CICIDS2017, UNSW-NB15, or custom)",
        type=["csv"],
        help="Upload a CSV with a label column ('Label', 'label', 'class', or 'target'). "
             "Feature columns must be numeric. The app auto-detects and processes it."
    )
    if uploaded is not None:
        ds_name = uploaded.name.replace(".csv", "")
        new_ds = process_uploaded_csv(uploaded, ds_name)
        if new_ds:
            available = list(dict.fromkeys([new_ds] + available))
            st.rerun()

    dataset = st.selectbox("Dataset", options=available, index=0,
                           help="Select a dataset. Uploaded CSVs appear here automatically.")
    st.session_state["dataset"] = dataset

    # Show dataset info
    try:
        arr = get_test_arrays(dataset)
        y = arr["y"]
        st.caption(f"🌐 {len(y)} test flows — Benign: {(y==0).sum()}, Attack: {(y==1).sum()}")
    except Exception:
        pass

    model_name = st.selectbox("Victim model", ALL_MODELS, index=0,
                              help="LR — fast, interpretable. RF/XGB — strong baselines. "
                                   "MLP/CNN1D — deep models, harder to attack.")

    try:
        vm = load_model(dataset, model_name)
        st.session_state["victim_model"] = vm
        st.session_state["model_trained"] = True
        st.session_state["model_name"] = model_name

        m = vm.metrics
        if m:
            st.markdown("**Clean-model performance**:")
            cols = st.columns(4)
            cols[0].metric("Accuracy", f"{m.get('accuracy', 0):.1%}")
            cols[1].metric("F1", f"{m.get('f1', 0):.1%}")
            cols[2].metric("Recall", f"{m.get('recall', 0):.1%}")
            cols[3].metric("AUROC", f"{m.get('auroc', 0):.1%}")
    except FileNotFoundError as e:
        st.warning(str(e))

    st.markdown("---")
    st.markdown("""
**Research question this tab addresses**: Given an IDS that achieves ~99% F1 on clean
data, does it break when an attacker slightly modifies network features?

- **F1** — balance of precision and recall. 98% = very few false alarms or misses.
- **AUROC** — model's ability to separate classes. 1.0 = perfect separation.
""")


# ---------------------------------------------------------------------------
# Tab: Attack Console
# ---------------------------------------------------------------------------
def render_attack():
    st.subheader("Attack Console")

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

    # --- Step 1: Pick a flow ---
    st.markdown("**Step 1 — Pick a flow**: Select a network flow (a packet) to attack.")
    st.caption("These are CICIDS-2017-style network flow records — each row is one connection "
               "with 78 features like packet counts, byte counts, and timing info.")
    label_filter = st.radio("Show", ("All flows", "Attack flows only"),
                            horizontal=True,
                            help="Attack flows are the evasion targets.")
    idxs = np.arange(len(y))
    if label_filter == "Attack flows only":
        idxs = idxs[y == ATTACK]

    idx = st.selectbox("Select flow", options=list(idxs),
                        format_func=lambda i: f"#{i:04d} · {LABEL_NAMES[y[i]]} · {families[i]}",
                        key="flow_idx")
    x_clean = X[idx:idx+1]
    true_label = int(y[idx])
    true_family = str(families[idx])

    # --- Step 2: Clean prediction ---
    st.markdown("---")
    st.markdown("**Step 2 — Before attack**: The model classifies this flow.")
    st.caption("The bar shows P(benign) and P(attack). If P(attack) > P(benign), "
               "the model correctly identifies this as an attack.")
    probs_before = vm.predict_proba(x_clean)[0]
    pred_before = int(np.argmax(probs_before))

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Model predicts", LABEL_NAMES[pred_before],
                  help=f"P(attack) = {probs_before[1]:.4f}")
        st.metric("True label", LABEL_NAMES[true_label])
        st.caption(f"Attack family: {true_family}")
    with c2:
        st.progress(float(probs_before[0]), text=f"P(benign) = {probs_before[0]:.4f}")
        st.progress(float(probs_before[1]), text=f"P(attack) = {probs_before[1]:.4f}")

    # --- Step 3: Attack ---
    st.markdown("---")
    st.markdown("**Step 3 — Launch attack**: Perturb features to flip the prediction.")
    st.caption("**FGSM** = one shot: nudge each feature by ε × sign(gradient). "
               "**PGD** = iterates FGSM multiple times with projection. "
               "**CW** = Carlini-Wagner optimization — finds the smallest perturbation. "
               "**ε** = max perturbation per feature (controls how much the features can change).")

    col_a, col_b = st.columns(2)
    with col_a:
        method = st.radio("Attack method", ("FGSM", "PGD", "CW"), horizontal=True,
                          help="**FGSM** = one shot: nudge each feature by ε × sign(gradient). "
                               "**PGD** = iterates FGSM multiple times with projection. "
                               "**CW** = Carlini-Wagner optimization: finds smallest perturbation that fools the model.")
        attack_fn = {"FGSM": fgsm, "PGD": pgd, "CW": cw}[method]
    with col_b:
        eps = st.select_slider("Epsilon (ε)", options=EPS_GRID, value=0.2,
                               help="Max allowed perturbation per feature. Higher = stronger attack.")

    if st.button("Run attack ▶", type="primary", use_container_width=True):
        result = attack_fn(vm, x_clean, eps, mask=mutable_mask)
        x_adv = result.x_adv  # AttackResult -> raw ndarray
        queries = result.n_queries if hasattr(result, 'n_queries') else 0
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

    r = st.session_state.get("result")
    if not r:
        st.caption("Pick a flow and click 'Run attack' to see the evasion.")
        return

    # --- Results ---
    st.markdown("---")
    st.subheader("Result — what happened?")

    if r["evasion"]:
        st.error("**Evasion succeeded** — the model now thinks this attack is benign!\n"
                 f"P(attack): {r['probs_before']['attack']:.4f} → {r['probs_after']['attack']:.4f}",
                 icon="🔴")
    elif not r["valid"]:
        st.warning(f"Invalid — {len(r['violations'])} constraint violations",
                   icon="⚠")
    else:
        st.success(f"Blocked — model still says {r['label_after']}",
                   icon="🟢")

    st.markdown("---")
    st.markdown("**After attack — model's new prediction**")
    col_e, col_f, col_g = st.columns(3)
    col_e.metric("Before", f"P(attack) = {r['probs_before']['attack']:.4f}")
    col_f.metric("After", f"P(attack) = {r['probs_after']['attack']:.4f}")
    col_g.metric("Status", r["msg"])

    # Perturbation cost
    st.markdown("---")
    st.markdown("**How much did we change the packet?**")
    st.caption("**L1** = average total change across all features. **L∞** = the single "
               "biggest change to any one feature. These tell you how much the attacker "
               "had to distort the packet to fool the model.")
    c1, c2, c3 = st.columns(3)
    c1.metric("L1 norm", f"{r['l1']:.4f}", help="Sum of |changes| across all features")
    c2.metric("L∞ norm", f"{r['linf']:.4f}", help="Largest single-feature change")
    c3.metric("Queries", f"{r.get('queries', 0)}", help="Model evaluations used by the attack")

    # Validity + behaviour checks
    st.markdown("---")
    st.markdown("**Did the attack play by the rules?**")
    st.caption("**Validity** = did we stay within allowed bounds? (no negative packet counts, "
               "immutable fields unchanged, perturbation ≤ ε)")
    st.caption("**Behaviour** = did we preserve the malicious semantics? (the packet "
               "is still fundamentally an attack, just perturbed)")
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
    def color_delta(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return "color: #06b7de"
            elif val < 0:
                return "color: #f56565"
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
    st.subheader("Results Gallery")
    dataset = st.session_state.get("dataset", "dummy")

    e1_path = RESULTS_DIR / "E1" / f"{dataset}_clean_metrics.json"
    if e1_path.exists():
        with st.expander("Clean model metrics (E1)"):
            data = json.load(open(e1_path))
            st.json(data)

    st.info("Run batch experiments to generate E2 (FGSM ASR) and E3 (PGD ASR):\n```\n"
            "python -m nids.experiments --e2 --e3\n```")
    st.markdown("""
**Research questions answered by batch experiments:**
- **RQ1** — Which model architectures are most vulnerable? Compare ASR across LR/RF/XGB/MLP/CNN1D
- **RQ2** — Which features are most susceptible? The feature diff table + gradient chart show this
- **RQ3** — Can adversarial training help? Run E8 to find out
""")


# ---------------------------------------------------------------------------
# Tab: About
# ---------------------------------------------------------------------------
def render_about():
    st.subheader("About")
    st.markdown("""
**Adversarial NIDS Evasion Console** — adversarial ML for network intrusion detection.

**The research problem**: Your IDS catches 98% of attacks. But what if an attacker
slightly tweaks their network packet features — without changing the malicious
intent — so the model misclassifies it as benign?

**Architecture (6 layers)**:
1. **config** — frozen seeds, epsilon grid, label convention
2. **data** — CICIDS-2017 / UNSW-NB15 network flow features
3. **models** — LR, RF, XGBoost, MLP, CNN1D (pre-trained, loaded from `artifacts/`)
4. **attacks** — FGSM (single-step), PGD (iterative), CW (optimization-based), projection (constraint enforcement)
5. **eval** — metrics (F1, AUROC, FPR), validity checks, transfer matrix
6. **dashboard** — this UI

**Key terms**:
- **Epsilon (ε)**: max perturbation per feature. Controls attack strength.
- **FGSM**: nudge each feature by ε × sign(gradient) — one shot.
- **PGD**: iterate FGSM with projection — stronger than FGSM.
- **CW**: Carlini-Wagner optimization — finds minimal perturbation via Adam + binary search — strongest white-box attack.
- **Validity**: perturbed sample stays within realistic bounds.
- **Evasion**: attack succeeded — malicious packet classified as benign.
- **L1/L∞ norms**: how much features changed (total vs. max single change).
""")
    st.code("python -m scripts.pretrain --dataset dummy --models lr rf xgb mlp cnn1d", language="bash")
    st.caption("Pre-trained models load instantly in the dashboard. Upload a CSV in Setup to use your own dataset. CW attack is ready — select 'CW' in the Attack Console.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("Adversarial NIDS Evasion Console")
    st.caption("Attack a pre-trained NIDS with FGSM, PGD, or CW. Load a model in Setup, attack in the console.")

    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False

    tabs = st.tabs(["Setup", "Attack Console", "Results Gallery", "About"])
    with tabs[0]: render_setup()
    with tabs[1]: render_attack()
    with tabs[2]: render_gallery()
    with tabs[3]: render_about()


if __name__ == "__main__":
    main()