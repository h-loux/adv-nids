# Adversarial NIDS Project — Crash Study Notes
> 15-min presentation prep. Read this before you present.

## 1. What is this project? (30-second pitch)

We have a Network Intrusion Detection System (NIDS) — a machine learning model that looks
at network traffic and says "attack" or "benign". We then try to FAKE the model: we slightly
perturb a malicious network packet so the model thinks it's benign. This is called **adversarial
evasion**.

The dashboard lets you pick a model, pick a network flow, pick an attack method, and see if
the attack succeeds.

---

## 2. The 6-Layer Architecture

```
1. config    — frozen seeds, epsilon grid, label constants
2. data      — CICIDS-2017 / UNSW-NB15 network flow features (78 features per flow)
3. models    — LR, RF, XGBoost, MLP, CNN1D (pre-trained, loaded from artifacts/)
4. attacks   — FGSM, PGD, CW, DeepFool, JSMA, NES
5. eval      — metrics, validity checks, transfer matrix
6. dashboard — Streamlit UI
```

**Key principle**: Each layer is independent. data doesn't import models. attacks don't
import dashboard. This modularity lets you swap components.

---

## 3. Data Flow — End to End (the story)

```
1. Raw CSV (CICIDS2017)  →  nids/data/__init__.py
2. Preprocess: 80/20 split, StandardScaler fit on train
3. Save: data/processed/{dataset}_train.npz, {dataset}_test.npz
       data/meta/{dataset}_meta.json
4. Train: python -m scripts.pretrain --dataset dummy --models lr rf xgb mlp cnn1d
5. Save model + scaler: artifacts/{dataset}_{model}.joblib
6. Dashboard loads: victim_model (model + scaler), test arrays, meta
7. User picks a flow → picks an attack → attacks
8. Attack takes x_clean → returns x_adv (perturbed)
9. Dashboard shows: evasion success, costs, validity, feature diffs
```

**What is a "flow"?** A single network connection (e.g., one TCP session). Each row in the
dataset = one flow. 78 features describe it: packet counts, byte counts, flow duration,
flag counts (SYN, PSH, RST), timing, etc.

---

## 4. The 5 Models — Know One Trait Each

| Model | Type | Key trait for the attack |
|-------|------|--------------------------|
| LR   | Logistic Regression | Linear — gradient is simple → easiest to attack |
| RF   | Random Forest       | Tree ensemble — no smooth gradient → gradient attacks fail, use transfer |
| XGB  | XGBoost             | Boosted trees — similar to RF, often weak to CW |
| MLP  | Multi-Layer Perceptron | Neural net — differentiable, medium difficulty |
| CNN1D | 1-D CNN             | Deep neural net on feature sequence — hardest to attack |

**For the presentation**: you don't need to explain the math. Just say: "simpler models (LR)
are easier to fool; tree-based (RF/XGB) are non-differentiable; deep models (MLP/CNN1D)
are in between."

---

## 5. The 6 Attack Methods — Know One Sentence Each

| Attack | Type | One-liner |
|--------|------|-----------|
| FGSM  | White-box, gradient | One-step: nudge by epsilon × sign(gradient) |
| PGD   | White-box, gradient | Iterate FGSM with projection — stronger |
| CW    | White-box, optimization | Adam + binary search — finds smallest perturbation |
| DeepFool | White-box, gradient | Line search to decision boundary — minimal L2 |
| JSMA  | White-box, Jacobian  | Targets specific features via Jacobian saliency map |
| NES   | Black-box, zeroth-order | Estimates gradient via random sampling — no model access |

**White-box** = attacker has full model access (weights, gradients). **Black-box** = attacker
only sees predictions (queries model).

**For the presentation**: "FGSM is the simplest (one push), PGD iterates that push, CW optimizes
to find the mathematically smallest push, NES works even if you don't have the model."

---

## 6. Key Terminologies — Be Ready to Define These

- **Evasion**: The attack succeeded — a malicious packet was classified as benign.
- **Validity**: Did the perturbation stay within realistic bounds? (No negative packet counts,
  no values below 0, perturbation ≤ epsilon.)
- **Behavior preservation**: Did the packet remain fundamentally malicious? (e.g., it's still
  a DDoS, just with slightly tweaked feature values.)
- **Epsilon (ε)**: Maximum allowed perturbation per feature. Controls attack strength.
- **L1 norm**: Total change across all features (sum of |deltas|). Lower = stealthier.
- **L∞ norm**: The single largest change to any one feature. Lower = less suspicious on any one feature.
- **ASR (Attack Success Rate)**: What % of attacks evade detection. Used in batch experiments.
- **Query count**: For black-box attacks (NES), how many times the model was queried.

---

## 7. Metrics & Validity Checks — The eval/ folder

**Metrics** (`nids/eval/metrics.py`):
- Accuracy, Precision, Recall, F1 — standard classification metrics.
- AUROC — area under ROC curve.
- Confusion matrix (TN, FP, FN, TP).

**Validity** (`nids/eval/validity.py`):
- `check_validity()`: Is x_adv within bounds? (|x_adv - x_clean| ≤ eps everywhere, no negatives)
- `check_behaviour_preserved()`: Did we change the malicious nature?
- `perturbation_magnitude()`: Returns (L1, L∞) of the perturbation.
- `feature_diff()`: Returns top-N changed features with before/after/delta.

---

## 8. Code Structure — What to Know for Q&A

```
adv-nids/
├── config.py          # EPS_GRID, LABEL_NAMES, seeds, ALL_MODELS
├── data/
│   └── __init__.py    # _build_dummy(), _build_cicids2017(), _build_unsw_nb15()
│                      # load_dataset(), get_train_arrays(), get_test_arrays(), get_meta()
├── models/
│   └── __init__.py    # VictimModel — wraps sklearn/torch model + scaler
├── attacks/
│   ├── fgsm.py        # FGSM
│   ├── pgd.py         # PGD
│   ├── carlini_wagner.py  # CW (Adam + binary search)
│   ├── deepfool.py    # DeepFool
│   ├── jsma.py        # JSMA
│   ├── zeroth_order.py # NES + transfer attacks
│   ├── __init__.py    # exports all + AttackResult
│   └── result.py      # AttackResult dataclass: x_adv, success, n_queries, runtime_ms
├── eval/
│   ├── metrics.py     # compute_metrics(), confusion_matrix_dict()
│   ├── validity.py    # check_validity(), check_behaviour_preserved(), etc.
│   └── transfer.py    # transfer matrix between models
├── scripts/
│   └── pretrain.py    # Trains models, saves to artifacts/
├── dashboard/
│   └── app.py         # Streamlit app — 4 tabs: Setup, Attack Console, Results Gallery, About
├── data/              # GENERATED (in .gitignore) — raw CSVs, processed .npz, meta JSON
├── artifacts/         # GENERATED — {dataset}_{model}.joblib (pre-trained models)
└── results/           # GENERATED — E1 clean metrics, E2/E3 batch experiment results
```

**Critical**: `nids/attacks/AttackResult` — all attacks return this. It has `.x_adv` (the
perturbed sample), `.n_queries`, `.runtime_ms`. The dashboard reads `.x_adv`.

---

## 9. Dashboard Flow — Step by Step

**Tab 1 — Setup**:
- Select dataset (dummy default, or uploaded CSV)
- Select model (LR, RF, XGB, MLP, CNN1D)
- Pre-trained model loads from `artifacts/`
- Shows clean-model performance (Accuracy, F1, Recall, AUROC)
- Shows confusion matrix + top 15 feature importances

**Tab 2 — Attack Console** (the main one):
1. **Step 1**: Pick a flow from test data. Filter by "Attack flows only" to see evasion targets.
2. **Step 2**: Model classifies the clean flow. Shows prediction card with confidence bar.
3. **Step 3**: Pick attack method (FGSM/PGD/CW/DeepFool/JSMA/NES) and epsilon. Click "Run attack".
   - Shows the loader while running
   - Disabled button prevents double-click
4. **Results**:
   - Big status: "Evasion succeeded" / "Blocked" / "Invalid"
   - Before/After P(attack) metrics
   - L1/L∞/Queries cost metrics
   - Validity + behavior checkmarks
   - Feature diff table (which features were changed, colored by sign)
   - Gradient magnitudes chart (expander)

**Tab 3 — Results Gallery**: Clean model metrics, RQ1/RQ2/RQ3.

**Tab 4 — About**: Project architecture, terminology glossary.

---

## 10. Presentation Flow (3 minutes setup, 12 minutes demo+Q&A)

### Opening (1 min)
- "This project explores **adversarial evasion** against NIDS models."
- "We train models on network flow data, then try to perturb malicious flows so models misclassify them."

### Architecture (1 min)
- Show the 6-layer diagram.
- "Data → Models → Attacks → Eval → Dashboard — each independent."

### Live Demo (3-4 min)
1. Tab 1: Load dummy dataset + LR model. Show 98% F1.
2. Tab 2: Pick an attack flow (PortScan). Show model says "Attack, 98%".
3. Step 3: Select FGSM, ε=0.2, click Run. Show evasion succeeds, P(attack) drops to near 0.
4. Switch to CNN1D. Show FGSM fails, CW succeeds. This is the key contrast.

### Key Findings (1 min)
- LR: trivially evadable.
- Tree models (RF/XGB): gradient attacks don't work (non-differentiable).
- Deep models (MLP/CNN1D): harder but CW still breaks them with enough queries.
- NES (black-box) works but needs many more queries.

### Q&A Prep

| Q | A |
|---|---|
| "What's the difference between validity and behavior?" | Validity = stayed in bounds (< eps). Behavior = still malicious in nature. Both must hold. |
| "Why is RF hard to attack?" | Tree ensembles are non-differentiable; gradients are zero almost everywhere. |
| "What's epsilon?" | Max perturbation per feature. Higher = stronger attack but more detectable. |
| "Why 78 features?" | CICIDS-2017 has 78 network flow features per connection. |
| "Why allow_pickle=True?" | y_family stores string arrays (attack family names). Safe because we generate the files ourselves. |
| "How does the attack work technically?" | Takes gradient of model loss w.r.t. input, pushes x in that direction to flip prediction. |
| "What's the AttackResult?" | Standardized return type. All attacks return x_adv, n_queries, success, runtime. |

---

## 11. Files to Have Open During Presentation

- `dashboard/app.py` — main UI (470 lines for render functions)
- `nids/attacks/__init__.py` — shows all 6 attack exports
- `nids/attacks/result.py` — AttackResult dataclass
- `nids/models/__init__.py` — VictimModel
- `nids/data/__init__.py` — data loading pipeline
- `scripts/pretrain.py` — training pipeline

---

## 12. Gotchas (if asked)

- **RF/XGB + gradient attacks**: gradient is zero everywhere → FGSM/PGD produce no change → L1/L∞ ≈ 0 → "Blocked". This is expected, not a bug.
- **NES**: black-box, slow — ~100+ queries. The dummy dataset is small so it's fast.
- **The "Run attack" button** can get stuck disabled if an exception crashes mid-attack. Fixed by `st.session_state.pop("_attacking", None)` at the top of render_attack().
- **Data generation**: `data/` is in .gitignore. Fresh clones need `python -m scripts.pretrain --dataset dummy --models lr rf xgb mlp cnn1d`.
