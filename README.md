# Adversarial NIDS Evasion Console

Gradient-based adversarial attacks against Network Intrusion Detection System (NIDS) models, with a Streamlit dashboard for interactive evasion demonstrations. Supports CICIDS2017 and UNSW-NB15 datasets; attacks include FGSM and PGD against LR, RF, XGBoost, MLP, 1D-CNN, and a soft-voting LR/RF/XGBoost ensemble.

## Prerequisites

- Python 3.11+ (3.12 recommended)
- pip or uv
- ~2 GB free disk space (for models + datasets)
- Network access (to download dependencies; CICIDS2017 CSV is optional — a built-in synthetic dataset is ready to use)

## Step-by-Step Setup

### 1. Clone the repo

```bash
git clone https://github.com/h-loux/adv-nids.git
cd adv-nids
```

### 2. Create a Python virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit numpy pandas scikit-learn torch xgboost joblib
```

### 4. Prepare a dataset

The built-in `dummy` dataset is generated automatically and ready to use — no download needed. It is synthetic and only suitable for checking that training and the dashboard work; do not use its results as evidence of detection quality.

For real CICIDS2017 training, the loader downloads the five-day **improved CICIDS2017** flow release from the researchers' companion site. It uses their documented labeling guidance: flows marked “Attempted” are labeled benign. The downloaded archive is retained at `data/raw/CICIDS2017_improved.zip`; cite the original CICIDS2017 dataset and the CNS 2022 correction study when reporting results.

Then preprocess it:

```bash
python -c "from nids.data import load_dataset; load_dataset('cicids2017')"
```

The loader processes CSV chunks to limit peak memory, drops non-finite rows, and creates a stratified 80/20 split. It writes train/test arrays and source metadata under `data/processed/` and `data/meta/`. UNSW-NB15 can be prepared the same way with `load_dataset('unsw_nb15')`.

Dataset references: Sharafaldin et al., *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization* (2018); Engelen et al., *Troubleshooting an intrusion detection dataset: the CICIDS2017 case study* (2021); Liu et al., *Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018* (IEEE CNS 2022).

### 5. Train models

```bash
python -m scripts.pretrain --dataset cicids2017 --models lr rf xgb mlp cnn1d ensemble
```

Training is performed in this order:

1. Load or preprocess the selected dataset and read its fixed train/test split.
2. Fit `StandardScaler` on **training features only**, then transform the training features.
3. Fit each selected model on the training split. PyTorch models shuffle batches and use the configured seed; sklearn models use the same seeded factory.
4. Evaluate once on the untouched test split and report accuracy, F1, precision, recall, FPR/FNR, AUROC, and AUPRC.
5. Save dashboard-ready models under `artifacts/{dataset}_{model}.joblib` and clean-test metrics under `results/E1/`.

The ensemble averages predicted class probabilities from Logistic Regression, Random Forest, and XGBoost. `--seed` selects a reproducible model seed; `--epochs` and `--batch-size` affect only MLP/CNN1D.

To compare a small set of settings for every model on both real datasets, use the validation-driven tuner:

```bash
python -m scripts.tune_models --datasets unsw_nb15 cicids2017 --models lr rf xgb mlp cnn1d ensemble --search-rows 500000 --batch-size 2048 --xgb-device cuda
```

The tuner selects settings using a validation slice taken only from the training partition, refits each selected model on the full training partition, and evaluates the held-out test partition once. It records validation trials in `results/E1/{dataset}_tuning_results.json` and updates the clean metrics and dashboard artifacts. Search work is limited to at most `--search-rows` examples per dataset to control memory; final fits use the full training partition. Use `--xgb-device cpu` on machines without an XGBoost-compatible CUDA GPU. PyTorch neural models use CUDA only when the installed PyTorch build includes CUDA support.

For a fast pipeline smoke test (not a real-world performance result):

```bash
python -m scripts.pretrain --dataset dummy --models lr --seed 0
```

### 6. (Optional) Run the terminal demo

```bash
python -m nids.demo --dataset dummy --model lr --method pgd --eps 0.5
```

### 7. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Then open **http://localhost:8501** in your browser.

## Project Structure

```
adv-nids/
├── nids/                    # Core research pipeline
│   ├── config.py           # Seeds, EPS_GRID, supported datasets/models
│   ├── data/               # Dataset loaders (CICIDS2017, UNSW-NB15, dummy)
│   ├── models/             # VictimModel, MLP, CNN1D
│   ├── attacks/            # projection, fgsm, pgd
│   ├── eval/               # metrics, validity checks
│   └── demo.py             # Terminal demo
├── dashboard/
│   └── app.py              # Streamlit demo dashboard
├── scripts/
│   └── pretrain.py         # Pre-train all models and save artifacts
├── artifacts/              # Saved .joblib model artifacts (gitignored)
├── data/                   # Raw + processed datasets (gitignored)
├── results/                # Experiment outputs (gitignored)
└── README.md
```

## Experiments

| Experiment | Flag | Description |
|---|---|---|
| **E1** | `--e1` | Clean model metrics (F1, recall, FPR, AUROC, AUPRC) per model |
| **E2** | `--e2` | FGSM attack success rate per model x epsilon |
| **E3** | `--e3` | PGD attack success rate per model x epsilon |
| **E5** | `--e5` | Cross-model transfer matrix (planned, not yet implemented) |
| **E8** | `--e8` | Adversarial training (planned, not yet implemented) |

```bash
# Run E1 + E2 + E3 on the dummy dataset
python -m nids.experiments --dataset dummy --models lr rf xgb mlp cnn1d ensemble --e1 --e2 --e3
```

## Dashboard

The sidebar navigation switches between Model setup, Attack console, Results gallery, and About. The interface uses a spatial observability theme with a deep navy canvas, subtle coordinate grid, layered surfaces, and cyan/violet data accents. Model setup can upload a CSV, train or refresh the selected model, and display held-out test metrics and diagnostics. The Results gallery includes a clean-set overview, grouped F1/precision/recall/AUROC chart, model ranking and error-count visuals, an interactive per-class outcome breakdown, and raw JSON in a secondary expander. Saved supplied-augmentation runs appear there as paired clean-set comparisons; if none exist, the page explains how to create one from a reviewed local archive. All graphs use saved clean-set metrics; missing measurements are not filled with sample values. Setup controls remain native Streamlit widgets so values stay synchronized with Python session state. The CLI supports training multiple models at once. Models and results are written locally; the app does not upload dataset contents. Shared styles live in `dashboard/styles.css`; theme colors are configured in `.streamlit/config.toml`.

### Supplied-augmentation training comparison

The Model setup tab can optionally train a **separate** model using a local `.npz` file of already-prepared training augmentations. The archive must contain `X_raw` (raw feature rows), `y` (binary labels), and `source_indices` (zero-based row indices into the selected dataset's clean training split). Each label must match the label of its source training row, and each augmented row must differ from that source. The scaler is fitted on clean training rows only and reused for both clean and supplied rows. The baseline artifact is not overwritten.

For example, package previously reviewed arrays as follows:

```python
import numpy as np

np.savez_compressed(
    "training_augmentations.npz",
    X_raw=augmented_X_raw,
    y=source_labels,
    source_indices=clean_train_row_indices,
)
```

The app records the input file hash, model seed, training counts, clean metrics, and a baseline-versus-augmented clean-test comparison under `results/E9/`. Source indices and label equality are checked, but the app cannot independently establish how an uploaded row was created or whether its manipulation preserves the underlying class; review that provenance before training. This workflow does not generate or evaluate adversarial examples, and clean-test results alone are not evidence of adversarial robustness.

**Workspace sections**:

1. **Model setup** — Select or upload a dataset, train/select a model, and view clean test metrics, confusion matrix, and feature importance
2. **Attack console** — Browse a flow, inspect its clean prediction, run a selected attack, and review constraint checks and feature changes
3. **Results gallery** — Compare locally generated clean metrics and supplied-augmentation training runs; missing outputs are identified rather than replaced with sample data
4. **About** — Research question, architecture, key terms, and training command

## Data Sources

### CICIDS2017
- Official dataset page: https://www.unb.ca/cic/datasets/ids-2017.html
- Optional automatic download: `https://iscx2016.secdataset.com/cicids2017/CICIDS2017.zip`
- 78 features, 80 columns (includes labels)

### UNSW-NB15
- Official dataset page: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- The loader accepts public Parquet train/test partitions in `data/raw/unsw_nb15_hf/`.
- The Parquet mirror used here derives from the UNSW-NB15 Kaggle dataset and contains fewer source columns than the full 49-feature release. Training uses its 175,341-row partition and testing its 82,332-row partition, matching the official row counts; the mirror filenames are reversed.
- Binary labels use `label` (0 benign, 1 attack); `attack_cat` is retained as the flow-family label.

CICIDS2017 is downloaded automatically by `nids.data.load_dataset()` when needed. UNSW-NB15 Parquet partitions must be placed in `data/raw/unsw_nb15_hf/`.

## Threat Model

- **Adversary**: white-box to one victim model (can access gradients via autograd for CNN1D, finite-difference estimation for sklearn models)
- **Goal**: cause a malicious network flow to be classified as benign (evasion)
- **Constraints**: L-inf perturbation <= eps; immutable features (protocol, state) cannot change; perturbed values must stay within feature bounds

## License

MIT