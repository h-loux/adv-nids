# Adversarial NIDS Evasion Console

Gradient-based adversarial attacks against Network Intrusion Detection System (NIDS) models, with a Streamlit demo dashboard for interactive evasion demonstrations. Supports CICIDS2017 and UNSW-NB15 datasets; attacks include FGSM and PGD against LR, RF, XGBoost, MLP, and 1D-CNN victim models.

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

### 4. Pre-build datasets

The built-in `dummy` dataset is generated automatically and ready to use — no download needed.

If you want real CICIDS2017 or UNSW-NB15 data:

```bash
python -c "from nids.data import load_dataset; load_dataset('cicids2017'); load_dataset('unsw_nb15')"
```

This downloads the raw CSVs (~100 MB each) into `data/raw/`, then preprocesses them into `data/processed/` and `data/meta/`.

### 5. Build pre-trained models

```bash
python scripts/pretrain.py --dataset dummy --model all
```

This trains all 5 models (LR, RF, XGB, MLP, CNN1D) on the `dummy` dataset and saves them to `artifacts/`.

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
python -m nids.experiments --dataset dummy --model all --e1 --e2 --e3
```

## Dashboard

The Streamlit app reads pre-trained models and precomputed results — it does not retrain anything live.

**Tabs**:

1. **Setup** — Select dataset + model; view clean E1 metrics
2. **Attack Console** — Browse a flow, see clean prediction, run live FGSM/PGD attack, view result panel with feature diff
3. **Results Gallery** — Precomputed figures (ASR curves, transfer heatmap, confusion matrices)
4. **About** — Project title, research questions, threat model

## Data Sources

### CICIDS2017
- Auto-download: `https://iscx2016.secdataset.com/cicids2017/CICIDS2017.zip`
- Manual download: https://www.unb.ca/cic/datasets/ids-2017.html
- 78 features, 80 columns (includes labels)

### UNSW-NB15
- Auto-download: `https://research.unsw.edu.au/projects/datasets/unsw-nisw-nids15-data/unswnb15-datasets.zip`
- Manual download: https://research.unsw.edu.au/projects/datasets/unswnb15-dataset
- 49 features (numeric + categorical), 9 attack categories + normal

Both datasets are auto-downloaded by `nids.data.load_dataset()` into `data/raw/` during first run.

## Threat Model

- **Adversary**: white-box to one victim model (can access gradients via autograd for CNN1D, finite-difference estimation for sklearn models)
- **Goal**: cause a malicious network flow to be classified as benign (evasion)
- **Constraints**: L-inf perturbation <= eps; immutable features (protocol, state) cannot change; perturbed values must stay within feature bounds

## License

MIT