# Adversarial NIDS Evasion Console

Gradient-based adversarial attacks against Network Intrusion Detection System (NIDS) models, with a Streamlit demo dashboard for interactive evasion demonstrations. Supports CICIDS2017 and UNSW-NB15 datasets; attacks include FGSM and PGD against LR, RF, XGBoost, MLP, and 1D-CNN victim models.

## Quick Start

```bash
# 1. Create the environment
conda env create -f environment.yml
conda activate adv-nids

# 2. Download and preprocess datasets
python -c "from nids.data import load_dataset; load_dataset('cicids2017'); load_dataset('unsw_nb15')"

# 3. Run all experiments (E1-E8)
python -m nids.experiments --dataset cicids2017 --model all --e1 --e2 --e3 --e5 --e8

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

## Project Structure

```
adv-nids/
├── nids/                    # Core research pipeline
│   ├── constants.py         # Paths, EPS_GRID, supported datasets/models
│   ├── data.py              # Dataset loaders (CICIDS2017, UNSW-NB15)
│   ├── models.py            # Victim models + registry + gradient
│   ├── attacks.py           # FGSM, PGD, projection
│   ├── validity.py          # Validity & behaviour-preservation checks
│   └── experiments.py       # CLI runner for E1-E8
├── dashboard/
│   └── app.py               # Streamlit demo dashboard
├── environment.yml          # Conda environment
└── pyproject.toml           # Packaging
```

## Experiments

| Experiment | Description |
|---|---|
| **E1** | Train & evaluate 5 clean victim models (LR, RF, XGB, MLP, CNN1D) |
| **E2** | FGSM attack sweep at eps in {0.05, 0.1, 0.2, 0.5} - compute ASR |
| **E3** | PGD attack sweep - compute ASR + validity rate |
| **E5** | Cross-model transfer matrix |
| **E8** | Adversarial training (robust MLP + CNN1D) |

```bash
# Run specific experiments
python -m nids.experiments --dataset cicids2017 --model cnn1d --e1 --e2 --e3 --e5 --e8

# All datasets, all models
python -m nids.experiments --dataset cicids2017 --model all --e1 --e2 --e3 --e5 --e8
python -m nids.experiments --dataset unsw_nb15 --model all --e1 --e2 --e3 --e5 --e8
```

## Dashboard

The Streamlit app reads trained models and precomputed results - it does not retrain anything live.

**Tabs**:
1. **Setup** - Select dataset + model; view clean E1 metrics
2. **Attack Console** - Browse/upload a flow, see clean prediction, run live FGSM/PGD attack, view result panel with feature diff
3. **Results Gallery** - Precomputed figures (ASR curves, transfer heatmap, confusion matrices)
4. **About** - Project title, research questions, threat model

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

- **Adversary**: white-box to one victim model (can probe gradients via NES for sklearn models, autograd for CNN1D)
- **Goal**: cause a malicious network flow to be classified as benign (evasion)
- **Constraints**: L-inf perturbation <= eps; immutable features (protocol, state) cannot change; perturbed values must stay within feature bounds

## License

MIT
