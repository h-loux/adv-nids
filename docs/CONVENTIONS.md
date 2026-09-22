# Adversarial NIDS — Conventions

## Team
- P1 — Data: datasets, leakage audit, constraint mask, preprocessing
- P2 — Models: victim models, FGSM/PGD, projection, adversarial training
- P3 — Eval: metrics, validity, transfer matrix, plots, writeup
(Single person can cover all roles.)

## Seeds
- `SPLIT_SEED = 42` — train/test split
- `MODEL_SEEDS = [0, 1, 2]` — model init (E1, E8)
- `ATTACK_SEED = 7` — FGSM/PGD randomness (E2, E3, E5, E8)
- `EPS_GRID = [0.05, 0.1, 0.2, 0.5]`

## Data Contract
- `data/processed/{dataset}_train.npz` — `X_scaled`, `X_raw`, `y`, `y_family`
- `data/processed/{dataset}_test.npz` — same keys
- `data/meta/{dataset}_meta.json` — `feature_names`, `mutable_mask`, `bounds`, `one_hot_groups`, `label_names`, `n_train`, `n_test`, `split_seed`

## Result Schema (every result JSON)
```json
{
  "experiment": "e1",
  "dataset": "cicids2017",
  "model": "logreg",
  "metrics": {"f1": 0.94, "recall": 0.95, ...},
  "metadata": {"seed": 42, "eps": 0.1, ...}
}
```

## Label Convention
- `0 = benign`, `1 = attack` (always, in every dataset)

## Feature Convention
- All features are `float64`
- `mutable_mask`: boolean per feature (True = attacker can perturb)
- `bounds`: per-feature `{"min": float, "max": float, "mutable": bool}`
- `one_hot_groups`: `{"group_name": [column_indices]}`

## Model Names
- Shallow: `lr`, `rf`, `xgb`
- Deep: `mlp`, `cnn1d`

## Attack Convention
- Target class for evasion: BENIGN (0) — gradient increases P(benign)
- eps is L-infinity, applied in SCALED feature space
- Projection happens in scaled space, then inverse-transformed to raw

## Module Layout
```
nids/
  config.py          — seeds, paths, constants
  data/              — loaders, preprocessing
  models/            — VictimModel, MLP, CNN1D
  attacks/           — projection, fgsm, pgd
  defenses/          — adversarial training
  eval/              — metrics, validity, transfer
```
