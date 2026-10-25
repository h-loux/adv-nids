"""Frozen seeds, epsilon grid, and project paths.

Conventions (from docs/CONVENTIONS.md):
  SPLIT_SEED    = 42   — train/test split
  MODEL_SEEDS   = [0, 1, 2]  — model init seeds (E1, E8)
  ATTACK_SEED   = 7     — FGSM/PGD randomness (E2, E3, E5, E8)
  EPS_GRID      = [0.05, 0.1, 0.2, 0.5]  — perturbation magnitudes
"""
from pathlib import Path

# --- Seeds ---
SPLIT_SEED = 42
MODEL_SEEDS = [0, 1, 2]
ATTACK_SEED = 7

# --- Epsilon grid (L-infinity perturbation) ---
EPS_GRID = [0.05, 0.1, 0.2, 0.5]

# --- Label convention ---
BENIGN = 0
ATTACK = 1
LABEL_NAMES = {0: "benign", 1: "attack"}

# --- Model names ---
SHALLOW_MODELS = ["lr", "rf", "xgb"]
DEEP_MODELS = ["mlp", "cnn1d"]
ENSEMBLE_MODELS = ["ensemble"]
ALL_MODELS = SHALLOW_MODELS + DEEP_MODELS + ENSEMBLE_MODELS

# --- Datasets ---
SUPPORTED_DATASETS = ["cicids2017", "unsw_nb15"]

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
META_DIR = DATA_DIR / "meta"
ARTIFACTS_DIR = ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# --- Result schema (every result JSON follows this) ---
RESULT_SCHEMA = {
    "experiment": str,
    "dataset": str,
    "model": str,
    "metrics": dict,   # from compute_metrics()
    "metadata": dict,  # seeds, eps, params
}
