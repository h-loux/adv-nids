"""Project-wide constants, paths, and the EPS_GRID used across experiments and dashboard."""
from pathlib import Path

# Root anchors (read at runtime so profile scoping / test isolation works)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
META_DIR = DATA_DIR / "meta"
ARTIFACTS_DIR = ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# EPS grid shared with the dashboard spec so live results stay comparable to report numbers
EPS_GRID = [0.05, 0.1, 0.2, 0.5]

# Datasets
SUPPORTED_DATASETS = ["cicids2017", "unsw_nb15"]
# Victim models from E1
SUPPORTED_MODELS = ["lr", "rf", "xgb", "mlp", "cnn1d"]

# Label convention: 0 = benign, 1 = attack
BENIGN = 0
ATTACK = 1
LABEL_NAMES = {0: "benign", 1: "attack"}
