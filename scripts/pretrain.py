"""Train selected victim models and save dashboard-ready artifacts.

Example:
  python -m scripts.pretrain --dataset cicids2017 --models lr rf ensemble mlp cnn1d
"""
import argparse
import json

from nids.config import ALL_MODELS, ARTIFACTS_DIR, MODEL_SEEDS, RESULTS_DIR
from nids.data import get_meta, get_test_arrays, get_train_arrays, load_dataset
from nids.training import train_and_save_model


def main():
    parser = argparse.ArgumentParser(
        description="Train NIDS models on a prepared dataset and save artifacts."
    )
    parser.add_argument("--dataset", default="dummy")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--seed", type=int, default=MODEL_SEEDS[0])
    parser.add_argument("--epochs", type=int, default=50,
                        help="Epochs for MLP/CNN1D; ignored by sklearn models.")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    print(f"Preparing '{args.dataset}' data...")
    load_dataset(args.dataset)
    train_arrays = get_train_arrays(args.dataset, include_family=False)
    test_arrays = get_test_arrays(args.dataset, include_family=False)
    meta = get_meta(args.dataset)

    results_path = RESULTS_DIR / "E1" / f"{args.dataset}_clean_metrics.json"
    results = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as previous:
            results.update(json.load(previous))
    for model_name in args.models:
        print(f"Training {model_name} (seed={args.seed})...", flush=True)
        _, metrics = train_and_save_model(
            args.dataset,
            model_name,
            train_arrays,
            test_arrays,
            meta,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        results[model_name] = metrics
        print(
            f"  test F1={metrics['f1']:.4f} "
            f"recall={metrics['recall']:.4f} AUROC={metrics['auroc']:.4f}"
        )

    import joblib
    for model_name in ALL_MODELS:
        artifact_path = ARTIFACTS_DIR / f"{args.dataset}_{model_name}.joblib"
        if model_name not in results and artifact_path.exists():
            artifact = joblib.load(artifact_path)
            results[model_name] = artifact["metrics"]

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)
    print(f"Saved clean test metrics to {results_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
