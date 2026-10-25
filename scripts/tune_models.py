"""Tune model settings on validation data, then evaluate the selected models once."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from nids.config import ALL_MODELS, MODEL_SEEDS, RESULTS_DIR, SUPPORTED_DATASETS
from nids.data import get_meta, get_test_arrays, get_train_arrays, load_dataset
from nids.eval.metrics import compute_metrics
from nids.training import fit_model, train_and_save_model


def _candidates(model_name: str, xgb_device: str) -> list[dict]:
    if model_name == "lr":
        return [
            {"model_params": {"C": 0.1}},
            {"model_params": {"C": 2.0}},
        ]
    if model_name == "rf":
        return [
            {"model_params": {
                "n_estimators": 300, "max_depth": 24,
                "max_leaf_nodes": 4096, "min_samples_leaf": 2,
                "max_samples": 0.8,
            }},
            {"model_params": {
                "n_estimators": 400, "max_depth": 24,
                "max_leaf_nodes": 4096, "min_samples_leaf": 1,
                "max_samples": 0.6,
            }},
        ]
    if model_name == "xgb":
        return [
            {"model_params": {
                "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.9, "colsample_bytree": 0.9,
                "tree_method": "hist", "device": xgb_device,
            }},
            {"model_params": {
                "n_estimators": 600, "max_depth": 6, "learning_rate": 0.03,
                "subsample": 0.9, "colsample_bytree": 0.9,
                "tree_method": "hist", "device": xgb_device,
            }},
        ]
    if model_name == "ensemble":
        candidates = []
        for estimators, depth, rate in ((300, 4, 0.05), (500, 6, 0.03)):
            candidates.append({
                "model_params": {
                    "lr__C": 1.0,
                    "rf__n_estimators": estimators,
                    "rf__max_depth": 24,
                    "rf__max_leaf_nodes": 4096,
                    "rf__min_samples_leaf": 2,
                    "rf__max_samples": 0.8,
                    "xgb__n_estimators": estimators,
                    "xgb__max_depth": depth,
                    "xgb__learning_rate": rate,
                    "xgb__subsample": 0.9,
                    "xgb__colsample_bytree": 0.9,
                    "xgb__tree_method": "hist",
                    "xgb__device": xgb_device,
                },
            })
        return candidates
    if model_name in {"mlp", "cnn1d"}:
        return [{"epochs": 25}, {"epochs": 50}]
    raise ValueError(f"No tuning candidates for model: {model_name}")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
    temporary_path.replace(path)


def _validation_indices(
    labels: np.ndarray, seed: int, search_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(len(labels))
    if len(all_indices) > search_rows:
        selected, _ = train_test_split(
            all_indices, train_size=search_rows, stratify=labels,
            random_state=seed,
        )
    else:
        selected = all_indices
    fit_indices, validation_indices = train_test_split(
        selected, test_size=0.2, stratify=labels[selected],
        random_state=seed,
    )
    return fit_indices, validation_indices


def _save_clean_metrics(path: Path, results: dict) -> None:
    _write_json(path, results)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select model settings using a validation split of training data, "
            "then refit on the full training split and score the held-out test set."
        ),
    )
    parser.add_argument(
        "--datasets", nargs="+", default=SUPPORTED_DATASETS,
        choices=SUPPORTED_DATASETS,
    )
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
    )
    parser.add_argument("--seed", type=int, default=MODEL_SEEDS[0])
    parser.add_argument("--search-rows", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--xgb-device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.search_rows < 100 or args.batch_size < 1:
        parser.error("--search-rows must be at least 100 and --batch-size must be positive")

    for dataset_name in args.datasets:
        print(f"Preparing '{dataset_name}' data...", flush=True)
        load_dataset(dataset_name)
        train_arrays = get_train_arrays(dataset_name, include_family=False)
        test_arrays = get_test_arrays(dataset_name, include_family=False)
        meta = get_meta(dataset_name)
        labels = np.asarray(train_arrays["y"], dtype=np.int64)
        fit_indices, validation_indices = _validation_indices(
            labels, args.seed, args.search_rows,
        )
        search_train = {
            "X_raw": train_arrays["X_raw"][fit_indices],
            "y": labels[fit_indices],
        }
        validation_x = train_arrays["X_raw"][validation_indices]
        validation_y = labels[validation_indices]

        clean_metrics_path = (
            RESULTS_DIR / "E1" / f"{dataset_name}_clean_metrics.json"
        )
        tuning_path = (
            RESULTS_DIR / "E1" / f"{dataset_name}_tuning_results.json"
        )
        clean_metrics = {}
        if clean_metrics_path.exists():
            with open(clean_metrics_path, encoding="utf-8") as previous:
                clean_metrics.update(json.load(previous))
        tuning_results = {}
        if tuning_path.exists():
            with open(tuning_path, encoding="utf-8") as previous:
                tuning_results.update(json.load(previous))

        for model_name in args.models:
            print(
                f"Tuning {dataset_name}/{model_name} using "
                f"{len(fit_indices):,} fit and {len(validation_indices):,} "
                "validation rows...",
                flush=True,
            )
            best_rank = None
            best_trial = None
            trial_results = []
            candidates = _candidates(model_name, args.xgb_device)
            for trial_number, candidate in enumerate(candidates, start=1):
                epochs = candidate.get("epochs", 50)
                model_params = candidate.get("model_params", {})
                print(
                    f"  Trial {trial_number}/{len(candidates)}: "
                    f"params={model_params or {'epochs': epochs}}",
                    flush=True,
                )
                victim = fit_model(
                    model_name,
                    search_train,
                    meta,
                    seed=args.seed,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    model_params=model_params,
                )
                validation_probabilities = victim.predict_proba(validation_x)
                validation_predictions = np.argmax(
                    validation_probabilities, axis=1,
                )
                validation_metrics = compute_metrics(
                    validation_y, validation_predictions, validation_probabilities,
                )
                rank = (
                    validation_metrics["accuracy"],
                    validation_metrics["f1"],
                    validation_metrics["auroc"],
                    -validation_metrics["fpr"],
                )
                result = {
                    "trial": trial_number,
                    "model_params": model_params,
                    "epochs": epochs if model_name in {"mlp", "cnn1d"} else None,
                    "validation_metrics": validation_metrics,
                }
                trial_results.append(result)
                print(
                    f"    validation accuracy={validation_metrics['accuracy']:.4f} "
                    f"F1={validation_metrics['f1']:.4f} "
                    f"AUROC={validation_metrics['auroc']:.4f}",
                    flush=True,
                )
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_trial = result
                del victim
                gc.collect()

            if best_trial is None:
                raise RuntimeError(f"No candidate evaluated for {dataset_name}/{model_name}")
            best_epochs = best_trial["epochs"] or 50
            best_params = best_trial["model_params"]
            print(
                f"  Refit selected configuration on all "
                f"{len(labels):,} training rows; evaluating held-out test once...",
                flush=True,
            )
            _, test_metrics = train_and_save_model(
                dataset_name,
                model_name,
                train_arrays,
                test_arrays,
                meta,
                seed=args.seed,
                epochs=best_epochs,
                batch_size=args.batch_size,
                model_params=best_params,
            )
            clean_metrics[model_name] = test_metrics
            tuning_results[model_name] = {
                "search_rows": len(fit_indices),
                "validation_rows": len(validation_indices),
                "selected_trial": best_trial,
                "trials": trial_results,
                "final_test_metrics": test_metrics,
                "seed": args.seed,
                "batch_size": (
                    args.batch_size
                    if model_name in {"mlp", "cnn1d"} else None
                ),
            }
            _save_clean_metrics(clean_metrics_path, clean_metrics)
            _write_json(tuning_path, tuning_results)
            print(
                f"  held-out test accuracy={test_metrics['accuracy']:.4f} "
                f"F1={test_metrics['f1']:.4f} "
                f"recall={test_metrics['recall']:.4f} "
                f"AUROC={test_metrics['auroc']:.4f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
