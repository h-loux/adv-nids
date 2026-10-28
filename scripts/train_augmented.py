"""Batch-generate training-only augmentations and fit comparison models.

Example:
  python -m scripts.train_augmented --datasets cicids2017 unsw_nb15 \
      --models lr rf xgb mlp cnn1d ensemble --method FGSM --eps 0.2 \
      --max-samples 64 --epochs 20 --batch-size 256

Existing compatible clean baseline artifacts are reused. Missing baselines are
trained and saved. Each dataset/model pair is isolated so one failure does not
stop the overnight batch. Outputs are saved under artifacts/ and results/E9/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from nids.augmentation import generate_adversarial_training_augmentations
from nids.config import ALL_MODELS, ARTIFACTS_DIR, RESULTS_DIR, SUPPORTED_DATASETS
from nids.data import get_meta, get_test_arrays, get_train_arrays, load_dataset
from nids.eval.metrics import compute_metrics
from nids.models import VictimModel
from nids.training import train_and_save_model, train_augmented_and_save_model


def _as_victim(artifact: dict, model_name: str) -> VictimModel:
    """Wrap a saved artifact and normalize XGBoost estimators to CPU inference."""
    victim = VictimModel(
        artifact["model"],
        is_torch=bool(artifact["is_torch"]),
        n_features=int(artifact["n_features"]),
    )
    victim.set_scaler(artifact.get("scaler"))
    victim.feature_names = list(artifact.get("feature_names", []))
    victim.mutable_mask = np.asarray(artifact.get("mutable_mask", []), dtype=bool)
    victim.meta = artifact.get("meta", {}) or {}
    victim.metrics = artifact.get("metrics", {}) or {}
    victim.model_params = artifact.get("model_params", {}) or {}
    victim.seed = artifact.get("seed", 0)
    victim.epochs = artifact.get("epochs")
    victim.batch_size = artifact.get("batch_size", 256)

    estimators = [victim.base]
    estimators.extend(getattr(victim.base, "estimators_", []) or [])
    for estimator in estimators:
        if estimator.__class__.__module__.lower().startswith("xgboost"):
            try:
                estimator.set_params(device="cpu")
            except (TypeError, ValueError):
                pass
            try:
                estimator.get_booster().set_param({"device": "cpu"})
            except (AttributeError, TypeError, ValueError):
                # An unfitted estimator has no booster yet; training will use
                # the CPU default supplied by the model factory.
                pass
    return victim


def _baseline_compatible(
    artifact: dict,
    dataset_meta: dict,
    train_arrays: dict,
) -> tuple[bool, str]:
    """Ensure an existing baseline was trained on this feature schema/split."""
    if list(artifact.get("feature_names", [])) != list(dataset_meta.get("feature_names", [])):
        return False, "feature order differs"
    prior_meta = artifact.get("meta", {}) or {}
    old_hash = prior_meta.get("source_hash")
    current_hash = dataset_meta.get("source_hash")
    if old_hash and current_hash and old_hash != current_hash:
        return False, "dataset source hash differs"
    scaler = artifact.get("scaler")
    if scaler is None:
        return False, "baseline has no saved scaler"
    expected = StandardScaler().fit(train_arrays["X_raw"])
    if (
        not hasattr(scaler, "mean_")
        or not hasattr(scaler, "scale_")
        or not np.allclose(scaler.mean_, expected.mean_)
        or not np.allclose(scaler.scale_, expected.scale_)
    ):
        return False, "scaler does not match the clean training split"
    return True, "compatible"


def _load_or_train_baseline(
    dataset: str,
    model_name: str,
    train_arrays: dict,
    test_arrays: dict,
    dataset_meta: dict,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    retrain: bool,
) -> tuple[VictimModel, dict, bool]:
    """Reuse a compatible baseline; otherwise train and save one."""
    baseline_path = ARTIFACTS_DIR / f"{dataset}_{model_name}.joblib"
    artifact = None
    if baseline_path.is_file() and not retrain:
        try:
            candidate = joblib.load(baseline_path)
            compatible, reason = _baseline_compatible(candidate, dataset_meta, train_arrays)
            if compatible:
                artifact = candidate
                print(f"  Reusing clean baseline: {baseline_path}", flush=True)
            else:
                print(f"  Refreshing baseline ({reason}): {baseline_path}", flush=True)
        except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
            print(f"  Refreshing unreadable baseline ({exc}): {baseline_path}", flush=True)

    trained_now = artifact is None
    if artifact is None:
        print(f"  Training clean baseline {dataset}/{model_name}...", flush=True)
        artifact, _ = train_and_save_model(
            dataset,
            model_name,
            train_arrays,
            test_arrays,
            dataset_meta,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size or 256,
        )
    victim = _as_victim(artifact, model_name)
    return victim, artifact, trained_now


def _metrics(model: VictimModel, test_arrays: dict) -> dict:
    probabilities = model.predict_proba(test_arrays["X_raw"])
    predictions = np.argmax(probabilities, axis=1)
    return compute_metrics(test_arrays["y"], predictions, probabilities)


def _train_pair(
    dataset: str,
    model_name: str,
    train_arrays: dict,
    test_arrays: dict,
    dataset_meta: dict,
    args,
) -> dict:
    started = time.time()
    baseline, baseline_artifact, baseline_trained_now = _load_or_train_baseline(
        dataset,
        model_name,
        train_arrays,
        test_arrays,
        dataset_meta,
        seed=args.seed,
        epochs=args.epochs if args.epochs is not None else 20,
        batch_size=args.batch_size,
        retrain=args.retrain_baselines,
    )
    baseline_metrics = _metrics(baseline, test_arrays)

    if model_name in {"mlp", "cnn1d"}:
        chosen_epochs = (
            args.epochs
            if args.epochs is not None
            else int(baseline_artifact.get("epochs") or 20)
        )
    else:
        chosen_epochs = args.epochs if args.epochs is not None else 20
    chosen_batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else (baseline_artifact.get("batch_size") or 256)
    )
    model_params = baseline_artifact.get("model_params") or None
    baseline_seed = (
        int(baseline_artifact["seed"])
        if baseline_artifact.get("seed") is not None
        else int(args.seed)
    )

    print(
        f"  Generating {args.method} rows (eps={args.eps}, limit={args.max_samples})...",
        flush=True,
    )
    augmented_arrays, generation_stats = generate_adversarial_training_augmentations(
        baseline,
        train_arrays,
        dataset_meta,
        method=args.method,
        eps=args.eps,
        max_samples=args.max_samples,
        seed=args.seed,
        pgd_steps=args.pgd_steps,
    )

    augmentation_dir = RESULTS_DIR / "E9" / "augmentations"
    augmentation_dir.mkdir(parents=True, exist_ok=True)
    eps_label = str(args.eps).replace(".", "p")
    archive_path = augmentation_dir / (
        f"{dataset}_{model_name}_{args.method.lower()}_eps{eps_label}.npz"
    )
    np.savez_compressed(archive_path, **augmented_arrays)
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    print(
        f"  Training augmented model on {len(train_arrays['y']):,} clean + "
        f"{len(augmented_arrays['y']):,} augmented rows "
        f"(epochs={chosen_epochs if model_name in {'mlp', 'cnn1d'} else 'n/a'})...",
        flush=True,
    )
    augmented_artifact, augmented_metrics = train_augmented_and_save_model(
        dataset,
        model_name,
        train_arrays,
        augmented_arrays,
        test_arrays,
        dataset_meta,
        seed=baseline_seed,
        epochs=chosen_epochs,
        batch_size=chosen_batch_size,
        model_params=model_params,
        augmentation_sha256=archive_hash,
    )

    preprocessing_matches = (
        np.allclose(
            baseline.scaler_.mean_, augmented_artifact["scaler"].mean_
        )
        and np.allclose(
            baseline.scaler_.scale_, augmented_artifact["scaler"].scale_
        )
    )
    comparison = {
        "experiment": "clean_performance_after_training_augmentation",
        "dataset": dataset,
        "model": model_name,
        "baseline_metrics": baseline_metrics,
        "augmented_metrics": augmented_metrics,
        "training": {
            "strategy": "clean_plus_training_only_adversarial_augmentations",
            "n_clean_train": int(len(train_arrays["y"])),
            "n_augmented_train": int(len(augmented_arrays["y"])),
            "seed": baseline_seed,
            "epochs": chosen_epochs if model_name in {"mlp", "cnn1d"} else None,
            "batch_size": chosen_batch_size if model_name in {"mlp", "cnn1d"} else None,
            "model_params": model_params or {},
            "augmentation_file_sha256": archive_hash,
            "augmentation_file": str(archive_path),
            "dataset_source_hash": dataset_meta.get("source_hash"),
            "split_seed": dataset_meta.get("split_seed"),
            "preprocessing_matches_baseline": bool(preprocessing_matches),
            "clean_baseline_trained_by_this_run": bool(baseline_trained_now),
            "generation": generation_stats,
            "method": args.method,
            "epsilon_scaled": float(args.eps),
            "max_samples_requested": int(args.max_samples),
            "pgd_steps": int(args.pgd_steps) if args.method == "PGD" else None,
        },
        "scope": (
            "Clean held-out test comparison only. Training augmentations came from "
            "the clean training split. This comparison does not itself prove "
            "adversarial robustness; matched adversarial evaluation is still required."
        ),
    }
    result_path = RESULTS_DIR / "E9" / f"{dataset}_{model_name}_augmented_comparison.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(comparison, stream, indent=2, default=str)
    return {
        "dataset": dataset,
        "model": model_name,
        "status": "success",
        "baseline_f1": float(baseline_metrics["f1"]),
        "augmented_f1": float(augmented_metrics["f1"]),
        "valid_augmentations": int(len(augmented_arrays["y"])),
        "evasion_successes_in_training_candidates": int(
            generation_stats["valid_candidates_predicted_benign"]
        ),
        "baseline_trained_now": bool(baseline_trained_now),
        "seconds": round(time.time() - started, 2),
        "result_file": str(result_path),
        "model_file": str(ARTIFACTS_DIR / f"{dataset}_{model_name}_augmented.joblib"),
        "augmentation_file": str(archive_path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch train augmented NIDS models across datasets/models."
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=SUPPORTED_DATASETS,
        default=["cicids2017", "unsw_nb15"],
    )
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument("--method", choices=["FGSM", "PGD"], default="FGSM")
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=32,
                        help="Maximum valid-source candidates per dataset/model pair.")
    parser.add_argument("--pgd-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Epochs for MLP/CNN1D; default is existing baseline epochs or 20.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override neural-model batch size; defaults to the baseline setting or 256.")
    parser.add_argument("--retrain-baselines", action="store_true",
                        help="Retrain and overwrite clean baseline artifacts before augmentation.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate/retrain augmented outputs even if both outputs already exist.")
    args = parser.parse_args(argv)

    if not np.isfinite(args.eps) or args.eps <= 0:
        parser.error("--eps must be finite and positive")
    if (
        args.max_samples < 1
        or args.pgd_steps < 1
        or (args.batch_size is not None and args.batch_size < 1)
    ):
        parser.error("--max-samples and --pgd-steps must be positive; --batch-size must be positive when supplied")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")

    summary = []
    total = len(args.datasets) * len(args.models)
    done = 0
    failures = 0
    print(f"Starting {total} dataset/model augmentation runs.", flush=True)
    print(f"Method={args.method}, epsilon={args.eps}, max_samples={args.max_samples}", flush=True)

    for dataset in args.datasets:
        try:
            print(f"\n=== Preparing dataset: {dataset} ===", flush=True)
            load_dataset(dataset)
            train_arrays = get_train_arrays(dataset, include_family=False)
            test_arrays = get_test_arrays(dataset, include_family=False)
            dataset_meta = get_meta(dataset)
        except Exception as exc:
            failures += len(args.models)
            message = f"Dataset preparation failed: {type(exc).__name__}: {exc}"
            print(f"ERROR {dataset}: {message}", flush=True)
            for model_name in args.models:
                summary.append({"dataset": dataset, "model": model_name,
                                "status": "failed", "error": message})
            continue

        for model_name in args.models:
            done += 1
            result_path = RESULTS_DIR / "E9" / f"{dataset}_{model_name}_augmented_comparison.json"
            model_path = ARTIFACTS_DIR / f"{dataset}_{model_name}_augmented.joblib"
            if (
                not args.force
                and not args.retrain_baselines
                and result_path.is_file()
                and model_path.is_file()
            ):
                print(f"\n[{done}/{total}] SKIP {dataset}/{model_name}: outputs already exist", flush=True)
                summary.append({"dataset": dataset, "model": model_name,
                                "status": "skipped_existing",
                                "result_file": str(result_path),
                                "model_file": str(model_path)})
                continue
            print(f"\n[{done}/{total}] {dataset}/{model_name}", flush=True)
            try:
                record = _train_pair(
                    dataset, model_name, train_arrays, test_arrays, dataset_meta, args
                )
                summary.append(record)
                print(
                    f"  DONE: baseline F1={record['baseline_f1']:.4f}, "
                    f"augmented F1={record['augmented_f1']:.4f}, "
                    f"rows={record['valid_augmentations']}, "
                    f"time={record['seconds']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                failures += 1
                message = f"{type(exc).__name__}: {exc}"
                print(f"  ERROR: {message}", flush=True)
                summary.append({
                    "dataset": dataset,
                    "model": model_name,
                    "status": "failed",
                    "error": message,
                })
            finally:
                print(f"Progress: {done}/{total} pairs visited.", flush=True)

    summary_path = RESULTS_DIR / "E9" / "batch_augmented_training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {"method": args.method, "eps": args.eps, "summary": summary},
            stream, indent=2, default=str,
        )

    successes = sum(record.get("status") == "success" for record in summary)
    skipped = sum(record.get("status") == "skipped_existing" for record in summary)
    print("\n=== Batch complete ===")
    print(f"Successful pairs: {successes}; skipped existing: {skipped}; failures: {failures}")
    print(f"Batch report: {summary_path}")
    print(f"Per-model comparison JSONs: {RESULTS_DIR / 'E9'}")
    print(f"Augmented model artifacts: {ARTIFACTS_DIR}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
