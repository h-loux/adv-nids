#!/usr/bin/env python3
"""Run a bounded smoke-test matrix over real CICIDS2017 and UNSW-NB15 data.

The runner trains lightweight models on deterministic, class-balanced samples of
each real training split, then executes each attack once against a held-out
attack flow. This is an execution/shape/finite-value smoke test, not a full ASR
benchmark. Results are written to results/attack_matrix_smoke.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from nids.attacks import cw, deepfool, fgsm, jsma, nes_attack, pgd
from nids.config import ALL_MODELS, ATTACK
from nids.data import get_meta, get_test_arrays, get_train_arrays, load_dataset
from nids.eval.validity import check_validity
from nids.training import fit_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "attack_matrix_smoke.json"

ATTACKS = {
    "FGSM": (fgsm, {"n_project": 2}),
    "PGD": (pgd, {"n_steps": 2, "n_project": 2}),
    "CW": (cw, {"n_steps": 3, "n_binary_search": 2, "n_project": 2}),
    "DeepFool": (deepfool, {"n_steps": 2}),
    "JSMA": (jsma, {"n_steps": 2, "gamma": 0.1}),
    "NES": (nes_attack, {"n_queries": 8, "n_directions": 2, "n_steps": 2}),
}


def balanced_subset(arrays: dict, max_rows: int, seed: int) -> dict:
    """Return a deterministic, balanced subset using only training rows."""
    X = np.asarray(arrays["X_raw"])
    y = np.asarray(arrays["y"]).astype(np.int64)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("Training features/labels have mismatched shapes")
    classes = np.unique(y)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"Expected both binary classes in training split; got {classes}")
    rng = np.random.default_rng(seed)
    quota = max(1, max_rows // 2)
    selected = []
    for label in (0, 1):
        candidates = np.flatnonzero(y == label)
        take = min(quota, len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return {"X_raw": X[selected], "y": y[selected]}


def model_parameters(model_name: str) -> dict:
    """Keep smoke tests fast without changing model families."""
    if model_name == "rf":
        return {"n_estimators": 20}
    if model_name == "xgb":
        return {"n_estimators": 20}
    if model_name == "ensemble":
        return {"lr__max_iter": 500, "rf__n_estimators": 20, "xgb__n_estimators": 20}
    return {}


def is_ignored_xgboost_device_error(model_name: str, exc: BaseException) -> bool:
    if model_name not in {"xgb", "ensemble"}:
        return False
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(token in message for token in ("cuda", "gpu", "device ordinal", "device mismatch"))


def emit_row(rows: list[dict], dataset: str, model: str, attack: str,
             status: str, elapsed_s: float = 0.0, **details) -> None:
    row = {
        "dataset": dataset, "model": model, "attack": attack, "status": status,
        "elapsed_seconds": round(elapsed_s, 3),
    }
    row.update(details)
    rows.append(row)
    suffix = f" — {details.get('error')}" if details.get("error") else ""
    print(f"[{status:30}] {dataset:12} {model:8} {attack:9} {elapsed_s:7.2f}s{suffix}",
          flush=True)


def run_matrix(datasets: list[str], models: list[str], max_rows: int,
               epsilon: float) -> list[dict]:
    rows: list[dict] = []
    # Prevent CPU oversubscription on hosted runners and repeatable timings.
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

    for d_idx, dataset in enumerate(datasets):
        print(f"\n=== Preparing dataset: {dataset} ===", flush=True)
        try:
            load_dataset(dataset)
            meta = get_meta(dataset)
            train = get_train_arrays(dataset)
            test = get_test_arrays(dataset)
            X_test = np.asarray(test["X_raw"])
            y_test = np.asarray(test["y"]).astype(np.int64)
            attack_rows = np.flatnonzero(y_test == ATTACK)
            if not len(attack_rows):
                raise ValueError("Held-out split contains no attack-labelled rows")
            x_clean = np.atleast_2d(X_test[attack_rows[0]]).astype(np.float32)
            mutable_mask = np.asarray(meta["mutable_mask"], dtype=bool)
            if x_clean.shape[1] != len(mutable_mask):
                raise ValueError("Feature count does not match the dataset mutable mask")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            for model_name in models:
                for attack_name in ATTACKS:
                    emit_row(rows, dataset, model_name, attack_name,
                             "DATASET_SETUP_ERROR", error=message)
            continue

        subset = balanced_subset(train, max_rows=max_rows, seed=910 + d_idx)
        for m_idx, model_name in enumerate(models):
            print(f"\n--- Training {dataset}/{model_name} on {len(subset['y'])} rows ---",
                  flush=True)
            start = time.perf_counter()
            try:
                model = fit_model(
                    model_name,
                    subset,
                    meta,
                    seed=123 + d_idx * 10 + m_idx,
                    epochs=1,
                    batch_size=64,
                    model_params=model_parameters(model_name) or None,
                )
                # Preflight both clean and held-out data through the public API.
                clean_probs = np.asarray(model.predict_proba(x_clean))
                if clean_probs.shape != (1, 2) or not np.isfinite(clean_probs).all():
                    raise ValueError(f"Invalid probability output shape/value: {clean_probs.shape}")
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                status = (
                    "SKIPPED_XGBOOST_CUDA"
                    if is_ignored_xgboost_device_error(model_name, exc)
                    else "MODEL_SETUP_ERROR"
                )
                for attack_name in ATTACKS:
                    emit_row(rows, dataset, model_name, attack_name, status,
                             time.perf_counter() - start, error=msg)
                continue

            for attack_name, (attack_fn, kwargs) in ATTACKS.items():
                started = time.perf_counter()
                try:
                    result = attack_fn(
                        model, x_clean.copy(), epsilon, mask=mutable_mask.copy(), **kwargs,
                    )
                    x_adv = np.asarray(result.x_adv)
                    if x_adv.shape != x_clean.shape:
                        raise AssertionError(
                            f"Expected adversarial output shape {x_clean.shape}; got {x_adv.shape}"
                        )
                    if not np.isfinite(x_adv).all():
                        raise AssertionError("Output contains NaN/Inf")
                    probabilities = np.asarray(model.predict_proba(x_adv))
                    if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
                        raise AssertionError("Model returned invalid post-run probabilities")
                    valid, violations = check_validity(
                        x_adv, x_clean, meta, epsilon, model=model,
                    )
                    success_values = np.asarray(getattr(result, "success", [False])).reshape(-1)
                    emit_row(
                        rows, dataset, model_name, attack_name, "PASS",
                        time.perf_counter() - started,
                        valid=bool(valid),
                        violations=list(violations),
                        success=bool(success_values[0]) if len(success_values) else False,
                        queries=int(getattr(result, "n_queries", 0)),
                        runtime_ms=float(getattr(result, "runtime_ms", 0.0)),
                    )
                except Exception as exc:
                    if is_ignored_xgboost_device_error(model_name, exc):
                        status = "SKIPPED_XGBOOST_CUDA"
                    else:
                        status = "FAIL"
                    emit_row(rows, dataset, model_name, attack_name, status,
                             time.perf_counter() - started,
                             error=f"{type(exc).__name__}: {exc}",
                             traceback=traceback.format_exc(limit=4))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cicids2017", "unsw_nb15"])
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--sample-rows", type=int, default=512,
                        help="maximum total training rows per model (balanced by class)")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    unknown_datasets = set(args.datasets) - {"cicids2017", "unsw_nb15"}
    unknown_models = set(args.models) - set(ALL_MODELS)
    if unknown_datasets or unknown_models:
        parser.error(f"unknown datasets={unknown_datasets}; unknown models={unknown_models}")
    if args.sample_rows < 4:
        parser.error("--sample-rows must be >= 4")
    if not 0 < args.epsilon <= 0.5:
        parser.error("--epsilon must be in (0, 0.5]")

    rows = run_matrix(args.datasets, args.models, args.sample_rows, args.epsilon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "scope": "bounded real-data smoke tests; not a full robustness/ASR benchmark",
        "epsilon": args.epsilon,
        "sample_rows": args.sample_rows,
        "matrix_dimensions": {
            "datasets": args.datasets, "models": args.models, "attacks": list(ATTACKS),
        },
        "total": len(rows),
        "passed": sum(r["status"] == "PASS" for r in rows),
        "failed": sum(r["status"] == "FAIL" for r in rows),
        "skipped_xgboost_cuda": sum(r["status"] == "SKIPPED_XGBOOST_CUDA" for r in rows),
        "setup_errors": sum(r["status"] in {"MODEL_SETUP_ERROR", "DATASET_SETUP_ERROR"} for r in rows),
        "rows": rows,
    }
    args.output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n=== MATRIX SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"rows", "matrix_dimensions"}}, indent=2))
    print(f"Results: {args.output}")
    return 1 if summary["failed"] or summary["setup_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
