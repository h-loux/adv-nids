"""Layer 4 — Evaluation metrics.

compute_metrics(y_true, y_pred, y_prob) -> dict with F1, recall, FPR, AUROC, AUPRC, etc.
write_result(path, result) -> enforces the fixed JSON result schema.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)


def confusion_matrix_dict(y_true, y_pred, labels=None) -> dict:
    """Return confusion matrix as a dict with named keys.

    Returns:
        {"tp": int, "fp": int, "tn": int, "fn": int,
         "matrix": [[tn, fp], [fn, tp]]}
    """
    if labels is None:
        labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "matrix": cm.tolist(),
    }


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute full metric set for binary classification.

    Args:
        y_true: (n,) ground truth labels (0=benign, 1=attack)
        y_pred: (n,) predicted labels
        y_prob: (n,) or (n,2) predicted attack probability

    Returns:
        dict with f1, precision, recall, fpr, fnr, auroc, auprc, tp, fp, tn, fn
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]  # P(attack)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = 0.5

    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = 0.0

    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def write_result(path: str | Path, experiment: str, dataset: str,
                model: str, metrics: dict, **metadata) -> None:
    """Write a result JSON following the fixed schema."""
    result = {
        "experiment": experiment,
        "dataset": dataset,
        "model": model,
        "metrics": metrics,
        "metadata": metadata,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
