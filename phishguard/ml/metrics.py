"""Evaluation metrics — every value computed from actual predictions.

Required metric set (P2): accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
false-positive rate, false-negative rate, inference latency, model size.
Binomial proportions additionally carry 95% Wilson confidence intervals so
small test sets are not over-read.
"""

from __future__ import annotations

import math
import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def wilson_ci(successes: int, total: int, z: float = 1.96) -> list[float | None]:
    """95% Wilson score interval for a binomial proportion."""
    if total <= 0:
        return [None, None]
    proportion = successes / total
    denom = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return [round(max(centre - margin, 0.0), 6), round(min(centre + margin, 1.0), 6)]


def _round(value: float) -> float:
    return round(float(value), 6)


def classification_metrics(y_true, y_pred, y_score) -> dict:
    """Full metric bundle from labels, hard predictions and scores."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    n_pos = tp + fn
    n_neg = tn + fp

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / n_neg if n_neg else None
    fnr = fn / n_pos if n_pos else None
    single_class = len(np.unique(y_true)) < 2
    roc_auc = None if single_class else roc_auc_score(y_true, y_score)
    pr_auc = None if single_class else average_precision_score(y_true, y_score)

    return {
        "accuracy": _round(accuracy),
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
        "roc_auc": None if roc_auc is None else _round(roc_auc),
        "pr_auc": None if pr_auc is None else _round(pr_auc),
        "fpr": None if fpr is None else _round(fpr),
        "fnr": None if fnr is None else _round(fnr),
        "ci95": {
            "accuracy": wilson_ci(round(accuracy * len(y_true)), len(y_true)),
            "precision": wilson_ci(tp, tp + fp),
            "recall": wilson_ci(tp, n_pos),
            "fpr": wilson_ci(fp, n_neg),
            "fnr": wilson_ci(fn, n_pos),
        },
        "counts": {
            "n_samples": int(len(y_true)),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
    }


def measure_latency_ms(model, frame, rounds: int = 100, warmup: int = 10) -> dict:
    """Single-row ``predict()`` latency — the API's actual serving shape."""
    row = frame.iloc[[0]]
    for _ in range(warmup):
        model.predict(row)
    samples: list[float] = []
    for _ in range(rounds):
        start = time.perf_counter()
        model.predict(row)
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": _round(float(np.mean(samples))),
        "p95_ms": _round(float(np.percentile(samples, 95))),
        "rounds": rounds,
        "note": "single-row predict() on this machine; not comparable across hardware",
    }


def artifact_size_bytes(path) -> int:
    from pathlib import Path

    return Path(path).stat().st_size
