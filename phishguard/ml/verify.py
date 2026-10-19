"""Comparison helpers shared by ``train_model.py --check`` and
``evaluate_models.py --check``.

Only *quality* metrics are compared — latency is machine-dependent and
timestamps/environment data are informational.
"""

from __future__ import annotations

QUALITY_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "fpr",
    "fnr",
)


def experiment_index(results: dict) -> dict[tuple[str, str, str], dict]:
    return {
        (record["model"], record["protocol"], record["feature_set"]): record
        for record in results["experiments"]
    }


def compare_metrics(committed: dict, recomputed: dict, tolerance: float = 1e-6) -> list[str]:
    """Return human-readable differences in quality metrics (empty == match)."""
    diffs: list[str] = []
    committed_index = experiment_index(committed)
    recomputed_index = experiment_index(recomputed)

    missing = sorted(set(committed_index) - set(recomputed_index))
    extra = sorted(set(recomputed_index) - set(committed_index))
    if missing:
        diffs.append(f"missing experiments: {missing}")
    if extra:
        diffs.append(f"unexpected experiments: {extra}")

    for key in sorted(set(committed_index) & set(recomputed_index)):
        left = committed_index[key]["metrics"]
        right = recomputed_index[key]["metrics"]
        label = "/".join(key)
        for metric in QUALITY_METRICS:
            a, b = left.get(metric), right.get(metric)
            if a is None or b is None:
                if a != b:
                    diffs.append(f"{label} {metric}: committed={a} recomputed={b}")
                continue
            if abs(float(a) - float(b)) > tolerance:
                diffs.append(
                    f"{label} {metric}: committed={a} recomputed={b} "
                    f"delta={abs(float(a) - float(b)):.3e} > {tolerance:g}"
                )
    return diffs
