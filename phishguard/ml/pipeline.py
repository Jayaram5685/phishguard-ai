"""Experiment orchestration: protocols x feature sets x models.

Design (P2):

- One split per protocol, shared by every model and ablation so comparisons
  are paired.
- Ablations run on the primary protocol (``domain_disjoint``); protocol
  comparisons run on the shipped feature set (:data:`SHIPPED_FEATURE_SET`).
- Every fit is seeded and single-threaded; metrics come only from predictions
  on the protocol's held-out test rows.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from phishguard.features import (
    ABLATIONS,
    SCHEMA_VERSION,
    SHIPPED_FEATURE_SET,
    ablation_features,
    schema_fingerprint,
)
from phishguard.ml.data import (
    DATASET_PATH,
    LABEL_COLUMN,
    dataset_report,
    dataset_sha256,
    deduplicate,
    drop_mixed_domains,
    load_raw,
)
from phishguard.ml.metrics import classification_metrics, measure_latency_ms
from phishguard.ml.models import MODEL_SPECS, SEED, build_model
from phishguard.ml.split import (
    TEST_FRACTION,
    Split,
    domain_disjoint_split,
    random_row_split,
    vector_disjoint_split,
)

PROTOCOLS = ("random", "domain_disjoint", "vector_disjoint")
ABLATION_PROTOCOL = "domain_disjoint"
MODEL_KEYS = tuple(MODEL_SPECS)


@dataclass(frozen=True)
class PreparedData:
    raw: pd.DataFrame
    training_pool: pd.DataFrame
    mixed_rows_dropped: int
    report: dict
    dataset_sha: str


def prepare_data() -> PreparedData:
    raw = load_raw()
    deduped = deduplicate(raw)
    pool, mixed_dropped = drop_mixed_domains(deduped)
    return PreparedData(
        raw=raw,
        training_pool=pool,
        mixed_rows_dropped=mixed_dropped,
        report=dataset_report(raw),
        dataset_sha=dataset_sha256(DATASET_PATH),
    )


def build_splits(prepared: PreparedData, seed: int = SEED) -> dict[str, Split]:
    return {
        "random": random_row_split(prepared.raw, TEST_FRACTION, seed),
        "domain_disjoint": domain_disjoint_split(prepared.training_pool, TEST_FRACTION, seed),
        "vector_disjoint": vector_disjoint_split(prepared.training_pool, TEST_FRACTION, seed),
    }


def experiment_combos() -> list[tuple[str, str]]:
    """(protocol, feature_set) pairs actually measured, in stable order."""
    combos = {(protocol, SHIPPED_FEATURE_SET) for protocol in PROTOCOLS}
    combos |= {(ABLATION_PROTOCOL, name) for name in ABLATIONS}
    protocol_order = {name: i for i, name in enumerate(PROTOCOLS)}
    feature_order = {name: i for i, name in enumerate(ABLATIONS)}
    return sorted(combos, key=lambda c: (protocol_order[c[0]], feature_order[c[1]]))


def _library_versions() -> dict[str, str]:
    import joblib
    import lightgbm
    import numpy
    import sklearn
    import xgboost

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "pandas": pd.__version__,
        "numpy": numpy.__version__,
        "joblib": joblib.__version__,
    }


def run_experiments(
    seed: int = SEED,
    model_keys: tuple[str, ...] = MODEL_KEYS,
    progress=None,
    keep_models: bool = False,
) -> tuple[dict, dict]:
    """Run the full benchmark.

    Returns ``(results_document, fitted_models)``. ``fitted_models`` only
    contains estimators when ``keep_models`` is true — the models fitted on
    the primary protocol (``domain_disjoint`` + shipped feature set), which
    are exactly the artifacts shipped to ``models/``.
    """
    prepared = prepare_data()
    splits = build_splits(prepared, seed)
    experiments: list[dict] = []
    combos = experiment_combos()
    kept: dict[str, object] = {}

    for protocol, feature_set in combos:
        split = splits[protocol]
        columns = ablation_features(feature_set)
        x_train = split.train[columns]
        x_test = split.test[columns]
        y_train = split.train[LABEL_COLUMN]
        y_test = split.test[LABEL_COLUMN]
        for model_key in model_keys:
            estimator, hyperparameters = build_model(model_key, seed, y_train)
            t0 = time.perf_counter()
            estimator.fit(x_train, y_train)
            fit_seconds = round(time.perf_counter() - t0, 3)
            y_pred = estimator.predict(x_test)
            y_score = estimator.predict_proba(x_test)[:, 1]
            record = {
                "model": model_key,
                "protocol": protocol,
                "feature_set": feature_set,
                "n_features": len(columns),
                "train_rows": int(len(x_train)),
                "test_rows": int(len(x_test)),
                "hyperparameters": hyperparameters,
                "metrics": classification_metrics(y_test, y_pred, y_score),
                "latency_ms": measure_latency_ms(estimator, x_test),
                "fit_seconds": fit_seconds,
            }
            experiments.append(record)
            if keep_models and protocol == ABLATION_PROTOCOL and feature_set == SHIPPED_FEATURE_SET:
                kept[model_key] = estimator
            if progress:
                m = record["metrics"]
                progress(
                    f"[{protocol}/{feature_set}/{model_key}] "
                    f"acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
                    f"roc_auc={m['roc_auc']} pr_auc={m['pr_auc']} "
                    f"fpr={m['fpr']} fnr={m['fnr']}"
                )

    results = {
        "metrics_schema": "metrics-v1",
        "feature_schema_version": SCHEMA_VERSION,
        "feature_schema_fingerprint": schema_fingerprint(),
        "seed": seed,
        "test_fraction": TEST_FRACTION,
        "generated_by": f"python train_model.py --seed {seed}",
        "dataset": {
            "path": str(DATASET_PATH.relative_to(DATASET_PATH.parents[1])),
            "sha256": prepared.dataset_sha,
            **prepared.report,
            "rows_training_pool": int(len(prepared.training_pool)),
            "mixed_label_rows_dropped": int(prepared.mixed_rows_dropped),
        },
        "split_info": {protocol: splits[protocol].info for protocol in PROTOCOLS},
        "experiments": experiments,
        "environment": {
            **_library_versions(),
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    }
    return results, kept


def split_for(results: dict, protocol: str, seed: int = SEED) -> Split:
    """Rebuild the exact split a results document was produced with."""
    prepared = prepare_data()
    if protocol == "random":
        return random_row_split(prepared.raw, results["test_fraction"], seed)
    if protocol == "domain_disjoint":
        return domain_disjoint_split(prepared.training_pool, results["test_fraction"], seed)
    if protocol == "vector_disjoint":
        return vector_disjoint_split(prepared.training_pool, results["test_fraction"], seed)
    raise ValueError(f"unknown protocol {protocol!r}")
