"""Model candidates for the P2 benchmark.

Only four candidates are evaluated, and each has a distinct job:

- ``logistic_regression`` — linear, interpretable baseline (calibrated-ish,
  fast, tiny artifact). Any nonlinear model must beat it to justify its size.
- ``random_forest`` — incumbent served model; bagged trees over the same
  features.
- ``xgboost`` — incumbent gradient-boosting model from the legacy pipeline.
- ``lightgbm`` — gradient boosting with a different tree-growth prior; kept
  only if its measured metrics earn it (decided from results, not decoration).

Determinism: every estimator receives an explicit ``random_state`` and is
restricted to a single thread so re-runs reproduce identical predictions
regardless of machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression

SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    key: str
    algorithm: str
    artifact: str
    hyperparameters: dict


MODEL_SPECS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        key="logistic_regression",
        algorithm="LogisticRegression",
        artifact="logistic_regression.joblib",
        hyperparameters={
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
    ),
    "random_forest": ModelSpec(
        key="random_forest",
        algorithm="RandomForestClassifier",
        artifact="random_forest.joblib",
        hyperparameters={
            "n_estimators": 500,
            "class_weight": "balanced",
            "n_jobs": 1,
        },
    ),
    "xgboost": ModelSpec(
        key="xgboost",
        algorithm="XGBClassifier",
        artifact="xgboost.joblib",
        hyperparameters={
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "nthread": 1,
        },
    ),
    "lightgbm": ModelSpec(
        key="lightgbm",
        algorithm="LGBMClassifier",
        artifact="lightgbm.joblib",
        hyperparameters={
            "n_estimators": 500,
            "learning_rate": 0.02,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": 1,
            "deterministic": True,
            "force_row_wise": True,
            "verbose": -1,
        },
    ),
}


def _estimator(spec: ModelSpec):
    if spec.algorithm == "LogisticRegression":
        return LogisticRegression(**spec.hyperparameters)
    if spec.algorithm == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**spec.hyperparameters)
    if spec.algorithm == "XGBClassifier":
        from xgboost import XGBClassifier

        return XGBClassifier(**spec.hyperparameters)
    if spec.algorithm == "LGBMClassifier":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**spec.hyperparameters)
    raise ValueError(f"unknown algorithm {spec.algorithm!r}")


def build_model(key: str, seed: int = SEED, y_train: pd.Series | None = None):
    """Construct an un-fitted estimator with deterministic settings.

    Returns ``(estimator, fitted_hyperparameters)`` where the recorded
    hyperparameters include the seed and, for boosting models, the exact
    ``scale_pos_weight`` computed from ``y_train``.
    """
    spec = MODEL_SPECS[key]
    params = dict(spec.hyperparameters)
    params["random_state"] = seed
    if key in ("xgboost", "lightgbm") and y_train is not None:
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        params["scale_pos_weight"] = round(n_neg / max(n_pos, 1), 6)
    estimator = _estimator(spec)
    estimator.set_params(**params)
    recorded = {"algorithm": spec.algorithm, **params}
    return estimator, recorded
