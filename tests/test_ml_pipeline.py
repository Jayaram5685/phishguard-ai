"""ML pipeline unit tests: combos, metrics, verification gates.

These stay fast (single-model runs on small slices); the full 24-fit
benchmark is exercised in CI by ``python train_model.py --check``.
"""

import json
from pathlib import Path

import joblib

from phishguard.features import FEATURE_NAMES, SHIPPED_FEATURE_SET, ablation_features
from phishguard.ml.metrics import (
    artifact_size_bytes,
    classification_metrics,
    measure_latency_ms,
    wilson_ci,
)
from phishguard.ml.models import MODEL_SPECS, build_model
from phishguard.ml.pipeline import (
    MODEL_KEYS as PIPELINE_MODEL_KEYS,
)
from phishguard.ml.pipeline import (
    PROTOCOLS,
    experiment_combos,
    run_experiments,
)
from phishguard.ml.registry import build_registry_entry
from phishguard.ml.verify import compare_metrics, experiment_index

ROOT = Path(__file__).resolve().parent.parent


def test_experiment_combos_are_six_pairs():
    combos = experiment_combos()
    assert len(combos) == 6
    for protocol in PROTOCOLS:
        assert (protocol, SHIPPED_FEATURE_SET) in combos
    domain_ablations = [f for p, f in combos if p == "domain_disjoint"]
    assert domain_ablations == ["url_only", "url_domain", "url_domain_page", "full_legacy16"]


def test_model_specs_cover_four_algorithms():
    assert tuple(MODEL_SPECS) == (
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
    )
    assert PIPELINE_MODEL_KEYS == tuple(MODEL_SPECS)
    algorithms = {spec.algorithm for spec in MODEL_SPECS.values()}
    assert len(algorithms) == 4


def test_build_model_is_seeded_and_records_scale_pos_weight():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    frame = pd.DataFrame(
        [[0] * len(FEATURE_NAMES) + [0], [1] * len(FEATURE_NAMES) + [1]],
        columns=FEATURE_NAMES + ["Label"],
    )
    y = frame["Label"]
    estimator, params = build_model("logistic_regression", 42, y)
    assert isinstance(estimator, LogisticRegression)
    assert params["random_state"] == 42

    _, xgb_params = build_model("xgboost", 42, y)
    assert xgb_params["scale_pos_weight"] == 1.0
    assert xgb_params["random_state"] == 42


def test_classification_metrics_perfect_and_random_cases():
    perfect = classification_metrics([0, 0, 1, 1], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert perfect["accuracy"] == 1.0
    assert perfect["f1"] == 1.0
    assert perfect["roc_auc"] == 1.0
    assert perfect["fpr"] == 0.0
    assert perfect["fnr"] == 0.0
    assert perfect["counts"]["tp"] == 2
    assert perfect["counts"]["tn"] == 2
    assert perfect["counts"]["fp"] == 0
    assert perfect["counts"]["fn"] == 0

    one_class = classification_metrics([1, 1], [1, 1], [0.9, 0.8])
    assert one_class["roc_auc"] is None
    assert one_class["accuracy"] == 1.0


def test_wilson_ci_brackets_the_point_estimate():
    low, high = wilson_ci(8, 10)
    assert 0.0 <= low <= 0.8 <= high <= 1.0


def test_measure_latency_and_artifact_size():
    estimator, _ = build_model("logistic_regression", 42, None)
    import pandas as pd

    frame = pd.DataFrame(
        [[0] * len(FEATURE_NAMES), [1] * len(FEATURE_NAMES)], columns=FEATURE_NAMES
    )
    estimator.fit(frame, [0, 1])  # lbfgs requires both classes to fit
    latency = measure_latency_ms(estimator, frame)
    assert latency["mean_ms"] >= 0
    assert latency["rounds"] == 100
    artifact = ROOT / "models" / "logistic_regression.joblib"
    assert artifact_size_bytes(artifact) > 0


def test_compare_metrics_detects_drift_and_tolerates_match():
    committed = {
        "experiments": [
            {"model": "m", "protocol": "p", "feature_set": "f",
             "metrics": {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5,
                          "roc_auc": 0.5, "pr_auc": 0.5, "fpr": 0.5, "fnr": 0.5}},
        ]
    }
    assert compare_metrics(committed, committed, 1e-9) == []

    drifted = json.loads(json.dumps(committed))
    drifted["experiments"][0]["metrics"]["accuracy"] = 0.51
    diffs = compare_metrics(committed, drifted, 1e-6)
    assert len(diffs) == 1 and "accuracy" in diffs[0]

    assert compare_metrics(committed, {"experiments": []}, 1e-6)


def test_run_experiments_lr_only_produces_valid_documents(tmp_path):
    results, kept = run_experiments(
        seed=42, model_keys=("logistic_regression",), progress=None, keep_models=True
    )
    assert len(results["experiments"]) == 6  # 6 protocol/feature-set combos
    assert list(kept) == ["logistic_regression"]
    assert results["feature_schema_version"] == "features-v2"
    assert results["dataset"]["sha256"]
    for record in results["experiments"]:
        assert record["n_features"] == len(ablation_features(record["feature_set"]))
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            value = record["metrics"][metric]
            if value is not None:
                assert 0.0 <= value <= 1.0

    index = experiment_index(results)
    experiment = index[("logistic_regression", "domain_disjoint", SHIPPED_FEATURE_SET)]

    artifact = tmp_path / "logistic_regression.joblib"
    joblib.dump(kept["logistic_regression"], artifact)
    entry = build_registry_entry(
        "logistic_regression", artifact, results, experiment, seed=42
    )
    assert entry["feature_columns"] == FEATURE_NAMES
    assert entry["feature_schema_version"] == "features-v2"
    assert entry["split"]["domain_overlap"] == 0
    assert entry["evaluation"]["feature_set"] == SHIPPED_FEATURE_SET
    assert entry["evaluation"]["metrics_status"] == "reproduced"
    assert len(entry["artifact_sha256"]) == 64


def test_registry_and_metrics_files_are_consistent():
    """Committed artifacts: metrics.json <-> registry.json <-> files."""
    results = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "models" / "registry.json").read_text(encoding="utf-8"))
    index = experiment_index(results)
    assert registry, "registry must not be empty"
    for artifact, entry in registry.items():
        key = (Path(artifact).stem, "domain_disjoint", SHIPPED_FEATURE_SET)
        assert key in index, f"{artifact}: no committed experiment for {key}"
        assert entry["evaluation"]["metrics"] == index[key]["metrics"]
        assert (ROOT / "models" / artifact).stat().st_size == entry["artifact_size_bytes"]
