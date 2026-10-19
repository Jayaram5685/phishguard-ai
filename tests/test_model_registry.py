"""Model registry integrity tests.

Registry entries must agree with the artifacts, the features-v2 schema and
the committed benchmark metrics — if any drift, CI fails.
"""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from phishguard.features import FEATURE_NAMES, SCHEMA_VERSION

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "models" / "registry.json"
METRICS_PATH = ROOT / "models" / "metrics.json"


def _load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _load_metrics() -> dict:
    return json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registry_lists_multiple_models():
    registry = _load_registry()
    assert registry, "models/registry.json must not be empty"
    algorithms = {entry["algorithm"] for entry in registry.values()}
    assert len(algorithms) >= 2, f"expected multiple algorithms, got {algorithms}"


def test_registry_artifacts_exist_and_match_checksums():
    registry = _load_registry()
    for name, entry in registry.items():
        artifact = ROOT / "models" / name
        assert artifact.is_file(), f"missing artifact {name}"
        assert _sha256(artifact) == entry["artifact_sha256"], (
            f"checksum drift for {name} - artifact changed without registry update"
        )
        assert artifact.stat().st_size == entry["artifact_size_bytes"]
        assert entry["parity_verified"] is True


def test_registry_declares_features_v2_schema():
    registry = _load_registry()
    for name, entry in registry.items():
        assert entry["feature_schema_version"] == SCHEMA_VERSION, name
        assert entry["feature_columns"] == FEATURE_NAMES, name
        assert len(entry["feature_columns"]) == 15
        assert entry["feature_columns"][0] == "Have_IP"
        assert entry["feature_columns"][-1] == "Web_Forwards"
        assert "Web_Traffic" not in entry["feature_columns"], name
        assert entry["label_semantics"] == {"0": "legitimate", "1": "phishing"}


def test_registry_entries_carry_provenance_and_metrics():
    registry = _load_registry()
    results = _load_metrics()
    index = {
        (record["model"], record["protocol"], record["feature_set"]): record
        for record in results["experiments"]
    }
    for name, entry in registry.items():
        assert len(entry["dataset"]["sha256"]) == 64, name
        assert entry["dataset"]["rows"] == 10000, name
        assert entry["split"]["protocol"] == "domain_disjoint", name
        assert entry["split"]["domain_overlap"] == 0, name
        assert entry["training"]["seed"] == 42, name
        assert entry["training"]["hyperparameters"], name

        evaluation = entry["evaluation"]
        assert evaluation["feature_set"] == "url_domain_page", name
        assert evaluation["metrics_status"] == "reproduced", name
        key = (Path(name).stem, "domain_disjoint", "url_domain_page")
        assert key in index, f"{name}: no committed experiment for {key}"
        assert evaluation["metrics"] == index[key]["metrics"], name


def test_registry_metrics_fall_in_valid_ranges():
    for entry in _load_registry().values():
        metrics = entry["evaluation"]["metrics"]
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            assert 0.0 <= metrics[metric] <= 1.0, f"{entry['algorithm']}: {metric}"
        counts = metrics["counts"]
        assert counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"] == counts["n_samples"]
        assert counts["n_samples"] == 878  # domain-disjoint test rows


@pytest.mark.parametrize("artifact_name", ["random_forest.joblib", "xgboost.joblib"])
def test_incumbent_models_still_predict_valid_labels(artifact_name):
    model = joblib.load(ROOT / "models" / artifact_name)
    frame = pd.DataFrame(
        [[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]],
        columns=_load_registry()[artifact_name]["feature_columns"],
    )
    prediction = model.predict(frame)
    assert int(prediction[0]) in (0, 1)

    probabilities = model.predict_proba(frame)
    assert probabilities.shape == (1, 2)
    assert abs(float(probabilities.sum()) - 1.0) < 1e-6


def test_every_registered_artifact_predicts():
    for name in _load_registry():
        model = joblib.load(ROOT / "models" / name)
        frame = pd.DataFrame(
            [[0] * len(FEATURE_NAMES)], columns=FEATURE_NAMES
        )
        assert int(model.predict(frame)[0]) in (0, 1), name


def test_artifacts_are_distinct_models_not_renamed_duplicates():
    """Guards against the upstream bug where two differently named files
    contained byte-identical Random Forest pickles."""
    registry = _load_registry()
    checksums = {entry["artifact_sha256"] for entry in registry.values()}
    assert len(checksums) == len(registry), "two registry entries share one artifact"

    for name, entry in registry.items():
        model = joblib.load(ROOT / "models" / name)
        assert type(model).__name__ == entry["algorithm"], (
            f"{name} declares {entry['algorithm']} but contains {type(model).__name__}"
        )
