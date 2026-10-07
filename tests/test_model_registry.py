"""Model registry integrity tests.

These tests are the safety net for the pickle -> joblib migration: if a
registry entry drifts from the artifact it describes, CI fails.
"""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "models" / "registry.json"


def _load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registry_lists_at_least_one_model():
    registry = _load_registry()
    assert registry, "models/registry.json must not be empty"


def test_registry_artifacts_exist_and_match_checksums():
    registry = _load_registry()
    for name, entry in registry.items():
        artifact = ROOT / "models" / name
        assert artifact.is_file(), f"missing artifact {name}"
        assert _sha256(artifact) == entry["artifact_sha256"], (
            f"checksum drift for {name} — artifact changed without registry update"
        )
        assert entry["parity_verified"] is True


def test_registry_declares_feature_schema():
    registry = _load_registry()
    for name, entry in registry.items():
        assert len(entry["feature_columns"]) == 16, f"{name}: expected 16 features"
        assert entry["feature_columns"][0] == "Have_IP"
        assert entry["feature_columns"][-1] == "Web_Forwards"
        assert entry["label_semantics"] == {"0": "legitimate", "1": "phishing"}


@pytest.mark.parametrize("artifact_name", ["random_forest.joblib", "xgboost.joblib"])
def test_model_predicts_valid_labels(artifact_name):
    model = joblib.load(ROOT / "models" / artifact_name)
    frame = pd.DataFrame(
        [[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0]],
        columns=_load_registry()[artifact_name]["feature_columns"],
    )
    prediction = model.predict(frame)
    assert int(prediction[0]) in (0, 1)

    probabilities = model.predict_proba(frame)
    assert probabilities.shape == (1, 2)
    assert abs(float(probabilities.sum()) - 1.0) < 1e-6


def test_artifacts_are_distinct_models_not_renamed_duplicates():
    """Guards against the upstream bug where two differently named files
    contained byte-identical Random Forest pickles."""
    registry = _load_registry()
    algorithms = {entry["algorithm"] for entry in registry.values()}
    assert len(algorithms) >= 2, f"expected multiple algorithms, got {algorithms}"

    checksums = {entry["artifact_sha256"] for entry in registry.values()}
    assert len(checksums) == len(registry), "two registry entries share one artifact"

    for name, entry in registry.items():
        model = joblib.load(ROOT / "models" / name)
        assert type(model).__name__ == entry["algorithm"], (
            f"{name} declares {entry['algorithm']} but contains {type(model).__name__}"
        )
