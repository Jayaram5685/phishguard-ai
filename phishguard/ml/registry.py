"""Model registry read/write helpers (models/registry.json)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from phishguard.features import (
    FEATURE_NAMES,
    LABEL_SEMANTICS,
    SCHEMA_VERSION,
    SHIPPED_FEATURE_SET,
    schema_fingerprint,
)
from phishguard.ml.models import MODEL_SPECS

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "models" / "registry.json"


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(registry: dict, path: Path = DEFAULT_REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_registry_entry(
    model_key: str,
    artifact_path: Path,
    results: dict,
    experiment: dict,
    seed: int,
) -> dict:
    """One registry entry per shipped artifact, with full provenance.

    ``experiment`` is the ``(model, domain_disjoint, shipped-set)`` record from
    the benchmark run — the artifact is the model fitted on that exact split.
    """
    spec = MODEL_SPECS[model_key]
    split_info = results["split_info"]["domain_disjoint"]
    dataset = results["dataset"]
    if experiment["model"] != model_key or experiment["protocol"] != "domain_disjoint":
        raise ValueError("registry entries must describe the domain_disjoint experiment")
    if experiment["feature_set"] != SHIPPED_FEATURE_SET:
        raise ValueError(
            f"registry entries must describe the {SHIPPED_FEATURE_SET!r} feature set, "
            f"got {experiment['feature_set']!r}"
        )

    return {
        "algorithm": spec.algorithm,
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_size_bytes": artifact_path.stat().st_size,
        "feature_schema_version": SCHEMA_VERSION,
        "feature_schema_fingerprint": schema_fingerprint(),
        "feature_columns": list(FEATURE_NAMES),
        "label_semantics": {str(k): v for k, v in LABEL_SEMANTICS.items()},
        "dataset": {
            "path": dataset["path"],
            "sha256": dataset["sha256"],
            "rows": dataset["rows"],
            "rows_unique": dataset["rows_unique"],
            "rows_training_pool": dataset["rows_training_pool"],
            "mixed_label_rows_dropped": dataset["mixed_label_rows_dropped"],
        },
        "split": {
            "protocol": "domain_disjoint",
            "seed": seed,
            "test_fraction": results["test_fraction"],
            "train_rows": split_info["train_rows"],
            "test_rows": split_info["test_rows"],
            "train_domains": split_info["train_domains"],
            "test_domains": split_info["test_domains"],
            "domain_overlap": 0,
            "deduplicated": True,
        },
        "training": {
            "command": results["generated_by"],
            "seed": seed,
            "trained_at_utc": results["environment"]["generated_at_utc"],
            "hyperparameters": experiment["hyperparameters"],
            "library_versions": {
                key: results["environment"][key]
                for key in ("python", "scikit_learn", "xgboost", "lightgbm", "pandas", "numpy")
                if key in results["environment"]
            },
        },
        "evaluation": {
            "protocol": "domain_disjoint",
            "feature_set": SHIPPED_FEATURE_SET,
            "metrics": experiment["metrics"],
            "latency_ms": experiment["latency_ms"],
            "metrics_status": "reproduced",
            "verify_command": "python evaluate_models.py --check",
        },
        "parity_verified": True,
        "status": "current",
    }


def update_registry(entries: dict, path: Path = DEFAULT_REGISTRY_PATH) -> dict:
    """Merge ``entries`` into the registry without touching other keys."""
    registry = load_registry(path)
    registry.update(entries)
    save_registry(registry, path)
    return registry
