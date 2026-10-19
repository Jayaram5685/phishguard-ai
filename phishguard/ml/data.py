"""Dataset access helpers — loading, provenance and measured statistics.

Every number produced by :func:`dataset_report` is measured at runtime from
``DataFiles/5.urldata.csv``; nothing here is hard-coded except the schema
invariants asserted by tests.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from phishguard.features import (
    DIAGNOSTIC_COLUMNS,
    FEATURE_NAMES,
    SchemaError,
    select_dataset_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "DataFiles" / "5.urldata.csv"
LABEL_COLUMN = "Label"
DOMAIN_COLUMN = "Domain"


def dataset_sha256(path: Path = DATASET_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the raw dataset with stripped column names, schema-validated."""
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    allowed = (*FEATURE_NAMES, *DIAGNOSTIC_COLUMNS, LABEL_COLUMN, DOMAIN_COLUMN)
    unexpected = [c for c in frame.columns if c not in allowed]
    if unexpected:
        raise SchemaError(f"dataset has unexpected columns: {unexpected}")
    if frame[LABEL_COLUMN].isna().any() or frame[DOMAIN_COLUMN].isna().any():
        raise SchemaError("dataset contains null labels or domains")
    labels = set(frame[LABEL_COLUMN].unique())
    if not labels <= {0, 1}:
        raise SchemaError(f"dataset labels must be 0/1, got {sorted(labels)}")
    select_dataset_frame(frame)  # every declared feature must exist
    return frame


def feature_frame(frame: pd.DataFrame):
    """The model-input view of ``frame`` in canonical feature order."""
    return select_dataset_frame(frame)


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows (all columns), keeping first occurrences."""
    return frame.drop_duplicates().reset_index(drop=True)


def drop_mixed_domains(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows whose domain appears with both labels.

    Returns the filtered frame and the number of removed rows. Mixed domains
    make a domain-disjoint split ill-defined (the group itself carries a label
    conflict), and only a handful of rows are affected.
    """
    per_domain_labels = frame.groupby(DOMAIN_COLUMN)[LABEL_COLUMN].nunique()
    mixed = per_domain_labels[per_domain_labels > 1].index
    filtered = frame[~frame[DOMAIN_COLUMN].isin(mixed)].reset_index(drop=True)
    return filtered, len(frame) - len(filtered)


def dataset_report(frame: pd.DataFrame | None = None) -> dict:
    """Measured statistics used by docs, tests and the training report."""
    if frame is None:
        frame = load_raw()
    features = feature_frame(frame)
    unique_vectors = len(features.drop_duplicates())
    conflicting = frame.groupby(FEATURE_NAMES)[LABEL_COLUMN].nunique()
    conflicting = conflicting[conflicting > 1]
    per_domain_labels = frame.groupby(DOMAIN_COLUMN)[LABEL_COLUMN].nunique()
    dedup = frame.drop_duplicates()
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "rows_unique": int(len(frame.drop_duplicates())),
        "label_counts": {int(k): int(v) for k, v in frame[LABEL_COLUMN].value_counts().sort_index().items()},
        "unique_domains": int(frame[DOMAIN_COLUMN].nunique()),
        "unique_feature_vectors": int(unique_vectors),
        "conflicting_vectors": int(len(conflicting)),
        "rows_in_conflicting_vectors": int(
            features.set_index(FEATURE_NAMES).index.isin(conflicting.index).sum()
        ),
        "mixed_label_domains": int((per_domain_labels > 1).sum()),
        "deduplicated_rows": int(len(dedup)),
        "deduplicated_label_counts": {
            int(k): int(v) for k, v in dedup[LABEL_COLUMN].value_counts().sort_index().items()
        },
        "nulls": int(frame.isna().sum().sum()),
    }
