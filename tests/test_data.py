"""Dataset provenance and measured-statistics tests.

Every constant here was measured against the committed
``DataFiles/5.urldata.csv`` during the P2 dataset investigation
(``DataFiles/DATASETS.md`` records the full audit).
"""

import pytest

from phishguard.features import FEATURE_NAMES, SchemaError
from phishguard.ml.data import (
    DATASET_PATH,
    DOMAIN_COLUMN,
    LABEL_COLUMN,
    dataset_report,
    dataset_sha256,
    deduplicate,
    drop_mixed_domains,
    load_raw,
)

#: sha256 of the committed training CSV — provenance for every metric.
DATASET_SHA256 = "5c74254ce576a730beaea2ed15c3acbfd7e02dad853b552d248b5af5a034b188"


def test_dataset_exists_with_expected_checksum():
    assert DATASET_PATH.is_file()
    assert dataset_sha256() == DATASET_SHA256


def test_load_raw_shape_and_labels():
    frame = load_raw()
    assert frame.shape == (10000, 18)
    assert sorted(frame[LABEL_COLUMN].unique()) == [0, 1]
    assert frame[LABEL_COLUMN].value_counts().to_dict() == {0: 5000, 1: 5000}
    assert not frame.isna().any().any()


def test_load_raw_rejects_unexpected_columns(tmp_path):
    frame = load_raw()
    frame["Bogus_Column"] = 0
    bad_csv = tmp_path / "unexpected.csv"
    frame.to_csv(bad_csv, index=False)
    with pytest.raises(SchemaError, match="unexpected columns"):
        load_raw(bad_csv)


def test_report_matches_audited_constants():
    report = dataset_report()
    assert report["rows"] == 10000
    assert report["rows_unique"] == 4374
    assert report["label_counts"] == {0: 5000, 1: 5000}
    assert report["unique_domains"] == 3553
    # features-v2 (15 columns): vectors that differ only in the dead
    # Web_Traffic column are the same vector for the shipped models.
    assert report["unique_feature_vectors"] == 581
    assert report["conflicting_vectors"] == 94
    assert report["rows_in_conflicting_vectors"] == 6593
    assert report["mixed_label_domains"] == 3
    assert report["deduplicated_rows"] == 4374
    assert report["deduplicated_label_counts"] == {0: 529, 1: 3845}
    assert report["nulls"] == 0


def test_deduplicate_drops_exact_duplicates_only():
    frame = load_raw()
    deduped = deduplicate(frame)
    assert len(deduped) == 4374
    assert deduped.duplicated().sum() == 0


def test_drop_mixed_domains_removes_eight_rows():
    frame = deduplicate(load_raw())
    filtered, removed = drop_mixed_domains(frame)
    assert removed == 8
    assert len(filtered) == 4366
    per_domain = filtered.groupby(DOMAIN_COLUMN)[LABEL_COLUMN].nunique()
    assert (per_domain > 1).sum() == 0


def test_feature_columns_are_a_subset_of_the_dataset():
    frame = load_raw()
    assert set(FEATURE_NAMES) <= set(frame.columns)
    assert DOMAIN_COLUMN in frame.columns
