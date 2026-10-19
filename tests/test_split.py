"""Leakage-free split protocol tests.

Constants were measured from the committed dataset with seed 42; the
invariants (zero overlap, coverage, stratification sanity) are what make
these protocols trustworthy regardless of seed.
"""

import pandas as pd
import pytest

from phishguard.ml.data import DOMAIN_COLUMN, LABEL_COLUMN, load_raw
from phishguard.ml.pipeline import prepare_data
from phishguard.ml.split import (
    TEST_FRACTION,
    Split,
    SplitError,
    domain_disjoint_split,
)

SEED = 42


@pytest.fixture(scope="module")
def splits() -> dict[str, Split]:
    from phishguard.ml.pipeline import build_splits

    return build_splits(prepare_data(), SEED)


def test_random_row_split_shape(splits):
    split = splits["random"]
    assert (len(split.train), len(split.test)) == (8000, 2000)
    assert split.test[LABEL_COLUMN].value_counts().to_dict() == {0: 1012, 1: 988}
    assert "leakage" in split.info  # explicitly documented as baseline-only


def test_domain_disjoint_split_shape_and_zero_overlap(splits):
    split = splits["domain_disjoint"]
    assert (len(split.train), len(split.test)) == (3488, 878)
    assert split.info["train_domains"] == 2839
    assert split.info["test_domains"] == 711
    assert split.info["domain_overlap"] == 0
    assert split.test[LABEL_COLUMN].value_counts().to_dict() == {0: 106, 1: 772}
    train_domains = set(split.train[DOMAIN_COLUMN])
    test_domains = set(split.test[DOMAIN_COLUMN])
    assert train_domains.isdisjoint(test_domains)


def test_vector_disjoint_split_shape_and_zero_overlap(splits):
    split = splits["vector_disjoint"]
    assert (len(split.train), len(split.test)) == (3223, 1143)
    assert split.info["train_vectors"] == 464
    assert split.info["test_vectors"] == 116
    assert split.info["vector_overlap"] == 0


def test_splits_cover_the_pool_exactly(splits):
    pool = prepare_data().training_pool
    for protocol in ("domain_disjoint", "vector_disjoint"):
        split = splits[protocol]
        combined = pd.concat([split.train, split.test])
        assert len(combined) == len(pool)
        assert not combined.index.duplicated().any()
        assert sorted(combined.index) == sorted(pool.index)


def test_domain_split_is_deterministic_for_a_seed():
    prepared = prepare_data()
    first = domain_disjoint_split(prepared.training_pool, TEST_FRACTION, SEED)
    second = domain_disjoint_split(prepared.training_pool, TEST_FRACTION, SEED)
    assert first.test.index.equals(second.test.index)
    other = domain_disjoint_split(prepared.training_pool, TEST_FRACTION, SEED + 1)
    assert not first.test.index.equals(other.test.index)


def test_domain_split_rejects_mixed_label_domains():
    """Mixed-label domains make the split ill-defined; the splitter must
    refuse rather than leak them (prepare_data drops them up front)."""
    from phishguard.ml.data import deduplicate

    deduped = deduplicate(load_raw())  # mixed domains still present
    with pytest.raises(SplitError, match="mixed-label domains"):
        domain_disjoint_split(deduped, TEST_FRACTION, SEED)
