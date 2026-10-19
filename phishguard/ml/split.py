"""Leakage-aware train/test split protocols.

Three protocols are evaluated in milestone P2:

``random``
    Row-level shuffle of the raw dataset — reproduces the legacy (leaked)
    procedure so its numbers are comparable, and is kept only as a baseline.
``domain_disjoint``
    **Primary protocol.** No domain appears on both sides, so metrics measure
    generalisation to unseen domains instead of memorisation of domains seen
    during training.
``vector_disjoint``
    Stress test: no feature vector is shared between train and test, i.e. the
    model must handle feature combinations it has never seen.

All splits are deterministic for a given seed and dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import train_test_split

from phishguard.features import FEATURE_NAMES
from phishguard.ml.data import DOMAIN_COLUMN, LABEL_COLUMN

SEED = 42
TEST_FRACTION = 0.2


class SplitError(RuntimeError):
    """A split invariant was violated (should never fire in normal operation)."""


def _row_tuple(row) -> tuple:
    return tuple(row)


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    info: dict = field(default_factory=dict)


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame[LABEL_COLUMN].value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in counts.items()}


def _base_info(protocol: str, split: Split) -> dict:
    return {
        "protocol": protocol,
        "train_rows": int(len(split.train)),
        "test_rows": int(len(split.test)),
        "train_label_counts": _label_counts(split.train),
        "test_label_counts": _label_counts(split.test),
    }


def _assert_exact_partition(frame: pd.DataFrame, split: Split, protocol: str) -> None:
    """Every source row lands in exactly one side, keeping its original row id.

    Splits deliberately do **not** reset the index: the index is the row's
    provenance (its line in the deduplicated training pool), so an audited test
    prediction can be traced back to the dataset, and coverage of the source
    frame is checkable as a set equality rather than a row count.
    """
    train_ids = set(split.train.index)
    test_ids = set(split.test.index)
    overlap = train_ids & test_ids
    if overlap:
        raise SplitError(f"{protocol} split has overlapping row ids: {sorted(overlap)[:5]}")
    if len(split.train) + len(split.test) != len(frame):
        raise SplitError(f"{protocol} split dropped or duplicated rows")
    if train_ids | test_ids != set(frame.index):
        raise SplitError(f"{protocol} split does not cover the source frame exactly")


def random_row_split(
    frame: pd.DataFrame, test_fraction: float = TEST_FRACTION, seed: int = SEED
) -> Split:
    """Legacy-compatible row-level shuffle (LEAKED on this dataset)."""
    train, test = train_test_split(
        frame, test_size=test_fraction, random_state=seed, shuffle=True
    )
    split = Split(train=train, test=test)
    _assert_exact_partition(frame, split, "random")
    split.info.update(_base_info("random", split))
    split.info["test_fraction"] = test_fraction
    split.info["seed"] = seed
    split.info["leakage"] = (
        "row-level split of a dataset with massive duplicate overlap - "
        "baseline only, not a generalisation estimate"
    )
    return split


def _grouped_domain_split(
    frame: pd.DataFrame, test_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    """Stratify domains by their (pure) label so both sides keep both classes."""
    domain_labels = frame.groupby(DOMAIN_COLUMN)[LABEL_COLUMN].agg(["mean", "nunique"])
    train_domains: set[str] = set()
    test_domains: set[str] = set()
    for label in (0, 1):
        pure = sorted(domain_labels.index[domain_labels["mean"] == label].tolist())
        if not pure:
            raise SplitError(f"no pure class-{label} domains available for splitting")
        train_part, test_part = train_test_split(
            pure, test_size=test_fraction, random_state=seed, shuffle=True
        )
        train_domains.update(train_part)
        test_domains.update(test_part)
    mixed = domain_labels.index[domain_labels["nunique"] > 1].tolist()
    if mixed:
        raise SplitError(f"frame still contains mixed-label domains: {mixed[:5]}")
    return train_domains, test_domains


def domain_disjoint_split(
    frame: pd.DataFrame, test_fraction: float = TEST_FRACTION, seed: int = SEED
) -> Split:
    """Primary protocol: train and test share zero domains.

    ``frame`` must already be deduplicated and stripped of mixed-label domains
    (see :func:`phishguard.ml.data.deduplicate` / ``drop_mixed_domains``).
    """
    train_domains, test_domains = _grouped_domain_split(frame, test_fraction, seed)
    if train_domains & test_domains:
        raise SplitError("domain-disjoint split has overlapping domains")
    train = frame[frame[DOMAIN_COLUMN].isin(train_domains)]
    test = frame[frame[DOMAIN_COLUMN].isin(test_domains)]
    split = Split(train=train, test=test)
    _assert_exact_partition(frame, split, "domain-disjoint")
    split.info.update(_base_info("domain_disjoint", split))
    split.info.update(
        {
            "test_fraction": test_fraction,
            "seed": seed,
            "train_domains": len(train_domains),
            "test_domains": len(test_domains),
            "domain_overlap": 0,
        }
    )
    return split


def vector_disjoint_split(
    frame: pd.DataFrame, test_fraction: float = TEST_FRACTION, seed: int = SEED
) -> Split:
    """Stress protocol: train and test share zero feature vectors.

    Vectors that carry both labels form their own stratum so the conflict
    rate is similar on both sides.
    """
    vectors = frame[FEATURE_NAMES].drop_duplicates().reset_index(drop=True)
    stats = frame.groupby(FEATURE_NAMES, sort=False)[LABEL_COLUMN].agg(["nunique", "first"])
    strata_by_key = {
        tuple(key): ("mixed" if int(nunique) > 1 else f"class_{int(first)}")
        for key, nunique, first in zip(stats.index, stats["nunique"], stats["first"], strict=True)
    }
    strata = [strata_by_key[key] for key in vectors.itertuples(index=False, name=None)]
    train_vectors, test_vectors = train_test_split(
        vectors, test_size=test_fraction, random_state=seed, shuffle=True,
        stratify=strata,
    )
    train_keys = set(train_vectors.itertuples(index=False, name=None))
    test_keys = set(test_vectors.itertuples(index=False, name=None))
    if train_keys & test_keys:
        raise SplitError("vector-disjoint split has overlapping feature vectors")
    row_keys = frame[FEATURE_NAMES].apply(_row_tuple, axis=1)
    train = frame[row_keys.isin(train_keys)]
    test = frame[row_keys.isin(test_keys)]
    split = Split(train=train, test=test)
    _assert_exact_partition(frame, split, "vector-disjoint")
    if set(map(tuple, train[FEATURE_NAMES].itertuples(index=False, name=None))) & set(
        map(tuple, test[FEATURE_NAMES].itertuples(index=False, name=None))
    ):
        raise SplitError("vector-disjoint split has overlapping feature vectors")
    split.info.update(_base_info("vector_disjoint", split))
    split.info.update(
        {
            "test_fraction": test_fraction,
            "seed": seed,
            "train_vectors": len(train_keys),
            "test_vectors": len(test_keys),
            "vector_overlap": 0,
        }
    )
    return split
