"""features-v2 — the single source of truth for PhishGuard's 15-feature contract.

One schema serves training, evaluation, inference and the API:

- ``train_model.py`` / ``evaluate_models.py`` select and order dataset columns
  through this module.
- ``URLFeatureExtraction.featureExtraction`` returns a mapping that is validated
  against this module before it reaches a model.
- ``app.py`` asserts at startup that the registry's ``feature_columns`` equal
  :data:`FEATURE_NAMES`.

Historical note (P2): the feature set and its exact value semantics were
recovered from the upstream extraction notebook and verified against
``DataFiles/5.urldata.csv`` by reconstructing 9,964 of the 10,000 rows from the
raw source URL lists (see ``DataFiles/DATASETS.md``). ``Have_IP`` in particular
is a hostname-scoped dotted-quad test — the legacy full-URL variant is
falsified by 37 dataset rows.

The legacy 16th column, ``Web_Traffic``, is **not** part of features-v2: its
source (Alexa rank) is dead, so it can only ever be served as a constant.
Measured on the domain-disjoint test split, forcing the serving value while
the model was trained on varying values costs ~4 accuracy points (see
``docs/METHODOLOGY.md``), so the column is dropped from the contract and kept
only as a diagnostic ablation level (:data:`ABLATIONS`).
"""

from __future__ import annotations

import hashlib
import json
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

SCHEMA_VERSION = "features-v2"
LABEL_SEMANTICS = {0: "legitimate", 1: "phishing"}

#: Sanity bound for URL_Depth (dataset max is 20; the cap only rejects
#: physically implausible values, it is not a learned threshold).
URL_DEPTH_MAX = 50


class SchemaError(ValueError):
    """A feature mapping or frame does not satisfy the features-v2 contract."""


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    minimum: int
    maximum: int
    description: str
    source: str


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("Have_IP", "url", 0, 1,
                "hostname contains a dotted-quad IPv4 pattern",
                "verified against dataset (4964/4964 reconstructed rows)"),
    FeatureSpec("Have_At", "url", 0, 1,
                "URL contains '@'",
                "reproduces upstream rule"),
    FeatureSpec("URL_Length", "url", 0, 1,
                "1 when len(url) >= 54 else 0",
                "reproduces upstream rule (threshold 54)"),
    FeatureSpec("URL_Depth", "url", 0, URL_DEPTH_MAX,
                "number of non-empty path segments",
                "reproduces upstream rule"),
    FeatureSpec("Redirection", "url", 0, 1,
                "'//' occurring beyond the scheme separator",
                "reproduces upstream rule"),
    FeatureSpec("https_Domain", "url", 0, 1,
                "substring 'https' present in hostname",
                "reproduces upstream rule"),
    FeatureSpec("TinyURL", "url", 0, 1,
                "hostname matches the legacy shortener regex list",
                "reproduces upstream rule"),
    FeatureSpec("Prefix/Suffix", "url", 0, 1,
                "hostname contains '-'",
                "reproduces upstream rule"),
    FeatureSpec("DNS_Record", "domain", 0, 1,
                "1 when WHOIS lookup fails (no registration intelligence)",
                "reproduces upstream rule (inverted legacy naming)"),
    FeatureSpec("Domain_Age", "domain", 0, 1,
                "1 when WHOIS is unavailable or registration lifetime < 6 months",
                "reproduces upstream rule"),
    FeatureSpec("Domain_End", "domain", 0, 1,
                "1 when WHOIS is unavailable or expiry is >= 6 months away",
                "reproduces upstream rule"),
    FeatureSpec("iFrame", "page", 0, 1,
                "1 when page fetch failed or iframe probe found no match",
                "reproduces upstream rule (incl. legacy regex quirk)"),
    FeatureSpec("Mouse_Over", "page", 0, 1,
                "1 when page fetch failed or onmouseover script found",
                "reproduces upstream rule"),
    FeatureSpec("Right_Click", "page", 0, 1,
                "1 when page fetch failed or right-click-disable script absent",
                "reproduces upstream rule"),
    FeatureSpec("Web_Forwards", "page", 0, 1,
                "1 when page fetch failed or redirect chain > 2 hops",
                "reproduces upstream rule"),
)

FEATURE_NAMES: list[str] = [spec.name for spec in FEATURES]
SPEC_BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in FEATURES}

GROUPS: dict[str, list[str]] = {}
for _spec in FEATURES:
    GROUPS.setdefault(_spec.group, []).append(_spec.name)

#: Dataset-only columns referenced by :data:`ABLATIONS` but never shipped:
#: the legacy ``Web_Traffic`` column exists in the training CSV, yet its
#: Alexa source is dead, so inference can only feed a constant (measured
#: serving skew: ~4 accuracy points — docs/METHODOLOGY.md).
DIAGNOSTIC_COLUMNS: tuple[str, ...] = ("Web_Traffic",)

#: The feature set actually shipped: full features-v2 (URL + domain + page).
SHIPPED_FEATURE_SET = "url_domain_page"

#: Ablation ladder — each level is a strict prefix of the next.
ABLATIONS: dict[str, list[str]] = {
    "url_only": GROUPS["url"],
    "url_domain": GROUPS["url"] + GROUPS["domain"],
    "url_domain_page": GROUPS["url"] + GROUPS["domain"] + GROUPS["page"],
    # Diagnostic only: the legacy 16-column dataset, dead column included.
    "full_legacy16": (
        GROUPS["url"] + GROUPS["domain"] + GROUPS["page"] + list(DIAGNOSTIC_COLUMNS)
    ),
}


def schema_fingerprint() -> str:
    """Stable hash of schema version, feature order, types and ranges."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "label_semantics": {str(k): v for k, v in LABEL_SEMANTICS.items()},
        "features": [
            {"name": s.name, "group": s.group, "min": s.minimum, "max": s.maximum}
            for s in FEATURES
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _coerce_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise SchemaError(f"feature {name!r} must be an integer, got {type(value).__name__}")
    return int(value)


def validate_features(features: Mapping[str, object]) -> dict[str, int]:
    """Validate a feature mapping against features-v2 and return it in order.

    Rejects missing features, unknown features, non-integer values and values
    outside each feature's declared range.
    """
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise SchemaError(f"missing features: {missing}")
    extra = [key for key in features if key not in SPEC_BY_NAME]
    if extra:
        raise SchemaError(f"unknown features: {extra}")
    validated: dict[str, int] = {}
    for name in FEATURE_NAMES:
        spec = SPEC_BY_NAME[name]
        value = _coerce_int(name, features[name])
        if not spec.minimum <= value <= spec.maximum:
            raise SchemaError(
                f"feature {name!r} out of range [{spec.minimum}, {spec.maximum}]: {value}"
            )
        validated[name] = value
    return validated


def ordered_values(features: Mapping[str, object]) -> list[int]:
    """Validate ``features`` and return the values in canonical order."""
    return list(validate_features(features).values())


def to_frame(records: Iterable[Mapping[str, object]]):
    """Build a DataFrame in canonical column order from feature mappings."""
    import pandas as pd

    rows = [ordered_values(record) for record in records]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def frame_from_features(features: Mapping[str, object]):
    """Single-record convenience wrapper around :func:`to_frame`."""
    return to_frame([features])


def select_dataset_frame(frame):
    """Return the dataset's feature columns in canonical order.

    Raises :class:`SchemaError` when a declared feature is absent. Extra
    columns (``Domain``, ``Label``, diagnostic-only ``Web_Traffic``) are
    ignored — they are not model inputs.
    """
    import pandas as pd

    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise SchemaError(f"dataset is missing feature columns: {missing}")
    return pd.DataFrame(
        {name: frame[name] for name in FEATURE_NAMES},
        index=frame.index,
    )


def ablation_features(name: str) -> list[str]:
    try:
        return list(ABLATIONS[name])
    except KeyError as exc:
        raise SchemaError(
            f"unknown ablation {name!r}; expected one of {sorted(ABLATIONS)}"
        ) from exc
