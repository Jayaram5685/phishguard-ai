"""features-v2 schema contract tests.

The schema is the single source of truth shared by training, evaluation,
inference and the API — these tests lock its exact shape.
"""

import pytest

from phishguard.features import (
    DIAGNOSTIC_COLUMNS,
    FEATURE_NAMES,
    LABEL_SEMANTICS,
    SCHEMA_VERSION,
    SHIPPED_FEATURE_SET,
    SPEC_BY_NAME,
    SchemaError,
    ablation_features,
    frame_from_features,
    ordered_values,
    schema_fingerprint,
    validate_features,
)

#: Exact canonical order — any change here is a breaking schema change and
#: must come with a schema-version bump plus a full retrain.
EXPECTED_FEATURES = [
    "Have_IP",
    "Have_At",
    "URL_Length",
    "URL_Depth",
    "Redirection",
    "https_Domain",
    "TinyURL",
    "Prefix/Suffix",
    "DNS_Record",
    "Domain_Age",
    "Domain_End",
    "iFrame",
    "Mouse_Over",
    "Right_Click",
    "Web_Forwards",
]


def _valid_mapping() -> dict:
    return dict.fromkeys(FEATURE_NAMES, 0)


def test_schema_version_and_labels():
    assert SCHEMA_VERSION == "features-v2"
    assert LABEL_SEMANTICS == {0: "legitimate", 1: "phishing"}


def test_feature_names_exact_order():
    assert FEATURE_NAMES == EXPECTED_FEATURES


def test_dead_web_traffic_column_is_not_in_the_contract():
    assert "Web_Traffic" not in FEATURE_NAMES
    assert DIAGNOSTIC_COLUMNS == ("Web_Traffic",)
    assert set(DIAGNOSTIC_COLUMNS).isdisjoint(FEATURE_NAMES)


def test_all_specs_binary_or_bounded_ints():
    for name in FEATURE_NAMES:
        spec = SPEC_BY_NAME[name]
        assert isinstance(spec.minimum, int) and isinstance(spec.maximum, int)
        assert spec.minimum <= spec.maximum
        assert spec.group in {"url", "domain", "page"}
        assert spec.description and spec.source


def test_ablation_ladder_is_nested_prefixes():
    url_only = ablation_features("url_only")
    url_domain = ablation_features("url_domain")
    shipped = ablation_features(SHIPPED_FEATURE_SET)
    legacy = ablation_features("full_legacy16")
    assert url_only[: len(url_only)] == url_only
    assert url_domain[: len(url_only)] == url_only
    assert shipped[: len(url_domain)] == url_domain
    assert legacy[: len(shipped)] == shipped
    assert len(url_only) == 8
    assert len(url_domain) == 11
    assert len(shipped) == 15
    assert len(legacy) == 16
    assert set(legacy) == set(shipped) | set(DIAGNOSTIC_COLUMNS)
    assert shipped == FEATURE_NAMES


def test_unknown_ablation_rejected():
    with pytest.raises(SchemaError, match="unknown ablation"):
        ablation_features("everything")


def test_validate_features_orders_and_accepts_ints():
    mapping = _valid_mapping()
    mapping["URL_Depth"] = 3
    validated = validate_features(mapping)
    assert list(validated) == FEATURE_NAMES
    assert ordered_values(mapping)[FEATURE_NAMES.index("URL_Depth")] == 3


def test_validate_features_rejects_missing_unknown_and_bad_values():
    missing = _valid_mapping()
    del missing["Have_IP"]
    with pytest.raises(SchemaError, match="missing features"):
        validate_features(missing)

    extra = _valid_mapping()
    extra["Web_Traffic"] = 1
    with pytest.raises(SchemaError, match="unknown features"):
        validate_features(extra)

    non_int = _valid_mapping()
    non_int["URL_Depth"] = "3"
    with pytest.raises(SchemaError, match="must be an integer"):
        validate_features(non_int)

    out_of_range = _valid_mapping()
    out_of_range["URL_Depth"] = 999
    with pytest.raises(SchemaError, match="out of range"):
        validate_features(out_of_range)


def test_frame_from_features_matches_schema():
    frame = frame_from_features(_valid_mapping())
    assert list(frame.columns) == FEATURE_NAMES
    assert len(frame) == 1


def test_schema_fingerprint_is_stable_hex_sha256():
    fingerprint = schema_fingerprint()
    assert fingerprint == schema_fingerprint()
    assert len(fingerprint) == 64
    int(fingerprint, 16)  # valid hex
