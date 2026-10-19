"""Explainability contract: reasons must be a faithful reading of the features.

The API's ``reasons`` are only trustworthy if every sentence is implied by the
feature vector the model actually scored -- these tests pin that down and
guard the two honesty rules (fetch-failure collapse, legacy page quirks).
"""

import pytest

from phishguard.explain import _RISK_RULES, PAGE_FEATURES, UNAVAILABLE_REASON, explain
from phishguard.features import FEATURE_NAMES, SPEC_BY_NAME, SchemaError

#: URL_Depth is capped by the schema; one past the cap must be rejected.
URL_DEPTH_OUT_OF_RANGE = SPEC_BY_NAME["URL_Depth"].maximum + 1


def _features(**overrides: int) -> dict[str, int]:
    base = dict.fromkeys(FEATURE_NAMES, 0)
    base.update(overrides)
    return base


def test_all_benign_features_yield_no_reasons():
    assert explain(_features()) == []


def test_missing_features_are_rejected_not_guessed():
    with pytest.raises(SchemaError):
        explain({"Have_IP": 1})


def test_values_outside_the_schema_are_rejected():
    with pytest.raises(SchemaError):
        explain(_features(URL_Depth=URL_DEPTH_OUT_OF_RANGE))


def test_lexical_signals_appear_in_rule_order():
    reasons = explain(_features(TinyURL=1, Have_IP=1, URL_Length=1))
    assert reasons == [
        "Host is a raw IP address instead of a domain name",
        "Long URL (54+ characters), typical of obfuscated phishing",
        "URL shortener hides the real destination",
    ]


def test_every_rule_is_reachable_and_traceable_to_a_declared_feature():
    for name, risk_value, message in _RISK_RULES:
        assert name in SPEC_BY_NAME, name
        assert risk_value == 1
        assert message.strip()
        assert explain(_features(**{name: risk_value})) == [message]


def test_all_page_features_one_reports_fetch_failure_not_four_findings():
    payload = _features(**dict.fromkeys(PAGE_FEATURES, 1))
    assert explain(payload) == [UNAVAILABLE_REASON]


def test_page_features_are_described_individually_when_fetch_succeeded():
    payload = _features(Mouse_Over=1, Right_Click=1)
    reasons = explain(payload)
    assert UNAVAILABLE_REASON not in reasons
    assert any("onmouseover" in reason for reason in reasons)
    assert any("Right-click" in reason for reason in reasons)


def test_a_single_flipped_value_changes_exactly_one_reason():
    assert explain(_features(Domain_Age=1)) == [
        "Domain is under 6 months old, or WHOIS was unavailable"
    ]
    assert explain(_features(Domain_Age=0)) == []


def test_domain_and_page_signals_never_claim_more_than_the_value_proves():
    """Legacy features are 1 both on 'pattern observed' and on 'lookup failed';
    the wording must stay ambiguous where the value is ambiguous."""
    reasons = explain(_features(DNS_Record=1, Web_Forwards=1, iFrame=1, Mouse_Over=0))
    assert any("No registration/DNS intelligence" in reason for reason in reasons)
    assert any("No iframe/frameBorder signal" in reason for reason in reasons)
    for reason in reasons:
        assert "confirmed" not in reason
        assert "definitely" not in reason
