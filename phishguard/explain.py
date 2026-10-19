"""Explainability: turn a features-v2 mapping into human-readable reasons.

``/predict`` answers *is this URL suspicious*; :func:`explain` answers *why*,
using exactly the inputs the model saw. No surrogate models, no SHAP cache --
the signals are the deterministic rules the extractor already computed, so the
explanation can never disagree with the feature vector that produced the score.

Two honesty rules shape the wording:

* Lexical signals (URL syntax) are stated firmly: they are computed from the
  URL alone and cannot fail.
* Domain/page signals are stated as what the *feature value* means. Several
  legacy features are ``1`` both when a pattern was observed **and** when the
  lookup/fetch failed (``Domain_Age``, ``Mouse_Over``, ...), and
  ``iFrame``/``Right_Click`` encode upstream quirks -- claiming more than the
  value proves would be a lie the user cannot check.
"""

from __future__ import annotations

from collections.abc import Mapping

from phishguard.features import validate_features

#: Page-level features all collapse to ``1`` when the fetch failed (the
#: extractor has no page to inspect), so four ``1``s together mean
#: "unavailable" rather than four independent observations.
PAGE_FEATURES: tuple[str, ...] = ("iFrame", "Mouse_Over", "Right_Click", "Web_Forwards")

UNAVAILABLE_REASON = (
    "Page inspection unavailable: the site could not be fetched "
    "(blocked, unreachable or timed out)"
)

#: ``(feature, risk_value, message)`` in presentation order: URL syntax,
#: then registration intelligence, then page signals.
_RISK_RULES: tuple[tuple[str, int, str], ...] = (
    ("Have_IP", 1, "Host is a raw IP address instead of a domain name"),
    ("Have_At", 1, "URL contains '@' - text before it is ignored by browsers"),
    ("URL_Length", 1, "Long URL (54+ characters), typical of obfuscated phishing"),
    ("Redirection", 1, "Extra '//' separator in the URL (open-redirect pattern)"),
    ("https_Domain", 1, "'https' appears inside the hostname (brand spoofing)"),
    ("TinyURL", 1, "URL shortener hides the real destination"),
    ("Prefix/Suffix", 1, "Hostname contains '-' (lookalike-domain trick)"),
    ("DNS_Record", 1, "No registration/DNS intelligence available for this domain"),
    ("Domain_Age", 1, "Domain is under 6 months old, or WHOIS was unavailable"),
    ("Domain_End", 1, "Domain expires in under 6 months, or WHOIS was unavailable"),
    ("iFrame", 1, "No iframe/frameBorder signal found in the page source"),
    ("Mouse_Over", 1, "onmouseover status-bar script found in the page source"),
    ("Right_Click", 1, "Right-click-disable script not found in the page source"),
    ("Web_Forwards", 1, "Page redirects more than twice"),
)


def explain(features: Mapping[str, object]) -> list[str]:
    """Return ordered, human-readable risk signals for ``features``.

    Empty list means "nothing to report", which is not the same as "safe" --
    a model can flag a URL whose lexical features are all benign. Callers must
    treat the verdict and the reasons as independent outputs.
    """
    validated = validate_features(features)

    if all(validated[name] == 1 for name in PAGE_FEATURES):
        reasons = [UNAVAILABLE_REASON]
        skip = set(PAGE_FEATURES)
    else:
        reasons = []
        skip = set()

    reasons.extend(
        message
        for name, risk_value, message in _RISK_RULES
        if name not in skip and validated[name] == risk_value
    )
    return reasons
