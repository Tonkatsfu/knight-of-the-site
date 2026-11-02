"""Unified 30-feature schema mapping Tier 1 (9 fast lexical features) and Tier 2 (21 heavy features).

Matches the exact attribute ordering and domain values in `phishing+websites/Training Dataset.arff`
and rules defined in `Phishing Websites Features.docx`.
"""

from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np

# Exact 30 feature names in the order they appear in `Training Dataset.arff`
ALL_FEATURES: list[str] = [
    "having_IP_Address",            # 0  - Tier 1 (Address Bar / Lexical)
    "URL_Length",                   # 1  - Tier 1 (Address Bar / Lexical)
    "Shortining_Service",           # 2  - Tier 1 (Address Bar / Lexical)
    "having_At_Symbol",             # 3  - Tier 1 (Address Bar / Lexical)
    "double_slash_redirecting",     # 4  - Tier 1 (Address Bar / Lexical)
    "Prefix_Suffix",                # 5  - Tier 1 (Address Bar / Lexical)
    "having_Sub_Domain",            # 6  - Tier 1 (Address Bar / Lexical)
    "SSLfinal_State",               # 7  - Tier 2 (SSL / Network)
    "Domain_registeration_length",  # 8  - Tier 2 (WHOIS)
    "Favicon",                      # 9  - Tier 2 (HTML / DOM)
    "port",                         # 10 - Tier 1 (Address Bar / Lexical)
    "HTTPS_token",                  # 11 - Tier 1 (Address Bar / Lexical)
    "Request_URL",                  # 12 - Tier 2 (HTML / DOM)
    "URL_of_Anchor",                # 13 - Tier 2 (HTML / DOM)
    "Links_in_tags",                # 14 - Tier 2 (HTML / DOM)
    "SFH",                          # 15 - Tier 2 (HTML / DOM)
    "Submitting_to_email",          # 16 - Tier 2 (HTML / JS)
    "Abnormal_URL",                 # 17 - Tier 2 (WHOIS)
    "Redirect",                     # 18 - Tier 2 (HTTP Network)
    "on_mouseover",                 # 19 - Tier 2 (HTML / JS)
    "RightClick",                   # 20 - Tier 2 (HTML / JS)
    "popUpWidnow",                  # 21 - Tier 2 (HTML / JS)
    "Iframe",                       # 22 - Tier 2 (HTML / JS)
    "age_of_domain",                # 23 - Tier 2 (WHOIS)
    "DNSRecord",                    # 24 - Tier 2 (DNS Network)
    "web_traffic",                  # 25 - Tier 2 (External Traffic / Tranco)
    "Page_Rank",                    # 26 - Tier 2 (External Reputation)
    "Google_Index",                 # 27 - Tier 2 (External Index)
    "Links_pointing_to_page",       # 28 - Tier 2 (HTML / External)
    "Statistical_report",           # 29 - Tier 2 (PhishTank / Threat Feed)
]

TARGET_ATTRIBUTE: str = "Result"

# Column indices in `ALL_FEATURES` for the 9 zero-network URL lexical features (Tier 1)
TIER1_INDICES: list[int] = [0, 1, 2, 3, 4, 5, 6, 10, 11]
TIER1_FEATURES: list[str] = [ALL_FEATURES[i] for i in TIER1_INDICES]

# Column indices in `ALL_FEATURES` for the 21 heavy network/SSL/WHOIS/DOM features (Tier 2)
TIER2_HEAVY_INDICES: list[int] = [
    i for i in range(len(ALL_FEATURES)) if i not in TIER1_INDICES
]
TIER2_HEAVY_FEATURES: list[str] = [ALL_FEATURES[i] for i in TIER2_HEAVY_INDICES]

# Allowed domain values per feature from @attribute declarations in `Training Dataset.arff`
ALLOWED_FEATURE_VALUES: dict[str, set[int]] = {
    "having_IP_Address": {-1, 1},
    "URL_Length": {-1, 0, 1},
    "Shortining_Service": {-1, 1},
    "having_At_Symbol": {-1, 1},
    "double_slash_redirecting": {-1, 1},
    "Prefix_Suffix": {-1, 1},
    "having_Sub_Domain": {-1, 0, 1},
    "SSLfinal_State": {-1, 0, 1},
    "Domain_registeration_length": {-1, 1},
    "Favicon": {-1, 1},
    "port": {-1, 1},
    "HTTPS_token": {-1, 1},
    "Request_URL": {-1, 1},
    "URL_of_Anchor": {-1, 0, 1},
    "Links_in_tags": {-1, 0, 1},
    "SFH": {-1, 0, 1},
    "Submitting_to_email": {-1, 1},
    "Abnormal_URL": {-1, 1},
    "Redirect": {0, 1},
    "on_mouseover": {-1, 1},
    "RightClick": {-1, 1},
    "popUpWidnow": {-1, 1},
    "Iframe": {-1, 1},
    "age_of_domain": {-1, 1},
    "DNSRecord": {-1, 1},
    "web_traffic": {-1, 0, 1},
    "Page_Rank": {-1, 1},
    "Google_Index": {-1, 1},
    "Links_pointing_to_page": {-1, 0, 1},
    "Statistical_report": {-1, 1},
}

# Label conventions:
# In UCI ARFF: -1 = Phishing, 1 = Legitimate
# In ML binary classification: 1 = Phishing (positive class), 0 = Legitimate (negative class)
ARFF_PHISHING_LABEL: int = -1
ARFF_LEGITIMATE_LABEL: int = 1
BINARY_PHISHING_LABEL: int = 1
BINARY_LEGITIMATE_LABEL: int = 0


def label_arff_to_binary(arff_label: int) -> int:
    """Convert UCI ARFF label (-1=Phishing, 1=Legitimate) to binary (1=Phishing, 0=Legitimate)."""
    if arff_label == ARFF_PHISHING_LABEL:
        return BINARY_PHISHING_LABEL
    if arff_label == ARFF_LEGITIMATE_LABEL:
        return BINARY_LEGITIMATE_LABEL
    raise ValueError(f"Unexpected ARFF label: {arff_label}. Expected -1 or 1.")


def label_binary_to_arff(binary_label: int) -> int:
    """Convert binary label (1=Phishing, 0=Legitimate) back to UCI ARFF label (-1 or 1)."""
    if binary_label == BINARY_PHISHING_LABEL:
        return ARFF_PHISHING_LABEL
    if binary_label == BINARY_LEGITIMATE_LABEL:
        return ARFF_LEGITIMATE_LABEL
    raise ValueError(f"Unexpected binary label: {binary_label}. Expected 0 or 1.")


def validate_feature_vector(
    features: Mapping[str, int] | Sequence[int],
    expected_names: Sequence[str],
    *,
    strict_domain: bool = True,
) -> np.ndarray:
    """Validate and convert a feature dict or sequence into a 1D float32 NumPy array."""
    if isinstance(features, Mapping):
        missing = [name for name in expected_names if name not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        values = [int(features[name]) for name in expected_names]
    else:
        if len(features) != len(expected_names):
            raise ValueError(
                f"Expected {len(expected_names)} features, got {len(features)}."
            )
        values = [int(v) for v in features]

    if strict_domain:
        for name, val in zip(expected_names, values):
            allowed = ALLOWED_FEATURE_VALUES[name]
            if val not in allowed:
                raise ValueError(
                    f"Invalid value {val} for feature '{name}'. Allowed: {sorted(allowed)}"
                )

    return np.asarray(values, dtype=np.float32)


def validate_tier1_features(
    features: Mapping[str, int] | Sequence[int],
    *,
    strict_domain: bool = True,
) -> np.ndarray:
    """Validate and return a 9-element float32 vector for Tier 1 inference."""
    return validate_feature_vector(
        features, TIER1_FEATURES, strict_domain=strict_domain
    )


def validate_all_features(
    features: Mapping[str, int] | Sequence[int],
    *,
    strict_domain: bool = True,
) -> np.ndarray:
    """Validate and return a 30-element float32 vector for Tier 2 inference."""
    return validate_feature_vector(
        features, ALL_FEATURES, strict_domain=strict_domain
    )


def merge_tier1_and_heavy_features(
    tier1_features: Mapping[str, int] | Sequence[int],
    heavy_features: Mapping[str, int] | Sequence[int],
    *,
    strict_domain: bool = True,
) -> np.ndarray:
    """Merge 9 Tier 1 lexical features and 21 Tier 2 heavy features into the canonical 30-feature vector."""
    t1_vec = validate_feature_vector(
        tier1_features, TIER1_FEATURES, strict_domain=strict_domain
    )
    t2_vec = validate_feature_vector(
        heavy_features, TIER2_HEAVY_FEATURES, strict_domain=strict_domain
    )

    merged = np.empty(len(ALL_FEATURES), dtype=np.float32)
    merged[TIER1_INDICES] = t1_vec
    merged[TIER2_HEAVY_INDICES] = t2_vec
    return merged
