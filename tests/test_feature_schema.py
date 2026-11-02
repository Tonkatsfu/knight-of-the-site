"""Unit tests for the unified 30-feature schema contract (`shared/feature_schema.py`)."""

import numpy as np
import pytest

from shared.feature_schema import (
    ALL_FEATURES,
    TIER1_FEATURES,
    TIER1_INDICES,
    TIER2_HEAVY_FEATURES,
    TIER2_HEAVY_INDICES,
    label_arff_to_binary,
    label_binary_to_arff,
    merge_tier1_and_heavy_features,
    validate_all_features,
    validate_tier1_features,
)


def test_feature_counts_and_disjoint_partition():
    assert len(ALL_FEATURES) == 30
    assert len(TIER1_FEATURES) == 9
    assert len(TIER2_HEAVY_FEATURES) == 21
    assert set(TIER1_INDICES).isdisjoint(set(TIER2_HEAVY_INDICES))
    assert sorted(TIER1_INDICES + TIER2_HEAVY_INDICES) == list(range(30))


def test_tier1_feature_names_and_indices():
    expected_tier1 = [
        "having_IP_Address",
        "URL_Length",
        "Shortining_Service",
        "having_At_Symbol",
        "double_slash_redirecting",
        "Prefix_Suffix",
        "having_Sub_Domain",
        "port",
        "HTTPS_token",
    ]
    assert TIER1_FEATURES == expected_tier1
    assert TIER1_INDICES == [0, 1, 2, 3, 4, 5, 6, 10, 11]


def test_label_conversions():
    assert label_arff_to_binary(-1) == 1  # Phishing
    assert label_arff_to_binary(1) == 0   # Legitimate
    assert label_binary_to_arff(1) == -1
    assert label_binary_to_arff(0) == 1

    with pytest.raises(ValueError):
        label_arff_to_binary(0)
    with pytest.raises(ValueError):
        label_binary_to_arff(2)


def test_validate_tier1_and_merge():
    tier1_dict = {name: 1 for name in TIER1_FEATURES}
    tier1_dict["URL_Length"] = 0
    vec = validate_tier1_features(tier1_dict)
    assert vec.shape == (9,)
    assert vec.dtype == np.float32

    heavy_dict = {name: -1 for name in TIER2_HEAVY_FEATURES}
    heavy_dict["Redirect"] = 0  # Redirect only allows {0, 1}
    merged = merge_tier1_and_heavy_features(tier1_dict, heavy_dict)
    assert merged.shape == (30,)
    assert np.all(merged[TIER1_INDICES] == vec)
    assert merged[ALL_FEATURES.index("Redirect")] == 0.0

    # Verify validate_all_features accepts the merged vector
    validated_all = validate_all_features(merged.astype(int).tolist())
    assert np.array_equal(merged, validated_all)


def test_invalid_domain_raises_value_error():
    bad_tier1 = {name: 1 for name in TIER1_FEATURES}
    bad_tier1["having_IP_Address"] = 0  # Only {-1, 1} allowed for having_IP_Address
    with pytest.raises(ValueError, match="having_IP_Address"):
        validate_tier1_features(bad_tier1)
