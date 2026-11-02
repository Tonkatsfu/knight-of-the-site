"""Unit tests for the UCI Phishing Websites ARFF loader and stratified splitter (`shared/data_loader.py`)."""

import numpy as np

from shared.data_loader import (
    DEFAULT_ARFF_PATH,
    load_stratified_splits,
    parse_arff_file,
)
from shared.feature_schema import ALL_FEATURES


def test_parse_arff_file_dimensions_and_labels():
    X_all, y, attrs = parse_arff_file(DEFAULT_ARFF_PATH)
    assert attrs == ALL_FEATURES
    assert X_all.shape == (11055, 30)
    assert X_all.dtype == np.float32
    assert y.shape == (11055,)
    # 6157 Legitimate (0) and 4898 Phishing (1)
    assert int(np.sum(y == 0)) == 6157
    assert int(np.sum(y == 1)) == 4898


def test_load_stratified_splits_counts_and_stratification():
    splits = load_stratified_splits(DEFAULT_ARFF_PATH, random_state=42)

    total_rows = (
        splits.train.num_samples
        + splits.stream.num_samples
        + splits.test.num_samples
    )
    assert total_rows == 11055
    assert splits.train.num_samples == 7738
    assert splits.stream.num_samples == 1658
    assert splits.test.num_samples == 1659

    # Check Tier 1 and Tier 2 feature slice shapes
    assert splits.train.X_tier1.shape == (7738, 9)
    assert splits.train.X_tier2_heavy.shape == (7738, 21)
    assert splits.stream.X_tier1.shape == (1658, 9)
    assert splits.test.X_tier1.shape == (1659, 9)

    # Check class balance preservation across splits (~44.3% phishing in each split)
    overall_phish_rate = 4898 / 11055
    for split in (splits.train, splits.stream, splits.test):
        split_rate = split.phishing_count / split.num_samples
        assert abs(split_rate - overall_phish_rate) < 0.005
