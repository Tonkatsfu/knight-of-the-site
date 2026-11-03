"""ARFF parser and deterministic stratified dataset splitter for the UCI Phishing Websites dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from shared.feature_schema import (
    ALL_FEATURES,
    TARGET_ATTRIBUTE,
    TIER1_INDICES,
    TIER2_HEAVY_INDICES,
    label_arff_to_binary,
)
from shared.lexical_extractor import extract_tier1_vector

DEFAULT_ARFF_PATH: Path = (
    Path(__file__).resolve().parent.parent
    / "phishing+websites"
    / "Training Dataset.arff"
)
DEFAULT_TIER1_URLS_PATH: Path = (
    Path(__file__).resolve().parent.parent
    / "phishing+websites"
    / "tier1_urls.csv"
)


@dataclass(frozen=True)
class DatasetSplit:
    """Holds a single dataset split (features X in float32 and binary labels y in int64)."""

    X_all: np.ndarray  # shape (N, 30)
    y: np.ndarray      # shape (N,), 1 = Phishing, 0 = Legitimate

    @property
    def X_tier1(self) -> np.ndarray:
        """Return the 9-column Tier 1 lexical feature matrix of shape (N, 9)."""
        return self.X_all[:, TIER1_INDICES]

    @property
    def X_tier2_heavy(self) -> np.ndarray:
        """Return the 21-column Tier 2 heavy feature matrix of shape (N, 21)."""
        return self.X_all[:, TIER2_HEAVY_INDICES]

    @property
    def num_samples(self) -> int:
        return int(self.X_all.shape[0])

    @property
    def phishing_count(self) -> int:
        return int(np.sum(self.y == 1))

    @property
    def legitimate_count(self) -> int:
        return int(np.sum(self.y == 0))


@dataclass(frozen=True)
class PhishingDatasetSplits:
    """Deterministic 3-way stratified split: Initial Train (70%), Streaming Pool (15%), Holdout Test (15%)."""

    train: DatasetSplit
    stream: DatasetSplit
    test: DatasetSplit


def parse_arff_file(arff_path: str | Path = DEFAULT_ARFF_PATH) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Parse an ARFF file into (X_all, y_binary, feature_names).

    Validates that the 30 feature attributes match `ALL_FEATURES` in exact order.
    """
    path = Path(arff_path)
    if not path.exists():
        raise FileNotFoundError(f"ARFF dataset not found at: {path}")

    attributes: list[str] = []
    raw_rows: list[list[int]] = []
    in_data = False

    with path.open("r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            lower = line.lower()
            if lower.startswith("@relation"):
                continue
            if lower.startswith("@attribute"):
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Malformed @attribute at line {line_num}: {line}")
                attributes.append(parts[1])
                continue
            if lower.startswith("@data"):
                in_data = True
                continue

            if in_data:
                tokens = [tok.strip() for tok in line.split(",") if tok.strip()]
                if len(tokens) != 31:
                    raise ValueError(
                        f"Expected 31 values at line {line_num}, got {len(tokens)}."
                    )
                raw_rows.append([int(tok) for tok in tokens])

    if not attributes:
        raise ValueError(f"No @attribute headers found in {path}")

    feature_attrs = attributes[:-1]
    target_attr = attributes[-1]

    if feature_attrs != ALL_FEATURES:
        raise ValueError(
            f"ARFF feature attributes do not match canonical ALL_FEATURES.\n"
            f"Expected: {ALL_FEATURES}\nGot: {feature_attrs}"
        )
    if target_attr != TARGET_ATTRIBUTE:
        raise ValueError(
            f"Expected target attribute '{TARGET_ATTRIBUTE}', got '{target_attr}'."
        )

    data_matrix = np.asarray(raw_rows, dtype=np.int32)
    X_all = data_matrix[:, :30].astype(np.float32)
    y_arff = data_matrix[:, 30]
    y_binary = np.asarray([label_arff_to_binary(int(lbl)) for lbl in y_arff], dtype=np.int64)

    return X_all, y_binary, feature_attrs


@lru_cache(maxsize=2)
def _load_real_tier1_vectors(csv_path_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and extract 9-feature Tier 1 vectors for (legitimate, phishing) URLs from CSV."""
    leg_vecs: list[np.ndarray] = []
    phi_vecs: list[np.ndarray] = []
    with Path(csv_path_str).open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vec = extract_tier1_vector(row["url"])
            if int(row["label"]) == 0:
                leg_vecs.append(vec)
            else:
                phi_vecs.append(vec)
    return np.asarray(leg_vecs, dtype=np.float32), np.asarray(phi_vecs, dtype=np.float32)


def load_stratified_splits(
    arff_path: str | Path = DEFAULT_ARFF_PATH,
    *,
    tier1_urls_path: str | Path | None = DEFAULT_TIER1_URLS_PATH,
    train_ratio: float = 0.70,
    stream_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> PhishingDatasetSplits:
    """Load the dataset and partition into stratified train, stream, and test splits.

    When `tier1_urls_path` exists, the 9 Tier 1 lexical columns are populated by running
    `extract_tier1_vector` on real legitimate and PhishTank phishing URLs so that Tier 1
    training and live URL inference share the exact same feature distribution.
    """
    if not np.isclose(train_ratio + stream_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + stream_ratio + test_ratio must equal 1.0")

    X_all, y, _ = parse_arff_file(arff_path)

    if tier1_urls_path is not None and Path(tier1_urls_path).exists():
        leg_vecs, phi_vecs = _load_real_tier1_vectors(str(Path(tier1_urls_path).resolve()))
        leg_idx = np.where(y == 0)[0]
        phi_idx = np.where(y == 1)[0]
        if len(leg_vecs) == len(leg_idx) and len(phi_vecs) == len(phi_idx):
            X_all = X_all.copy()
            X_all[leg_idx[:, None], TIER1_INDICES] = leg_vecs
            X_all[phi_idx[:, None], TIER1_INDICES] = phi_vecs

    # First split: Train (70%) vs Temp (30% = Stream + Test)
    temp_ratio = stream_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all,
        y,
        test_size=temp_ratio,
        stratify=y,
        random_state=random_state,
    )

    # Second split: partition Temp into Stream (15%) and Holdout Test (15%)
    relative_test_ratio = test_ratio / temp_ratio
    X_stream, X_test, y_stream, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_ratio,
        stratify=y_temp,
        random_state=random_state,
    )

    return PhishingDatasetSplits(
        train=DatasetSplit(X_all=X_train, y=y_train),
        stream=DatasetSplit(X_all=X_stream, y=y_stream),
        test=DatasetSplit(X_all=X_test, y=y_test),
    )
