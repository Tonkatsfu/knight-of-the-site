"""Model training and ONNX export utilities shared by the initial training script and the Tier 2 retrainer.

Tier 1 uses a shallow decision tree on the 9 lexical features: on the UCI dataset it matches the
accuracy of random forests and gradient boosting while exporting to only a few KB of ONNX, which
keeps OTA updates to edge devices cheap.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from shared.data_loader import DatasetSplit
from shared.feature_schema import TIER1_FEATURES

MAX_TIER1_ONNX_BYTES: int = 100 * 1024  # Edge size budget from the project plan (< 100 KB)
ONNX_TARGET_OPSET: int = 17
RANDOM_STATE: int = 42


def train_tier1_model(train: DatasetSplit) -> DecisionTreeClassifier:
    """Train the 9-feature Tier 1 edge classifier."""
    model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=RANDOM_STATE)
    model.fit(train.X_tier1, train.y)
    return model


def train_tier2_model(train: DatasetSplit) -> RandomForestClassifier:
    """Train the 30-feature Tier 2 cloud classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(train.X_all, train.y)
    return model


def export_tier1_onnx(model: DecisionTreeClassifier, path: str | Path, *, version: str) -> Path:
    """Export a fitted Tier 1 model to ONNX with version + feature-order metadata.

    The graph takes a float32 input of shape (N, 9) and returns (label, probabilities) where
    probabilities has shape (N, 2) — column 1 is P(phishing). ZipMap is disabled so the output is
    a plain tensor that onnxruntime can return without Python dict conversion.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = np.zeros((1, len(TIER1_FEATURES)), dtype=np.float32)
    onnx_model = to_onnx(
        model, sample, options={id(model): {"zipmap": False}}, target_opset=ONNX_TARGET_OPSET
    )
    metadata = {
        "model_version": version,
        "feature_names": ",".join(TIER1_FEATURES),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save_model(onnx_model, str(path))
    return path


def read_onnx_metadata(path: str | Path) -> dict[str, str]:
    """Read the custom metadata map embedded by `export_tier1_onnx`."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return dict(session.get_modelmeta().custom_metadata_map)


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Precision, recall, F1, false-positive rate, and accuracy with phishing (1) as the positive class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else 0.0,
    }
