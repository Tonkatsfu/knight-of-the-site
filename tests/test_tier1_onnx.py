"""Tests for Tier 1 ONNX export (`shared/model_training.py`) and the committed `models/tier1_v1.onnx`."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from shared.data_loader import load_stratified_splits
from shared.feature_schema import TIER1_FEATURES
from shared.lexical_extractor import extract_tier1_vector
from shared.model_training import (
    MAX_TIER1_ONNX_BYTES,
    export_tier1_onnx,
    read_onnx_metadata,
    train_tier1_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMITTED_TIER1_MODEL = PROJECT_ROOT / "models" / "tier1_v1.onnx"


@pytest.fixture(scope="module")
def splits():
    return load_stratified_splits()


@pytest.fixture(scope="module")
def trained_model(splits):
    return train_tier1_model(splits.train)


def _onnx_proba(path: Path, X: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    _, proba = session.run(None, {input_name: X.astype(np.float32)})
    return proba[:, 1]


def test_onnx_probabilities_match_sklearn(tmp_path, splits, trained_model):
    path = export_tier1_onnx(trained_model, tmp_path / "tier1_test.onnx", version="test")
    expected = trained_model.predict_proba(splits.test.X_tier1)[:, 1]
    actual = _onnx_proba(path, splits.test.X_tier1)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_onnx_model_is_under_edge_size_budget(tmp_path, trained_model):
    path = export_tier1_onnx(trained_model, tmp_path / "tier1_test.onnx", version="test")
    assert path.stat().st_size < MAX_TIER1_ONNX_BYTES


def test_onnx_metadata_records_version_and_feature_order(tmp_path, trained_model):
    path = export_tier1_onnx(trained_model, tmp_path / "tier1_v7.onnx", version="v7")
    meta = read_onnx_metadata(path)
    assert meta["model_version"] == "v7"
    assert meta["feature_names"] == ",".join(TIER1_FEATURES)


def test_tier1_model_beats_majority_baseline(splits, trained_model):
    accuracy = float(np.mean(trained_model.predict(splits.test.X_tier1) == splits.test.y))
    majority_rate = float(np.mean(splits.test.y == 0))
    assert accuracy > majority_rate + 0.10


def test_committed_v1_model_is_valid():
    if not COMMITTED_TIER1_MODEL.exists():
        pytest.fail("models/tier1_v1.onnx missing — run `python scripts/train_initial_models.py`.")
    assert COMMITTED_TIER1_MODEL.stat().st_size < MAX_TIER1_ONNX_BYTES
    assert read_onnx_metadata(COMMITTED_TIER1_MODEL)["model_version"] == "v1"
    proba = _onnx_proba(COMMITTED_TIER1_MODEL, np.ones((1, len(TIER1_FEATURES)), dtype=np.float32))
    assert 0.0 <= float(proba[0]) <= 1.0


def test_committed_v1_model_routes_real_urls_sensibly():
    benign_p = float(
        _onnx_proba(COMMITTED_TIER1_MODEL, extract_tier1_vector("https://www.google.com/")[None, :])[0]
    )
    ambig_p = float(
        _onnx_proba(
            COMMITTED_TIER1_MODEL, extract_tier1_vector("http://paypal-account.com/login")[None, :]
        )[0]
    )
    phish_p = float(
        _onnx_proba(
            COMMITTED_TIER1_MODEL,
            extract_tier1_vector("http://192.168.10.5:8080/paypal/login@verify//secure")[None, :],
        )[0]
    )
    assert benign_p <= 0.35
    assert 0.35 < ambig_p < 0.65
    assert phish_p >= 0.65
