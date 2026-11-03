"""Train the initial (v1) Tier 1 and Tier 2 models and write them to `models/`.

Usage:
    python scripts/train_initial_models.py

Outputs:
    models/tier1_v1.onnx    9-feature edge classifier (ONNX, < 100 KB)
    models/tier2_v1.joblib  30-feature cloud classifier (scikit-learn)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from shared.data_loader import load_stratified_splits  # noqa: E402
from shared.model_training import (  # noqa: E402
    evaluate_binary,
    export_tier1_onnx,
    train_tier1_model,
    train_tier2_model,
)

MODELS_DIR = PROJECT_ROOT / "models"
LOW_THRESHOLD, HIGH_THRESHOLD = 0.35, 0.65


def _format_metrics(metrics: dict[str, float]) -> str:
    return "  ".join(f"{name}={value:.3f}" for name, value in metrics.items())


def main() -> None:
    splits = load_stratified_splits()
    test = splits.test

    tier1 = train_tier1_model(splits.train)
    tier1_path = export_tier1_onnx(tier1, MODELS_DIR / "tier1_v1.onnx", version="v1")
    p_phish = tier1.predict_proba(test.X_tier1)[:, 1]
    confident = (p_phish <= LOW_THRESHOLD) | (p_phish >= HIGH_THRESHOLD)
    tier1_pred = (p_phish >= 0.5).astype(int)
    print(f"[Tier 1] {tier1_path.relative_to(PROJECT_ROOT)} ({tier1_path.stat().st_size / 1024:.1f} KB)")
    print(f"         holdout: {_format_metrics(evaluate_binary(test.y, tier1_pred))}")
    print(
        f"         gate [{LOW_THRESHOLD}, {HIGH_THRESHOLD}]: resolved locally={confident.mean():.1%}, "
        f"local accuracy={np.mean(tier1_pred[confident] == test.y[confident]):.3f}"
    )

    tier2 = train_tier2_model(splits.train)
    tier2_path = MODELS_DIR / "tier2_v1.joblib"
    joblib.dump(tier2, tier2_path, compress=3)
    tier2_pred = tier2.predict(test.X_all)
    print(f"[Tier 2] {tier2_path.relative_to(PROJECT_ROOT)} ({tier2_path.stat().st_size / 1024:.1f} KB)")
    print(f"         holdout: {_format_metrics(evaluate_binary(test.y, tier2_pred))}")


if __name__ == "__main__":
    main()
