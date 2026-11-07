"""Tier 2 runtime settings, read from environment variables.

    TIER2_DATABASE_URL         SQLAlchemy URL. Default: SQLite file `tier2_cloud/tier2.db`.
                               Point at PostgreSQL in deployment, e.g.
                               postgresql+psycopg2://user:pass@db:5432/knight
    TIER2_MODELS_DIR           Directory holding tier1_vN.onnx / tier2_vN.joblib. Default: `models/`.
    TIER2_EXTRACTION_BUDGET_S  Hard wall-clock budget for heavy feature extraction. Default: 2.5 s,
                               chosen to fit inside the Tier 1 client's 3 s HTTP timeout.
    TIER2_TRANCO_PATH          Optional Tranco top-1M CSV (rank,domain) for the reputation features.
    TIER2_BLOCKLIST_PATH       URL corpus CSV (url,label) whose phishing hosts act as a threat feed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATABASE_URL = f"sqlite:///{(PROJECT_ROOT / 'tier2_cloud' / 'tier2.db').as_posix()}"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_EXTRACTION_BUDGET_S = 2.5
DEFAULT_TRANCO_PATH = PROJECT_ROOT / "phishing+websites" / "tranco_top1m.csv"
DEFAULT_BLOCKLIST_PATH = PROJECT_ROOT / "phishing+websites" / "tier1_urls.csv"


@dataclass(frozen=True)
class Settings:
    database_url: str = DEFAULT_DATABASE_URL
    models_dir: Path = DEFAULT_MODELS_DIR
    extraction_budget_s: float = DEFAULT_EXTRACTION_BUDGET_S
    tranco_path: Path | None = DEFAULT_TRANCO_PATH
    blocklist_path: Path | None = DEFAULT_BLOCKLIST_PATH

    def __post_init__(self) -> None:
        if self.extraction_budget_s <= 0:
            raise ValueError(f"extraction_budget_s must be positive, got {self.extraction_budget_s}.")

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            database_url=env.get("TIER2_DATABASE_URL", DEFAULT_DATABASE_URL),
            models_dir=Path(env.get("TIER2_MODELS_DIR", DEFAULT_MODELS_DIR)),
            extraction_budget_s=float(env.get("TIER2_EXTRACTION_BUDGET_S", DEFAULT_EXTRACTION_BUDGET_S)),
            tranco_path=Path(env.get("TIER2_TRANCO_PATH", DEFAULT_TRANCO_PATH)),
            blocklist_path=Path(env.get("TIER2_BLOCKLIST_PATH", DEFAULT_BLOCKLIST_PATH)),
        )
