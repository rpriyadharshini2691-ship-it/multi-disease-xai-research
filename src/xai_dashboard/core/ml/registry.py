"""Cached registry for versioned, locally persisted model artifacts."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from xai_dashboard.settings import Settings


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_latest(self, disease_id: str) -> dict[str, Any] | None:
        files = sorted((self.settings.artifacts_dir / disease_id).glob("models_*.joblib"))
        if not files:
            return None
        path = files[-1]
        return self._load(path.as_posix(), path.stat().st_mtime_ns)

    @staticmethod
    @lru_cache(maxsize=32)
    def _load(path: str, modified_ns: int) -> dict[str, Any]:
        del modified_ns
        artifact = joblib.load(Path(path))
        if not isinstance(artifact, dict) or not isinstance(artifact.get("models"), dict):
            raise ValueError(f"Invalid model artifact: {path}")
        return artifact
