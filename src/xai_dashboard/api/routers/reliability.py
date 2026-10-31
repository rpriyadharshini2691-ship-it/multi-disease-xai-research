"""Serve stored reliability experiment artifacts without fabricating results."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from xai_dashboard.core.data.diseases import load_disease_configs
from xai_dashboard.settings import Settings, get_settings

router = APIRouter(prefix="/reliability", tags=["reliability"])


def latest_reliability(directory: Path) -> Path | None:
    files = sorted(directory.glob("reliability_*.json"))
    return files[-1] if files else None


@router.get("/{disease_id}")
def reliability(disease_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    if disease_id not in load_disease_configs(settings.configs_dir):
        return {
            "status": "not_found",
            "disease_id": disease_id,
            "message": "Disease configuration was not found.",
        }
    path = latest_reliability(settings.artifacts_dir / disease_id)
    if path is None:
        return {
            "status": "not_generated",
            "disease_id": disease_id,
            "message": "Reliability experiments have not yet been generated.",
            "interpretation": (
                "Reliability results measure consistency under tested conditions only; "
                "they do not imply medical correctness."
            ),
            "perturbation_note": (
                "Deltas 0.01, 0.05, and 0.10 are experimental settings on standardized "
                "numeric features, not clinical tolerances."
            ),
        }
    return {"status": "ready", "artifact": json.loads(path.read_text(encoding="utf-8"))}
