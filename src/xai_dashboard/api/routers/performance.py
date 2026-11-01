"""Read evaluation metrics produced by the training pipeline."""

import base64
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from xai_dashboard.api.routers.predictions import latest_model_file
from xai_dashboard.core.data.diseases import load_disease_configs
from xai_dashboard.settings import Settings, get_settings

router = APIRouter(prefix="/performance", tags=["performance"])


def latest_evaluation(directory: Path) -> Path | None:
    files = sorted(directory.glob("evaluation_*.json"))
    return files[-1] if files else None


@router.get("/{disease_id}")
def performance(disease_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    configs = load_disease_configs(settings.configs_dir)
    if disease_id not in configs:
        return {
            "status": "not_found",
            "disease_id": disease_id,
            "message": "Disease configuration was not found.",
        }
    directory = settings.artifacts_dir / disease_id
    path = latest_evaluation(directory)
    if path is None or latest_model_file(directory) is None:
        return {
            "status": "not_generated",
            "disease_id": disease_id,
            "message": "Evaluation artifacts are not yet generated.",
        }
    plot_files = sorted(
        [*directory.glob("model_comparison_*.png"), *directory.glob("evaluation_*.png")]
    )
    plots = [
        {
            "filename": image.name,
            "png_base64": base64.b64encode(image.read_bytes()).decode("ascii"),
        }
        for image in plot_files
    ]
    return {
        "status": "ready",
        "artifact": json.loads(path.read_text(encoding="utf-8")),
        "plots": plots,
    }
