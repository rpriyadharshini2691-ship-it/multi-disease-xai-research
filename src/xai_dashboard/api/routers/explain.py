"""Local probability-space explanations for generated models."""

import base64
import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from xai_dashboard.api.deps import get_current_user, get_model_registry
from xai_dashboard.api.schemas import SCORE_DISCLAIMER
from xai_dashboard.core.data.diseases import load_disease_configs
from xai_dashboard.core.data.loaders import load_dataset
from xai_dashboard.core.ml.registry import ModelRegistry
from xai_dashboard.core.xai.shap_service import explain_instance
from xai_dashboard.db.models import User
from xai_dashboard.settings import Settings, get_settings

router = APIRouter(prefix="/explain", tags=["explainability"])


class ExplainRequest(BaseModel):
    features: dict[str, Any]


@router.post("/{disease_id}/local")
def explain_local(
    disease_id: str,
    payload: ExplainRequest,
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
    _user: User = Depends(get_current_user),
) -> dict[str, Any]:
    config = load_disease_configs(settings.configs_dir).get(disease_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "unknown_disease", "message": "Disease configuration was not found."},
        )
    if set(payload.features) != set(config.feature_columns):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_features",
                "message": "Feature names must match the disease configuration.",
            },
        )
    artifact = registry.load_latest(disease_id)
    dataset = Path(config.dataset_path)
    evaluation_files = sorted((settings.artifacts_dir / disease_id).glob("evaluation_*.json"))
    if artifact is None or not dataset.exists() or not evaluation_files:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "explanation_not_generated",
                "message": "A model, evaluation artifact, and background dataset are required.",
            },
        )
    model = artifact["models"]["weighted_ensemble"]
    frame = pd.DataFrame(
        [{name: payload.features[name] for name in config.feature_columns}],
        columns=config.feature_columns,
    )
    evaluation = json.loads(evaluation_files[-1].read_text(encoding="utf-8"))
    features, target_labels = load_dataset(config, dataset)
    target = (target_labels == config.positive_label).astype(int)
    background, _, _, _ = train_test_split(
        features,
        target,
        test_size=float(evaluation["test_size"]),
        stratify=target,
        random_state=int(evaluation["seed"]),
    )
    background = background.sample(n=min(30, len(background)), random_state=42)
    try:
        explanation = explain_instance(model, background, frame)
    except (ArithmeticError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail={"code": "explanation_failed", "message": str(exc)}
        ) from exc
    explanation["disease_id"] = disease_id
    explanation["disclaimer"] = SCORE_DISCLAIMER
    return explanation


@router.get("/{disease_id}/global")
def explain_global(disease_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    config = load_disease_configs(settings.configs_dir).get(disease_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "unknown_disease", "message": "Disease configuration was not found."},
        )
    directory = settings.artifacts_dir / disease_id
    table_path = directory / "global_shap_importance.csv"
    plot_path = directory / "global_shap_beeswarm.png"
    if not table_path.is_file() or not plot_path.is_file():
        return {
            "status": "not_generated",
            "message": "Global SHAP artifacts are not yet generated.",
        }
    return {
        "status": "ready",
        "features": pd.read_csv(table_path).to_dict(orient="records"),
        "beeswarm_png_base64": base64.b64encode(plot_path.read_bytes()).decode("ascii"),
        "disclaimer": SCORE_DISCLAIMER,
    }
