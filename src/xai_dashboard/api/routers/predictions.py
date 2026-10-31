"""Authenticated model inference; submitted feature values are never persisted."""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from xai_dashboard.api.deps import get_current_user, get_model_registry, get_session
from xai_dashboard.api.schemas import PredictionRequest, PredictionResponse
from xai_dashboard.core.data.diseases import load_disease_configs
from xai_dashboard.core.ml.registry import ModelRegistry
from xai_dashboard.db.models import PredictionLog, User
from xai_dashboard.settings import Settings, get_settings

router = APIRouter(prefix="/predict", tags=["prediction"])


def latest_model_file(directory: Path) -> Path | None:
    files = sorted(directory.glob("models_*.joblib"))
    return files[-1] if files else None


@router.post("", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
) -> PredictionResponse:
    config = load_disease_configs(settings.configs_dir).get(payload.disease_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "unknown_disease", "message": "Disease configuration was not found."},
        )
    expected = set(config.feature_columns)
    if set(payload.features) != expected:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_features",
                "message": "Feature names must match the disease configuration.",
            },
        )
    for name, bounds in config.valid_ranges.items():
        if name in payload.features:
            try:
                value = float(payload.features[name])
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail={"code": "invalid_feature", "message": f"{name} must be numeric."},
                ) from None
            if not bounds[0] <= value <= bounds[1]:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "out_of_range",
                        "message": f"{name} is outside its configured input range.",
                    },
                )
    artifact = registry.load_latest(config.id)
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_not_generated",
                "message": "Model artifacts are not yet generated for this disease.",
            },
        )
    model = artifact["models"][payload.model_name]
    frame = pd.DataFrame(
        [{name: payload.features[name] for name in config.feature_columns}],
        columns=config.feature_columns,
    )
    score = float(model.predict_proba(frame)[0, 1])
    if payload.consent:
        db.add(
            PredictionLog(
                user_id=user.id,
                disease_id=config.id,
                model_name=payload.model_name,
                positive_score=score,
            )
        )
        db.commit()
    return PredictionResponse(
        disease_id=config.id,
        model_name=payload.model_name,
        positive_class_score=score,
        stored=payload.consent,
    )
