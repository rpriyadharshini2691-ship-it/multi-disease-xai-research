"""Authenticated model inference; submitted feature values are never persisted."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from xai_dashboard.api.deps import get_current_user, get_model_registry, get_session
from xai_dashboard.api.schemas import PredictionRequest, PredictionResponse
from xai_dashboard.core.data.diseases import load_disease_configs
from xai_dashboard.core.ml.registry import ModelRegistry
from xai_dashboard.db.models import PredictionLog, User
from xai_dashboard.services.prediction import PredictionInputError, create_prediction_details
from xai_dashboard.settings import Settings, get_settings

router = APIRouter(prefix="/predict", tags=["prediction"])


def latest_model_file(directory: Path) -> Path | None:
    files = sorted(directory.glob("models_*.joblib"))
    return files[-1] if files else None


def execute_prediction(
    payload: PredictionRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
) -> tuple[dict[str, object], dict[str, object]]:
    config = load_disease_configs(settings.configs_dir).get(payload.disease_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "unknown_disease", "message": "Disease configuration was not found."},
        )
    directory = settings.artifacts_dir / config.id
    model_path = latest_model_file(directory)
    artifact = registry.load_latest(config.id)
    if artifact is None or model_path is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_not_generated",
                "message": "Model artifacts are not yet generated for this disease.",
            },
        )
    model_version = model_path.stem.removeprefix("models_")
    try:
        details = create_prediction_details(
            config,
            payload.features,
            payload.model_name,
            artifact,
            model_version,
            Path(config.dataset_path),
            settings.artifacts_dir,
        )
    except PredictionInputError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except ArithmeticError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "explanation_failed", "message": str(exc)},
        ) from exc
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "prediction_unavailable",
                "message": "The selected model, dataset, or explanation background is unavailable.",
            },
        ) from exc
    if payload.consent:
        score = float(details["positive_class_score"])
        db.add(
            PredictionLog(
                user_id=user.id,
                disease_id=config.id,
                model_name=payload.model_name,
                positive_score=score,
            )
        )
        db.commit()
    details["stored"] = payload.consent
    return {"id": config.id, "name": config.name}, details


@router.post("", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
) -> PredictionResponse:
    _, details = execute_prediction(payload, user, db, settings, registry)
    return PredictionResponse.model_validate(details)
