"""Background model-training jobs and status polling."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from xai_dashboard.api.deps import get_current_user, require_roles
from xai_dashboard.core.data.diseases import load_disease_configs
from xai_dashboard.core.ml.jobs import get_job, submit_job
from xai_dashboard.core.ml.training import train_disease
from xai_dashboard.core.reliability.runner import run_reliability
from xai_dashboard.db.models import User
from xai_dashboard.settings import Settings, get_settings

router = APIRouter(prefix="/jobs", tags=["jobs"])


class TrainRequest(BaseModel):
    dataset_path: Path | None = None


@router.post("/train/{disease_id}", status_code=202)
def start_training(
    disease_id: str,
    payload: TrainRequest,
    _user: User = Depends(require_roles("admin", "researcher")),
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    config = load_disease_configs(settings.configs_dir).get(disease_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "unknown_disease", "message": "Disease configuration was not found."},
        )
    dataset = payload.dataset_path or Path(config.dataset_path)
    if not dataset.exists():
        raise HTTPException(
            status_code=422,
            detail={
                "code": "dataset_missing",
                "message": "Provide a local dataset file before training.",
            },
        )
    job_id = submit_job("training", lambda: train_disease(config, dataset, settings.artifacts_dir))
    return {"job_id": job_id, "status": "queued"}


@router.get("/{job_id}")
def job_status(job_id: str, _user: User = Depends(get_current_user)) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "job_not_found", "message": "Job was not found in this API process."},
        )
    return job


@router.post("/reliability/{disease_id}", status_code=202)
def start_reliability(
    disease_id: str,
    payload: TrainRequest,
    _user: User = Depends(require_roles("admin", "researcher")),
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    config = load_disease_configs(settings.configs_dir).get(disease_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "unknown_disease", "message": "Disease configuration was not found."},
        )
    dataset = payload.dataset_path or Path(config.dataset_path)
    if not dataset.exists():
        raise HTTPException(
            status_code=422,
            detail={
                "code": "dataset_missing",
                "message": "Provide a local dataset file before running reliability experiments.",
            },
        )
    job_id = submit_job(
        "reliability", lambda: run_reliability(config, dataset, settings.artifacts_dir)
    )
    return {"job_id": job_id, "status": "queued"}
