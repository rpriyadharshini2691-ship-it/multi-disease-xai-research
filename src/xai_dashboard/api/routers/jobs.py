"""Background model-training jobs and status polling."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from xai_dashboard.api.deps import get_current_user, require_roles
from xai_dashboard.core.data.diseases import load_disease_configs
from xai_dashboard.core.ml.jobs import get_job, submit_job
from xai_dashboard.db.models import User
from xai_dashboard.services.reliability import run_and_record_reliability
from xai_dashboard.services.training import train_and_register_models
from xai_dashboard.settings import Settings, get_settings

router = APIRouter(prefix="/jobs", tags=["jobs"])


class TrainRequest(BaseModel):
    dataset_path: Path | None = None


class ReliabilityRequest(BaseModel):
    dataset_path: Path | None = None
    repetitions: int = Field(default=30, ge=30)
    sample_size: int = Field(default=8, ge=2)


def _resolve_dataset_path(
    configured_path: str, requested_path: Path | None, configs_dir: Path
) -> Path:
    project_root = configs_dir.resolve().parent
    configured = Path(configured_path)
    if not configured.is_absolute():
        configured = project_root / configured
    candidate = requested_path or configured
    if not candidate.is_absolute():
        candidate = project_root / candidate
    resolved = candidate.resolve()
    allowed_root = (project_root / "data").resolve()
    if resolved != configured.resolve() and not resolved.is_relative_to(allowed_root):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "dataset_path_not_allowed",
                "message": (
                    "Dataset paths must use the configured source or a file under the "
                    "workspace data directory."
                ),
            },
        )
    if not resolved.is_file():
        raise HTTPException(
            status_code=422,
            detail={
                "code": "dataset_missing",
                "message": "Provide an existing local dataset under the workspace data directory.",
            },
        )
    return resolved


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
    dataset = _resolve_dataset_path(config.dataset_path, payload.dataset_path, settings.configs_dir)
    job_id = submit_job(
        "training",
        lambda: train_and_register_models(
            config,
            dataset,
            settings.artifacts_dir,
            created_by_id=_user.id,
        ),
    )
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
    payload: ReliabilityRequest,
    user: User = Depends(require_roles("admin", "researcher")),
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    config = load_disease_configs(settings.configs_dir).get(disease_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "unknown_disease", "message": "Disease configuration was not found."},
        )
    dataset = _resolve_dataset_path(config.dataset_path, payload.dataset_path, settings.configs_dir)
    job_id = submit_job(
        "reliability",
        lambda: run_and_record_reliability(
            config,
            dataset,
            settings.artifacts_dir,
            user.id,
            repetitions=payload.repetitions,
            sample_size=payload.sample_size,
        ),
    )
    return {"job_id": job_id, "status": "queued"}
