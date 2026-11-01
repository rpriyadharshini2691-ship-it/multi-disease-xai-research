"""PDF reports built only from stored evaluation artifacts."""

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from xai_dashboard.api.deps import get_current_user, get_session
from xai_dashboard.api.routers.performance import latest_evaluation
from xai_dashboard.api.routers.reliability import latest_reliability
from xai_dashboard.core.data.diseases import load_disease_configs
from xai_dashboard.core.reports.builder import build_evaluation_report, build_research_report
from xai_dashboard.db.models import ReportLog, User
from xai_dashboard.settings import Settings, get_settings

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{disease_id}/evaluation.pdf")
def evaluation_report(
    disease_id: str,
    _user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> Response:
    config = load_disease_configs(settings.configs_dir).get(disease_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "unknown_disease", "message": "Disease configuration was not found."},
        )
    path: Path | None = latest_evaluation(settings.artifacts_dir / disease_id)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "report_not_generated",
                "message": "Evaluation artifacts are not yet generated.",
            },
        )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    pdf = build_evaluation_report(config.name, artifact)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{disease_id}-evaluation.pdf"'},
    )


@router.get("/{disease_id}/research.pdf")
def research_report(
    disease_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Response:
    config = load_disease_configs(settings.configs_dir).get(disease_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "unknown_disease", "message": "Disease configuration was not found."},
        )
    directory = settings.artifacts_dir / disease_id
    evaluation_path = latest_evaluation(directory)
    if evaluation_path is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "report_not_generated",
                "message": "Evaluation artifacts are not yet generated.",
            },
        )
    reliability_path = latest_reliability(directory)
    reliability = (
        json.loads(reliability_path.read_text(encoding="utf-8")) if reliability_path else None
    )
    pdf = build_research_report(
        config.name,
        json.loads(evaluation_path.read_text(encoding="utf-8")),
        reliability,
        visualizations_dir=directory,
    )
    db.add(
        ReportLog(
            user_id=user.id,
            disease_id=disease_id,
            report_type="research_pdf",
            artifact_path=str(evaluation_path),
        )
    )
    db.commit()
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{disease_id}-research.pdf"'},
    )
