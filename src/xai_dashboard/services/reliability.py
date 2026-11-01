"""Reliability experiment orchestration and append-only database audit records."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from xai_dashboard.core.data.diseases import DiseaseConfig
from xai_dashboard.core.reliability.runner import run_reliability
from xai_dashboard.db.models import ExperimentLog
from xai_dashboard.db.session import SessionLocal


def record_reliability_experiment(
    db: Session, result: dict[str, Any], user_id: int | None = None
) -> int:
    """Append one audit row pointing to the durable JSON artifact; never replace rows."""
    parameters = {
        "schema_version": result["schema_version"],
        "model_version": result["model_version"],
        "dataset_fingerprint": result["dataset_fingerprint"],
        "sample_count": result["sample_count"],
        "model_configurations": result["model_configurations"],
        "experiment_config": result["experiment_config"],
    }
    record = ExperimentLog(
        user_id=user_id,
        disease_id=str(result["disease_id"]),
        experiment_type="shap_explanation_reliability",
        status="completed",
        parameters=parameters,
        artifact_path=str(result["artifact_path"]),
        completed_at=datetime.now(UTC),
    )
    db.add(record)
    db.flush()
    record_id = int(record.id)
    db.commit()
    return record_id


def run_and_record_reliability(
    config: DiseaseConfig,
    dataset_path: Any,
    artifacts_dir: Any,
    user_id: int,
    repetitions: int = 30,
    sample_size: int = 8,
) -> dict[str, Any]:
    result = run_reliability(
        config,
        dataset_path,
        artifacts_dir,
        repetitions=repetitions,
        sample_size=sample_size,
    )
    with SessionLocal() as db:
        record_id = record_reliability_experiment(db, result, user_id)
    return {
        "disease_id": result["disease_id"],
        "model_version": result["model_version"],
        "dataset_fingerprint": result["dataset_fingerprint"],
        "artifact_path": result["artifact_path"],
        "sample_count": result["sample_count"],
        "sample_result_count": len(result["sample_results"]),
        "experiment_log_id": record_id,
        "warnings": result["warnings"],
    }