"""Training orchestration and database registration for generated model bundles."""

from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from xai_dashboard.core.data.diseases import DiseaseConfig
from xai_dashboard.core.ml.training import train_disease
from xai_dashboard.db.models import ModelRegistry
from xai_dashboard.db.session import SessionLocal


def register_model_artifacts(
    db: Session,
    config: DiseaseConfig,
    result: dict[str, Any],
    artifacts_dir: Path,
    created_by_id: int | None = None,
) -> int:
    """Register each trained estimator against its persisted versioned bundle."""
    version = str(result["model_version"])
    artifact_path = str((artifacts_dir / config.id / str(result["model_artifact"])).resolve())
    model_names = tuple(result["holdout_metrics"])
    registered = 0
    for model_name in model_names:
        previous = db.scalars(
            select(ModelRegistry).where(
                ModelRegistry.disease_id == config.id,
                ModelRegistry.model_name == model_name,
                ModelRegistry.is_active.is_(True),
            )
        ).all()
        for entry in previous:
            entry.is_active = entry.version == version
        current = db.scalar(
            select(ModelRegistry).where(
                ModelRegistry.disease_id == config.id,
                ModelRegistry.model_name == model_name,
                ModelRegistry.version == version,
            )
        )
        if current is None:
            db.add(
                ModelRegistry(
                    disease_id=config.id,
                    model_name=model_name,
                    version=version,
                    artifact_path=artifact_path,
                    is_active=True,
                    created_by_id=created_by_id,
                )
            )
            registered += 1
        else:
            current.artifact_path = artifact_path
            current.is_active = True
            if created_by_id is not None:
                current.created_by_id = created_by_id
    db.commit()
    return registered


def train_and_register_models(
    config: DiseaseConfig,
    dataset_path: Path,
    artifacts_dir: Path,
    seed: int = 42,
    test_size: float = 0.2,
    cv_splits: int = 5,
    cv_repeats: int = 3,
    created_by_id: int | None = None,
) -> dict[str, Any]:
    """Run training and persist registry metadata using an independent DB session."""
    result = train_disease(
        config,
        dataset_path,
        artifacts_dir,
        seed=seed,
        test_size=test_size,
        cv_splits=cv_splits,
        cv_repeats=cv_repeats,
    )
    with SessionLocal() as db:
        result["registered_model_count"] = register_model_artifacts(
            db, config, result, artifacts_dir, created_by_id
        )
    return result


def print_training_summary(result: dict[str, Any]) -> None:
    print(
        f"{result['disease_id']} version {result['model_version']} "
        f"({result['registered_model_count']} model records registered)"
    )
    for model_name, metrics in result["holdout_metrics"].items():
        values = " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        cv_auc = result["cv"]["training_metrics"][model_name]["roc_auc"]
        lower, upper = cv_auc["bootstrap_ci_95"]
        print(
            f"  {model_name}: {values}; OOF-CV ROC-AUC="
            f"{cv_auc['mean']:.4f} [{lower:.4f}, {upper:.4f}]"
        )
