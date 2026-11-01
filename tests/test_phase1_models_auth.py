from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from xai_dashboard.core.auth.security import (
    PUBLIC_ROLES,
    ROLES,
    hash_password,
    verify_password,
)
from xai_dashboard.db.models import (
    Base,
    ExperimentLog,
    ModelRegistry,
    PredictionLog,
    ReportLog,
    User,
)
from xai_dashboard.services.auth import register_user


def _session() -> tuple[Session, object]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine), engine


def test_user_creation_hashes_password_and_only_allows_public_roles() -> None:
    db, engine = _session()
    try:
        user = register_user(
            db,
            "Researcher@Example.com",
            "Researcher One",
            "secure-research-password",
            "researcher",
            "researcher1",
        )
        assert user.email == "researcher@example.com"
        assert user.role == "researcher"
        assert user.password_hash != "secure-research-password"
        assert verify_password("secure-research-password", user.password_hash)
        assert ROLES == {"admin", "researcher", "clinician"}
        assert PUBLIC_ROLES == {"researcher", "clinician"}
        with pytest.raises(ValueError, match="Public registration"):
            register_user(
                db, "admin@example.com", "Admin", "secure-admin-password", "admin", "adminuser"
            )
    finally:
        db.close()
        engine.dispose()


def test_phase_one_registry_and_audit_models_persist() -> None:
    db, engine = _session()
    try:
        user = User(
            username="adminuser",
            email="admin@example.com",
            full_name="Admin User",
            password_hash=hash_password("secure-admin-password"),
            role="admin",
        )
        db.add(user)
        db.flush()
        model = ModelRegistry(
            disease_id="diabetes",
            model_name="weighted_ensemble",
            version="v1",
            artifact_path="artifacts/diabetes/model.joblib",
            created_by_id=user.id,
        )
        experiment = ExperimentLog(
            user_id=user.id,
            disease_id="diabetes",
            experiment_type="shap_stability",
            status="completed",
            parameters={"seed": 42},
            artifact_path="artifacts/diabetes/reliability.json",
            completed_at=datetime.now(UTC),
        )
        prediction = PredictionLog(
            user_id=user.id,
            disease_id="diabetes",
            model_name="weighted_ensemble",
            positive_score=0.61,
        )
        report = ReportLog(
            user_id=user.id,
            disease_id="diabetes",
            report_type="evaluation_pdf",
            artifact_path="artifacts/diabetes/report.pdf",
        )
        db.add_all([model, experiment, prediction, report])
        db.commit()
        assert model.creator == user
        assert experiment.user == user
        assert report.user == user
        assert prediction.model_name == "weighted_ensemble"
    finally:
        db.close()
        engine.dispose()


def test_database_rejects_unknown_role() -> None:
    db, engine = _session()
    try:
        db.add(
            User(
                username="invalid",
                email="invalid@example.com",
                full_name="Invalid Role",
                password_hash=hash_password("secure-invalid-password"),
                role="superuser",
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
    finally:
        db.rollback()
        db.close()
        engine.dispose()
