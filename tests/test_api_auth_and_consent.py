from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from xai_dashboard.api import deps, main
from xai_dashboard.core.auth.security import hash_password
from xai_dashboard.db.models import Base, PredictionLog, User
from xai_dashboard.settings import Settings


class FixedScoreModel:
    def predict_proba(self, frame: Any) -> np.ndarray:
        return np.tile([0.35, 0.65], (len(frame), 1))


def test_registration_rbac_and_consent_only_prediction_logging(tmp_path, monkeypatch) -> None:
    settings = Settings(
        environment="testing",
        secret_key="a-test-secret-key-that-is-at-least-thirty-two-characters",
        configs_dir=Path("configs").resolve(),
        artifacts_dir=tmp_path,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_session():
        with session_factory() as session:
            yield session

    main.app.dependency_overrides[deps.get_session] = override_session
    main.app.dependency_overrides[deps.get_settings] = lambda: settings
    model_dir = tmp_path / "diabetes"
    model_dir.mkdir()
    joblib.dump(
        {"models": {"weighted_ensemble": FixedScoreModel()}},
        model_dir / "models_test.joblib",
    )
    client = TestClient(main.app)
    try:
        admin_registration = client.post(
            "/api/v1/auth/register",
            json={
                "username": "notadmin",
                "email": "admin@example.com",
                "full_name": "Not an Admin",
                "password": "valid-password-123",
                "role": "admin",
            },
        )
        assert admin_registration.status_code == 422

        registration = client.post(
            "/api/v1/auth/register",
            json={
                "username": "researcher1",
                "email": "researcher@example.com",
                "full_name": "Researcher One",
                "password": "valid-password-123",
                "role": "researcher",
            },
        )
        assert registration.status_code == 201
        token = client.post(
            "/api/v1/auth/login",
            data={"username": "researcher@example.com", "password": "valid-password-123"},
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        assert client.get("/api/v1/users/me", headers=headers).status_code == 200
        forbidden = client.get("/api/v1/users", headers=headers)
        assert forbidden.status_code == 403
        assert forbidden.json()["error"]["code"] == "forbidden"

        features = {
            "age": 40,
            "gender": "Female",
            "polyuria": "No",
            "polydipsia": "No",
            "sudden_weight_loss": "No",
            "weakness": "No",
            "polyphagia": "No",
            "genital_thrush": "No",
            "visual_blurring": "No",
            "itching": "No",
            "irritability": "No",
            "delayed_healing": "No",
            "partial_paresis": "No",
            "muscle_stiffness": "No",
            "alopecia": "No",
            "obesity": "No",
        }
        prediction = client.post(
            "/api/v1/predict",
            headers=headers,
            json={"disease_id": "diabetes", "features": features, "consent": True},
        )
        assert prediction.status_code == 200
        body = prediction.json()
        assert body["positive_class_score"] == 0.65
        assert "not a clinical probability" in body["disclaimer"]
        with session_factory() as session:
            logs = session.scalars(select(PredictionLog)).all()
            assert len(logs) == 1
            assert logs[0].disease_id == "diabetes"
            assert logs[0].positive_score == 0.65
            assert logs[0].model_name == "weighted_ensemble"
            assert not hasattr(logs[0], "features")

        no_consent = client.post(
            "/api/v1/predict",
            headers=headers,
            json={"disease_id": "diabetes", "features": features, "consent": False},
        )
        assert no_consent.status_code == 200
        with session_factory() as session:
            assert len(session.scalars(select(PredictionLog)).all()) == 1

        with session_factory() as session:
            session.add(
                User(
                    username="adminuser",
                    email="admin@example.com",
                    full_name="Admin User",
                    password_hash=hash_password("valid-admin-password-123"),
                    role="admin",
                )
            )
            session.commit()
        admin_token = client.post(
            "/api/v1/auth/login",
            data={"username": "admin@example.com", "password": "valid-admin-password-123"},
        ).json()["access_token"]
        admin_users = client.get(
            "/api/v1/users", headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert admin_users.status_code == 200
    finally:
        main.app.dependency_overrides.clear()
        client.close()
        engine.dispose()
