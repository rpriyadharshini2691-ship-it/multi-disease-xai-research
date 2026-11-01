from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xai_dashboard.api import deps, main
from xai_dashboard.core.auth.rate_limit import login_limiter
from xai_dashboard.core.data.diseases import load_disease_config
from xai_dashboard.db.models import Base, PredictionLog, ReportLog, User
from xai_dashboard.settings import Settings

ROOT = Path(__file__).parents[1]
SECRET = "phase-six-end-to-end-test-secret-at-least-thirty-two-characters"


def _dataset_features(disease_id: str) -> dict[str, Any]:
    config = load_disease_config(ROOT / "configs" / f"{disease_id}.yaml")
    frame = pd.read_csv(ROOT / config.dataset_path, na_values=["?"])
    values: dict[str, Any] = {}
    for column in config.feature_columns:
        value = frame[column].dropna().iloc[0]
        values[column] = value.item() if hasattr(value, "item") else value
    return values


def test_phase_six_complete_researcher_journey_isolated_sqlite(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        environment="testing",
        secret_key=SECRET,
        database_url="sqlite:///:memory:",
        configs_dir=ROOT / "configs",
        artifacts_dir=ROOT / "artifacts",
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_session() -> Iterator[Session]:
        with factory() as session:
            yield session

    main.app.dependency_overrides[deps.get_session] = override_session
    main.app.dependency_overrides[deps.get_settings] = lambda: settings
    login_limiter.reset()
    client = TestClient(main.app)
    try:
        registration = client.post(
            "/api/v1/auth/register",
            json={
                "username": "phase6researcher",
                "email": "phase6researcher@example.org",
                "full_name": "Phase Six Researcher",
                "password": "phase-six-password-123",
                "role": "researcher",
            },
        )
        assert registration.status_code == 201
        login = client.post(
            "/api/v1/auth/login",
            data={
                "username": "phase6researcher@example.org",
                "password": "phase-six-password-123",
            },
        )
        assert login.status_code == 200
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        assert client.get("/api/v1/health").json() == {"status": "ok"}
        assert client.get("/api/v1/users/me", headers=headers).status_code == 200
        assert client.get("/api/v1/users", headers=headers).status_code == 403

        dashboard_modules = client.get("/api/v1/diseases").json()
        assert {module["id"] for module in dashboard_modules} == {"diabetes", "heart", "ckd"}
        for disease_id in ("diabetes", "heart", "ckd"):
            config = load_disease_config(ROOT / "configs" / f"{disease_id}.yaml")
            features = _dataset_features(disease_id)
            payload = {
                "disease_id": disease_id,
                "features": features,
                "model_name": "weighted_ensemble",
                "consent": False,
            }
            invalid = client.post(
                "/api/v1/predict",
                headers=headers,
                json={**payload, "features": {**features, "unexpected": "value"}},
            )
            assert invalid.status_code == 422

            prediction = client.post("/api/v1/predict", headers=headers, json=payload)
            assert prediction.status_code == 200, prediction.text
            prediction_body = prediction.json()
            assert prediction_body["disease_id"] == disease_id
            assert prediction_body["model_name"] == "weighted_ensemble"
            assert prediction_body["model_version"]
            assert 0 <= prediction_body["positive_class_score"] <= 1
            assert prediction_body["important_features"]
            assert prediction_body["stored"] is False
            assert "diagnosis" in prediction_body["disclaimer"].lower()

            local_explanation = client.post(
                f"/api/v1/explain/{disease_id}/local",
                headers=headers,
                json={"features": features, "model_name": "weighted_ensemble"},
            )
            assert local_explanation.status_code == 200, local_explanation.text
            assert local_explanation.json()["additivity_passed"] is True
            invalid_explanation_features = {
                **features,
                config.categorical_columns[0]: "not-a-configured-category",
            }
            invalid_explanation = client.post(
                f"/api/v1/explain/{disease_id}/local",
                headers=headers,
                json={
                    "features": invalid_explanation_features,
                    "model_name": "weighted_ensemble",
                },
            )
            assert invalid_explanation.status_code == 422
            assert invalid_explanation.json()["error"]["code"] == "invalid_category"

            individual_pdf = client.post(
                "/api/v1/reports/prediction.pdf", headers=headers, json=payload
            )
            assert individual_pdf.status_code == 200
            assert individual_pdf.headers["content-type"] == "application/pdf"
            assert individual_pdf.content.startswith(b"%PDF")
            assert individual_pdf.headers["x-prediction-reference"]

            global_explanation = client.get(
                f"/api/v1/explain/{disease_id}/global?model_name=weighted_ensemble"
            )
            assert global_explanation.status_code == 200
            assert global_explanation.json()["status"] == "ready"
            assert global_explanation.json()["features"]

            performance = client.get(f"/api/v1/performance/{disease_id}")
            assert performance.status_code == 200
            assert performance.json()["status"] == "ready"
            assert set(performance.json()["artifact"]["holdout_metrics"]) == {
                "logistic_regression",
                "random_forest",
                "xgboost",
                "equal_weight_ensemble",
                "weighted_ensemble",
            }

            reliability = client.get(f"/api/v1/reliability/{disease_id}")
            assert reliability.status_code == 200
            reliability_body = reliability.json()
            assert reliability_body["status"] == "ready"
            assert reliability_body["artifact"]["schema_version"] == 2
            assert len(reliability_body["artifact"]["model_configurations"]) == 5

            research_pdf = client.get(f"/api/v1/reports/{disease_id}/research.pdf", headers=headers)
            assert research_pdf.status_code == 200
            assert research_pdf.headers["content-type"] == "application/pdf"
            assert research_pdf.content.startswith(b"%PDF")

        with factory() as db:
            assert db.scalar(select(func.count()).select_from(PredictionLog)) == 0
            assert db.scalar(select(func.count()).select_from(ReportLog)) == 3
            researcher = db.scalar(select(User).where(User.username == "phase6researcher"))
            assert researcher is not None and researcher.is_active

        logout = client.post("/api/v1/auth/logout", headers=headers)
        assert logout.status_code == 204
        assert client.get("/api/v1/users/me", headers=headers).status_code == 401
    finally:
        main.app.dependency_overrides.clear()
        login_limiter.reset()
        client.close()
        engine.dispose()
