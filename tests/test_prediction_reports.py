import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xai_dashboard.api import deps, main
from xai_dashboard.api.routers import predictions as prediction_router
from xai_dashboard.api.routers.diseases import list_diseases
from xai_dashboard.core.auth.security import create_access_token, hash_password
from xai_dashboard.core.data.diseases import DiseaseConfig, load_disease_config
from xai_dashboard.core.data.preprocessing import build_pipeline
from xai_dashboard.core.data.quality import file_sha256
from xai_dashboard.core.reports.builder import build_prediction_report
from xai_dashboard.db.models import Base, PredictionLog, User
from xai_dashboard.services.prediction import (
    PredictionInputError,
    create_prediction_details,
    validate_prediction_features,
)
from xai_dashboard.settings import Settings

SECRET = "phase-four-test-secret-key-at-least-thirty-two-characters"


def test_disease_metadata_returns_verified_categorical_options() -> None:
    modules = list_diseases(Settings(configs_dir=Path("configs").resolve()))
    diabetes = next(module for module in modules if module["id"] == "diabetes")
    gender = next(feature for feature in diabetes["features"] if feature["id"] == "gender")
    assert gender["options"] == ["Female", "Male"]


def test_disease_metadata_exposes_processed_categorical_options() -> None:
    result = list_diseases(Settings(configs_dir=Path("configs").resolve()))
    diabetes = next(item for item in result if item["id"] == "diabetes")
    gender = next(feature for feature in diabetes["features"] if feature["id"] == "gender")
    assert gender["options"] == ["Female", "Male"]


def _features(config: DiseaseConfig) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for column in config.numeric_columns:
        bounds = config.valid_ranges.get(column)
        values[column] = (bounds[0] + bounds[1]) / 2 if bounds else 40
    categories = {
        "diabetes": {"gender": "Female"},
        "heart": {
            "sex": 1,
            "cp": 1,
            "fbs": 0,
            "restecg": 0,
            "exang": 0,
            "slope": 1,
            "ca": 0,
            "thal": 3,
        },
        "ckd": {
            "sg": 1.02,
            "al": 0,
            "su": 0,
            "rbc": "normal",
            "pc": "normal",
            "pcc": "notpresent",
            "ba": "notpresent",
            "htn": "no",
            "dm": "no",
            "cad": "no",
            "appet": "good",
            "pe": "no",
            "ane": "no",
        },
    }
    values.update(categories[config.id])
    for column in config.categorical_columns:
        values.setdefault(column, "No")
    return values


@pytest.mark.parametrize("disease_id", ["diabetes", "heart", "ckd"])
def test_prediction_inputs_accept_configured_categories(disease_id: str) -> None:
    config = load_disease_config(Path("configs") / f"{disease_id}.yaml")
    features = _features(config)
    reference = pd.DataFrame([features])

    validated = validate_prediction_features(config, features, reference)

    assert tuple(validated.columns) == config.feature_columns
    assert len(validated) == 1


def test_registration_rejects_bcrypt_overlength_passwords() -> None:
    from pydantic import ValidationError

    from xai_dashboard.api.schemas import RegisterRequest

    with pytest.raises(ValidationError, match="72 UTF-8 bytes"):
        RegisterRequest(
            username="longpass",
            email="longpass@example.org",
            full_name="Long Password",
            password="x" * 73,
        )
    with pytest.raises(ValueError, match="72 UTF-8 bytes"):
        hash_password("x" * 73)


@pytest.mark.parametrize("disease_id", ["diabetes", "heart", "ckd"])
def test_prediction_inputs_reject_unknown_categories(disease_id: str) -> None:
    config = load_disease_config(Path("configs") / f"{disease_id}.yaml")
    features = _features(config)
    category_column = config.categorical_columns[0]
    reference = pd.DataFrame([features])
    features[category_column] = "unexpected-category"

    with pytest.raises(PredictionInputError, match="configured dataset category"):
        validate_prediction_features(config, features, reference)


def test_prediction_inputs_reject_missing_extra_null_and_out_of_range_values() -> None:
    config = load_disease_config(Path("configs/diabetes.yaml"))
    features = _features(config)
    reference = pd.DataFrame([features])
    with pytest.raises(PredictionInputError, match="Feature names"):
        validate_prediction_features(
            config,
            {key: value for key, value in features.items() if key != "age"},
            reference,
        )
    with pytest.raises(PredictionInputError, match="Feature names"):
        validate_prediction_features(config, {**features, "extra": 1}, reference)

    null_features = {**features, "gender": None}
    with pytest.raises(PredictionInputError, match="cannot be null"):
        validate_prediction_features(config, null_features, reference)

    out_of_range = {**features, "age": 121}
    with pytest.raises(PredictionInputError, match="outside its configured input range"):
        validate_prediction_features(config, out_of_range, reference)


def test_missing_model_artifact_is_reported_without_prediction(tmp_path: Path) -> None:
    config = load_disease_config(Path("configs/diabetes.yaml"))
    features = _features(config)
    dataset_path = tmp_path / "diabetes.csv"
    pd.DataFrame([features | {"target": 1}]).to_csv(
        dataset_path, index=False
    )
    with pytest.raises(FileNotFoundError, match="not present"):
        create_prediction_details(
            config,
            features,
            "weighted_ensemble",
            {"models": {}},
            "missing-version",
            dataset_path,
            tmp_path,
        )


def test_stored_pipeline_generates_prediction_and_grouped_shap(tmp_path: Path) -> None:
    features = pd.DataFrame(
        {
            "age": [float(value) for value in range(40)],
            "group": ["A" if value % 2 == 0 else "B" for value in range(40)],
        }
    )
    target = pd.Series([int(value % 2 == 0) for value in range(40)])
    dataset = features.assign(target=target)
    dataset_path = tmp_path / "toy.csv"
    dataset.to_csv(dataset_path, index=False)
    config = DiseaseConfig(
        id="toy",
        name="Toy disease",
        dataset_path=str(dataset_path),
        target_column="target",
        positive_label=1,
        numeric_columns=("age",),
        categorical_columns=("group",),
        binary_columns=(),
        display_names={"age": "Age", "group": "Group"},
        units={},
        valid_ranges={"age": (0, 100)},
    )
    model = build_pipeline(config, LogisticRegression()).fit(features, target)
    version = "toy-v1"
    artifacts_dir = tmp_path / "artifacts"
    model_dir = artifacts_dir / "toy"
    model_dir.mkdir(parents=True)
    (model_dir / f"evaluation_{version}.json").write_text(
        json.dumps(
            {
                "dataset_sha256": file_sha256(dataset_path),
                "seed": 13,
                "test_size": 0.25,
            }
        ),
        encoding="utf-8",
    )

    result = create_prediction_details(
        config,
        {"age": 31, "group": "B"},
        "logistic_regression",
        {"models": {"logistic_regression": model}},
        version,
        dataset_path,
        artifacts_dir,
    )

    assert result["model_version"] == version
    assert result["predicted_class"] in {"positive", "negative"}
    assert 0 <= result["positive_class_score"] <= 1
    assert result["additivity_passed"] is True
    assert result["important_features"]
    assert {item["feature"] for item in result["important_features"]} <= {"Age", "Group"}


def test_individual_prediction_pdf_contains_report_and_disclaimer() -> None:
    details = {
        "prediction_reference": "PRED-ABC123",
        "created_at": datetime(2026, 11, 1, tzinfo=UTC),
        "disease_name": "Diabetes",
        "model_name": "weighted_ensemble",
        "model_version": "v1",
        "predicted_class": "positive",
        "positive_class_score": 0.73,
        "decision_threshold": 0.5,
        "important_features": [
            {
                "feature": "Polyuria",
                "value": "Yes",
                "shap_value": 0.12,
                "direction": "increases",
            }
        ],
        "additivity_passed": True,
        "score_match_passed": True,
        "additivity_error": 1e-8,
    }
    pdf = build_prediction_report(details)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


class FakeRegistry:
    def load_latest(self, _disease_id: str) -> dict[str, Any]:
        return {"models": {"weighted_ensemble": object()}}


@pytest.fixture
def api_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, Any, Any, Any]]:
    settings = Settings(
        environment="testing",
        secret_key=SECRET,
        configs_dir=Path("configs").resolve(),
        artifacts_dir=tmp_path,
    )
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_session() -> Iterator[Session]:
        with factory() as session:
            yield session

    main.app.dependency_overrides[deps.get_session] = override_session
    main.app.dependency_overrides[deps.get_settings] = lambda: settings
    main.app.dependency_overrides[deps.get_model_registry] = FakeRegistry
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    model_dir = tmp_path / "diabetes"
    model_dir.mkdir()
    (model_dir / "models_test-version.joblib").write_bytes(b"stub")
    with factory() as db:
        active = User(
            username="researcher",
            email="researcher@example.com",
            full_name="Researcher",
            password_hash=hash_password("test-password-123"),
            role="researcher",
            is_active=True,
        )
        inactive = User(
            username="inactive",
            email="inactive@example.com",
            full_name="Inactive",
            password_hash=hash_password("test-password-123"),
            role="researcher",
            is_active=False,
        )
        db.add_all([active, inactive])
        db.commit()
        active_id, inactive_id = active.id, inactive.id
    token = create_access_token(str(active_id), "researcher", settings)
    inactive_token = create_access_token(str(inactive_id), "researcher", settings)
    client = TestClient(main.app)
    try:
        yield (
            client,
            {"Authorization": f"Bearer {token}"},
            {"Authorization": f"Bearer {inactive_token}"},
            factory,
        )
    finally:
        main.app.dependency_overrides.clear()
        client.close()
        engine.dispose()


def test_prediction_and_report_endpoints_are_authenticated_and_consent_gated(
    api_env: tuple[TestClient, Any, Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, headers, inactive_headers, factory = api_env
    details = {
        "prediction_reference": "PRED-API123456",
        "created_at": datetime(2026, 11, 1, tzinfo=UTC),
        "disease_id": "diabetes",
        "disease_name": "Diabetes",
        "model_name": "weighted_ensemble",
        "model_version": "test-version",
        "predicted_class": "positive",
        "positive_class_score": 0.65,
        "decision_threshold": 0.5,
        "important_features": [],
        "additivity_error": 0.0,
        "additivity_passed": True,
        "score_match_passed": True,
    }
    monkeypatch.setattr(prediction_router, "create_prediction_details", lambda *args: details)
    features = _features(load_disease_config(Path("configs/diabetes.yaml")))
    payload = {"disease_id": "diabetes", "features": features}

    denied = client.post("/api/v1/predict", json=payload)
    assert denied.status_code == 401
    unknown_disease = client.post(
        "/api/v1/predict",
        json={**payload, "disease_id": "unknown"},
        headers=headers,
    )
    assert unknown_disease.status_code == 404
    inactive = client.post("/api/v1/predict", json=payload, headers=inactive_headers)
    assert inactive.status_code == 403
    prediction = client.post("/api/v1/predict", json=payload, headers=headers)
    assert prediction.status_code == 200
    assert prediction.json()["model_version"] == "test-version"
    assert prediction.json()["prediction_reference"] == "PRED-API123456"
    assert prediction.json()["stored"] is False
    with monkeypatch.context() as patch:
        patch.setattr(prediction_router, "latest_model_file", lambda _directory: None)
        missing_model = client.post("/api/v1/predict", json=payload, headers=headers)
    assert missing_model.status_code == 503

    pdf = client.post("/api/v1/reports/prediction.pdf", json=payload, headers=headers)
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.headers["x-prediction-reference"] == "PRED-API123456"
    assert pdf.content.startswith(b"%PDF")
    with factory() as db:
        assert db.scalars(select(PredictionLog)).all() == []

    consented_pdf = client.post(
        "/api/v1/reports/prediction.pdf",
        json={**payload, "consent": True},
        headers=headers,
    )
    assert consented_pdf.status_code == 200
    with factory() as db:
        logs = db.scalars(select(PredictionLog)).all()
        assert len(logs) == 1
        assert logs[0].disease_id == "diabetes"
        assert logs[0].model_name == "weighted_ensemble"
        assert logs[0].positive_score == 0.65
        assert not hasattr(logs[0], "features")