import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from xai_dashboard.api.routers import jobs
from xai_dashboard.api.routers.reliability import latest_reliability, reliability
from xai_dashboard.core.data.diseases import DiseaseConfig
from xai_dashboard.core.data.preprocessing import build_pipeline
from xai_dashboard.core.data.quality import file_sha256
from xai_dashboard.core.ml.training import WeightedProbabilityEnsemble
from xai_dashboard.core.reliability import runner as reliability_runner
from xai_dashboard.db.models import Base, ExperimentLog
from xai_dashboard.services.reliability import record_reliability_experiment
from xai_dashboard.settings import Settings


def _result(artifact_path: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "disease_id": "diabetes",
        "model_version": "test-version",
        "dataset_fingerprint": "abc123",
        "sample_count": 8,
        "model_configurations": ["logistic_regression", "weighted_ensemble"],
        "experiment_config": {"repetitions_per_model_level": 30},
        "artifact_path": artifact_path,
    }


def test_reliability_log_persistence_appends_rows(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'experiments.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    try:
        with factory() as db:
            first = record_reliability_experiment(db, _result("artifacts/one.json"))
            second = record_reliability_experiment(db, _result("artifacts/two.json"))
            rows = db.scalars(select(ExperimentLog).order_by(ExperimentLog.id)).all()
            assert first != second
            assert len(rows) == 2
            assert rows[0].status == "completed"
            assert rows[0].experiment_type == "shap_explanation_reliability"
            assert rows[0].parameters["dataset_fingerprint"] == "abc123"
            assert rows[0].artifact_path == "artifacts/one.json"
    finally:
        engine.dispose()


def test_legacy_reliability_is_not_exposed_as_phase_three(tmp_path: Path) -> None:
    directory = tmp_path / "diabetes"
    directory.mkdir()
    legacy = directory / "reliability_legacy.json"
    legacy.write_text(json.dumps({"disease_id": "diabetes", "stability": {}}), encoding="utf-8")
    assert latest_reliability(directory) is None
    response = reliability(
        "diabetes",
        Settings(configs_dir=Path("configs").resolve(), artifacts_dir=tmp_path),
    )
    assert response["status"] == "not_generated"


def test_reliability_api_serves_current_schema_artifact(tmp_path: Path) -> None:
    directory = tmp_path / "diabetes"
    directory.mkdir()
    artifact = _result("artifacts/diabetes/result.json")
    artifact.update(
        {
            "schema_version": 2,
            "generated_at": "2026-11-01T00:00:00+00:00",
            "summary": [],
            "sample_results": [],
            "visualizations": {},
        }
    )
    (directory / "reliability_phase3.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )
    response = reliability(
        "diabetes",
        Settings(configs_dir=Path("configs").resolve(), artifacts_dir=tmp_path),
    )
    assert response["status"] == "ready"
    assert response["artifact"]["schema_version"] == 2
    assert response["artifact"]["dataset_fingerprint"] == "abc123"


def test_reliability_job_requires_thirty_repetitions_and_queues_valid_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValidationError):
        jobs.ReliabilityRequest(repetitions=29)

    queued: dict[str, object] = {}

    def fake_submit(kind: str, operation: object) -> str:
        queued["kind"] = kind
        queued["operation"] = operation
        return "job-id"

    monkeypatch.setattr(jobs, "submit_job", fake_submit)
    settings = Settings(configs_dir=Path("configs").resolve(), artifacts_dir=tmp_path)
    response = jobs.start_reliability(
        "diabetes", jobs.ReliabilityRequest(), SimpleNamespace(id=7), settings
    )
    assert response == {"job_id": "job-id", "status": "queued"}
    assert queued["kind"] == "reliability"


def test_runner_executes_all_configurations_and_persists_genuine_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "toy.csv"
    positions = np.arange(40)
    frame = pd.DataFrame(
        {
            "age": 20 + positions,
            "sex": np.where(positions % 2 == 0, "F", "M"),
            "target": positions % 2,
        }
    )
    frame.to_csv(dataset_path, index=False)
    config = DiseaseConfig(
        id="toy",
        name="Toy",
        dataset_path=str(dataset_path),
        target_column="target",
        positive_label=1,
        numeric_columns=("age",),
        categorical_columns=("sex",),
        binary_columns=(),
        display_names={},
        units={},
        valid_ranges={},
    )
    pipelines = {
        "logistic_regression": build_pipeline(config, LogisticRegression()),
        "random_forest": build_pipeline(
            config, RandomForestClassifier(n_estimators=8, random_state=3)
        ),
        "xgboost": build_pipeline(config, LogisticRegression(C=0.5)),
    }
    features = frame.loc[:, config.feature_columns]
    target = frame["target"].to_numpy(dtype=np.int64)
    for pipeline in pipelines.values():
        pipeline.fit(features, target)
    equal_weights = {name: 1 / 3 for name in pipelines}
    model_bundle = {
        "models": {
            **pipelines,
            "equal_weight_ensemble": WeightedProbabilityEnsemble(pipelines, equal_weights),
            "weighted_ensemble": WeightedProbabilityEnsemble(pipelines, equal_weights),
        }
    }
    disease_dir = tmp_path / "artifacts" / "toy"
    disease_dir.mkdir(parents=True)
    model_version = "smoke"
    joblib.dump(model_bundle, disease_dir / f"models_{model_version}.joblib")
    (disease_dir / f"evaluation_{model_version}.json").write_text(
        json.dumps(
            {
                "dataset_sha256": file_sha256(dataset_path),
                "seed": 42,
                "test_size": 0.25,
            }
        ),
        encoding="utf-8",
    )

    class FakeExplainer:
        def __init__(self, score_function, _background, algorithm: str, seed: int) -> None:
            assert algorithm == "permutation"
            assert seed == 42
            self.score_function = score_function

        def __call__(
            self, values: pd.DataFrame, max_evals: int, silent: bool = True
        ) -> SimpleNamespace:
            assert max_evals > values.shape[1]
            assert silent
            scores = self.score_function(values.to_numpy())
            weights = np.arange(1, values.shape[1] + 1, dtype=float)
            weights /= weights.sum()
            attributions = (scores - 0.5)[:, None] * weights[None, :]
            return SimpleNamespace(
                values=attributions,
                base_values=np.full(len(values), 0.5),
            )

    monkeypatch.setattr(reliability_runner.shap, "Explainer", FakeExplainer)
    result = reliability_runner.run_reliability(
        config,
        dataset_path,
        tmp_path / "artifacts",
        repetitions=30,
        sample_size=8,
        random_seeds=(42, 123, 2026, 2027),
    )

    assert len(result["model_configurations"]) == 5
    assert len(result["summary"]) == 15
    assert len(result["global_stability"]) == 15
    assert all(len(row["repetition_results"]) == 30 for row in result["global_stability"])
    assert len(result["sample_results"]) == 5 * 3 * 30 * 8
    assert len(set(result["heldout_sample_ids"])) == 8
    assert all(row["additivity_passed"] for row in result["sample_results"])
    assert all(
        row["model_name"] in result["model_configurations"]
        for row in result["sample_results"]
    )
    assert all((disease_dir / name).is_file() for name in result["visualizations"].values())
    persisted = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert persisted["dataset_fingerprint"] == file_sha256(dataset_path)
    assert persisted["experiment_config"]["sample_size_evaluated"] == 8


def test_runner_additivity_check_rejects_non_additive_explanations() -> None:
    class NonAdditiveExplainer:
        def __call__(self, values: pd.DataFrame, max_evals: int, silent: bool) -> SimpleNamespace:
            return SimpleNamespace(
                values=np.zeros((len(values), values.shape[1])),
                base_values=np.zeros(len(values)),
            )

    frame = pd.DataFrame({"feature": [1.0, 2.0]})

    def score_function(values: np.ndarray) -> np.ndarray:
        return np.full(len(values), 0.7, dtype=float)

    _, _, _, errors, passed = reliability_runner._explain_batch(
        NonAdditiveExplainer(),
        frame,
        score_function,
        tolerance=1e-4,
        max_evals=3,
    )
    assert errors.tolist() == [-0.7, -0.7]
    assert passed.tolist() == [False, False]