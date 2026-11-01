from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from xai_dashboard.core.data.diseases import DiseaseConfig
from xai_dashboard.db.models import Base, ModelRegistry
from xai_dashboard.services.training import register_model_artifacts


def test_model_artifacts_register_idempotently_and_activate_latest(tmp_path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    config = DiseaseConfig(
        id="toy",
        name="Toy",
        dataset_path="data/toy.csv",
        target_column="target",
        positive_label=1,
        numeric_columns=("feature",),
        categorical_columns=(),
        binary_columns=(),
        display_names={},
        units={},
        valid_ranges={},
    )
    metrics = {
        name: {"accuracy": 0.5}
        for name in (
            "logistic_regression",
            "random_forest",
            "xgboost",
            "equal_weight_ensemble",
            "weighted_ensemble",
        )
    }
    try:
        with session_factory() as session:
            first_result = {
                "model_version": "v1",
                "model_artifact": "models_v1.joblib",
                "holdout_metrics": metrics,
            }
            assert register_model_artifacts(session, config, first_result, tmp_path) == 5
            assert register_model_artifacts(session, config, first_result, tmp_path) == 0

            second_result = {
                "model_version": "v2",
                "model_artifact": "models_v2.joblib",
                "holdout_metrics": metrics,
            }
            assert register_model_artifacts(session, config, second_result, tmp_path) == 5
            entries = session.scalars(
                select(ModelRegistry).order_by(ModelRegistry.model_name, ModelRegistry.version)
            ).all()
            assert len(entries) == 10
            for model_name in metrics:
                versions = [item for item in entries if item.model_name == model_name]
                assert [(item.version, item.is_active) for item in versions] == [
                    ("v1", False),
                    ("v2", True),
                ]
                assert versions[1].artifact_path.endswith("models_v2.joblib")
    finally:
        engine.dispose()
