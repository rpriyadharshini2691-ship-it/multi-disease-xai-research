from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

from xai_dashboard.settings import Settings
from xai_dashboard.ui import components


class FakeClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    def get(self, path: str) -> dict[str, Any]:
        if path.startswith("/reliability/"):
            return self.response
        return {"status": "completed"}

    def post(self, _path: str, _payload: dict[str, Any]) -> dict[str, str]:
        return {"job_id": "test-job"}


def _app_path() -> Path:
    return Path(__file__).parents[1] / "src" / "xai_dashboard" / "ui" / "pages" / "5_Reliability.py"


def _ready_artifact() -> dict[str, Any]:
    models = [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "equal_weight_ensemble",
        "weighted_ensemble",
    ]
    summary = [
        {
            "model_name": model,
            "perturbation_level": 0.05,
            "spearman": {
                "mean": 0.9,
                "sd": 0.05,
                "n_repetitions": 30,
                "confidence_interval_95": [0.85, 0.95],
            },
            "top5_jaccard": {
                "mean": 0.8,
                "sd": 0.1,
                "n_repetitions": 30,
                "confidence_interval_95": [0.7, 0.9],
            },
            "additivity_abs_error": {"mean": 1e-8, "sd": 1e-8},
            "sample_comparisons": 240,
            "additivity_passed": 240,
            "additivity_failed": 0,
        }
        for model in models
    ]
    return {
        "schema_version": 2,
        "model_configurations": models,
        "experiment_config": {
            "sample_size_evaluated": 8,
            "heldout_test_rows": 20,
            "repetitions_per_model_level": 30,
            "random_seeds": [42, 123, 2026, 2027],
            "perturbation_levels": [0.05],
        },
        "summary": summary,
        "sample_results": [
            {
                "model_name": "logistic_regression",
                "perturbation_level": 0.05,
                "sample_id": 12,
                "repetition": 1,
                "random_seed": 42,
                "spearman_rho": 0.9,
                "original_top5_features": ["age", "sex"],
                "perturbed_top5_features": ["age", "sex"],
                "common_top5_features": ["age", "sex"],
                "top5_jaccard": 1.0,
                "perturbed_additivity_error": 1e-8,
                "additivity_passed": True,
            }
        ],
        "global_stability": [
            {
                "model_name": "logistic_regression",
                "perturbation_level": 0.05,
                "spearman": {"mean": 0.95, "sd": 0.02, "n_repetitions": 30},
                "baseline_feature_importance": [
                    {"feature": "age", "mean_abs_shap": 0.3},
                    {"feature": "sex", "mean_abs_shap": 0.2},
                ],
                "perturbed_feature_importance": [
                    {"feature": "age", "mean_abs_shap": 0.28},
                    {"feature": "sex", "mean_abs_shap": 0.22},
                ],
            }
        ],
        "shap_vs_permutation": [
            {
                "model_name": "logistic_regression",
                "spearman_rho": 0.8,
                "features": ["age", "sex"],
                "shap_mean_abs": [0.3, 0.2],
                "permutation_mean": [0.25, 0.1],
                "permutation_sd": [0.02, 0.01],
                "evaluation_rows": 8,
                "permutation_repetitions": 30,
                "seed": 42,
                "feature_representation": "original features",
            }
        ],
        "additivity_summary": [
            {
                "model_name": "equal_weight_ensemble",
                "is_ensemble": True,
                "score_space": "positive_class_probability",
                "passed": 728,
                "failed": 0,
                "max_abs_error": 1e-8,
            },
            {
                "model_name": "weighted_ensemble",
                "is_ensemble": True,
                "score_space": "positive_class_probability",
                "passed": 728,
                "failed": 0,
                "max_abs_error": 1e-8,
            },
        ],
        "visualizations": {},
        "warnings": [],
    }


def _run_page(monkeypatch, response: dict[str, Any], tmp_path: Path) -> AppTest:
    client = FakeClient(response)
    monkeypatch.setattr(components, "require_login", lambda: (client, {"role": "researcher"}))
    monkeypatch.setattr(
        components,
        "load_diseases",
        lambda _client: [
            {"id": "diabetes", "name": "Diabetes"},
            {"id": "heart", "name": "Heart Disease"},
            {"id": "ckd", "name": "CKD"},
        ],
    )
    monkeypatch.setattr(
        "xai_dashboard.settings.get_settings",
        lambda: Settings(artifacts_dir=tmp_path / "artifacts"),
    )
    return AppTest.from_file(str(_app_path())).run()


def test_reliability_page_shows_not_yet_generated(monkeypatch, tmp_path: Path) -> None:
    app = _run_page(monkeypatch, {"status": "not_generated"}, tmp_path)
    assert not app.exception
    assert any(element.value == "Not yet generated." for element in app.info)


def test_reliability_page_renders_metrics_and_named_feature_comparison(
    monkeypatch, tmp_path: Path
) -> None:
    app = _run_page(monkeypatch, {"status": "ready", "artifact": _ready_artifact()}, tmp_path)
    assert not app.exception
    assert len(app.metric) == 4
    assert len(app.dataframe) >= 7
    assert any("Common features: age, sex" in element.value for element in app.markdown)
    assert any("Seed: 42" in element.value for element in app.markdown)
    assert any("Global explanation stability" in element.value for element in app.subheader)
    assert any("SHAP versus permutation importance" in element.value for element in app.subheader)
    assert any("Ensemble additivity validation" in element.value for element in app.subheader)
    assert any("Disease-wise stability" in element.value for element in app.subheader)
