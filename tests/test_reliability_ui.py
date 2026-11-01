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
    return {
        "schema_version": 2,
        "model_configurations": ["logistic_regression"],
        "experiment_config": {
            "sample_size_evaluated": 8,
            "heldout_test_rows": 20,
            "repetitions_per_model_level": 30,
            "random_seeds": [42, 123, 2026, 2027],
            "perturbation_levels": [0.05],
        },
        "summary": [
            {
                "model_name": "logistic_regression",
                "perturbation_level": 0.05,
                "spearman": {"mean": 0.9, "sd": 0.05, "n_repetitions": 30},
                "top5_jaccard": {"mean": 0.8, "sd": 0.1, "n_repetitions": 30},
                "additivity_abs_error": {"mean": 1e-8, "sd": 1e-8},
                "sample_comparisons": 240,
                "additivity_passed": 240,
                "additivity_failed": 0,
            }
        ],
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
        "visualizations": {},
        "warnings": [],
    }


def _run_page(monkeypatch, response: dict[str, Any], tmp_path: Path) -> AppTest:
    client = FakeClient(response)
    monkeypatch.setattr(components, "require_login", lambda: (client, {"role": "researcher"}))
    monkeypatch.setattr(
        components,
        "load_diseases",
        lambda _client: [{"id": "diabetes", "name": "Diabetes"}],
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
    assert len(app.metric) == 3
    assert any("Common features: age, sex" in element.value for element in app.markdown)
    assert any("Seed: 42" in element.value for element in app.markdown)