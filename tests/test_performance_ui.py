import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from streamlit.testing.v1 import AppTest

from xai_dashboard.ui import components
from xai_dashboard.ui.client import ApiError


def _png() -> str:
    buffer = BytesIO()
    Image.new("RGB", (6, 6), color="white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class PerformanceClient:
    def __init__(self, response: dict[str, Any] | None = None, fail: bool = False) -> None:
        self.response = response or {"status": "not_generated"}
        self.fail = fail

    def get(self, _path: str) -> dict[str, Any]:
        if self.fail:
            raise ApiError(503, "Performance service unavailable")
        return self.response


def _run_page(monkeypatch, client: PerformanceClient) -> AppTest:
    monkeypatch.setattr(components, "require_login", lambda: (client, {"role": "researcher"}))
    monkeypatch.setattr(
        components,
        "load_diseases",
        lambda _client: [{"id": "diabetes", "name": "Diabetes"}],
    )
    page = Path(__file__).parents[1] / "src" / "xai_dashboard" / "ui" / "pages" / "4_Performance.py"
    return AppTest.from_file(str(page)).run()


def _ready_response() -> dict[str, Any]:
    metrics = {
        "accuracy": 0.9,
        "precision": 0.8,
        "sensitivity": 0.75,
        "specificity": 0.95,
        "f1": 0.77,
        "roc_auc": 0.96,
        "pr_auc": 0.94,
        "brier_score": 0.12,
    }
    return {
        "status": "ready",
        "artifact": {
            "dataset_path": "data/processed/diabetes.csv",
            "generated_at": "2026-11-01T00:00:00Z",
            "dataset_rows": 251,
            "dataset_sha256": "a" * 64,
            "seed": 42,
            "test_size": 0.2,
            "holdout_metrics": {"weighted_ensemble": metrics},
            "cv": {
                "repeats": 3,
                "splits": 5,
                "interval_method": "bootstrap",
                "training_metrics": {
                    "weighted_ensemble": {
                        "roc_auc": {
                            "mean": 0.91,
                            "sd": 0.02,
                            "bootstrap_ci_95": [0.87, 0.95],
                            "folds": 15,
                        }
                    }
                },
            },
        },
        "plots": [
            {"filename": "model_comparison_test.png", "png_base64": _png()},
            {"filename": "evaluation_weighted_ensemble_test.png", "png_base64": _png()},
        ],
    }


def test_performance_page_shows_not_yet_generated(monkeypatch) -> None:
    app = _run_page(monkeypatch, PerformanceClient())
    assert not app.exception
    assert any("Not yet generated" in item.value for item in app.info)


def test_performance_page_renders_metrics_ci_and_stored_plots(monkeypatch) -> None:
    app = _run_page(monkeypatch, PerformanceClient(_ready_response()))
    assert not app.exception
    assert len(app.dataframe) == 2
    assert len(app.get("imgs")) == 2
    assert any("Best by held-out ROC-AUC" in item.value for item in app.success)
    assert any("95% bootstrap CI" in str(frame.value) for frame in app.dataframe)


def test_performance_page_surfaces_api_failure(monkeypatch) -> None:
    app = _run_page(monkeypatch, PerformanceClient(fail=True))
    assert not app.exception
    assert any("Performance service unavailable" in item.value for item in app.error)