import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from streamlit.testing.v1 import AppTest

from xai_dashboard.ui import components


def _png_data() -> str:
    image = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(image, format="PNG")
    return base64.b64encode(image.getvalue()).decode("ascii")


class ExplainClient:
    def __init__(self, fail_global: bool = False) -> None:
        self.fail_global = fail_global

    def get(self, path: str) -> dict[str, Any]:
        if self.fail_global:
            from xai_dashboard.ui.client import ApiError

            raise ApiError(503, "SHAP artifact is unavailable")
        return {
            "status": "ready",
            "model_name": "weighted_ensemble",
            "features": [{"feature": "numeric__age", "mean_abs_shap": 0.25}],
            "beeswarm_png_base64": _png_data(),
        }

    def post(self, _path: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "positive_class_score": 0.72,
            "disclaimer": "Research estimate, not diagnosis.",
            "additivity_sum": 0.72,
            "raw_model_score": 0.72,
            "additivity_error": 0.0,
            "additivity_passed": True,
            "score_match_passed": True,
            "waterfall_png_base64": _png_data(),
            "features": [
                {
                    "feature": "numeric__age",
                    "value": 50.0,
                    "shap_value": 0.3,
                    "direction": "increases",
                }
            ],
        }


def _run_page(monkeypatch, client: ExplainClient) -> AppTest:
    disease = {
        "id": "diabetes",
        "name": "Diabetes",
        "features": [
            {
                "id": "age",
                "name": "Age",
                "kind": "numeric",
                "valid_range": [1, 120],
            }
        ],
    }
    monkeypatch.setattr(components, "require_login", lambda: (client, {"role": "researcher"}))
    monkeypatch.setattr(components, "load_diseases", lambda _client: [disease])
    path = (
        Path(__file__).parents[1]
        / "src"
        / "xai_dashboard"
        / "ui"
        / "pages"
        / "3_Explainability.py"
    )
    return AppTest.from_file(str(path)).run()


def test_explainability_page_renders_local_and_global_shap(monkeypatch) -> None:
    app = _run_page(monkeypatch, ExplainClient())
    assert not app.exception
    assert len(app.get("imgs")) == 1

    app.number_input[0].set_value(50)
    submit = next(button for button in app.button if button.label == "Generate local explanation")
    app = submit.click().run()

    assert not app.exception
    assert len(app.get("imgs")) == 2
    assert len(app.dataframe) >= 2
    assert any("Top feature contributions" in item.value for item in app.subheader)


def test_explainability_page_handles_missing_global_artifacts(monkeypatch) -> None:
    app = _run_page(monkeypatch, ExplainClient(fail_global=True))
    assert not app.exception
    assert any("Global explanation unavailable" in item.value for item in app.error)