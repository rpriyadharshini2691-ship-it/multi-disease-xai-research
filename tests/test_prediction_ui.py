from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

from xai_dashboard.ui import components


class FakePredictionClient:
    def __init__(self) -> None:
        self.prediction_payload: dict[str, Any] | None = None
        self.report_payload: dict[str, Any] | None = None

    def post(self, _path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.prediction_payload = payload
        return {
            "predicted_class": "positive",
            "positive_class_score": 0.7,
            "disclaimer": "Research score only.",
            "stored": payload["consent"],
            "model_name": payload["model_name"],
            "model_version": "test-version",
            "prediction_reference": "PRED-UI123456",
            "additivity_passed": True,
            "important_features": [
                {"feature": "Age", "value": 50, "shap_value": 0.1, "direction": "increases"}
            ],
        }

    def post_download(
        self, _path: str, payload: dict[str, Any]
    ) -> tuple[bytes, str]:
        self.report_payload = payload
        return b"%PDF-test", "PRED-UI123456"


def test_prediction_page_requests_pdf_without_repeating_consent(
    monkeypatch,
) -> None:
    client = FakePredictionClient()
    monkeypatch.setattr(components, "require_login", lambda: (client, {"role": "researcher"}))
    monkeypatch.setattr(
        components,
        "load_diseases",
        lambda _client: [
            {
                "id": "diabetes",
                "name": "Diabetes",
                "features": [
                    {
                        "id": "age",
                        "name": "Age",
                        "kind": "numeric",
                        "valid_range": [1, 120],
                    },
                    {
                        "id": "gender",
                        "name": "Gender",
                        "kind": "categorical",
                        "options": ["Female", "Male"],
                    },
                ],
            }
        ],
    )
    page = Path(__file__).parents[1] / "src" / "xai_dashboard" / "ui" / "pages" / "2_Prediction.py"
    app = AppTest.from_file(str(page)).run()
    app.number_input[0].set_value(50)
    submit = next(
        button for button in app.button if button.label == "Estimate model score"
    )
    submit.click().run(timeout=10)
    assert not app.exception
    assert any(
        metric.label == "Model prediction" and metric.value == "Positive"
        for metric in app.metric
    )
    assert client.prediction_payload is not None
    assert client.prediction_payload["consent"] is False
    assert client.prediction_payload["features"]["gender"] == "Female"

    next(button for button in app.button if button.label == "Generate prediction PDF").click().run(
        timeout=10
    )

    assert not app.exception
    assert any(
        metric.label == "Model prediction" and metric.value == "Positive"
        for metric in app.metric
    )
    assert client.report_payload is not None
    assert client.report_payload["consent"] is False
    assert len(app.get("download_button")) == 1