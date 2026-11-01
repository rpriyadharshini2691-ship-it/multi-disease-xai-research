from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from xai_dashboard.ui import client as client_module
from xai_dashboard.ui.client import ApiError


class DashboardClient:
    def __init__(self, _token: str | None = None) -> None:
        pass

    def get(self, path: str) -> Any:
        if path == "/diseases":
            return [
                {"id": "diabetes", "name": "Diabetes", "features": [{"id": "age"}]},
                {"id": "heart", "name": "Heart Disease", "features": [{"id": "age"}]},
                {"id": "ckd", "name": "CKD", "features": [{"id": "age"}]},
            ]
        if path.endswith("/diabetes"):
            return {
                "status": "ready",
                "artifact": {
                    "holdout_metrics": {
                        "weighted_ensemble": {
                            "accuracy": 0.9,
                            "precision": 0.8,
                            "sensitivity": 0.75,
                            "specificity": 0.95,
                            "f1": 0.77,
                            "roc_auc": 0.96,
                        },
                        "random_forest": {
                            "accuracy": 0.8,
                            "precision": 0.7,
                            "sensitivity": 0.65,
                            "specificity": 0.85,
                            "f1": 0.67,
                            "roc_auc": 0.88,
                        },
                    },
                    "dataset_rows": 251,
                    "dataset_sha256": "a" * 64,
                    "seed": 42,
                    "test_size": 0.2,
                    "cv": {"repeats": 3, "splits": 5, "training_metrics": {}},
                },
            }
        if path.endswith("/heart"):
            return {"status": "not_generated"}
        raise ApiError(503, "Performance API unavailable")


@pytest.mark.parametrize("role, has_admin_link", [("researcher", False), ("admin", True)])
def test_dashboard_renders_actual_metrics_and_graceful_empty_states(
    monkeypatch: pytest.MonkeyPatch, role: str, has_admin_link: bool
) -> None:
    monkeypatch.setattr(client_module, "ApiClient", DashboardClient)
    navigation_targets: list[str] = []
    monkeypatch.setattr(
        st,
        "page_link",
        lambda page, **_kwargs: navigation_targets.append(str(page)),
    )
    page = Path(__file__).parents[1] / "src" / "xai_dashboard" / "ui" / "Home.py"
    app = AppTest.from_file(str(page))
    app.session_state["access_token"] = "test-token"
    app.session_state["user"] = {"full_name": "Test User", "role": role}
    app = app.run()

    assert not app.exception
    assert len(app.metric) >= 6
    assert any(
        "Best holdout ROC-AUC: Weighted Ensemble (0.960)" in item.value
        for item in app.caption
    )
    assert any("Not yet generated" in item.value for item in app.info)
    assert any("Performance results unavailable" in item.value for item in app.error)
    assert "Home.py" in navigation_targets
    assert "pages/2_Prediction.py" in navigation_targets
    assert "pages/3_Explainability.py" in navigation_targets
    assert "pages/4_Performance.py" in navigation_targets
    assert "pages/5_Reliability.py" in navigation_targets
    assert "pages/6_Reports.py" in navigation_targets
    assert ("pages/7_User_Management.py" in navigation_targets) is has_admin_link