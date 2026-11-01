from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

from xai_dashboard.ui import components


class UserClient:
    def get(self, _path: str) -> list[dict[str, Any]]:
        return [
            {
                "id": 1,
                "username": "admin",
                "email": "admin@example.com",
                "full_name": "Admin User",
                "role": "admin",
                "is_active": True,
                "created_at": "2026-01-01T00:00:00Z",
                "prediction_count": 0,
                "activity": [],
            }
        ]

    def patch(self, _path: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {}


def _run_page(monkeypatch, role: str) -> AppTest:
    client = UserClient()
    user = {
        "id": 1,
        "username": "test-user",
        "email": "test@example.com",
        "full_name": "Test User",
        "role": role,
        "is_active": True,
        "created_at": "2026-01-01T00:00:00Z",
    }
    monkeypatch.setattr(components, "require_login", lambda: (client, user))
    page = (
        Path(__file__).parents[1]
        / "src"
        / "xai_dashboard"
        / "ui"
        / "pages"
        / "7_User_Management.py"
    )
    return AppTest.from_file(str(page)).run()


def test_user_management_hides_admin_controls_for_researcher(monkeypatch) -> None:
    app = _run_page(monkeypatch, "researcher")
    assert not app.exception
    assert any("restricted to Admin users" in item.value for item in app.info)
    assert not any(button.label == "Apply role change" for button in app.button)


def test_user_management_shows_admin_controls_for_admin(monkeypatch) -> None:
    app = _run_page(monkeypatch, "admin")
    assert not app.exception
    assert any(item.label == "Apply role change" for item in app.button)