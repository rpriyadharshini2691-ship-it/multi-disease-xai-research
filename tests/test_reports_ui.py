from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from xai_dashboard.ui import components
from xai_dashboard.ui.client import ApiError


class ReportClient:
    def __init__(self, missing: bool) -> None:
        self.missing = missing

    def download(self, path: str) -> bytes:
        if self.missing:
            raise ApiError(404, "not generated")
        return b"%PDF-test"


def _run_page(monkeypatch, missing: bool) -> AppTest:
    client = ReportClient(missing)
    monkeypatch.setattr(components, "require_login", lambda: (client, {"role": "researcher"}))
    monkeypatch.setattr(
        components,
        "load_diseases",
        lambda _client: [{"id": "diabetes", "name": "Diabetes"}],
    )
    monkeypatch.setattr(st, "page_link", lambda *_args, **_kwargs: None)
    page = Path(__file__).parents[1] / "src" / "xai_dashboard" / "ui" / "pages" / "6_Reports.py"
    return AppTest.from_file(str(page)).run()


def test_reports_page_handles_missing_artifacts(monkeypatch) -> None:
    app = _run_page(monkeypatch, missing=True)
    evaluation_button = next(
        button for button in app.button if button.label == "Prepare evaluation PDF"
    )
    app = evaluation_button.click().run()
    assert not app.exception
    assert any("Not yet generated" in item.value for item in app.info)
    research_button = next(
        button for button in app.button if button.label == "Prepare research report PDF"
    )
    app = research_button.click().run()
    assert not app.exception
    assert any("Not yet generated" in item.value for item in app.info)


def test_reports_page_offers_downloads_for_generated_pdfs(monkeypatch) -> None:
    app = _run_page(monkeypatch, missing=False)
    evaluation_button = next(
        button for button in app.button if button.label == "Prepare evaluation PDF"
    )
    app = evaluation_button.click().run()
    assert not app.exception
    assert any(button.label == "Download evaluation PDF" for button in app.get("download_button"))

    research_button = next(
        button for button in app.button if button.label == "Prepare research report PDF"
    )
    app = research_button.click().run()
    assert not app.exception
    assert any(button.label == "Download research PDF" for button in app.get("download_button"))