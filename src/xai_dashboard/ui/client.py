"""Small typed HTTP client for the versioned API."""

import json
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from xai_dashboard.settings import get_settings


class ApiError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class ApiClient:
    def __init__(self, token: str | None = None) -> None:
        self.base_url = get_settings().api_base_url.rstrip("/")
        self.token = token

    def _request(
        self, method: str, path: str, payload: Any = None, form: dict[str, str] | None = None
    ) -> Any:
        headers = {"Accept": "application/json"}
        body: bytes | None = None
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if form is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            body = urlencode(form).encode("utf-8")
        elif payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")
        request = Request(f"{self.base_url}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                content = response.read()
                return json.loads(content) if content else None
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                details = json.loads(raw)
                message = details.get("error", {}).get("message", details.get("detail", raw))
            except json.JSONDecodeError:
                message = raw or "API request failed"
            raise ApiError(exc.code, str(message)) from exc
        except URLError as exc:
            raise ApiError(503, "API is unavailable. Start the FastAPI service and retry.") from exc

    def get(self, path: str) -> Any:
        return self._request("GET", path)

    def post(self, path: str, payload: Any = None) -> Any:
        return self._request("POST", path, payload)

    def patch(self, path: str, payload: Any = None) -> Any:
        return self._request("PATCH", path, payload)

    def login(self, identifier: str, password: str) -> Any:
        return self._request(
            "POST", "/auth/login", form={"username": identifier, "password": password}
        )

    def logout(self) -> None:
        self._request("POST", "/auth/logout")

    def download(self, path: str) -> bytes:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        request = Request(f"{self.base_url}{path}", headers=headers)
        try:
            with urlopen(request, timeout=30) as response:
                return cast(bytes, response.read())
        except (HTTPError, URLError) as exc:
            raise ApiError(getattr(exc, "code", 503), "Report is not available yet.") from exc
