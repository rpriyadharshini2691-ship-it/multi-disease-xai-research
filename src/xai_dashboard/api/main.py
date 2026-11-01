"""FastAPI application entry point."""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from xai_dashboard.api.routers import (
    auth,
    diseases,
    explain,
    health,
    jobs,
    performance,
    predictions,
    reliability,
    reports,
    users,
)
from xai_dashboard.db import models  # noqa: F401
from xai_dashboard.db.session import Base, engine
from xai_dashboard.settings import get_settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        for field in ("request_id", "path", "method", "status"):
            if hasattr(record, field):
                entry[field] = getattr(record, field)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
for handler in logging.getLogger().handlers:
    handler.setFormatter(JsonFormatter())
logger = logging.getLogger("xai_dashboard.api")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.environment.lower() == "development":
        Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Multi-Disease XAI Research API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Any:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:100]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_complete",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status": response.status_code,
        },
    )
    return response


def error_payload(code: str, message: str, details: Any = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details}}


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        body = error_payload(
            str(detail["code"]),
            str(detail.get("message", "Request failed.")),
            detail.get("details"),
        )
    else:
        body = error_payload("http_error", str(detail), None)
    return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {
            key: error[key]
            for key in ("loc", "msg", "type")
            if key in error
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=error_payload("validation_error", "Request validation failed.", details),
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "unhandled_request_error", extra={"request_id": getattr(request.state, "request_id", None)}
    )
    return JSONResponse(
        status_code=500, content=error_payload("internal_error", "An unexpected error occurred.")
    )


for route in (
    health.router,
    auth.router,
    users.router,
    diseases.router,
    predictions.router,
    explain.router,
    performance.router,
    reliability.router,
    reports.router,
    jobs.router,
):
    app.include_router(route, prefix="/api/v1")
