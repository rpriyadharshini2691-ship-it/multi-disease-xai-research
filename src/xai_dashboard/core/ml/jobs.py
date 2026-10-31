"""Small in-process threadpool job registry for local research workloads."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="xai-job")
_lock = Lock()
_jobs: dict[str, dict[str, Any]] = {}


def submit_job(kind: str, operation: Callable[[], Any]) -> str:
    job_id = str(uuid4())
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": datetime.now(UTC).isoformat(),
        }

    def run() -> Any:
        with _lock:
            _jobs[job_id]["status"] = "running"
        try:
            result = operation()
        except Exception as exc:
            with _lock:
                _jobs[job_id].update(
                    status="failed", error=str(exc), completed_at=datetime.now(UTC).isoformat()
                )
            raise
        with _lock:
            _jobs[job_id].update(
                status="completed", result=result, completed_at=datetime.now(UTC).isoformat()
            )
        return result

    future: Future[Any] = _executor.submit(run)
    future.add_done_callback(
        lambda completed: completed.exception() if not completed.cancelled() else None
    )
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        value = _jobs.get(job_id)
        return dict(value) if value else None
