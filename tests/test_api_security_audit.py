import asyncio
import json

from fastapi.exceptions import RequestValidationError

from xai_dashboard.api.main import validation_error_handler


def test_validation_error_response_never_echoes_submitted_input() -> None:
    sentinel = "PRIVATE-HEALTH-INPUT-DO-NOT-RETURN"
    error = RequestValidationError(
        [
            {
                "type": "string_type",
                "loc": ("body", "password"),
                "msg": "Input should be a valid string",
                "input": {"password": sentinel},
                "ctx": {"submitted": sentinel},
            }
        ]
    )

    response = asyncio.run(validation_error_handler(None, error))
    payload = json.loads(response.body)

    assert response.status_code == 422
    assert sentinel not in response.body.decode("utf-8")
    assert payload["error"]["details"] == [
        {
            "type": "string_type",
            "loc": ["body", "password"],
            "msg": "Input should be a valid string",
        }
    ]