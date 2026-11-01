"""Password hashing, access-token issuance, and role names."""

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

import bcrypt
import jwt

from xai_dashboard.settings import Settings

ROLES = frozenset({"admin", "researcher", "clinician"})
PUBLIC_ROLES = frozenset({"researcher", "clinician"})
TOKEN_ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    if len(password) < 10:
        raise ValueError("Password must contain at least 10 characters")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str, role: str, settings: Settings) -> str:
    expires = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "exp": expires,
        "iat": datetime.now(UTC),
        "jti": uuid4().hex,
    }
    return jwt.encode(payload, settings.secret_key.get_secret_value(), algorithm=TOKEN_ALGORITHM)


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    payload = jwt.decode(
        token,
        settings.secret_key.get_secret_value(),
        algorithms=[TOKEN_ALGORITHM],
        options={"require": ["exp", "sub", "jti"]},
    )
    if not payload.get("sub"):
        raise jwt.InvalidTokenError("Token subject is missing")
    return cast(dict[str, Any], payload)
