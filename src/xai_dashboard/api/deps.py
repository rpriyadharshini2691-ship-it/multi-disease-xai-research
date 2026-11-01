"""FastAPI dependencies for settings, DB sessions, authentication, and RBAC."""

from collections.abc import Callable, Generator
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from xai_dashboard.core.auth.security import decode_access_token
from xai_dashboard.core.ml.registry import ModelRegistry
from xai_dashboard.db.models import User
from xai_dashboard.db.session import get_db
from xai_dashboard.services.auth import is_token_revoked
from xai_dashboard.settings import Settings, get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_session() -> Generator[Session, None, None]:
    yield from get_db()


def get_model_registry(settings: Settings = Depends(get_settings)) -> ModelRegistry:
    return ModelRegistry(settings)


def _unauthorized(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": code, "message": message},
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_token_payload(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    try:
        payload = decode_access_token(token, settings)
    except jwt.ExpiredSignatureError:
        raise _unauthorized(
            "token_expired", "Your session has expired. Please sign in again."
        ) from None
    except jwt.InvalidTokenError:
        raise _unauthorized("token_invalid", "The authentication token is invalid.") from None
    if is_token_revoked(db, str(payload["jti"])):
        raise _unauthorized("token_revoked", "This session was signed out. Please sign in again.")
    return payload


def get_current_user(
    payload: dict[str, Any] = Depends(get_token_payload),
    db: Session = Depends(get_session),
) -> User:
    try:
        user = db.scalar(select(User).where(User.id == int(payload["sub"])))
    except (KeyError, TypeError, ValueError):
        raise _unauthorized("token_invalid", "The authentication token is invalid.") from None
    if user is None:
        raise _unauthorized("authentication_required", "Valid authentication is required.")
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "account_inactive", "message": "This account has been deactivated."},
        )
    return user


def require_roles(*roles: str) -> Callable[..., User]:
    def role_dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=403, detail={"code": "forbidden", "message": "Insufficient role."}
            )
        return user

    return role_dependency
