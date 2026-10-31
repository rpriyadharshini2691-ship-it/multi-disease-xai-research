"""FastAPI dependencies for settings, DB sessions, authentication, and RBAC."""

from collections.abc import Callable, Generator

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from xai_dashboard.core.auth.security import decode_access_token
from xai_dashboard.core.ml.registry import ModelRegistry
from xai_dashboard.db.models import User
from xai_dashboard.db.session import get_db
from xai_dashboard.settings import Settings, get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_session() -> Generator[Session, None, None]:
    yield from get_db()


def get_model_registry(settings: Settings = Depends(get_settings)) -> ModelRegistry:
    return ModelRegistry(settings)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": "authentication_required", "message": "Valid authentication is required."},
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token, settings)
        user = db.scalar(select(User).where(User.id == int(payload["sub"])))
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError):
        raise unauthorized from None
    if user is None:
        raise unauthorized
    return user


def require_roles(*roles: str) -> Callable[..., User]:
    def role_dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=403, detail={"code": "forbidden", "message": "Insufficient role."}
            )
        return user

    return role_dependency
