"""Registration, OAuth2 password login with rate limiting, and logout."""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from xai_dashboard.api.deps import get_session, get_token_payload
from xai_dashboard.api.schemas import RegisterRequest, TokenResponse, UserProfile
from xai_dashboard.core.auth.rate_limit import login_limiter
from xai_dashboard.core.auth.security import create_access_token, verify_password
from xai_dashboard.services.auth import (
    AccountError,
    find_user_by_identifier,
    register_user,
    revoke_token,
)
from xai_dashboard.settings import Settings, get_settings

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_session)) -> UserProfile:
    try:
        user = register_user(
            db,
            payload.email,
            payload.full_name,
            payload.password,
            payload.role,
            payload.username,
        )
    except AccountError as exc:
        raise HTTPException(
            status_code=409, detail={"code": "registration_failed", "message": str(exc)}
        ) from exc
    return UserProfile.model_validate(user)


@router.post("/login", response_model=TokenResponse)
def login(
    request: Request,
    username: str = Form(),
    password: str = Form(),
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    client = request.client.host if request.client else "unknown"
    login_limiter.max_attempts = settings.login_max_attempts
    login_limiter.window_seconds = settings.login_window_seconds
    keys = (f"ip:{client}", f"id:{username.strip().lower()}")
    wait = max(login_limiter.retry_after(key) for key in keys)
    if wait:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "too_many_attempts",
                "message": f"Too many failed sign-in attempts. Try again in {wait} seconds.",
            },
            headers={"Retry-After": str(wait)},
        )
    user = find_user_by_identifier(db, username)
    if user is None or not verify_password(password, user.password_hash):
        for key in keys:
            login_limiter.record_failure(key)
        raise HTTPException(
            status_code=401,
            detail={
                "code": "invalid_credentials",
                "message": "Username/email or password is incorrect.",
            },
        )
    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "account_inactive",
                "message": "This account has been deactivated. Contact an administrator.",
            },
        )
    login_limiter.reset(f"id:{username.strip().lower()}")
    token = create_access_token(str(user.id), user.role, settings)
    return TokenResponse(access_token=token, user=UserProfile.model_validate(user))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    payload: dict[str, Any] = Depends(get_token_payload), db: Session = Depends(get_session)
) -> Response:
    revoke_token(db, str(payload["jti"]), datetime.fromtimestamp(int(payload["exp"]), tz=UTC))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
