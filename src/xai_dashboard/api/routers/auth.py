"""Registration and OAuth2 password login."""

from fastapi import APIRouter, Depends, Form, HTTPException, status
from sqlalchemy.orm import Session

from xai_dashboard.api.deps import get_session
from xai_dashboard.api.schemas import RegisterRequest, TokenResponse, UserProfile
from xai_dashboard.core.auth.security import create_access_token, verify_password
from xai_dashboard.db.models import User
from xai_dashboard.services.auth import register_user
from xai_dashboard.settings import Settings, get_settings

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_session)) -> UserProfile:
    try:
        user = register_user(db, payload.email, payload.full_name, payload.password, payload.role)
        return UserProfile.model_validate(user)
    except ValueError as exc:
        raise HTTPException(
            status_code=409, detail={"code": "registration_failed", "message": str(exc)}
        ) from exc


@router.post("/login", response_model=TokenResponse)
def login(
    username: str = Form(),
    password: str = Form(),
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    from sqlalchemy import select

    user = db.scalar(select(User).where(User.email == username.strip().lower()))
    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail={"code": "invalid_credentials", "message": "Email or password is incorrect."},
        )
    token = create_access_token(str(user.id), user.role, settings)
    return TokenResponse(access_token=token, user=UserProfile.model_validate(user))
