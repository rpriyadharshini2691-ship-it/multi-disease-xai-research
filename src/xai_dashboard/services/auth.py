"""User account operations."""

import re
from datetime import UTC, datetime

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from xai_dashboard.core.auth.security import PUBLIC_ROLES, ROLES, hash_password
from xai_dashboard.db.models import RevokedToken, User

USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,49}$")


class AccountError(ValueError):
    """Business-rule violation on a user account; message is safe to show to clients."""


def normalize_username(username: str) -> str:
    normalized = username.strip().lower()
    if not USERNAME_PATTERN.fullmatch(normalized):
        raise AccountError(
            "Username must be 3-50 characters: lowercase letters, digits, '_', '.', '-'"
        )
    return normalized


def register_user(
    db: Session,
    email: str,
    full_name: str,
    password: str,
    role: str,
    username: str,
    allow_admin: bool = False,
) -> User:
    """Create an account. Admin creation is only reachable from trusted scripts."""
    if role not in PUBLIC_ROLES and not (allow_admin and role == "admin"):
        raise AccountError("Public registration is limited to researcher and clinician roles")
    normalized_email = email.strip().lower()
    normalized_username = normalize_username(username)
    if db.scalar(select(User).where(User.email == normalized_email)):
        raise AccountError("An account with this email already exists")
    if db.scalar(select(User).where(User.username == normalized_username)):
        raise AccountError("This username is already taken")
    user = User(
        username=normalized_username,
        email=normalized_email,
        full_name=full_name.strip(),
        password_hash=hash_password(password),
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def find_user_by_identifier(db: Session, identifier: str) -> User | None:
    value = identifier.strip().lower()
    return db.scalar(select(User).where(or_(User.email == value, User.username == value)))


def _active_admin_count(db: Session) -> int:
    return int(
        db.scalar(
            select(func.count()).select_from(User).where(User.role == "admin", User.is_active)
        )
        or 0
    )


def set_user_active(db: Session, actor: User, target: User, active: bool) -> User:
    if not active:
        if target.id == actor.id:
            raise AccountError("Administrators cannot deactivate their own account")
        if target.role == "admin" and target.is_active and _active_admin_count(db) <= 1:
            raise AccountError("The last active administrator cannot be deactivated")
    target.is_active = active
    db.commit()
    db.refresh(target)
    return target


def set_user_role(db: Session, actor: User, target: User, role: str) -> User:
    if role not in ROLES:
        raise AccountError(f"Role must be one of: {', '.join(sorted(ROLES))}")
    if target.role == "admin" and role != "admin":
        if target.id == actor.id:
            raise AccountError("Administrators cannot remove their own admin role")
        if target.is_active and _active_admin_count(db) <= 1:
            raise AccountError("The last active administrator cannot lose the admin role")
    target.role = role
    db.commit()
    db.refresh(target)
    return target


def revoke_token(db: Session, jti: str, expires_at: datetime) -> None:
    if db.get(RevokedToken, jti) is None:
        db.add(RevokedToken(jti=jti, expires_at=expires_at))
    now = datetime.now(UTC)
    for stale in db.scalars(select(RevokedToken).where(RevokedToken.expires_at < now)):
        db.delete(stale)
    db.commit()


def is_token_revoked(db: Session, jti: str) -> bool:
    return db.get(RevokedToken, jti) is not None
