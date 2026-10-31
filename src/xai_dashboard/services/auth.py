"""User account operations."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from xai_dashboard.core.auth.security import PUBLIC_ROLES, hash_password
from xai_dashboard.db.models import User


def register_user(db: Session, email: str, full_name: str, password: str, role: str) -> User:
    if role not in PUBLIC_ROLES:
        raise ValueError("Public registration is limited to researcher and clinician roles")
    normalized_email = email.strip().lower()
    if db.scalar(select(User).where(User.email == normalized_email)):
        raise ValueError("An account with this email already exists")
    user = User(
        email=normalized_email,
        full_name=full_name.strip(),
        password_hash=hash_password(password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
