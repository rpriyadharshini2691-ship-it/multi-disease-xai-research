"""Create the initial administrator through an interactive local setup command."""

from getpass import getpass

from sqlalchemy import select

from xai_dashboard.core.auth.security import hash_password
from xai_dashboard.db.models import User
from xai_dashboard.db.session import Base, SessionLocal, engine


def main() -> None:
    Base.metadata.create_all(bind=engine)
    email = input("Admin email: ").strip().lower()
    full_name = input("Admin full name: ").strip()
    password = getpass("Admin password (10+ characters): ")
    with SessionLocal() as db:
        if db.scalar(select(User).where(User.email == email)):
            raise SystemExit("An account with that email already exists.")
        user = User(
            email=email, full_name=full_name, password_hash=hash_password(password), role="admin"
        )
        db.add(user)
        db.commit()
    print("Administrator account created.")


if __name__ == "__main__":
    main()
