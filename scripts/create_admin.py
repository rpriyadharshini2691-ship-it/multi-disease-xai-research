"""Create the initial administrator through an interactive local setup command."""

from getpass import getpass

from xai_dashboard.db.session import Base, SessionLocal, engine
from xai_dashboard.services.auth import AccountError, register_user


def main() -> None:
    Base.metadata.create_all(bind=engine)
    email = input("Admin email: ").strip().lower()
    username = input("Admin username: ").strip().lower()
    full_name = input("Admin full name: ").strip()
    password = getpass("Admin password (10+ characters): ")
    with SessionLocal() as db:
        try:
            register_user(db, email, full_name, password, "admin", username, allow_admin=True)
        except (AccountError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
    print("Administrator account created.")


if __name__ == "__main__":
    main()
