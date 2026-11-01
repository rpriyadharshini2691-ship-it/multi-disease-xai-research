"""Phase 1: admin user management, account status, logout/revocation and login limiting."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jwt
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xai_dashboard.api import deps, main
from xai_dashboard.core.auth.rate_limit import login_limiter
from xai_dashboard.core.auth.security import hash_password
from xai_dashboard.db.models import Base, User
from xai_dashboard.settings import Settings, get_settings

PASSWORD = "valid-password-123"
SECRET = "a-test-secret-key-that-is-at-least-thirty-two-characters"


class Env:
    def __init__(self, client: TestClient, factory: sessionmaker[Session], settings: Settings):
        self.client = client
        self.factory = factory
        self.settings = settings

    def add_user(self, username: str, role: str, active: bool = True) -> int:
        with self.factory() as db:
            user = User(
                username=username,
                email=f"{username}@example.com",
                full_name=username.title(),
                password_hash=hash_password(PASSWORD),
                role=role,
                is_active=active,
            )
            db.add(user)
            db.commit()
            return user.id

    def login(self, identifier: str) -> dict[str, str]:
        response = self.client.post(
            "/api/v1/auth/login", data={"username": identifier, "password": PASSWORD}
        )
        assert response.status_code == 200, response.text
        return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    settings = Settings(
        environment="testing",
        secret_key=SECRET,
        configs_dir=Path("configs").resolve(),
        artifacts_dir=tmp_path,
    )
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_session() -> Iterator[Session]:
        with factory() as session:
            yield session

    main.app.dependency_overrides[deps.get_session] = override_session
    main.app.dependency_overrides[deps.get_settings] = lambda: settings
    login_limiter.reset()
    client = TestClient(main.app)
    try:
        yield Env(client, factory, settings)
    finally:
        main.app.dependency_overrides.clear()
        login_limiter.reset()
        client.close()
        engine.dispose()


def test_admin_lists_users_with_required_fields(env: Env) -> None:
    env.add_user("boss", "admin")
    env.add_user("res", "researcher")
    response = env.client.get("/api/v1/users", headers=env.login("boss"))
    assert response.status_code == 200
    rows = {row["username"]: row for row in response.json()}
    assert set(rows) == {"boss", "res"}
    for field in ("email", "role", "is_active", "created_at"):
        assert field in rows["res"]


def test_user_management_migration_preserves_and_backfills_existing_users(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'legacy-users.db').as_posix()}"
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE users ("
                    "id INTEGER PRIMARY KEY, email VARCHAR(320) NOT NULL, "
                    "username VARCHAR(50), full_name VARCHAR(160) NOT NULL, "
                    "password_hash VARCHAR(255) NOT NULL, role VARCHAR(20) NOT NULL, "
                    "created_at DATETIME NOT NULL)"
                )
            )
            connection.execute(
                text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
            )
            connection.execute(
                text("INSERT INTO alembic_version (version_num) VALUES ('0002_phase1_foundation')")
            )
            connection.exec_driver_sql(
                "INSERT INTO users "
                "(id, email, username, full_name, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        1,
                        "alice@example.com",
                        "alice",
                        "Alice",
                        "hash",
                        "admin",
                        "2026-01-01",
                    ),
                    (
                        2,
                        "alice@example.com",
                        None,
                        "Alice Two",
                        "hash",
                        "researcher",
                        "2026-01-02",
                    ),
                    (
                        3,
                        "alice@example.com",
                        "alice_2",
                        "Alice Three",
                        "hash",
                        "researcher",
                        "2026-01-03",
                    ),
                    (
                        4,
                        "alice@example.com",
                        "alice_2_2",
                        "Alice Four",
                        "hash",
                        "clinician",
                        "2026-01-04",
                    ),
                    (
                        5,
                        "_alice@example.com",
                        None,
                        "Alice Five",
                        "hash",
                        "researcher",
                        "2026-01-05",
                    ),
                ],
            )

        monkeypatch.setenv("DATABASE_URL", database_url)
        get_settings.cache_clear()
        config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
        command.upgrade(config, "head")
        get_settings.cache_clear()

        with engine.connect() as connection:
            rows = connection.execute(
                text("SELECT id, email, username, is_active FROM users ORDER BY id")
            ).mappings().all()
            assert [row["id"] for row in rows] == [1, 2, 3, 4, 5]
            assert rows[1]["username"] == "alice_2_3"
            assert rows[4]["username"] == "alice_5"
            assert all(row["is_active"] for row in rows)
            assert connection.execute(
                text("SELECT COUNT(*) FROM revoked_tokens")
            ).scalar_one() == 0
    finally:
        get_settings.cache_clear()
        engine.dispose()


def test_admin_changes_role_and_rejects_invalid_role(env: Env) -> None:
    env.add_user("boss", "admin")
    target = env.add_user("res", "researcher")
    headers = env.login("boss")
    ok = env.client.patch(
        f"/api/v1/users/{target}/role", json={"role": "clinician"}, headers=headers
    )
    assert ok.status_code == 200 and ok.json()["role"] == "clinician"
    bad = env.client.patch(f"/api/v1/users/{target}/role", json={"role": "root"}, headers=headers)
    assert bad.status_code == 409
    missing = env.client.patch("/api/v1/users/9999/role", json={"role": "admin"}, headers=headers)
    assert missing.status_code == 404


def test_deactivate_activate_and_inactive_login_rejected(env: Env) -> None:
    env.add_user("boss", "admin")
    target = env.add_user("res", "researcher")
    headers = env.login("boss")
    off = env.client.patch(
        f"/api/v1/users/{target}/status", json={"is_active": False}, headers=headers
    )
    assert off.status_code == 200 and off.json()["is_active"] is False
    denied = env.client.post(
        "/api/v1/auth/login", data={"username": "res", "password": PASSWORD}
    )
    assert denied.status_code == 403
    assert denied.json()["error"]["code"] == "account_inactive"
    on = env.client.patch(
        f"/api/v1/users/{target}/status", json={"is_active": True}, headers=headers
    )
    assert on.json()["is_active"] is True
    env.login("res")


def test_existing_token_of_deactivated_user_is_rejected(env: Env) -> None:
    env.add_user("boss", "admin")
    target = env.add_user("res", "researcher")
    user_headers = env.login("res")
    env.client.patch(
        f"/api/v1/users/{target}/status", json={"is_active": False}, headers=env.login("boss")
    )
    assert env.client.get("/api/v1/users/me", headers=user_headers).status_code == 403


@pytest.mark.parametrize("role", ["researcher", "clinician"])
def test_non_admins_receive_403_on_admin_endpoints(env: Env, role: str) -> None:
    env.add_user("boss", "admin")
    other = env.add_user("someone", role)
    headers = env.login("someone")
    assert env.client.get("/api/v1/users", headers=headers).status_code == 403
    assert (
        env.client.patch(
            f"/api/v1/users/{other}/role", json={"role": "admin"}, headers=headers
        ).status_code
        == 403
    )
    assert (
        env.client.patch(
            f"/api/v1/users/{other}/status", json={"is_active": False}, headers=headers
        ).status_code
        == 403
    )


def test_public_registration_cannot_create_admin_and_rejects_duplicates(env: Env) -> None:
    payload = {
        "username": "newuser",
        "email": "new@example.com",
        "full_name": "New",
        "password": PASSWORD,
        "role": "admin",
    }
    assert env.client.post("/api/v1/auth/register", json=payload).status_code == 422
    payload["role"] = "clinician"
    assert env.client.post("/api/v1/auth/register", json=payload).status_code == 201
    dup_username = {**payload, "email": "other@example.com"}
    assert env.client.post("/api/v1/auth/register", json=dup_username).status_code == 409
    dup_email = {**payload, "username": "another"}
    assert env.client.post("/api/v1/auth/register", json=dup_email).status_code == 409
    env.login("newuser")
    env.login("new@example.com")


def test_last_active_admin_and_self_protection(env: Env) -> None:
    admin_id = env.add_user("boss", "admin")
    headers = env.login("boss")
    for body, path in (
        ({"is_active": False}, "status"),
        ({"role": "researcher"}, "role"),
    ):
        response = env.client.patch(f"/api/v1/users/{admin_id}/{path}", json=body, headers=headers)
        assert response.status_code == 409


def test_cannot_remove_last_active_admin_even_via_another_inactive_admin(env: Env) -> None:
    env.add_user("boss", "admin")
    env.add_user("sleepy", "admin", active=False)
    other = env.add_user("second", "admin")
    headers = env.login("second")
    boss_id = 1
    # second may demote boss only while another active admin (second) remains
    assert (
        env.client.patch(
            f"/api/v1/users/{boss_id}/role", json={"role": "researcher"}, headers=headers
        ).status_code
        == 200
    )
    assert (
        env.client.patch(
            f"/api/v1/users/{other}/status", json={"is_active": False}, headers=headers
        ).status_code
        == 409
    )


def test_logout_revokes_token(env: Env) -> None:
    env.add_user("res", "researcher")
    headers = env.login("res")
    assert env.client.get("/api/v1/users/me", headers=headers).status_code == 200
    assert env.client.post("/api/v1/auth/logout", headers=headers).status_code == 204
    revoked = env.client.get("/api/v1/users/me", headers=headers)
    assert revoked.status_code == 401
    assert revoked.json()["error"]["code"] == "token_revoked"


def test_expired_and_malformed_tokens_have_clear_codes(env: Env) -> None:
    uid = env.add_user("res", "researcher")
    expired = jwt.encode(
        {
            "sub": str(uid),
            "jti": "x1",
            "exp": datetime.now(UTC) - timedelta(minutes=1),
        },
        SECRET,
        algorithm="HS256",
    )
    response = env.client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {expired}"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "token_expired"
    no_exp = jwt.encode({"sub": str(uid), "jti": "x2"}, SECRET, algorithm="HS256")
    response = env.client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {no_exp}"})
    assert response.json()["error"]["code"] == "token_invalid"
    response = env.client.get("/api/v1/users/me", headers={"Authorization": "Bearer garbage"})
    assert response.json()["error"]["code"] == "token_invalid"


def test_login_rate_limiting(env: Env) -> None:
    env.add_user("res", "researcher")
    for _ in range(env.settings.login_max_attempts):
        bad = env.client.post("/api/v1/auth/login", data={"username": "res", "password": "wrong"})
        assert bad.status_code == 401
    blocked = env.client.post(
        "/api/v1/auth/login", data={"username": "res", "password": PASSWORD}
    )
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0
    assert blocked.json()["error"]["code"] == "too_many_attempts"


def test_production_rejects_default_or_short_secret() -> None:
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings(
            environment="production", secret_key="development-only-change-before-deploy"
        ).validate_deployment_secret()
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings(environment="production", secret_key="short").validate_deployment_secret()
    Settings(environment="production", secret_key="x" * 40).validate_deployment_secret()
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings(
            environment="prod",
            secret_key="development-only-change-before-deploy",
        ).validate_deployment_secret()
