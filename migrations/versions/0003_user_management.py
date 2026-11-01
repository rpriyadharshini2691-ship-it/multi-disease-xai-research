"""Add user username/is_active and a revoked-token table without altering existing rows."""

import re

import sqlalchemy as sa
from alembic import op

revision = "0003_user_management"
down_revision = "0002_phase1_foundation"
branch_labels = None
depends_on = None

_USERNAME_CHARS = re.compile(r"[^a-z0-9_.-]")


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _derive_username(email: str, taken: set[str], user_id: int) -> str:
    base = _USERNAME_CHARS.sub("", email.split("@", 1)[0].lower())[:40].lstrip("_.-")
    base = base or f"user{user_id}"
    candidate = base
    suffix_number = 1
    while candidate in taken:
        suffix = f"_{user_id}" if suffix_number == 1 else f"_{user_id}_{suffix_number}"
        candidate = f"{base[: 50 - len(suffix)]}{suffix}"
        suffix_number += 1
    taken.add(candidate)
    return candidate


def upgrade() -> None:
    columns = {column["name"] for column in _inspector().get_columns("users")}
    if "is_active" not in columns:
        with op.batch_alter_table("users") as batch_op:
            batch_op.add_column(
                sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1"))
            )
    if "username" not in columns:
        with op.batch_alter_table("users") as batch_op:
            batch_op.add_column(sa.Column("username", sa.String(length=50), nullable=True))

    connection = op.get_bind()
    users = sa.table(
        "users",
        sa.column("id", sa.Integer),
        sa.column("email", sa.String),
        sa.column("username", sa.String),
    )
    rows = connection.execute(sa.select(users.c.id, users.c.email, users.c.username)).all()
    taken = {row.username for row in rows if row.username}
    for row in rows:
        if not row.username:
            connection.execute(
                users.update()
                .where(users.c.id == row.id)
                .values(username=_derive_username(row.email, taken, row.id))
            )

    with op.batch_alter_table("users") as batch_op:
        batch_op.alter_column(
            "username", existing_type=sa.String(length=50), nullable=False
        )
    index_names = {index["name"] for index in _inspector().get_indexes("users")}
    if "ix_users_username" not in index_names:
        op.create_index("ix_users_username", "users", ["username"], unique=True)

    if not _inspector().has_table("revoked_tokens"):
        op.create_table(
            "revoked_tokens",
            sa.Column("jti", sa.String(length=64), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("jti"),
        )
        op.create_index("ix_revoked_tokens_expires_at", "revoked_tokens", ["expires_at"])


def downgrade() -> None:
    raise RuntimeError(
        "Downgrade is blocked: it would drop usernames and revoked-token records."
    )
