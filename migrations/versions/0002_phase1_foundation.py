"""Adopt Phase 1 tables, normalize roles, and add version/audit metadata."""

import sqlalchemy as sa
from alembic import op

revision = "0002_phase1_foundation"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _has_table(name: str) -> bool:
    return _inspector().has_table(name)


def _has_index(table: str, name: str) -> bool:
    return any(index["name"] == name for index in _inspector().get_indexes(table))


def _ensure_index(name: str, table: str, columns: list[str]) -> None:
    if _has_table(table) and not _has_index(table, name):
        op.create_index(name, table, columns)


def upgrade() -> None:
    if _has_table("users"):
        op.execute(
            "UPDATE users SET role = LOWER(role) "
            "WHERE role IN ('Admin', 'Researcher', 'Clinician')"
        )
        inspector = _inspector()
        user_columns = {column["name"] for column in inspector.get_columns("users")}
        check_names = {
            constraint.get("name") for constraint in inspector.get_check_constraints("users")
        }
        if "role" in user_columns and "ck_users_role" not in check_names:
            with op.batch_alter_table("users") as batch_op:
                batch_op.alter_column(
                    "role",
                    existing_type=sa.String(length=20),
                    existing_nullable=False,
                    server_default="researcher",
                )
                batch_op.create_check_constraint(
                    "ck_users_role", "role IN ('admin', 'researcher', 'clinician')"
                )

    if _has_table("prediction_logs"):
        prediction_columns = {
            column["name"] for column in _inspector().get_columns("prediction_logs")
        }
        if "model_name" not in prediction_columns:
            with op.batch_alter_table("prediction_logs") as batch_op:
                batch_op.add_column(
                    sa.Column(
                        "model_name",
                        sa.String(length=80),
                        nullable=False,
                        server_default="weighted_ensemble",
                    )
                )

    if not _has_table("model_registry"):
        op.create_table(
            "model_registry",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("disease_id", sa.String(length=40), nullable=False),
            sa.Column("model_name", sa.String(length=80), nullable=False),
            sa.Column("version", sa.String(length=80), nullable=False),
            sa.Column("artifact_path", sa.String(length=1024), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
            sa.Column("created_by_id", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["created_by_id"], ["users.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "disease_id", "model_name", "version", name="uq_model_registry_version"
            ),
        )
    _ensure_index("ix_model_registry_disease_id", "model_registry", ["disease_id"])

    if not _has_table("experiment_logs"):
        op.create_table(
            "experiment_logs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=True),
            sa.Column("disease_id", sa.String(length=40), nullable=False),
            sa.Column("experiment_type", sa.String(length=80), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("parameters", sa.JSON(), nullable=True),
            sa.Column("artifact_path", sa.String(length=1024), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
        )
    _ensure_index("ix_experiment_logs_user_id", "experiment_logs", ["user_id"])
    _ensure_index("ix_experiment_logs_disease_id", "experiment_logs", ["disease_id"])

    if not _has_table("report_logs"):
        op.create_table(
            "report_logs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=True),
            sa.Column("disease_id", sa.String(length=40), nullable=False),
            sa.Column("report_type", sa.String(length=80), nullable=False),
            sa.Column("artifact_path", sa.String(length=1024), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
        )
    _ensure_index("ix_report_logs_user_id", "report_logs", ["user_id"])
    _ensure_index("ix_report_logs_disease_id", "report_logs", ["disease_id"])


def downgrade() -> None:
    raise RuntimeError(
        "This adoption migration may include pre-existing user tables and rows; "
        "downgrade is intentionally blocked to prevent data loss."
    )
