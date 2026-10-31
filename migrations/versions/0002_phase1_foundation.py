"""Add phase-one registry and audit tables and normalize roles."""

import sqlalchemy as sa
from alembic import op

revision = "0002_phase1_foundation"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "UPDATE users SET role = LOWER(role) "
        "WHERE role IN ('Admin', 'Researcher', 'Clinician')"
    )
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
    with op.batch_alter_table("prediction_logs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "model_name",
                sa.String(length=80),
                nullable=False,
                server_default="weighted_ensemble",
            )
        )

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
    op.create_index("ix_model_registry_disease_id", "model_registry", ["disease_id"])

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
    op.create_index("ix_experiment_logs_user_id", "experiment_logs", ["user_id"])
    op.create_index("ix_experiment_logs_disease_id", "experiment_logs", ["disease_id"])

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
    op.create_index("ix_report_logs_user_id", "report_logs", ["user_id"])
    op.create_index("ix_report_logs_disease_id", "report_logs", ["disease_id"])


def downgrade() -> None:
    op.drop_index("ix_report_logs_disease_id", table_name="report_logs")
    op.drop_index("ix_report_logs_user_id", table_name="report_logs")
    op.drop_table("report_logs")
    op.drop_index("ix_experiment_logs_disease_id", table_name="experiment_logs")
    op.drop_index("ix_experiment_logs_user_id", table_name="experiment_logs")
    op.drop_table("experiment_logs")
    op.drop_index("ix_model_registry_disease_id", table_name="model_registry")
    op.drop_table("model_registry")
    with op.batch_alter_table("prediction_logs") as batch_op:
        batch_op.drop_column("model_name")
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_constraint("ck_users_role", type_="check")
        batch_op.alter_column(
            "role",
            existing_type=sa.String(length=20),
            existing_nullable=False,
            server_default=None,
        )
