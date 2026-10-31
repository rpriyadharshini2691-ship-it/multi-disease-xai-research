"""Create user and consented prediction-log tables."""

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("full_name", sa.String(length=160), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("disease_id", sa.String(length=40), nullable=False),
        sa.Column("positive_score", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_prediction_logs_user_id", "prediction_logs", ["user_id"])
    op.create_index("ix_prediction_logs_disease_id", "prediction_logs", ["disease_id"])


def downgrade() -> None:
    op.drop_index("ix_prediction_logs_disease_id", table_name="prediction_logs")
    op.drop_index("ix_prediction_logs_user_id", table_name="prediction_logs")
    op.drop_table("prediction_logs")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
