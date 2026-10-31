"""SQLAlchemy persistence models. Prediction logs intentionally exclude input features."""

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from xai_dashboard.db.session import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(
            "role IN ('admin', 'researcher', 'clinician')", name="ck_users_role"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(160))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(
        String(20), default="researcher", server_default="researcher"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    predictions: Mapped[list["PredictionLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    registered_models: Mapped[list["ModelRegistry"]] = relationship(back_populates="creator")
    experiments: Mapped[list["ExperimentLog"]] = relationship(back_populates="user")
    reports: Mapped[list["ReportLog"]] = relationship(back_populates="user")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    disease_id: Mapped[str] = mapped_column(String(40), index=True)
    model_name: Mapped[str] = mapped_column(
        String(80), default="weighted_ensemble", server_default="weighted_ensemble"
    )
    positive_score: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    user: Mapped[User] = relationship(back_populates="predictions")


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    __table_args__ = (
        UniqueConstraint(
            "disease_id", "model_name", "version", name="uq_model_registry_version"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    disease_id: Mapped[str] = mapped_column(String(40), index=True)
    model_name: Mapped[str] = mapped_column(String(80))
    version: Mapped[str] = mapped_column(String(80))
    artifact_path: Mapped[str] = mapped_column(String(1024))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    creator: Mapped[User | None] = relationship(back_populates="registered_models")


class ExperimentLog(Base):
    __tablename__ = "experiment_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True
    )
    disease_id: Mapped[str] = mapped_column(String(40), index=True)
    experiment_type: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    parameters: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    user: Mapped[User | None] = relationship(back_populates="experiments")


class ReportLog(Base):
    __tablename__ = "report_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True
    )
    disease_id: Mapped[str] = mapped_column(String(40), index=True)
    report_type: Mapped[str] = mapped_column(String(80))
    artifact_path: Mapped[str] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    user: Mapped[User | None] = relationship(back_populates="reports")
