"""Typed request and response schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

SCORE_DISCLAIMER = (
    "This is a model-estimated positive-class score, not a clinical probability or diagnosis. "
    "This research and education tool is not a clinical device."
)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=160)
    password: str = Field(min_length=10, max_length=128)
    role: Literal["researcher", "clinician"] = "researcher"


class UserProfile(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    email: EmailStr
    full_name: str
    role: str
    is_active: bool
    created_at: datetime


class RoleUpdate(BaseModel):
    role: str = Field(min_length=1, max_length=20)


class StatusUpdate(BaseModel):
    is_active: bool


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile


class PredictionRequest(BaseModel):
    disease_id: str
    features: dict[str, Any]
    model_name: Literal[
        "logistic_regression",
        "random_forest",
        "xgboost",
        "equal_weight_ensemble",
        "weighted_ensemble",
    ] = "weighted_ensemble"
    consent: bool = False


class PredictionResponse(BaseModel):
    disease_id: str
    model_name: str
    positive_class_score: float
    score_type: str = "model_estimated_positive_class_score"
    disclaimer: str = SCORE_DISCLAIMER
    stored: bool


class ErrorBody(BaseModel):
    error: dict[str, Any]
