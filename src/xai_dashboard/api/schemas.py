"""Typed request and response schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, StrictBool, field_validator

SCORE_DISCLAIMER = (
    "This is a model-estimated positive-class score, not a clinical probability or diagnosis. "
    "This research and education tool is not a clinical device."
)

ModelName = Literal[
    "logistic_regression",
    "random_forest",
    "xgboost",
    "equal_weight_ensemble",
    "weighted_ensemble",
]


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=160)
    password: str = Field(min_length=10, max_length=128)
    role: Literal["researcher", "clinician"] = "researcher"

    @field_validator("password")
    @classmethod
    def password_fits_bcrypt_input_limit(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must not exceed 72 UTF-8 bytes")
        return value


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
    disease_id: str = Field(min_length=1, max_length=40)
    features: dict[str, Any]
    model_name: ModelName = "weighted_ensemble"
    consent: StrictBool = False


class PredictionContribution(BaseModel):
    feature: str
    value: Any
    shap_value: float
    direction: Literal["increases", "decreases", "neutral"]


class PredictionResponse(BaseModel):
    disease_id: str
    disease_name: str
    prediction_reference: str
    model_version: str
    model_name: str
    predicted_class: Literal["positive", "negative"]
    positive_class_score: float
    decision_threshold: float
    created_at: datetime
    important_features: list[PredictionContribution]
    additivity_error: float
    additivity_passed: bool
    score_match_passed: bool
    validation_warning: str | None = None
    score_type: str = "model_estimated_positive_class_score"
    disclaimer: str = SCORE_DISCLAIMER
    stored: bool


class ErrorBody(BaseModel):
    error: dict[str, Any]
