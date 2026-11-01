"""Validated inference and display-safe local SHAP details for API predictions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from math import isclose, isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from xai_dashboard.core.data.diseases import DiseaseConfig
from xai_dashboard.core.data.loader import load_dataset, stratified_split
from xai_dashboard.core.data.quality import file_sha256
from xai_dashboard.core.xai.ensemble import encode_ensemble_inputs
from xai_dashboard.core.xai.shap_service import explain_instance


class PredictionInputError(ValueError):
    """A submitted feature is absent, invalid, or outside the verified schema."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _feature_value(
    name: str,
    value: Any,
    config: DiseaseConfig,
    reference: pd.DataFrame,
) -> Any:
    if value is None:
        raise PredictionInputError("invalid_feature", f"{name} is required and cannot be null.")
    if isinstance(value, dict | list | tuple | set):
        raise PredictionInputError("invalid_feature", f"{name} must be a scalar value.")
    if bool(pd.isna(value)):
        raise PredictionInputError("invalid_feature", f"{name} is required and cannot be null.")
    if name in config.numeric_columns:
        if isinstance(value, bool):
            raise PredictionInputError("invalid_feature", f"{name} must be numeric.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise PredictionInputError("invalid_feature", f"{name} must be numeric.") from None
        if not isfinite(number):
            raise PredictionInputError("invalid_feature", f"{name} must be finite.")
        bounds = config.valid_ranges.get(name)
        if bounds is not None and not bounds[0] <= number <= bounds[1]:
            raise PredictionInputError(
                "out_of_range", f"{name} is outside its configured input range."
            )
        return number

    allowed = reference[name].dropna().unique().tolist()
    if pd.api.types.is_numeric_dtype(reference[name].dtype):
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise PredictionInputError(
                "invalid_category", f"{name} must match a configured dataset category."
            ) from None
        if not isfinite(number) or not any(isclose(number, float(item)) for item in allowed):
            raise PredictionInputError(
                "invalid_category", f"{name} must match a configured dataset category."
            )
        return number

    if not isinstance(value, str):
        raise PredictionInputError(
            "invalid_category", f"{name} must match a configured dataset category."
        )
    category = value.strip()
    if category not in {str(item).strip() for item in allowed}:
        raise PredictionInputError(
            "invalid_category", f"{name} must match a configured dataset category."
        )
    return category


def validate_prediction_features(
    config: DiseaseConfig,
    features: dict[str, Any],
    reference: pd.DataFrame,
) -> pd.DataFrame:
    expected = set(config.feature_columns)
    supplied = set(features)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise PredictionInputError(
            "invalid_features",
            "Feature names must match the disease configuration "
            f"(missing={missing}, extra={extra}).",
        )
    normalized = {
        name: _feature_value(name, features[name], config, reference)
        for name in config.feature_columns
    }
    return pd.DataFrame([normalized], columns=config.feature_columns)


def _display_contributions(
    config: DiseaseConfig,
    features: dict[str, Any],
    explanation: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    encoded = explanation["features"]
    grouped: list[dict[str, Any]] = []
    for name in config.feature_columns:
        prefix = f"numeric__{name}" if name in config.numeric_columns else f"categorical__{name}_"
        values = [
            float(item["shap_value"])
            for item in encoded
            if item["feature"] == prefix
            or (name in config.categorical_columns and item["feature"].startswith(prefix))
        ]
        if not values:
            continue
        attribution = float(sum(values))
        grouped.append(
            {
                "feature": config.display_names.get(name, name),
                "value": features[name],
                "shap_value": attribution,
                "direction": (
                    "increases"
                    if attribution > 0
                    else "decreases" if attribution < 0 else "neutral"
                ),
            }
        )
    grouped.sort(key=lambda item: -abs(float(item["shap_value"])))
    return grouped[:limit]


def create_prediction_details(
    config: DiseaseConfig,
    features: dict[str, Any],
    model_name: str,
    model_bundle: dict[str, Any],
    model_version: str,
    dataset_path: Path,
    artifacts_dir: Path,
) -> dict[str, Any]:
    reference = load_dataset(config, dataset_path)
    frame = validate_prediction_features(config, features, reference)
    model = model_bundle.get("models", {}).get(model_name)
    if model is None:
        raise FileNotFoundError(f"Model '{model_name}' is not present in the stored model artifact")
    score = float(model.predict_proba(frame)[0, 1])
    if not isfinite(score) or not 0 <= score <= 1:
        raise ValueError("The model returned an invalid positive-class score")

    evaluation_path = artifacts_dir / config.id / f"evaluation_{model_version}.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError("The evaluation metadata for the stored model is missing")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("dataset_sha256") != file_sha256(dataset_path):
        raise ValueError("Model evaluation metadata does not match the current dataset")
    split = stratified_split(
        reference,
        config,
        test_size=float(evaluation["test_size"]),
        seed=int(evaluation["seed"]),
    )
    background = split.x_train.sample(n=min(30, len(split.x_train)), random_state=42)
    combined = pd.concat([background, frame], ignore_index=True)
    explanation_model, encoded_combined = encode_ensemble_inputs(
        model_bundle, combined, model_name
    )
    local_explanation = explain_instance(
        explanation_model,
        encoded_combined.iloc[:-1],
        encoded_combined.iloc[[-1]],
    )
    raw_score = float(model.predict_proba(frame)[0, 1])
    score_match = bool(
        np.isclose(
            local_explanation["positive_class_score"],
            raw_score,
            atol=1e-4,
            rtol=1e-4,
        )
    )
    additivity_passed = bool(local_explanation["additivity_passed"] and score_match)
    details = {
        "prediction_reference": f"PRED-{uuid4().hex[:12].upper()}",
        "created_at": datetime.now(UTC),
        "disease_id": config.id,
        "disease_name": config.name,
        "model_name": model_name,
        "model_version": model_version,
        "predicted_class": "positive" if score >= 0.5 else "negative",
        "positive_class_score": score,
        "decision_threshold": 0.5,
        "important_features": _display_contributions(
            config, frame.iloc[0].to_dict(), local_explanation
        ),
        "additivity_error": float(local_explanation["additivity_error"]),
        "additivity_passed": additivity_passed,
        "score_match_passed": score_match,
    }
    if not score_match:
        details["validation_warning"] = (
            "SHAP score does not match the selected stored model; "
            "this explanation is not validated."
        )
    return details