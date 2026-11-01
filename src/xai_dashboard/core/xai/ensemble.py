"""Probability-space adapters for explaining fitted preprocessing/model bundles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin


class ProbabilitySpaceEnsemble(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Weighted positive-class scorer over already-fitted base estimators."""

    def __init__(self, estimators: dict[str, Any], weights: dict[str, float]) -> None:
        self.estimators = estimators
        self.weights = weights
        self.classes_ = np.array([0, 1])
        if set(estimators) != set(weights):
            raise ValueError("Every base estimator must have exactly one ensemble weight")
        if not estimators or any(weight < 0 for weight in weights.values()):
            raise ValueError("Ensemble weights must be non-negative and include base estimators")
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Ensemble weights must sum to one")

    def fit(self, features: Any, target: NDArray[Any]) -> ProbabilitySpaceEnsemble:
        matrix = np.asarray(features)
        for estimator in self.estimators.values():
            estimator.fit(matrix, target)
        self.classes_ = np.unique(target)
        return self

    def predict_proba(self, features: Any) -> NDArray[np.float64]:
        matrix = np.asarray(features)
        positive = np.zeros(matrix.shape[0], dtype=float)
        for name, estimator in self.estimators.items():
            positive += self.weights[name] * estimator.predict_proba(matrix)[:, 1]
        return np.column_stack((1.0 - positive, positive))


def encode_ensemble_inputs(
    model_bundle: dict[str, Any], frame: pd.DataFrame, model_name: str = "weighted_ensemble"
) -> tuple[ProbabilitySpaceEnsemble, pd.DataFrame]:
    """Transform rows and wrap a selected estimator or weighted ensemble in score space."""
    models = model_bundle.get("models", {})
    selected = models.get(model_name)
    if model_name in {"weighted_ensemble", "equal_weight_ensemble"}:
        if selected is None or not getattr(selected, "models", None):
            raise ValueError(f"Model artifact does not contain the '{model_name}' ensemble")
        pipelines = selected.models
        weights = selected.weights
    else:
        pipeline = models.get(model_name)
        if pipeline is None or not hasattr(pipeline, "named_steps"):
            raise ValueError(f"Model artifact does not contain '{model_name}'")
        pipelines = {model_name: pipeline}
        weights = {model_name: 1.0}
    first_pipeline = next(iter(pipelines.values()))
    preprocessor = first_pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    for pipeline in pipelines.values():
        other_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        if not np.array_equal(feature_names, other_names):
            raise ValueError("Fitted model pipelines produce inconsistent transformed features")
    transformed = preprocessor.transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    encoded_frame = pd.DataFrame(
        np.asarray(transformed, dtype=float),
        index=frame.index,
        columns=feature_names,
    )
    estimators = {
        name: pipeline.named_steps["model"] for name, pipeline in pipelines.items()
    }
    return ProbabilitySpaceEnsemble(estimators, weights), encoded_frame
