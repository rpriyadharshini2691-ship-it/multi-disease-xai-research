"""Leakage-safe preprocessing pipelines for config-defined disease features."""

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xai_dashboard.core.data.diseases import DiseaseConfig


def build_preprocessor(config: DiseaseConfig) -> ColumnTransformer:
    """Build transformers; all imputers/scalers/encoders fit only during pipeline.fit."""
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if config.numeric_columns:
        numeric = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric, list(config.numeric_columns)))
    if config.categorical_columns:
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical, list(config.categorical_columns)))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(config: DiseaseConfig, estimator: Any) -> Pipeline:
    """Put all learned preprocessing and estimator fitting inside one sklearn Pipeline."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(config)),
            ("model", estimator),
        ]
    )
