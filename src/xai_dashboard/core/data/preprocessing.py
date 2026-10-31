"""Disease-configured preprocessing fitted only within model training pipelines."""

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xai_dashboard.core.data.diseases import DiseaseConfig


def build_preprocessor(config: DiseaseConfig) -> ColumnTransformer:
    """Build the numeric/categorical transform declared by a disease YAML config."""
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if config.numeric_columns:
        numeric = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric, list(config.numeric_columns)))
    if config.categorical_columns:
        categorical = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                ),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical, list(config.categorical_columns))
        )
    if not transformers:
        raise ValueError(f"Disease {config.id!r} must define at least one feature column")
    return ColumnTransformer(transformers, remainder="drop")


def build_model_pipeline(config: DiseaseConfig, estimator: Any) -> Pipeline:
    """Compose preprocessing and estimator so transforms fit only on each train fold."""
    return Pipeline(
        [("preprocess", build_preprocessor(config)), ("model", estimator)]
    )
