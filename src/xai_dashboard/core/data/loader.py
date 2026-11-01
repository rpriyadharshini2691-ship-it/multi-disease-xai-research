"""Config-driven CSV loading, schema checks, profiling, and stratified splitting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from xai_dashboard.core.data.diseases import DiseaseConfig


class DatasetSchemaError(ValueError):
    """Dataset columns or values do not match a disease configuration."""


class StratifiedSplitError(ValueError):
    """Dataset is too small or imbalanced for the requested stratified split."""


@dataclass(frozen=True)
class DataSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: NDArray[np.int64]
    y_test: NDArray[np.int64]


def validate_dataset_schema(frame: pd.DataFrame, config: DiseaseConfig) -> None:
    """Validate configured feature/target fields and their basic value semantics."""
    if frame.empty:
        raise DatasetSchemaError(f"Dataset for '{config.id}' contains no rows.")
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DatasetSchemaError(f"Dataset has duplicate columns: {duplicates}.")

    required = set(config.feature_columns) | {config.target_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DatasetSchemaError(
            f"Dataset for '{config.id}' is missing required columns: {', '.join(missing)}. "
            f"Configured target: '{config.target_column}'."
        )

    target = frame[config.target_column]
    if target.isna().any():
        rows = (target.index[target.isna()] + 1).tolist()[:5]
        raise DatasetSchemaError(
            f"Target column '{config.target_column}' contains missing values at data rows {rows}."
        )
    if not (target == config.positive_label).any():
        observed = target.astype(str).drop_duplicates().head(8).tolist()
        raise DatasetSchemaError(
            f"Positive label {config.positive_label!r} is absent from target column "
            f"'{config.target_column}'. Observed values include: {observed}."
        )

    for column in config.numeric_columns:
        values = frame[column].dropna()
        converted = pd.to_numeric(values, errors="coerce")
        invalid = converted.isna()
        if invalid.any():
            examples = values.loc[invalid].astype(str).drop_duplicates().head(5).tolist()
            raise DatasetSchemaError(
                f"Configured numeric column '{column}' contains non-numeric values: {examples}."
            )


def load_dataset(config: DiseaseConfig, dataset_path: Path | None = None) -> pd.DataFrame:
    """Load a disease CSV and validate it against its YAML configuration."""
    path = dataset_path or Path(config.dataset_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset for '{config.id}' was not found at '{path}'. "
            "Supply the authorized CSV or update its YAML dataset_path."
        )
    frame = pd.read_csv(path, na_values=["?"])
    validate_dataset_schema(frame, config)
    for column in config.numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def stratified_split(
    frame: pd.DataFrame,
    config: DiseaseConfig,
    test_size: float = 0.2,
    seed: int = 42,
) -> DataSplit:
    """Split raw columns before any fitting so preprocessing learns from train rows only."""
    validate_dataset_schema(frame, config)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    features = frame.loc[:, config.feature_columns].copy()
    target = np.asarray(frame[config.target_column] == config.positive_label, dtype=np.int64)
    counts = np.bincount(target, minlength=2)
    if counts.min() < 2:
        raise StratifiedSplitError(
            "Stratified train/test split requires at least two rows in each target class; "
            f"observed class counts are negative={counts[0]}, positive={counts[1]}."
        )
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            stratify=target,
            random_state=seed,
        )
    except ValueError as exc:
        raise StratifiedSplitError(
            f"Cannot create a stratified split with test_size={test_size}: {exc}"
        ) from exc
    return DataSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=np.asarray(y_train, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
    )


def profile_dataset(frame: pd.DataFrame, config: DiseaseConfig) -> dict[str, Any]:
    """Return a compact schema and missingness profile without deriving model metrics."""
    validate_dataset_schema(frame, config)
    target_counts = frame[config.target_column].value_counts(dropna=False)
    return {
        "disease_id": config.id,
        "dataset_name": config.name,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "configured_features": list(config.feature_columns),
        "target_column": config.target_column,
        "positive_label": config.positive_label,
        "target_counts": {str(label): int(count) for label, count in target_counts.items()},
        "missing_by_column": {
            str(column): int(count)
            for column, count in frame.isna().sum().items()
            if count > 0
        },
    }
