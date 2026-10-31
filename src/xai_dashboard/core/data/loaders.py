"""Config-driven CSV loading, schema checks, and stratified dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from xai_dashboard.core.data.diseases import DiseaseConfig


class DataSchemaError(ValueError):
    """Raised when a dataset does not satisfy its disease configuration."""


@dataclass(frozen=True)
class DatasetSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: NDArray[Any]
    y_test: NDArray[Any]


def load_dataset(
    config: DiseaseConfig, dataset_path: Path | None = None
) -> tuple[pd.DataFrame, NDArray[Any]]:
    """Load configured features/target and validate/coerce their declared types."""
    path = dataset_path or Path(config.dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset for {config.id} was not found: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DataSchemaError(f"Could not read dataset {path}: {exc}") from exc
    if frame.empty:
        raise DataSchemaError(f"Dataset {path} contains no rows")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DataSchemaError(f"Dataset {path} has duplicate columns: {duplicates}")

    required = (*config.feature_columns, config.target_column)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DataSchemaError(
            f"Dataset for {config.id} at {path} is missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )

    features = frame.loc[:, config.feature_columns].copy()
    for column in config.numeric_columns:
        raw = features[column].replace(["?", ""], pd.NA)
        converted = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & converted.isna()
        if invalid.any():
            values = raw.loc[invalid].astype(str).unique().tolist()[:5]
            raise DataSchemaError(
                f"Dataset for {config.id} has non-numeric values in column "
                f"{column!r}: {values}"
            )
        features[column] = converted.replace([np.inf, -np.inf], np.nan)

    for column in config.categorical_columns:
        values = features[column].astype("string").str.strip()
        features[column] = values.replace({"?": pd.NA, "": pd.NA})

    target = frame[config.target_column]
    if isinstance(config.positive_label, str):
        target = target.astype("string").str.strip().replace({"?": pd.NA, "": pd.NA})
    if target.isna().any():
        raise DataSchemaError(
            f"Dataset for {config.id} contains missing values in target column "
            f"{config.target_column!r}"
        )
    labels = pd.unique(target)
    if config.positive_label not in labels:
        raise DataSchemaError(
            f"Configured positive label {config.positive_label!r} was not found "
            f"in target column {config.target_column!r}; found labels: {labels.tolist()}"
        )
    if len(labels) != 2:
        raise DataSchemaError(
            f"Target column {config.target_column!r} must contain exactly two classes; "
            f"found {labels.tolist()}"
        )
    return features, target.to_numpy()


def stratified_train_test_split(
    features: pd.DataFrame,
    target: NDArray[Any],
    test_size: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    """Return a reproducible stratified split without fitting any transforms."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be greater than 0 and less than 1")
    labels, counts = np.unique(target, return_counts=True)
    if len(labels) != 2 or int(counts.min()) < 2:
        raise ValueError("A stratified binary split requires at least two rows per class")
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=seed,
            stratify=target,
        )
    except ValueError as exc:
        raise ValueError(f"Could not create stratified train/test split: {exc}") from exc
    return DatasetSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
    )
