from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from xai_dashboard.core.data.diseases import DiseaseConfig
from xai_dashboard.core.data.loaders import (
    DataSchemaError,
    load_dataset,
    stratified_train_test_split,
)
from xai_dashboard.core.data.preprocessing import build_model_pipeline


def _config() -> DiseaseConfig:
    return DiseaseConfig(
        id="toy",
        name="Toy disease",
        dataset_path="toy.csv",
        target_column="outcome",
        positive_label=1,
        numeric_columns=("value",),
        categorical_columns=("group",),
        binary_columns=(),
        display_names={},
        units={},
        valid_ranges={},
    )


def test_schema_error_lists_missing_and_available_columns(tmp_path: Path) -> None:
    source = tmp_path / "incomplete.csv"
    pd.DataFrame({"value": [1, 2], "outcome": [0, 1]}).to_csv(source, index=False)
    with pytest.raises(DataSchemaError) as error:
        load_dataset(_config(), source)
    assert "'group'" in str(error.value)
    assert "Available columns" in str(error.value)
    assert str(source) in str(error.value)


def test_stratified_split_is_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "toy.csv"
    pd.DataFrame(
        {
            "value": np.arange(40, dtype=float),
            "group": ["a", "b"] * 20,
            "outcome": [0, 1] * 20,
        }
    ).to_csv(source, index=False)
    features, target = load_dataset(_config(), source)
    first = stratified_train_test_split(features, target, test_size=0.25, seed=17)
    second = stratified_train_test_split(features, target, test_size=0.25, seed=17)
    assert first.x_train.index.tolist() == second.x_train.index.tolist()
    assert first.x_test.index.tolist() == second.x_test.index.tolist()
    assert np.array_equal(first.y_train, second.y_train)
    assert np.array_equal(first.y_test, second.y_test)
    assert first.y_test.mean() == pytest.approx(0.5)


def test_preprocessor_statistics_use_training_rows_only(tmp_path: Path) -> None:
    source = tmp_path / "toy.csv"
    pd.DataFrame(
        {
            "value": np.arange(40, dtype=float),
            "group": ["a", "b"] * 20,
            "outcome": [0, 1] * 20,
        }
    ).to_csv(source, index=False)
    features, target = load_dataset(_config(), source)
    split = stratified_train_test_split(features, target, test_size=0.25, seed=17)
    pipeline = build_model_pipeline(_config(), LogisticRegression())
    pipeline.fit(split.x_train, split.y_train)
    scaler = (
        pipeline.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["scaler"]
    )
    assert scaler.mean_[0] == pytest.approx(split.x_train["value"].mean())
    assert scaler.mean_[0] != pytest.approx(features["value"].mean())
