import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from xai_dashboard.core.data.diseases import DiseaseConfig
from xai_dashboard.core.data.loader import (
    DatasetSchemaError,
    load_dataset,
    profile_dataset,
    stratified_split,
    validate_dataset_schema,
)
from xai_dashboard.core.data.preprocessing import build_pipeline


def _config() -> DiseaseConfig:
    return DiseaseConfig(
        id="toy",
        name="Toy dataset",
        dataset_path="data/toy.csv",
        target_column="outcome",
        positive_label=1,
        numeric_columns=("signal",),
        categorical_columns=("group",),
        binary_columns=(),
        display_names={},
        units={},
        valid_ranges={},
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": np.arange(40, dtype=float),
            "group": ["a", "b"] * 20,
            "outcome": [0] * 20 + [1] * 20,
        }
    )


def test_schema_errors_describe_missing_columns_and_numeric_values() -> None:
    config = _config()
    with pytest.raises(DatasetSchemaError, match="missing required columns: group"):
        validate_dataset_schema(pd.DataFrame({"signal": [1], "outcome": [1]}), config)

    invalid = _frame()
    invalid["signal"] = invalid["signal"].astype(object)
    invalid.loc[0, "signal"] = "not-a-number"
    with pytest.raises(DatasetSchemaError, match="numeric column 'signal'"):
        validate_dataset_schema(invalid, config)


def test_stratified_split_is_seeded_and_preserves_class_stratification() -> None:
    config = _config()
    frame = _frame()
    first = stratified_split(frame, config, test_size=0.25, seed=19)
    second = stratified_split(frame, config, test_size=0.25, seed=19)
    pd.testing.assert_frame_equal(first.x_train, second.x_train)
    pd.testing.assert_frame_equal(first.x_test, second.x_test)
    np.testing.assert_array_equal(first.y_train, second.y_train)
    np.testing.assert_array_equal(first.y_test, second.y_test)
    assert first.y_train.sum() == 15
    assert first.y_test.sum() == 5


def test_pipeline_learns_preprocessing_statistics_from_training_rows_only() -> None:
    config = _config()
    split = stratified_split(_frame(), config, test_size=0.25, seed=19)
    pipeline = build_pipeline(config, LogisticRegression())
    pipeline.fit(split.x_train, split.y_train)

    preprocessor = pipeline.named_steps["preprocess"]
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    learned_mean = numeric_pipeline.named_steps["scale"].mean_[0]
    assert learned_mean == pytest.approx(split.x_train["signal"].mean())
    assert learned_mean != pytest.approx(pd.concat([split.x_train, split.x_test])["signal"].mean())


def test_profile_uses_actual_dataset_counts() -> None:
    profile = profile_dataset(_frame(), _config())
    assert profile["rows"] == 40
    assert profile["target_counts"] == {"0": 20, "1": 20}
    assert profile["disease_id"] == "toy"


def test_loader_normalizes_parseable_numeric_strings(tmp_path) -> None:
    config = _config()
    frame = _frame()
    frame["signal"] = frame["signal"].astype(str)
    path = tmp_path / "toy.csv"
    frame.to_csv(path, index=False)
    loaded = load_dataset(config, path)
    assert pd.api.types.is_numeric_dtype(loaded["signal"])
