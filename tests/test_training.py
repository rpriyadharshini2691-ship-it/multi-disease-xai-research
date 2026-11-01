import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xai_dashboard.core.data.diseases import DiseaseConfig
from xai_dashboard.core.ml.training import (
    WeightedProbabilityEnsemble,
    _metrics,
    _oof_roc_auc_weights,
    _pipeline,
)
from xai_dashboard.core.xai.artifacts import generate_global_explanations
from xai_dashboard.core.xai.ensemble import encode_ensemble_inputs
from xai_dashboard.core.xai.shap_service import explain_instance


def _config(columns: tuple[str, ...]) -> DiseaseConfig:
    return DiseaseConfig(
        id="toy",
        name="Toy",
        dataset_path="",
        target_column="outcome",
        positive_label=1,
        numeric_columns=columns,
        categorical_columns=(),
        binary_columns=(),
        display_names={},
        units={},
        valid_ranges={},
    )


def test_all_eight_metrics_match_expected_binary_classification_values() -> None:
    target = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.7, 0.4, 0.9])
    metrics = _metrics(target, scores)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["sensitivity"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["pr_auc"] == pytest.approx(5 / 6)
    assert metrics["brier_score"] == pytest.approx(0.2175)


def test_ensemble_enforces_nonnegative_normalized_weights() -> None:
    first = LogisticRegression()
    second = LogisticRegression()
    ensemble = WeightedProbabilityEnsemble(
        {"first": first, "second": second}, {"first": 0.25, "second": 0.75}
    )
    assert sum(ensemble.weights.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-negative and sum to one"):
        WeightedProbabilityEnsemble(
            {"first": first, "second": second}, {"first": -0.1, "second": 1.1}
        )
    with pytest.raises(ValueError, match="sum to one"):
        WeightedProbabilityEnsemble(
            {"first": first, "second": second}, {"first": 0.4, "second": 0.4}
        )


def test_equal_weight_soft_voting_averages_base_probability_scores() -> None:
    values, target = make_classification(
        n_samples=40, n_features=3, n_informative=2, n_redundant=0, random_state=8
    )
    frame = pd.DataFrame(values, columns=("a", "b", "c"))
    logistic = LogisticRegression(random_state=8).fit(frame, target)
    forest = RandomForestClassifier(n_estimators=20, random_state=8).fit(frame, target)
    ensemble = WeightedProbabilityEnsemble(
        {"logistic_regression": logistic, "random_forest": forest},
        {"logistic_regression": 0.5, "random_forest": 0.5},
    )
    expected = (
        logistic.predict_proba(frame) + forest.predict_proba(frame)
    ) / 2
    np.testing.assert_allclose(ensemble.predict_proba(frame), expected)


def test_oof_ensemble_weights_are_deterministic_and_normalized() -> None:
    rng = np.random.default_rng(12)
    values = rng.normal(size=(60, 3))
    target = np.asarray([0, 1] * 30, dtype=np.int64)
    frame = pd.DataFrame(values, columns=("f0", "f1", "f2"))
    config = _config(tuple(frame.columns))
    pipelines = {
        "logistic_regression": _pipeline(
            config, LogisticRegression(max_iter=1000, random_state=12)
        ),
        "random_forest": _pipeline(
            config, RandomForestClassifier(n_estimators=20, random_state=12)
        ),
    }
    first = _oof_roc_auc_weights(pipelines, frame, target, splits=3, seed=12)
    second = _oof_roc_auc_weights(pipelines, frame, target, splits=3, seed=12)
    assert first == second
    assert all(weight >= 0 for weight in first.values())
    assert sum(first.values()) == pytest.approx(1.0)


def test_global_shap_uses_numeric_encoded_category_features(tmp_path) -> None:
    config = DiseaseConfig(
        id="toy-categorical",
        name="Toy categorical",
        dataset_path="",
        target_column="outcome",
        positive_label=1,
        numeric_columns=("signal",),
        categorical_columns=("group",),
        binary_columns=(),
        display_names={},
        units={},
        valid_ranges={},
    )
    frame = pd.DataFrame(
        {
            "signal": np.linspace(-2, 2, 12),
            "group": ["alpha", "beta"] * 6,
        }
    )
    target = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    pipelines = {
        "logistic_regression": _pipeline(config, LogisticRegression(max_iter=1000)),
    }
    for pipeline in pipelines.values():
        pipeline.fit(frame, target)
    ensemble = WeightedProbabilityEnsemble(pipelines, {"logistic_regression": 1.0})
    model_bundle = {
        "models": {**pipelines, "weighted_ensemble": ensemble},
        "weights": ensemble.weights,
    }
    encoded_model, encoded_frame = encode_ensemble_inputs(model_bundle, frame)
    assert all(pd.api.types.is_numeric_dtype(encoded_frame[column]) for column in encoded_frame)
    np.testing.assert_allclose(
        encoded_model.predict_proba(encoded_frame), ensemble.predict_proba(frame)
    )
    artifacts = generate_global_explanations(
        encoded_model, encoded_frame, tmp_path, sample_size=8
    )
    assert (tmp_path / artifacts["beeswarm_png"]).is_file()


def test_ensemble_shap_additivity_matches_raw_categorical_pipeline_output() -> None:
    config = DiseaseConfig(
        id="toy-local",
        name="Toy local",
        dataset_path="",
        target_column="outcome",
        positive_label=1,
        numeric_columns=("signal",),
        categorical_columns=("group",),
        binary_columns=(),
        display_names={},
        units={},
        valid_ranges={},
    )
    frame = pd.DataFrame(
        {
            "signal": np.linspace(-2, 2, 16),
            "group": ["alpha", "beta"] * 8,
        }
    )
    target = np.asarray([0] * 8 + [1] * 8, dtype=np.int64)
    pipeline = _pipeline(config, LogisticRegression(max_iter=1000)).fit(frame, target)
    weights = {"logistic_regression": 1.0}
    ensemble = WeightedProbabilityEnsemble({"logistic_regression": pipeline}, weights)
    model_bundle = {
        "models": {
            "logistic_regression": pipeline,
            "weighted_ensemble": ensemble,
        },
        "weights": weights,
    }
    raw_background = frame.iloc[:8]
    raw_instance = frame.iloc[[10]]
    raw_rows = pd.concat([raw_background, raw_instance], ignore_index=True)
    explanation_model, encoded_rows = encode_ensemble_inputs(model_bundle, raw_rows)
    explanation = explain_instance(
        explanation_model, encoded_rows.iloc[:8], encoded_rows.iloc[[8]]
    )
    raw_score = float(ensemble.predict_proba(raw_instance)[0, 1])
    assert explanation["additivity_passed"] is True
    assert explanation["positive_class_score"] == pytest.approx(raw_score, abs=1e-4)
    assert explanation["base_value"] + sum(
        item["shap_value"] for item in explanation["features"]
    ) == pytest.approx(raw_score, abs=1e-4)
