import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from xai_dashboard.core.ml.training import WeightedProbabilityEnsemble
from xai_dashboard.core.reliability.metrics import (
    additivity_diagnostics,
    aggregate_shap_by_groups,
    compare_shap_and_permutation,
    grouped_permutation_importance,
    jaccard_similarity,
    jaccard_top_k,
    perturb_standardized_features,
    run_stability_experiment,
    spearman_rho,
    summarize_repetitions,
    top_k_features,
)


def test_jaccard_top_k_identical_and_disjoint_rankings() -> None:
    assert jaccard_top_k([9, 8, 1, 0], [9, 8, 1, 0], k=2) == 1.0
    assert jaccard_top_k([9, 8, 1, 0], [0, 1, 8, 9], k=2) == 0.0


def test_spearman_handles_ties_constants_and_missing_values() -> None:
    assert spearman_rho([1, 1, 2, 3], [3, 3, 2, 1]) == -1.0
    assert spearman_rho([1, 1], [1, 2]) is None
    assert spearman_rho([1, np.nan, 3], [3, 8, 1]) == pytest.approx(-1.0)


def test_top_five_preserves_names_and_resolves_ties_deterministically() -> None:
    names = ["age", "gender_F", "gender_M", "symptom", "other", "missing"]
    values = [0.8, -0.8, 0.2, np.nan, 0.1, np.nan]
    assert top_k_features(values, names, k=5) == ["age", "gender_F", "gender_M", "other"]
    assert jaccard_similarity(["age", "gender_F"], ["age", "symptom"]) == 1 / 3
    assert top_k_features([np.nan, np.nan], ["age", "sex"]) == []


def test_standardized_perturbation_is_seeded_and_preserves_categorical_values() -> None:
    frame = pd.DataFrame({"numeric__age": [-1.0, 0.5], "categorical__sex_F": [1.0, 0.0]})
    original = frame.copy(deep=True)
    first = perturb_standardized_features(frame, ["numeric__age"], 0.05, seed=7)
    second = perturb_standardized_features(frame, ["numeric__age"], 0.05, seed=7)
    assert first.equals(second)
    assert first["categorical__sex_F"].equals(frame["categorical__sex_F"])
    assert frame.equals(original)
    with pytest.raises(ValueError, match="finite non-negative"):
        perturb_standardized_features(frame, ["numeric__age"], -0.1, seed=7)


def test_additivity_diagnostics_marks_invalid_probability_explanations() -> None:
    values = np.array([[0.1, -0.1], [0.2, 0.0]])
    errors, passed = additivity_diagnostics(values, [0.5, 0.5], [0.5, 0.8])
    assert np.allclose(errors, [0.0, -0.1])
    assert passed.tolist() == [True, False]


def test_repetition_summary_uses_only_valid_repetition_units() -> None:
    summary = summarize_repetitions([0.5, None, 1.0], seed=2)
    assert summary["n_repetitions"] == 2
    assert summary["mean"] == 0.75
    assert summary["sd"] == pytest.approx(np.std([0.5, 1.0], ddof=1))
    assert summarize_repetitions([None])["mean"] is None


def test_grouped_shap_and_permutation_keep_categorical_blocks_valid() -> None:
    frame = pd.DataFrame(
        {
            "numeric__age": [0.0, 1.0, 0.2, 0.8, 0.1, 0.9],
            "categorical__sex_F": [1, 0, 1, 0, 1, 0],
            "categorical__sex_M": [0, 1, 0, 1, 0, 1],
        }
    )
    groups = {
        "age": ["numeric__age"],
        "sex": ["categorical__sex_F", "categorical__sex_M"],
    }
    shap_values = aggregate_shap_by_groups(
        np.array([[0.1, 0.2, -0.3], [0.2, 0.1, -0.4]]),
        list(frame.columns),
        groups,
    )
    assert shap_values.tolist() == [[0.1, 0.5], [0.2, 0.5]]
    missing_values = aggregate_shap_by_groups(
        np.array([[np.nan, np.nan, np.nan]]), list(frame.columns), groups
    )
    assert np.isnan(missing_values).all()

    class BlockCheckingEstimator:
        def predict_proba(self, values: object) -> np.ndarray:
            matrix = np.asarray(values, dtype=float)
            assert np.allclose(matrix[:, 1:3].sum(axis=1), 1.0)
            positive = 0.1 + 0.8 * matrix[:, 2]
            return np.column_stack((1.0 - positive, positive))

    importance = grouped_permutation_importance(
        BlockCheckingEstimator(), frame, [0, 1, 0, 1, 0, 1], groups, repetitions=3, seed=8
    )
    assert importance["features"] == ["age", "sex"]
    assert len(importance["mean"]) == 2


def test_stability_is_seeded_and_has_required_deltas() -> None:
    frame = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.9, 0.8, 0.7]})

    def explain(values: pd.DataFrame) -> np.ndarray:
        return values.to_numpy()

    first = run_stability_experiment(frame, explain, ["a"], repetitions=4, seed=13)
    second = run_stability_experiment(frame, explain, ["a"], repetitions=4, seed=13)
    assert first == second
    assert [row["delta"] for row in first["results"]] == [0.01, 0.05, 0.1]
    assert all(row["repetitions"] == 4.0 for row in first["results"])


def test_stability_rejects_missing_numeric_column() -> None:
    frame = pd.DataFrame({"x": [1.0]})
    try:
        run_stability_experiment(frame, lambda values: values.to_numpy(), ["not_here"])
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing feature should be rejected")


def test_permutation_comparison_accepts_probability_ensemble() -> None:
    values, target = make_classification(
        n_samples=40, n_features=3, n_informative=2, n_redundant=0, random_state=4
    )
    frame = pd.DataFrame(values, columns=["a", "b", "c"])
    estimator = LogisticRegression().fit(frame, target)
    ensemble = WeightedProbabilityEnsemble({"baseline": estimator}, {"baseline": 1.0})
    result = compare_shap_and_permutation(
        ensemble, frame, target, shap_values=frame.to_numpy(), repetitions=2, seed=4
    )
    assert result["experiment"] == "shap_vs_permutation_importance"
    assert len(result["permutation_mean"]) == frame.shape[1]
