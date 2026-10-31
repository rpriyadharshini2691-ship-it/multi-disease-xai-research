import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from xai_dashboard.core.ml.training import WeightedProbabilityEnsemble
from xai_dashboard.core.reliability.metrics import (
    compare_shap_and_permutation,
    jaccard_top_k,
    run_stability_experiment,
)


def test_jaccard_top_k_identical_and_disjoint_rankings() -> None:
    assert jaccard_top_k([9, 8, 1, 0], [9, 8, 1, 0], k=2) == 1.0
    assert jaccard_top_k([9, 8, 1, 0], [0, 1, 8, 9], k=2) == 0.0


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
