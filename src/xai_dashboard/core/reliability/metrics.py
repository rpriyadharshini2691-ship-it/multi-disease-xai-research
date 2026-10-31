"""Reliability metrics quantify consistency under explicitly tested perturbations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

ShapFunction = Callable[[pd.DataFrame], NDArray[np.float64]]


def _mean_abs_shap(values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("SHAP output must have shape (samples, features)")
    return np.asarray(np.mean(np.abs(array), axis=0), dtype=np.float64)


def _spearman(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    result = spearmanr(left, right).statistic
    return float(result) if np.isfinite(result) else 0.0


def jaccard_top_k(
    original: Sequence[float] | NDArray[np.float64],
    perturbed: Sequence[float] | NDArray[np.float64],
    k: int = 5,
) -> float:
    """Return Jaccard overlap of top-k absolute feature magnitudes."""
    if k <= 0:
        raise ValueError("k must be positive")
    if len(original) != len(perturbed):
        raise ValueError("importance vectors must have equal lengths")
    if len(original) == 0:
        return 1.0
    count = min(k, len(original))
    first = set(np.argsort(np.abs(original))[-count:].tolist())
    second = set(np.argsort(np.abs(perturbed))[-count:].tolist())
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def run_stability_experiment(
    frame: pd.DataFrame,
    shap_fn: ShapFunction,
    numeric_columns: Sequence[str],
    deltas: Sequence[float] = (0.01, 0.05, 0.10),
    repetitions: int = 30,
    seed: int = 42,
    top_k: int = 5,
) -> dict[str, Any]:
    """Perturb standardized numeric inputs and compare global SHAP rankings."""
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    missing = set(numeric_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"numeric columns missing from input: {sorted(missing)}")
    baseline = _mean_abs_shap(shap_fn(frame.copy()))
    rng = np.random.default_rng(seed)
    results: list[dict[str, float]] = []
    for delta in deltas:
        if delta < 0:
            raise ValueError("delta must be non-negative")
        correlations: list[float] = []
        overlaps: list[float] = []
        for _ in range(repetitions):
            perturbed = frame.copy()
            noise = rng.normal(0.0, float(delta), size=(len(frame), len(numeric_columns)))
            for index, column in enumerate(numeric_columns):
                perturbed[column] = pd.to_numeric(perturbed[column]) + noise[:, index]
            explanation = _mean_abs_shap(shap_fn(perturbed))
            correlations.append(_spearman(baseline, explanation))
            overlaps.append(jaccard_top_k(baseline, explanation, top_k))
        results.append(
            {
                "delta": float(delta),
                "spearman_rho_mean": float(np.mean(correlations)),
                "spearman_rho_sd": float(np.std(correlations, ddof=1)) if repetitions > 1 else 0.0,
                "top_k_jaccard_mean": float(np.mean(overlaps)),
                "top_k_jaccard_sd": float(np.std(overlaps, ddof=1)) if repetitions > 1 else 0.0,
                "repetitions": float(repetitions),
            }
        )
    return {
        "experiment": "shap_stability_and_top_k_overlap",
        "interpretation": "Consistency under tested perturbations only; not medical correctness.",
        "perturbation_note": (
            "Delta is an experimental setting on standardized numeric features, "
            "not a clinical tolerance."
        ),
        "seed": seed,
        "results": results,
    }


def compare_shap_and_permutation(
    estimator: Any,
    frame: pd.DataFrame,
    target: Sequence[Any],
    shap_values: NDArray[np.float64],
    repetitions: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare global mean-absolute-SHAP and permutation rankings."""
    shap_importance = _mean_abs_shap(shap_values)
    permutation = permutation_importance(
        estimator, frame, target, n_repeats=repetitions, random_state=seed, scoring="roc_auc"
    )
    if len(shap_importance) != len(permutation.importances_mean):
        raise ValueError("SHAP and permutation feature dimensions differ")
    rho = _spearman(shap_importance, permutation.importances_mean)
    return {
        "experiment": "shap_vs_permutation_importance",
        "spearman_rho": rho,
        "features": list(frame.columns),
        "shap_mean_abs": shap_importance.tolist(),
        "permutation_mean": permutation.importances_mean.tolist(),
        "permutation_sd": permutation.importances_std.tolist(),
        "interpretation": (
            "Ranking agreement under the tested evaluation data; not medical correctness."
        ),
    }
