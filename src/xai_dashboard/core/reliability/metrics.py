"""Reliability metrics quantify consistency under explicitly tested perturbations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

ShapFunction = Callable[[pd.DataFrame], NDArray[np.float64]]


def mean_abs_shap(values: NDArray[np.float64]) -> NDArray[np.float64]:
    array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("SHAP output must have shape (samples, features)")
    finite = np.isfinite(array)
    totals = np.where(finite, np.abs(array), 0.0).sum(axis=0)
    counts = finite.sum(axis=0)
    result: NDArray[np.float64] = np.full(array.shape[1], np.nan, dtype=np.float64)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def spearman_rho(
    left: Sequence[float] | NDArray[np.float64],
    right: Sequence[float] | NDArray[np.float64],
) -> float | None:
    """Return a tied-rank Spearman coefficient, or None when it is undefined."""
    first: NDArray[np.float64] = np.asarray(left, dtype=np.float64).reshape(-1)
    second: NDArray[np.float64] = np.asarray(right, dtype=np.float64).reshape(-1)
    if first.size != second.size:
        raise ValueError("rank vectors must have equal lengths")
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if first.size < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return None
    result = float(spearmanr(first, second).statistic)
    return result if np.isfinite(result) else None


def top_k_features(
    values: Sequence[float] | NDArray[np.float64],
    feature_names: Sequence[str],
    k: int = 5,
) -> list[str]:
    """Rank finite attributions by absolute magnitude, resolving ties by input order."""
    importance: NDArray[np.float64] = np.asarray(values, dtype=np.float64).reshape(-1)
    if k <= 0:
        raise ValueError("k must be positive")
    if importance.size != len(feature_names):
        raise ValueError("feature names and attribution values must have equal lengths")
    valid_indices = np.flatnonzero(np.isfinite(importance))
    order = valid_indices[np.argsort(-np.abs(importance[valid_indices]), kind="stable")]
    return [str(feature_names[index]) for index in order[:k]]


def jaccard_similarity(first: Sequence[str], second: Sequence[str]) -> float:
    first_set, second_set = set(first), set(second)
    union = first_set | second_set
    return len(first_set & second_set) / len(union) if union else 1.0


def perturb_standardized_features(
    frame: pd.DataFrame,
    numeric_columns: Sequence[str],
    delta: float,
    seed: int,
) -> pd.DataFrame:
    """Add seeded Gaussian noise in standardized space without touching other columns."""
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("delta must be a finite non-negative value")
    if len(set(numeric_columns)) != len(numeric_columns):
        raise ValueError("numeric columns must be unique")
    missing = set(numeric_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"numeric columns missing from input: {sorted(missing)}")
    perturbed = frame.copy(deep=True)
    if not numeric_columns or frame.empty or delta == 0:
        return perturbed
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, delta, size=(len(frame), len(numeric_columns)))
    for index, column in enumerate(numeric_columns):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"numeric feature '{column}' contains non-finite values")
        perturbed[column] = values + noise[:, index]
    return perturbed


def additivity_diagnostics(
    shap_values: NDArray[np.float64],
    base_values: Sequence[float] | NDArray[np.float64] | float,
    scores: Sequence[float] | NDArray[np.float64],
    tolerance: float = 1e-4,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Check probability-space SHAP sums against model scores row by row."""
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 2:
        raise ValueError("SHAP output must have shape (samples, features)")
    expected: NDArray[np.float64] = np.asarray(base_values, dtype=np.float64).reshape(-1)
    actual: NDArray[np.float64] = np.asarray(scores, dtype=np.float64).reshape(-1)
    if expected.size == 1 and actual.size > 1:
        expected = np.repeat(expected, actual.size)
    if expected.size != actual.size or values.shape[0] != actual.size:
        raise ValueError("SHAP values, base values, and scores must have matching rows")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite positive value")
    totals = expected + np.sum(values, axis=1)
    errors = totals - actual
    passed = np.isfinite(totals) & np.isfinite(actual) & np.isclose(
        totals, actual, atol=tolerance, rtol=tolerance
    )
    return np.asarray(errors, dtype=np.float64), np.asarray(passed, dtype=bool)


def summarize_repetitions(values: Sequence[float | None], seed: int = 42) -> dict[str, Any]:
    """Summarize repetition-level values with sample SD and a bootstrap percentile CI."""
    observed = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)], dtype=float
    )
    if observed.size == 0:
        return {
            "n_repetitions": 0,
            "mean": None,
            "sd": None,
            "confidence_interval_95": [None, None],
        }
    mean = float(np.mean(observed))
    sd = float(np.std(observed, ddof=1)) if observed.size > 1 else 0.0
    if observed.size == 1:
        interval = [mean, mean]
    else:
        rng = np.random.default_rng(seed)
        samples = rng.choice(observed, size=(2000, observed.size), replace=True).mean(axis=1)
        interval = [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
    return {
        "n_repetitions": int(observed.size),
        "mean": mean,
        "sd": sd,
        "confidence_interval_95": interval,
    }


def aggregate_shap_by_groups(
    shap_values: NDArray[np.float64],
    encoded_columns: Sequence[str],
    feature_groups: Mapping[str, Sequence[str]],
) -> NDArray[np.float64]:
    """Sum absolute encoded attributions within each original input feature."""
    values: NDArray[np.float64] = np.asarray(shap_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(encoded_columns):
        raise ValueError("SHAP output and encoded feature columns do not match")
    positions = {name: index for index, name in enumerate(encoded_columns)}
    aggregated: list[NDArray[np.float64]] = []
    for feature, columns in feature_groups.items():
        if not columns:
            raise ValueError(f"feature group '{feature}' has no encoded columns")
        missing = set(columns) - set(positions)
        if missing:
            raise ValueError(f"encoded feature columns are missing: {sorted(missing)}")
        indices = [positions[column] for column in columns]
        group_values = np.abs(values[:, indices])
        finite = np.isfinite(group_values)
        totals = np.where(finite, group_values, 0.0).sum(axis=1)
        counts = finite.sum(axis=1)
        grouped = np.asarray(totals, dtype=np.float64)
        grouped[counts == 0] = np.nan
        aggregated.append(grouped)
    return np.asarray(np.column_stack(aggregated), dtype=np.float64)


def grouped_permutation_importance(
    estimator: Any,
    frame: pd.DataFrame,
    target: Sequence[Any] | NDArray[np.int64],
    feature_groups: Mapping[str, Sequence[str]],
    repetitions: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Permute original feature groups, keeping one-hot categorical blocks valid."""
    if repetitions < 1:
        raise ValueError("permutation repetitions must be at least one")
    target_values = np.asarray(target)
    if len(frame) != len(target_values) or len(frame) < 2:
        raise ValueError("permutation inputs must contain matching multiple rows")
    if not feature_groups:
        raise ValueError("at least one feature group is required")
    positions = {column: index for index, column in enumerate(frame.columns)}
    group_positions: dict[str, list[int]] = {}
    for feature, columns in feature_groups.items():
        if not columns:
            raise ValueError(f"feature group '{feature}' has no encoded columns")
        missing = set(columns) - set(positions)
        if missing:
            raise ValueError(f"encoded feature columns are missing: {sorted(missing)}")
        group_positions[feature] = [positions[column] for column in columns]

    baseline = np.asarray(estimator.predict_proba(frame), dtype=float)[:, 1]
    baseline_score = float(roc_auc_score(target_values, baseline))
    rng = np.random.default_rng(seed)
    importance = np.zeros((len(group_positions), repetitions), dtype=float)
    matrix = frame.to_numpy(copy=True)
    for group_index, column_indices in enumerate(group_positions.values()):
        for repetition in range(repetitions):
            permutation = rng.permutation(len(frame))
            changed = matrix.copy()
            changed[:, column_indices] = matrix[permutation][:, column_indices]
            scores = np.asarray(estimator.predict_proba(changed), dtype=float)[:, 1]
            importance[group_index, repetition] = baseline_score - roc_auc_score(
                target_values, scores
            )
    return {
        "features": list(group_positions),
        "mean": np.mean(importance, axis=1).tolist(),
        "sd": np.std(importance, axis=1, ddof=1).tolist()
        if repetitions > 1
        else np.zeros(len(group_positions)).tolist(),
    }


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
    names = [str(index) for index in range(len(original))]
    first = top_k_features(original, names, k)
    second = top_k_features(perturbed, names, k)
    return jaccard_similarity(first, second)


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
    baseline = mean_abs_shap(shap_fn(frame.copy()))
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    for delta in deltas:
        if delta < 0:
            raise ValueError("delta must be non-negative")
        correlations: list[float | None] = []
        overlaps: list[float] = []
        for _ in range(repetitions):
            perturbed = frame.copy()
            noise = rng.normal(0.0, float(delta), size=(len(frame), len(numeric_columns)))
            for index, column in enumerate(numeric_columns):
                perturbed[column] = pd.to_numeric(perturbed[column]) + noise[:, index]
            explanation = mean_abs_shap(shap_fn(perturbed))
            correlations.append(spearman_rho(baseline, explanation))
            overlaps.append(jaccard_top_k(baseline, explanation, top_k))
        valid_correlations = [value for value in correlations if value is not None]
        results.append(
            {
                "delta": float(delta),
                "spearman_rho_mean": (
                    float(np.mean(valid_correlations)) if valid_correlations else None
                ),
                "spearman_rho_sd": (
                    float(np.std(valid_correlations, ddof=1))
                    if len(valid_correlations) > 1
                    else 0.0 if valid_correlations else None
                ),
                "valid_spearman_comparisons": len(valid_correlations),
                "top_k_jaccard_mean": float(np.mean(overlaps)),
                "top_k_jaccard_sd": float(np.std(overlaps, ddof=1)) if repetitions > 1 else 0.0,
                "repetitions": repetitions,
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
    target: Sequence[Any] | NDArray[np.int64],
    shap_values: NDArray[np.float64],
    repetitions: int = 10,
    seed: int = 42,
    feature_groups: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Compare SHAP and permutation rankings on the same held-out feature representation."""
    if feature_groups is None:
        shap_importance = mean_abs_shap(shap_values)
        permutation = permutation_importance(
            estimator, frame, target, n_repeats=repetitions, random_state=seed, scoring="roc_auc"
        )
        features = list(frame.columns)
        permutation_mean = permutation.importances_mean
        permutation_sd = permutation.importances_std
        representation = "encoded columns permuted independently"
    else:
        grouped_values = aggregate_shap_by_groups(shap_values, list(frame.columns), feature_groups)
        shap_importance = mean_abs_shap(grouped_values)
        permutation = grouped_permutation_importance(
            estimator, frame, target, feature_groups, repetitions=repetitions, seed=seed
        )
        features = permutation["features"]
        permutation_mean = np.asarray(permutation["mean"], dtype=float)
        permutation_sd = np.asarray(permutation["sd"], dtype=float)
        representation = "original input features; one-hot columns permuted as a block"
    if len(shap_importance) != len(permutation_mean):
        raise ValueError("SHAP and permutation feature dimensions differ")
    rho = spearman_rho(shap_importance, permutation_mean)
    return {
        "experiment": "shap_vs_permutation_importance",
        "spearman_rho": rho,
        "spearman_valid": rho is not None,
        "features": features,
        "shap_mean_abs": shap_importance.tolist(),
        "permutation_mean": permutation_mean.tolist(),
        "permutation_sd": permutation_sd.tolist(),
        "evaluation_rows": int(len(frame)),
        "permutation_repetitions": repetitions,
        "seed": seed,
        "feature_representation": representation,
        "interpretation": (
            "Ranking agreement under the tested evaluation data; not medical correctness."
        ),
    }
