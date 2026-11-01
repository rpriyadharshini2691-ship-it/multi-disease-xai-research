"""Run reproducible, held-out SHAP reliability experiments for all model configurations."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from xai_dashboard.core.data.diseases import DiseaseConfig
from xai_dashboard.core.data.loader import load_dataset, stratified_split
from xai_dashboard.core.data.quality import file_sha256
from xai_dashboard.core.reliability.metrics import (
    additivity_diagnostics,
    aggregate_shap_by_groups,
    compare_shap_and_permutation,
    jaccard_similarity,
    mean_abs_shap,
    perturb_standardized_features,
    spearman_rho,
    summarize_repetitions,
    top_k_features,
)
from xai_dashboard.core.reliability.visualizations import (
    create_disease_comparison,
    create_reliability_visualizations,
)
from xai_dashboard.core.xai.ensemble import encode_ensemble_inputs

MODEL_CONFIGURATIONS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "equal_weight_ensemble",
    "weighted_ensemble",
)
PERTURBATION_LEVELS = (0.01, 0.05, 0.10)
DEFAULT_RANDOM_SEEDS = (42, 123, 2026, 2027)
DEFAULT_REPETITIONS = 30
DEFAULT_SAMPLE_SIZE = 8
SELECTION_SEED = 42
SHAP_BACKGROUND_SIZE = 30
SHAP_ADDITIVITY_TOLERANCE = 1e-4
ShapBatch = tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _finite_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _ordered_importance(
    feature_names: Sequence[str], importances: NDArray[np.float64]
) -> list[dict[str, Any]]:
    values = np.asarray(importances, dtype=float)
    order = sorted(
        range(len(feature_names)),
        key=lambda index: (
            not np.isfinite(values[index]),
            -abs(values[index]) if np.isfinite(values[index]) else 0.0,
            index,
        ),
    )
    return [
        {"feature": str(feature_names[index]), "mean_abs_shap": _finite_number(values[index])}
        for index in order
    ]


def _explain_batch(
    explainer: Any,
    frame: pd.DataFrame,
    score_function: Callable[[Any], NDArray[np.float64]],
    tolerance: float,
    max_evals: int,
) -> ShapBatch:
    explanation = explainer(frame, max_evals=max_evals, silent=True)
    raw_values = np.asarray(explanation.values, dtype=np.float64)
    selected_values: Any = (
        raw_values[..., 0]
        if raw_values.ndim == 3 and raw_values.shape[-1] == 1
        else raw_values
    )
    values: NDArray[np.float64] = np.asarray(selected_values, dtype=np.float64)
    if values.ndim != 2 or values.shape != (len(frame), frame.shape[1]):
        raise ValueError("SHAP returned an unexpected sample/feature shape")
    bases: NDArray[np.float64] = np.asarray(
        explanation.base_values, dtype=np.float64
    ).reshape(-1)
    if bases.size == 1 and len(frame) > 1:
        bases = np.repeat(bases, len(frame))
    scores = np.asarray(score_function(frame.to_numpy()), dtype=float).reshape(-1)
    errors, passed = additivity_diagnostics(values, bases, scores, tolerance)
    return values, bases, scores, errors, passed


def _make_score_function(
    model: Any, encoded_columns: Sequence[str]
) -> Callable[[Any], NDArray[np.float64]]:
    def score_function(values: Any) -> NDArray[np.float64]:
        probabilities = model.predict_proba(pd.DataFrame(values, columns=encoded_columns))
        return np.asarray(probabilities[:, 1], dtype=float)

    return score_function


def _seed_for_repetition(seed: int, repetition: int) -> int:
    state = np.random.SeedSequence([seed, repetition]).generate_state(1)
    return int(state[0])


def _select_heldout_sample(
    features: pd.DataFrame, target: NDArray[np.int64], sample_size: int, seed: int
) -> tuple[pd.DataFrame, NDArray[np.int64], list[int]]:
    if sample_size < 2:
        raise ValueError("sample_size must be at least two for a stratified sample")
    count = min(sample_size, len(features))
    if count == len(features):
        selected_positions: Any = np.arange(len(features), dtype=np.int64)
    else:
        selected_positions, _ = train_test_split(
            np.arange(len(features)),
            train_size=count,
            stratify=target,
            random_state=seed,
        )
    positions: NDArray[np.int64] = np.asarray(selected_positions, dtype=np.int64)
    positions.sort()
    selected = features.iloc[positions].copy()
    labels = np.asarray(target[positions], dtype=np.int64)
    row_ids = [int(features.index[position]) for position in positions]
    return selected, labels, row_ids


def _encoded_feature_groups(
    config: DiseaseConfig, encoded_columns: Sequence[str]
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for column in config.numeric_columns:
        encoded = f"numeric__{column}"
        if encoded in encoded_columns:
            groups[column] = [encoded]
    for column in config.categorical_columns:
        prefix = f"categorical__{column}_"
        matches = [name for name in encoded_columns if name.startswith(prefix)]
        if matches:
            groups[column] = matches
    missing = set(config.feature_columns) - set(groups)
    if missing:
        raise ValueError(f"Preprocessing did not encode configured features: {sorted(missing)}")
    return groups


def _build_summary(records: list[dict[str, Any]], repetitions: int) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    model_names = list(dict.fromkeys(row["model_name"] for row in records))
    levels = sorted({float(row["perturbation_level"]) for row in records})
    for model_name in model_names:
        for level_index, level in enumerate(levels):
            selected = [
                row
                for row in records
                if row["model_name"] == model_name
                and float(row["perturbation_level"]) == level
            ]
            spearman_by_repetition: list[float | None] = []
            jaccard_by_repetition: list[float | None] = []
            error_by_repetition: list[float | None] = []
            valid_comparisons = 0
            for repetition in range(1, repetitions + 1):
                group = [row for row in selected if row["repetition"] == repetition]
                correlations = [
                    float(row["spearman_rho"])
                    for row in group
                    if row["spearman_rho"] is not None
                ]
                valid_comparisons += len(correlations)
                spearman_by_repetition.append(
                    float(np.mean(correlations)) if correlations else None
                )
                jaccards = [
                    float(row["top5_jaccard"])
                    for row in group
                    if row["top5_jaccard"] is not None
                ]
                jaccard_by_repetition.append(
                    float(np.mean(jaccards)) if jaccards else None
                )
                errors = [
                    abs(float(row["perturbed_additivity_error"]))
                    for row in group
                    if row["perturbed_additivity_error"] is not None
                ]
                error_by_repetition.append(float(np.mean(errors)) if errors else None)
            additivity_passed = sum(bool(row["additivity_passed"]) for row in selected)
            summaries.append(
                {
                    "model_name": model_name,
                    "perturbation_level": level,
                    "repetitions_requested": repetitions,
                    "sample_count": len(selected) // repetitions,
                    "sample_comparisons": len(selected),
                    "valid_spearman_comparisons": valid_comparisons,
                    "spearman": summarize_repetitions(
                        spearman_by_repetition, seed=SELECTION_SEED + level_index
                    ),
                    "top5_jaccard": summarize_repetitions(
                        jaccard_by_repetition, seed=SELECTION_SEED + level_index
                    ),
                    "additivity_abs_error": summarize_repetitions(
                        error_by_repetition, seed=SELECTION_SEED + level_index
                    ),
                    "additivity_passed": additivity_passed,
                    "additivity_failed": len(selected) - additivity_passed,
                }
            )
    return summaries


def run_reliability(
    config: DiseaseConfig,
    dataset_path: Path,
    artifacts_dir: Path,
    repetitions: int = DEFAULT_REPETITIONS,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    deltas: Sequence[float] = PERTURBATION_LEVELS,
) -> dict[str, Any]:
    """Explain paired perturbations of a stratified sample from the held-out test partition."""
    if repetitions < DEFAULT_REPETITIONS:
        raise ValueError(f"repetitions must be at least {DEFAULT_REPETITIONS}")
    if len(random_seeds) < 2 or len(set(random_seeds)) != len(random_seeds):
        raise ValueError("at least two unique predefined random seeds are required")
    if not deltas or any(not np.isfinite(delta) or delta <= 0 for delta in deltas):
        raise ValueError("perturbation levels must be finite positive values")
    if len(set(deltas)) != len(deltas):
        raise ValueError("perturbation levels must be unique")
    if sample_size < 2:
        raise ValueError("sample_size must be at least two")

    disease_dir = artifacts_dir / config.id
    model_files = list(disease_dir.glob("models_*.joblib"))
    if not model_files:
        raise FileNotFoundError("Train and persist a model before running reliability experiments")
    model_path = max(model_files, key=lambda path: path.stat().st_mtime_ns)
    model_version = model_path.stem.removeprefix("models_")
    evaluation_path = disease_dir / f"evaluation_{model_version}.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"Evaluation artifact for model {model_version} is missing")

    dataset = load_dataset(config, dataset_path)
    dataset_fingerprint = file_sha256(dataset_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("dataset_sha256") != dataset_fingerprint:
        raise ValueError("Latest model evaluation does not match the current dataset fingerprint")
    split = stratified_split(
        dataset,
        config,
        test_size=float(evaluation["test_size"]),
        seed=int(evaluation["seed"]),
    )
    artifact = joblib.load(model_path)
    models = artifact.get("models", {})
    missing_models = set(MODEL_CONFIGURATIONS) - set(models)
    if missing_models:
        raise ValueError(
            "Trained artifact is missing model configurations: "
            f"{sorted(missing_models)}"
        )

    _, encoded_train = encode_ensemble_inputs(artifact, split.x_train, "weighted_ensemble")
    _, encoded_test = encode_ensemble_inputs(artifact, split.x_test, "weighted_ensemble")
    sample, sample_target, sample_row_ids = _select_heldout_sample(
        encoded_test, split.y_test, sample_size, SELECTION_SEED
    )
    background = encoded_train.sample(
        n=min(SHAP_BACKGROUND_SIZE, len(encoded_train)), random_state=SELECTION_SEED
    )
    encoded_feature_names = [str(column) for column in sample.columns]
    feature_groups = _encoded_feature_groups(config, encoded_feature_names)
    feature_names = list(feature_groups)
    numeric_features = [
        f"numeric__{column}"
        for column in config.numeric_columns
        if f"numeric__{column}" in sample.columns
    ]
    if not numeric_features:
        raise ValueError("No standardized numeric features are available to perturb")

    perturbations: dict[float, tuple[pd.DataFrame, list[dict[str, int]]]] = {}
    for level in deltas:
        frames: list[pd.DataFrame] = []
        metadata: list[dict[str, int]] = []
        for repetition in range(1, repetitions + 1):
            seed = int(random_seeds[(repetition - 1) % len(random_seeds)])
            perturbation_seed = _seed_for_repetition(seed, repetition)
            changed = perturb_standardized_features(
                sample, numeric_features, float(level), perturbation_seed
            )
            frames.append(changed.reset_index(drop=True))
            metadata.extend(
                {
                    "repetition": repetition,
                    "seed": seed,
                    "perturbation_seed": perturbation_seed,
                    "sample_position": position,
                    "sample_id": sample_row_ids[position],
                }
                for position in range(len(sample))
            )
        perturbations[float(level)] = (pd.concat(frames, ignore_index=True), metadata)

    records: list[dict[str, Any]] = []
    global_stability: list[dict[str, Any]] = []
    shap_permutation: list[dict[str, Any]] = []
    additivity_summary: list[dict[str, Any]] = []
    warnings: list[str] = []
    max_evals = 2 * sample.shape[1] + 1

    for model_name in MODEL_CONFIGURATIONS:
        model, model_sample = encode_ensemble_inputs(
            artifact, split.x_test.iloc[:1], model_name
        )
        if not model_sample.columns.equals(sample.columns):
            raise ValueError(f"Encoded feature columns changed for model '{model_name}'")

        score_function = _make_score_function(model, encoded_feature_names)
        explainer = shap.Explainer(
            score_function,
            background,
            algorithm="permutation",
            seed=SELECTION_SEED,
        )
        baseline_values, _, _, baseline_errors, baseline_passed = _explain_batch(
            explainer,
            sample,
            score_function,
            SHAP_ADDITIVITY_TOLERANCE,
            max_evals,
        )
        baseline_group_values = aggregate_shap_by_groups(
            baseline_values, encoded_feature_names, feature_groups
        )
        baseline_importance = mean_abs_shap(baseline_group_values)
        baseline_ordered = _ordered_importance(feature_names, baseline_importance)
        comparison = compare_shap_and_permutation(
            model,
            sample,
            sample_target,
            baseline_values,
            repetitions=30,
            seed=SELECTION_SEED,
            feature_groups=feature_groups,
        )
        shap_permutation.append({"model_name": model_name, **comparison})
        model_errors = [abs(float(error)) for error in baseline_errors if np.isfinite(error)]
        model_passed = int(np.count_nonzero(baseline_passed))
        model_total = len(baseline_passed)
        is_ensemble = model_name in {"equal_weight_ensemble", "weighted_ensemble"}

        for level in deltas:
            perturbed, metadata = perturbations[float(level)]
            perturbed_values, _, _, errors, passed = _explain_batch(
                explainer,
                perturbed,
                score_function,
                SHAP_ADDITIVITY_TOLERANCE,
                max_evals,
            )
            perturbed_group_values = aggregate_shap_by_groups(
                perturbed_values, encoded_feature_names, feature_groups
            )
            model_errors.extend(abs(float(error)) for error in errors if np.isfinite(error))
            model_passed += int(np.count_nonzero(passed))
            model_total += len(passed)
            global_by_repetition: list[dict[str, Any]] = []
            for index, info in enumerate(metadata):
                sample_position = info["sample_position"]
                original_attributions = baseline_group_values[sample_position]
                changed_attributions = perturbed_group_values[index]
                rho = spearman_rho(
                    np.abs(original_attributions), np.abs(changed_attributions)
                )
                original_top = top_k_features(original_attributions, feature_names, 5)
                changed_top = top_k_features(changed_attributions, feature_names, 5)
                changed_set = set(changed_top)
                common = [feature for feature in original_top if feature in changed_set]
                passed_pair = bool(baseline_passed[sample_position] and passed[index])
                records.append(
                    {
                        "disease_id": config.id,
                        "model_name": model_name,
                        "model_version": model_version,
                        "dataset_fingerprint": dataset_fingerprint,
                        "random_seed": info["seed"],
                        "perturbation_seed": info["perturbation_seed"],
                        "repetition": info["repetition"],
                        "perturbation_level": float(level),
                        "sample_id": info["sample_id"],
                        "sample_count": len(sample),
                        "spearman_rho": rho,
                        "spearman_valid": rho is not None,
                        "original_top5_features": original_top,
                        "perturbed_top5_features": changed_top,
                        "common_top5_features": common,
                        "top5_jaccard": (
                            jaccard_similarity(original_top, changed_top)
                            if original_top and changed_top
                            else None
                        ),
                        "original_additivity_error": _finite_number(
                            float(baseline_errors[sample_position])
                        ),
                        "perturbed_additivity_error": _finite_number(float(errors[index])),
                        "additivity_tolerance": SHAP_ADDITIVITY_TOLERANCE,
                        "additivity_passed": passed_pair,
                        "additivity_validated": passed_pair,
                        "score_space": "positive_class_probability",
                        "experiment_timestamp": datetime.now(UTC).isoformat(),
                    }
                )

            for repetition in range(1, repetitions + 1):
                positions = [
                    index
                    for index, info in enumerate(metadata)
                    if info["repetition"] == repetition
                ]
                repetition_importance = mean_abs_shap(perturbed_group_values[positions])
                repetition_seed = metadata[positions[0]]["seed"]
                global_by_repetition.append(
                    {
                        "repetition": repetition,
                        "random_seed": repetition_seed,
                        "spearman_rho": spearman_rho(
                            baseline_importance, repetition_importance
                        ),
                    }
                )
            all_perturbed_importance = mean_abs_shap(perturbed_group_values)
            global_stability.append(
                {
                    "model_name": model_name,
                    "perturbation_level": float(level),
                    "spearman": summarize_repetitions(
                        [row["spearman_rho"] for row in global_by_repetition],
                        seed=SELECTION_SEED + int(round(float(level) * 100)),
                    ),
                    "repetition_results": global_by_repetition,
                    "baseline_feature_importance": baseline_ordered,
                    "perturbed_feature_importance": _ordered_importance(
                        feature_names, all_perturbed_importance
                    ),
                }
            )
        additivity_summary.append(
            {
                "model_name": model_name,
                "is_ensemble": is_ensemble,
                "score_space": "positive_class_probability",
                "tolerance": SHAP_ADDITIVITY_TOLERANCE,
                "total": model_total,
                "passed": model_passed,
                "failed": model_total - model_passed,
                "mean_abs_error": (
                    float(np.mean(model_errors)) if model_errors else None
                ),
                "sd_abs_error": (
                    float(np.std(model_errors, ddof=1)) if len(model_errors) > 1 else 0.0
                ),
                "max_abs_error": max(model_errors) if model_errors else None,
            }
        )
        failed = model_total - model_passed
        if is_ensemble and failed:
            warnings.append(
                f"{model_name}: {failed} of {model_total} probability-space SHAP checks failed; "
                "these explanations are marked unvalidated."
            )

    summary = _build_summary(records, repetitions)
    result: dict[str, Any] = {
        "schema_version": 2,
        "disease_id": config.id,
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": model_version,
        "model_artifact": model_path.name,
        "dataset_fingerprint": dataset_fingerprint,
        "heldout_sample_ids": sample_row_ids,
        "sample_count": len(sample),
        "sample_class_distribution": {
            str(label): int(count)
            for label, count in zip(*np.unique(sample_target, return_counts=True), strict=True)
        },
        "experiment_config": {
            "perturbation_levels": [float(level) for level in deltas],
            "repetitions_per_model_level": repetitions,
            "random_seeds": [int(value) for value in random_seeds],
            "seed_assignment": (
                "Cycle configured seeds by repetition; derive a stream from "
                "(seed, repetition)."
            ),
            "selection_seed": SELECTION_SEED,
            "sample_strategy": (
                "Stratified sample from held-out test partition; same row IDs for all "
                "models and levels."
            ),
            "sample_size_requested": sample_size,
            "sample_size_evaluated": len(sample),
            "heldout_test_rows": int(len(split.x_test)),
            "heldout_seed": int(evaluation["seed"]),
            "heldout_test_fraction": float(evaluation["test_size"]),
            "shap_background_size": int(len(background)),
            "shap_algorithm": "permutation",
            "shap_feature_representation": (
                "Encoded columns; attributions summed by original configured feature."
            ),
            "shap_max_evaluations": max_evals,
            "score_space": "positive_class_probability",
            "perturbation_space": (
                "Standardized numeric outputs from the fitted preprocessing pipeline."
            ),
            "perturbed_features": numeric_features,
            "feature_groups": feature_groups,
            "categorical_features_perturbed": [],
            "clipping": (
                "None; standardized numeric features are perturbed and categorical "
                "one-hot columns remain unchanged."
            ),
            "repetition_unit": (
                "Mean across the fixed held-out sample for one seed/repetition; confidence "
                "intervals resample repetitions, not sample rows."
            ),
            "top_k": 5,
            "permutation_repetitions": 30,
            "permutation_evaluation_data": (
                "Same stratified held-out sample; one-hot columns permuted as a block."
            ),
        },
        "model_configurations": list(MODEL_CONFIGURATIONS),
        "summary": summary,
        "sample_results": records,
        "global_stability": global_stability,
        "shap_vs_permutation": shap_permutation,
        "additivity_summary": additivity_summary,
        "warnings": warnings,
        "interpretation": (
            "Results quantify explanation consistency under the specified perturbations only; "
            "they do not establish medical correctness, causality, or clinical reliability."
        ),
    }
    result = _json_safe(result)
    result["visualizations"] = create_reliability_visualizations(result, disease_dir)
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = disease_dir / f"reliability_{version}.json"
    result["artifact_path"] = str(destination)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    create_disease_comparison(artifacts_dir)
    return result
