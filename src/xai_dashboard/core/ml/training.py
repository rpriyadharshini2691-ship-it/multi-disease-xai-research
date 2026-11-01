"""Training, evaluation, and artifact persistence for configured disease datasets."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from xai_dashboard.core.data.diseases import DiseaseConfig
from xai_dashboard.core.data.loader import load_dataset, stratified_split
from xai_dashboard.core.data.preprocessing import build_pipeline
from xai_dashboard.core.data.quality import file_sha256
from xai_dashboard.core.xai.artifacts import generate_global_explanations
from xai_dashboard.core.xai.ensemble import encode_ensemble_inputs

plt.switch_backend("Agg")

METRIC_NAMES = (
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
)


class WeightedProbabilityEnsemble(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    def __init__(self, models: dict[str, Any], weights: dict[str, float]) -> None:
        self.models = models
        self.weights = weights
        self.classes_ = np.array([0, 1])
        if set(models) != set(weights):
            raise ValueError("Every ensemble model must have exactly one weight")
        if any(weight < 0 for weight in weights.values()) or not np.isclose(
            sum(weights.values()), 1.0
        ):
            raise ValueError("Ensemble weights must be non-negative and sum to one")

    def fit(
        self, frame: pd.DataFrame, target: NDArray[Any]
    ) -> WeightedProbabilityEnsemble:
        for model in self.models.values():
            model.fit(frame, target)
        self.classes_ = np.unique(target)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        positive = sum(
            self.weights[name] * model.predict_proba(frame)[:, 1]
            for name, model in self.models.items()
        )
        return np.asarray(np.column_stack((1.0 - positive, positive)), dtype=float)


def _metrics(
    target: NDArray[Any], scores: NDArray[np.float64], threshold: float = 0.5
) -> dict[str, float]:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, scores)),
        "pr_auc": float(average_precision_score(target, scores)),
        "brier_score": float(brier_score_loss(target, scores)),
    }


def _pipeline(config: DiseaseConfig, estimator: Any) -> Pipeline:
    return build_pipeline(config, estimator)


def _oof_roc_auc_weights(
    pipelines: dict[str, Pipeline],
    features: pd.DataFrame,
    target: NDArray[np.int64],
    splits: int,
    seed: int,
) -> dict[str, float]:
    """Normalize each model's ROC-AUC from pooled out-of-fold train predictions."""
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    aucs: dict[str, float] = {}
    for name, pipeline in pipelines.items():
        out_of_fold = cross_val_predict(
            pipeline,
            features,
            target,
            cv=splitter,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        aucs[name] = max(0.0, float(roc_auc_score(target, out_of_fold)))
    total = sum(aucs.values())
    if total == 0:
        return {name: 1.0 / len(aucs) for name in aucs}
    return {name: auc / total for name, auc in aucs.items()}


def train_disease(
    config: DiseaseConfig,
    dataset_path: Path,
    artifacts_dir: Path,
    seed: int = 42,
    test_size: float = 0.2,
    cv_splits: int = 5,
    cv_repeats: int = 3,
) -> dict[str, Any]:
    """Fit four estimators and save metrics/plots/models only from actual data."""
    frame = load_dataset(config, dataset_path)
    split = stratified_split(frame, config, test_size=test_size, seed=seed)
    x_train, x_test = split.x_train, split.x_test
    y_train, y_test = split.y_train, split.y_test
    counts = np.bincount(y_train, minlength=2)
    splits = min(cv_splits, int(counts.min()))
    if splits < 2:
        raise ValueError("At least two examples per class are required for stratified evaluation")
    estimators = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        ),
    }
    pipelines = {name: _pipeline(config, estimator) for name, estimator in estimators.items()}
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=cv_repeats, random_state=seed)
    fold_metrics: dict[str, list[dict[str, float]]] = {
        **{name: [] for name in pipelines},
        "equal_weight_ensemble": [],
        "weighted_ensemble": [],
    }
    for fold_index, (training_indices, validation_indices) in enumerate(cv.split(x_train, y_train)):
        fold_x_train = x_train.iloc[training_indices]
        fold_y_train = y_train[training_indices]
        fold_x_valid = x_train.iloc[validation_indices]
        fold_y_valid = y_train[validation_indices]
        inner_splits = min(3, int(np.bincount(fold_y_train, minlength=2).min()))
        if inner_splits >= 2:
            fold_weights = _oof_roc_auc_weights(
                pipelines,
                fold_x_train,
                fold_y_train,
                inner_splits,
                seed + fold_index,
            )
        else:
            fold_weights = {name: 1.0 / len(pipelines) for name in pipelines}
        fold_scores: dict[str, NDArray[np.float64]] = {}
        for name, pipeline in pipelines.items():
            pipeline.fit(fold_x_train, fold_y_train)
            fold_scores[name] = np.asarray(
                pipeline.predict_proba(fold_x_valid)[:, 1], dtype=float
            )
            fold_metrics[name].append(_metrics(fold_y_valid, fold_scores[name]))
        ensemble_scores = np.zeros(len(fold_y_valid), dtype=float)
        for name, values in fold_scores.items():
            ensemble_scores += fold_weights[name] * values
        fold_metrics["weighted_ensemble"].append(_metrics(fold_y_valid, ensemble_scores))
        equal_scores = np.mean(np.column_stack(list(fold_scores.values())), axis=1)
        fold_metrics["equal_weight_ensemble"].append(_metrics(fold_y_valid, equal_scores))

    rng = np.random.default_rng(seed)
    cv_results: dict[str, dict[str, Any]] = {}
    for model_name, observations in fold_metrics.items():
        cv_results[model_name] = {}
        for metric_name in METRIC_NAMES:
            values = np.asarray([item[metric_name] for item in observations], dtype=float)
            bootstrap_means = np.mean(
                rng.choice(values, size=(1000, len(values)), replace=True), axis=1
            )
            cv_results[model_name][metric_name] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "bootstrap_ci_95": [
                    float(np.quantile(bootstrap_means, 0.025)),
                    float(np.quantile(bootstrap_means, 0.975)),
                ],
                "folds": float(len(values)),
            }
    weights = _oof_roc_auc_weights(pipelines, x_train, y_train, splits, seed)
    equal_weights = {name: 1.0 / len(pipelines) for name in pipelines}
    for pipeline in pipelines.values():
        pipeline.fit(x_train, y_train)
    ensemble = WeightedProbabilityEnsemble(pipelines, weights)
    equal_ensemble = WeightedProbabilityEnsemble(pipelines, equal_weights)
    models: dict[str, Any] = {
        **pipelines,
        "equal_weight_ensemble": equal_ensemble,
        "weighted_ensemble": ensemble,
    }
    scores: dict[str, NDArray[np.float64]] = {
        name: np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
        for name, model in models.items()
    }
    metrics = {name: _metrics(y_test, prediction) for name, prediction in scores.items()}
    output = artifacts_dir / config.id
    output.mkdir(parents=True, exist_ok=True)
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    model_path = output / f"models_{version}.joblib"
    joblib.dump(
        {"models": models, "weights": weights, "features": list(config.feature_columns)}, model_path
    )
    result = {
        "disease_id": config.id,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "dataset_rows": int(len(frame)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "experiment_label": "post_cleaning_v2",
        "experiment_note": (
            "New experiment on the verified, de-duplicated dataset; not comparable with "
            "pre-cleaning metrics, which were computed on data containing duplicate rows."
        ),
        "leakage_safeguards": {
            "duplicates_handled_before_split": True,
            "preprocessing_fitted_on_training_rows_only": True,
            "ensemble_weights_from_training_oof_only": True,
            "test_set_used_only_for_final_metrics": True,
        },
        "seed": seed,
        "test_size": test_size,
        "cv": {
            "repeats": cv_repeats,
            "splits": splits,
            "interval_method": "bootstrap percentile interval over repeated-fold metric means",
            "training_metrics": cv_results,
        },
        "holdout_metrics": metrics,
        "equal_ensemble_weights": equal_weights,
        "ensemble_weights": weights,
        "model_artifact": model_path.name,
        "model_version": version,
        "reliability_interpretation": (
            "Consistency under tested conditions only; not medical correctness."
        ),
    }
    metrics_path = output / f"evaluation_{version}.json"
    _write_plots(output, version, y_test, scores)
    global_shap_artifacts: dict[str, dict[str, str]] = {}
    model_bundle = {"models": models, "weights": weights}
    for model_name in models:
        explanation_model, explanation_frame = encode_ensemble_inputs(
            model_bundle, x_train, model_name
        )
        global_shap_artifacts[model_name] = generate_global_explanations(
            explanation_model,
            explanation_frame,
            output,
            seed=seed,
            artifact_prefix=f"global_shap_{model_name}",
        )
    result["global_shap_artifacts"] = global_shap_artifacts
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _write_plots(
    output: Path,
    version: str,
    target: NDArray[Any],
    scores: dict[str, NDArray[np.float64]],
) -> None:
    names = list(scores)
    figure, axis = plt.subplots(figsize=(9, 4))
    auc_values = [roc_auc_score(target, scores[name]) for name in names]
    axis.bar(names, auc_values)
    axis.set_ylabel("ROC-AUC (held-out test set)")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output / f"model_comparison_{version}.png", dpi=140)
    plt.close(figure)
    for name, prediction in scores.items():
        figure, axes = plt.subplots(1, 4, figsize=(16, 3.5))
        axes[0].imshow(confusion_matrix(target, prediction >= 0.5, labels=[0, 1]), cmap="Blues")
        axes[0].set_title(f"{name}: confusion")
        false_positive, true_positive, _ = roc_curve(target, prediction)
        axes[1].plot(false_positive, true_positive)
        axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axes[1].set_title("ROC curve")
        precision, recall, _ = precision_recall_curve(target, prediction)
        axes[2].plot(recall, precision)
        axes[2].set_title("Precision-recall")
        observed, predicted = calibration_curve(
            target, prediction, n_bins=min(10, max(2, len(target) // 10)), strategy="quantile"
        )
        axes[3].plot(predicted, observed, marker="o")
        axes[3].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axes[3].set_title("Calibration (quantile bins)")
        figure.tight_layout()
        figure.savefig(output / f"evaluation_{name}_{version}.png", dpi=140)
        plt.close(figure)
