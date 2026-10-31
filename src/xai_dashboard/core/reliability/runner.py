"""Generate and persist the three specified explanation reliability experiments."""

from __future__ import annotations

import json
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
from xai_dashboard.core.data.loaders import load_dataset
from xai_dashboard.core.reliability.metrics import (
    compare_shap_and_permutation,
    run_stability_experiment,
)


def run_reliability(
    config: DiseaseConfig,
    dataset_path: Path,
    artifacts_dir: Path,
    seed: int = 42,
    repetitions: int = 30,
) -> dict[str, Any]:
    """Use the trained model and its training partition to create real reliability results."""
    disease_dir = artifacts_dir / config.id
    model_files = sorted(disease_dir.glob("models_*.joblib"))
    if not model_files:
        raise FileNotFoundError("Train and persist a model before running reliability experiments")
    x, target_labels = load_dataset(config, dataset_path)
    y = np.asarray(target_labels == config.positive_label, dtype=int)
    evaluation_files = sorted(disease_dir.glob("evaluation_*.json"))
    if not evaluation_files:
        raise FileNotFoundError("Evaluation artifact is missing; train the model first")
    evaluation = json.loads(evaluation_files[-1].read_text(encoding="utf-8"))
    x_train, _, y_train, _ = train_test_split(
        x,
        y,
        test_size=float(evaluation["test_size"]),
        stratify=y,
        random_state=int(evaluation["seed"]),
    )
    artifact = joblib.load(model_files[-1])
    model = artifact["models"]["weighted_ensemble"]
    background = x_train.sample(n=min(30, len(x_train)), random_state=seed)
    def score_function(values: Any) -> NDArray[np.float64]:
        probabilities = model.predict_proba(pd.DataFrame(values, columns=x.columns))
        return np.asarray(probabilities[:, 1], dtype=float)

    explainer = shap.Explainer(score_function, background, algorithm="permutation")
    max_evals = 2 * len(config.feature_columns) + 1

    def explain(frame: pd.DataFrame) -> NDArray[np.float64]:
        values = explainer(frame, max_evals=max_evals).values
        return np.asarray(values, dtype=float)

    numeric = list(config.numeric_columns)
    means = x_train[numeric].astype(float).mean() if numeric else pd.Series(dtype=float)
    scales = (
        x_train[numeric].astype(float).std(ddof=0).replace(0, 1)
        if numeric
        else pd.Series(dtype=float)
    )
    standardized = x_train.sample(n=min(30, len(x_train)), random_state=seed).copy()
    for column in numeric:
        standardized[column] = (standardized[column].astype(float) - means[column]) / scales[column]

    def explain_standardized(frame: pd.DataFrame) -> NDArray[np.float64]:
        raw = frame.copy()
        for column in numeric:
            raw[column] = raw[column].astype(float) * scales[column] + means[column]
        return explain(raw)

    stability = run_stability_experiment(
        standardized,
        explain_standardized,
        numeric,
        deltas=(0.01, 0.05, 0.10),
        repetitions=repetitions,
        seed=seed,
        top_k=5,
    )
    global_sample = background
    global_shap = explain(global_sample)
    comparison = compare_shap_and_permutation(
        model, x_train, y_train, global_shap, repetitions=10, seed=seed
    )
    result: dict[str, Any] = {
        "disease_id": config.id,
        "generated_at": datetime.now(UTC).isoformat(),
        "model_artifact": model_files[-1].name,
        "training_partition_rows": int(len(x_train)),
        "stability": stability,
        "shap_vs_permutation": comparison,
        "interpretation": (
            "Reliability measures consistency under tested conditions only; "
            "it does not imply medical correctness."
        ),
        "perturbation_note": (
            "Deltas 0.01, 0.05, and 0.10 are experimental settings on standardized "
            "numeric features, not clinical tolerances."
        ),
    }
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = disease_dir / f"reliability_{version}.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
