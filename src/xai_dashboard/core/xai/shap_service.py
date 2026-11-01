"""Probability-space SHAP values with an explicit local additivity check."""

import base64
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray


def explain_instance(
    model: Any, background: pd.DataFrame, instance: pd.DataFrame
) -> dict[str, Any]:
    if instance.shape[0] != 1:
        raise ValueError("Exactly one input row is required for a local explanation")
    def score_function(values: Any) -> NDArray[np.float64]:
        probabilities = model.predict_proba(
            pd.DataFrame(values, columns=background.columns)
        )
        return np.asarray(probabilities[:, 1], dtype=float)

    explainer = shap.Explainer(score_function, background, algorithm="permutation")
    explanation = explainer(instance, max_evals=2 * len(background.columns) + 1)
    values = np.asarray(explanation.values).reshape(-1)
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    score = float(score_function(instance)[0])
    total = base_value + float(values.sum())
    tolerance = 1e-4
    if not np.isclose(total, score, atol=tolerance, rtol=tolerance):
        raise ArithmeticError(f"SHAP additivity check failed: {total} != {score}")
    shap.plots.waterfall(explanation[0], max_display=min(20, len(instance.columns)), show=False)
    image = BytesIO()
    plt.gcf().savefig(image, format="png", dpi=140, bbox_inches="tight")
    plt.close(plt.gcf())
    return {
        "base_value": base_value,
        "positive_class_score": score,
        "additivity_sum": total,
        "additivity_error": total - score,
        "additivity_tolerance": tolerance,
        "additivity_passed": True,
        "waterfall_png_base64": base64.b64encode(image.getvalue()).decode("ascii"),
        "features": [
            {
                "feature": str(name),
                "value": instance.iloc[0][name].item()
                if hasattr(instance.iloc[0][name], "item")
                else instance.iloc[0][name],
                "shap_value": float(value),
                "direction": "increases" if value > 0 else "decreases" if value < 0 else "neutral",
            }
            for name, value in zip(instance.columns, values, strict=True)
        ],
    }
