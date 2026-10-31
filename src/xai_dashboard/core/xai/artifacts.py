"""Generate persisted global SHAP importance and beeswarm artifacts."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray


def generate_global_explanations(
    model: Any,
    training_frame: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    sample_size: int = 30,
) -> dict[str, str]:
    """Explain the probability-space positive score using a reproducible training sample."""
    if training_frame.empty:
        raise ValueError("Training data is required for global SHAP artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = training_frame.sample(n=min(sample_size, len(training_frame)), random_state=seed)

    def score_function(values: Any) -> NDArray[np.float64]:
        frame = pd.DataFrame(values, columns=training_frame.columns)
        return np.asarray(model.predict_proba(frame)[:, 1], dtype=float)

    explainer = shap.Explainer(score_function, sample, algorithm="permutation")
    explanation = explainer(sample, max_evals=2 * len(training_frame.columns) + 1)
    values = np.asarray(explanation.values, dtype=float)
    importance = np.mean(np.abs(values), axis=0)
    table = pd.DataFrame(
        {"feature": list(training_frame.columns), "mean_abs_shap": importance}
    ).sort_values("mean_abs_shap", ascending=False)
    table_path = output_dir / "global_shap_importance.csv"
    table.to_csv(table_path, index=False)

    shap.plots.beeswarm(
        explanation, max_display=min(20, len(training_frame.columns)), show=False
    )
    plot_path = output_dir / "global_shap_beeswarm.png"
    plt.gcf().tight_layout()
    plt.gcf().savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(plt.gcf())
    return {"importance_csv": table_path.name, "beeswarm_png": plot_path.name}
