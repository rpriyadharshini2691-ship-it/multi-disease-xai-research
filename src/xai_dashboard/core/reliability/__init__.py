"""Explanation reliability experiments."""

from xai_dashboard.core.reliability.metrics import (
    compare_shap_and_permutation,
    jaccard_top_k,
    run_stability_experiment,
)

__all__ = ["compare_shap_and_permutation", "jaccard_top_k", "run_stability_experiment"]
