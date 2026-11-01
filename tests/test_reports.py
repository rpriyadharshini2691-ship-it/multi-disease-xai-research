from pathlib import Path

from PIL import Image

from xai_dashboard.core.reports.builder import build_research_report

EVALUATION = {
    "seed": 42,
    "test_size": 0.2,
    "holdout_metrics": {"weighted_ensemble": {"accuracy": 0.9, "roc_auc": 0.95}},
    "ensemble_weights": {"logistic_regression": 0.5, "xgboost": 0.5},
}
RELIABILITY = {
    "stability": {
        "results": [
            {
                "delta": 0.05,
                "spearman_rho_mean": 0.99,
                "spearman_rho_sd": 0.01,
                "top_k_jaccard_mean": 0.8,
                "repetitions": 2.0,
            }
        ]
    },
    "shap_vs_permutation": {"spearman_rho": 0.85},
}


def test_research_report_is_pdf_with_and_without_reliability() -> None:
    with_reliability = build_research_report("Toy", EVALUATION, RELIABILITY)
    without_reliability = build_research_report("Toy", EVALUATION, None)
    assert with_reliability.startswith(b"%PDF")
    assert without_reliability.startswith(b"%PDF")
    assert with_reliability != without_reliability


def test_research_report_includes_phase_three_summaries_and_visualizations(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (8, 8), color="white").save(tmp_path / "spearman.png")
    reliability = {
        "schema_version": 2,
        "model_version": "test-version",
        "dataset_fingerprint": "abc123",
        "experiment_config": {
            "sample_size_evaluated": 8,
            "heldout_test_rows": 20,
            "random_seeds": [42, 123, 2026, 2027],
            "repetitions_per_model_level": 30,
            "perturbation_levels": [0.01, 0.05, 0.1],
        },
        "summary": [
            {
                "model_name": "weighted_ensemble",
                "perturbation_level": 0.05,
                "repetitions_requested": 30,
                "spearman": {"mean": 0.9, "sd": 0.02},
                "top5_jaccard": {"mean": 0.8, "sd": 0.1},
                "additivity_abs_error": {"mean": 1e-8},
                "additivity_passed": 240,
                "sample_comparisons": 240,
            }
        ],
        "global_stability": [
            {
                "model_name": "weighted_ensemble",
                "perturbation_level": 0.05,
                "spearman": {"mean": 0.95, "sd": 0.01, "n_repetitions": 30},
            }
        ],
        "shap_vs_permutation": [
            {
                "model_name": "weighted_ensemble",
                "spearman_rho": 0.85,
                "evaluation_rows": 8,
                "permutation_repetitions": 30,
            }
        ],
        "additivity_summary": [
            {
                "model_name": "weighted_ensemble",
                "score_space": "positive_class_probability",
                "passed": 728,
                "failed": 0,
                "max_abs_error": 1e-7,
            }
        ],
        "warnings": [],
        "visualizations": {"spearman": "spearman.png"},
    }
    pdf = build_research_report("Toy", EVALUATION, reliability, visualizations_dir=tmp_path)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000
