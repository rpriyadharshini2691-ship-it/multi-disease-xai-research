"""Build PDF reports only from persisted experiment artifacts."""

from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_evaluation_report(disease_name: str, artifact: dict[str, Any]) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"{disease_name} model evaluation", styles["Title"]), Spacer(1, 12)]
    story.append(
        Paragraph(
            "Generated from stored evaluation artifact: "
            f"{artifact.get('generated_at', 'timestamp unavailable')}",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "Metrics are research evaluation results, not clinical performance claims.",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph("This research and education tool is not a clinical device.", styles["BodyText"])
    )
    story.append(
        Paragraph(
            "Reliability results, where included, measure consistency under tested "
            "conditions only and do not imply medical correctness.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 16))
    table_data = [["Model", "Metric", "Value"]]
    for model_name, values in artifact.get("holdout_metrics", {}).items():
        for metric, value in values.items():
            table_data.append([model_name, metric, f"{float(value):.4f}"])
    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#254c45")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d4d0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f2")]),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(table)
    document.build(story)
    return buffer.getvalue()


def _styled_table(rows: list[list[str]]) -> Table:
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#254c45")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d4d0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f2")]),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build_research_report(
    disease_name: str,
    evaluation: dict[str, Any],
    reliability: dict[str, Any] | None,
    visualizations_dir: Path | None = None,
) -> bytes:
    """Research PDF from stored evaluation and (when present) reliability artifacts only."""
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    story: list[Any] = [
        Paragraph(f"{disease_name} research evaluation report", styles["Title"]),
        Paragraph(
            "Model scores are research outputs, not calibrated clinical probabilities. "
            "This tool is not a clinical device.",
            body,
        ),
        Spacer(1, 10),
        Paragraph("Methodology", styles["Heading2"]),
        Paragraph(
            f"Seed {evaluation.get('seed')}; held-out test fraction {evaluation.get('test_size')}. "
            "Preprocessing is fitted on training rows only. Ensemble weights come from "
            "training-only out-of-fold ROC-AUC and are normalized to sum to one.",
            body,
        ),
        Paragraph("Held-out metrics", styles["Heading2"]),
    ]
    metric_rows = [["Model", "Metric", "Value"]]
    for model_name, values in evaluation.get("holdout_metrics", {}).items():
        metric_rows.extend([model_name, metric, f"{float(v):.4f}"] for metric, v in values.items())
    story.append(_styled_table(metric_rows))
    weights = evaluation.get("ensemble_weights", {})
    if weights:
        story.append(Paragraph("Weighted-ensemble weights", styles["Heading2"]))
        story.append(
            _styled_table(
                [["Model", "Weight"], *[[name, f"{float(w):.4f}"] for name, w in weights.items()]]
            )
        )
    story.append(Paragraph("Explanation reliability", styles["Heading2"]))
    if reliability is None:
        story.append(Paragraph("Not yet generated.", body))
    elif reliability.get("schema_version") == 2:
        experiment = reliability["experiment_config"]
        perturbation_levels = ", ".join(
            f"{float(value):.2f}" for value in experiment["perturbation_levels"]
        )
        story.append(
            Paragraph(
                f"Model version {reliability['model_version']}; dataset SHA-256 "
                f"{reliability['dataset_fingerprint']}. The fixed stratified sample contains "
                f"{experiment['sample_size_evaluated']} of {experiment['heldout_test_rows']} "
                f"held-out rows. Seeds: {', '.join(map(str, experiment['random_seeds']))}. "
                f"Each model/level has {experiment['repetitions_per_model_level']} perturbation "
                "repetitions. Repetition means are the analysis unit; confidence intervals "
                "resample repetitions, not held-out rows.",
                body,
            )
        )
        story.append(
            Paragraph(
                "Gaussian noise is applied only to standardized numeric features at levels "
                f"{perturbation_levels}. "
                "Categorical one-hot blocks are unchanged for input perturbation and permuted "
                "as a block in permutation importance. SHAP is calculated in positive-class "
                "probability space; encoded attributions are summed by original feature.",
                body,
            )
        )
        summary_rows = [
            [
                "Model",
                "Delta",
                "Reps",
                "Spearman mean (SD)",
                "Jaccard mean (SD)",
                "Mean |additivity error|",
                "Additivity checks",
            ]
        ]
        for row in reliability.get("summary", []):
            spearman = row["spearman"]
            jaccard = row["top5_jaccard"]
            additivity = row["additivity_abs_error"]
            summary_rows.append(
                [
                    str(row["model_name"]),
                    f"{float(row['perturbation_level']):.2f}",
                    str(row["repetitions_requested"]),
                    _mean_sd(spearman),
                    _mean_sd(jaccard),
                    _format_value(additivity.get("mean"), 4),
                    f"{row['additivity_passed']}/{row['sample_comparisons']}",
                ]
            )
        story.append(Paragraph("Stability summary", styles["Heading3"]))
        story.append(_styled_table(summary_rows))

        global_rows = [["Model", "Delta", "Global Spearman mean (SD)", "Valid repetitions"]]
        for row in reliability.get("global_stability", []):
            values = row["spearman"]
            global_rows.append(
                [
                    str(row["model_name"]),
                    f"{float(row['perturbation_level']):.2f}",
                    _mean_sd(values),
                    str(values["n_repetitions"]),
                ]
            )
        story.append(Paragraph("Global SHAP importance stability", styles["Heading3"]))
        story.append(_styled_table(global_rows))

        comparison_rows = [["Model", "Spearman rho", "Evaluation rows", "Permutation repeats"]]
        for row in reliability.get("shap_vs_permutation", []):
            comparison_rows.append(
                [
                    str(row["model_name"]),
                    _format_value(row.get("spearman_rho"), 4),
                    str(row["evaluation_rows"]),
                    str(row["permutation_repetitions"]),
                ]
            )
        story.append(Paragraph("SHAP versus permutation importance", styles["Heading3"]))
        story.append(_styled_table(comparison_rows))

        additivity_rows = [["Model", "Score space", "Passed", "Failed", "Maximum absolute error"]]
        for row in reliability.get("additivity_summary", []):
            additivity_rows.append(
                [
                    str(row["model_name"]),
                    str(row["score_space"]),
                    str(row["passed"]),
                    str(row["failed"]),
                    _format_value(row.get("max_abs_error"), 6),
                ]
            )
        story.append(Paragraph("SHAP additivity validation", styles["Heading3"]))
        story.append(_styled_table(additivity_rows))
        for warning in reliability.get("warnings", []):
            story.append(Paragraph(f"Warning: {warning}", body))

        if visualizations_dir is not None:
            story.append(Paragraph("Reliability visualizations", styles["Heading3"]))
            for key in (
                "spearman",
                "jaccard",
                "model_comparison",
                "global_importance",
                "shap_vs_permutation",
                "additivity",
            ):
                filename = reliability.get("visualizations", {}).get(key)
                image_path = visualizations_dir / str(filename) if filename else None
                if image_path is not None and image_path.is_file():
                    story.append(Paragraph(key.replace("_", " ").title(), styles["Heading4"]))
                    story.append(Image(str(image_path), width=450, height=260, kind="proportional"))
                    story.append(Spacer(1, 8))
    else:
        rows = [["Delta", "Spearman mean", "Spearman SD", "Top-5 Jaccard", "Repetitions"]]
        for row in reliability.get("stability", {}).get("results", []):
            rows.append(
                [
                    f"{row['delta']:.2f}",
                    f"{row['spearman_rho_mean']:.4f}",
                    f"{row['spearman_rho_sd']:.4f}",
                    f"{row['top_k_jaccard_mean']:.4f}",
                    f"{int(row['repetitions'])}",
                ]
            )
        story.append(_styled_table(rows))
        permutation = reliability.get("shap_vs_permutation", {})
        if "spearman_rho" in permutation:
            story.append(
                Paragraph(
                    f"SHAP vs permutation importance Spearman rho: "
                    f"{float(permutation['spearman_rho']):.4f}.",
                    body,
                )
            )
        story.append(
            Paragraph(
                "Reliability reflects consistency under the tested conditions only, not "
                "medical correctness. Perturbation deltas are experiment settings, not "
                "clinical tolerances.",
                body,
            )
        )
    story.append(Paragraph("Limitations", styles["Heading2"]))
    story.append(
        Paragraph(
            "Results depend on dataset provenance and size; SHAP describes model behaviour "
            "and does not establish causation or clinical validity.",
            body,
        )
    )
    document.build(story)
    return buffer.getvalue()


def _format_value(value: Any, digits: int) -> str:
    return "undefined" if value is None else f"{float(value):.{digits}f}"


def _mean_sd(summary: dict[str, Any]) -> str:
    mean = summary.get("mean")
    sd = summary.get("sd")
    if mean is None:
        return "undefined"
    return f"{float(mean):.3f} ({float(sd):.3f})" if sd is not None else f"{float(mean):.3f} (n/a)"
