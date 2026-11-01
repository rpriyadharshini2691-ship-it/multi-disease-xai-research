"""Data-backed plots for stored explanation reliability results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


def create_reliability_visualizations(
    result: dict[str, Any], output_dir: Path
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    models = list(dict.fromkeys(row["model_name"] for row in summary))
    levels = sorted({float(row["perturbation_level"]) for row in summary})
    paths = {
        "spearman": "reliability_spearman.png",
        "jaccard": "reliability_jaccard.png",
        "model_comparison": "reliability_model_comparison.png",
        "global_importance": "reliability_global_shap_stability.png",
        "shap_vs_permutation": "reliability_shap_vs_permutation.png",
        "additivity": "reliability_ensemble_additivity.png",
    }

    for metric, title, filename, bounds in (
        ("spearman", "SHAP rank correlation by perturbation", paths["spearman"], (-1, 1)),
        ("top5_jaccard", "Top-five SHAP overlap by perturbation", paths["jaccard"], (0, 1)),
    ):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for model in models:
            rows = [row for row in summary if row["model_name"] == model]
            rows.sort(key=lambda row: float(row["perturbation_level"]))
            level_values = [float(row["perturbation_level"]) for row in rows]
            stats = [row[metric] for row in rows]
            means = [np.nan if stat["mean"] is None else stat["mean"] for stat in stats]
            deviations = [0.0 if stat["sd"] is None else stat["sd"] for stat in stats]
            axis.errorbar(
                level_values, means, yerr=deviations, marker="o", capsize=3, label=model
            )
        axis.set(title=title, xlabel="Perturbation level (standardized numeric units)")
        axis.set_ylabel("Spearman rho" if metric == "spearman" else "Jaccard similarity")
        axis.set_ylim(*bounds)
        axis.set_xticks(levels, [f"{level:.2f}" for level in levels])
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="best", fontsize="small")
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(len(levels), 1)
    model_positions = np.arange(len(models), dtype=float)
    for level_index, level in enumerate(levels):
        rows = [
            next(
                row
                for row in summary
                if row["model_name"] == model
                and float(row["perturbation_level"]) == level
            )
            for model in models
        ]
        offset = (level_index - (len(levels) - 1) / 2) * width
        axis.bar(
            model_positions + offset,
            [
                np.nan if row["spearman"]["mean"] is None else row["spearman"]["mean"]
                for row in rows
            ],
            width=width,
            label=f"delta={level:.2f}",
        )
    axis.set_xticks(model_positions, models, rotation=20, ha="right")
    axis.set_ylim(-1, 1)
    axis.set(title="Model-wise mean explanation rank stability", ylabel="Mean Spearman rho")
    axis.legend(title="Perturbation")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / paths["model_comparison"], dpi=150)
    plt.close(figure)

    global_rows = result["global_stability"]
    figure, axes = plt.subplots(3, 2, figsize=(14, 12), squeeze=False)
    for axis, model in zip(axes.flat, models, strict=False):
        model_rows = [row for row in global_rows if row["model_name"] == model]
        first = model_rows[0]
        baseline = first["baseline_feature_importance"]
        top_features = [row["feature"] for row in baseline[:8]]
        baseline_values = {
            row["feature"]: row["mean_abs_shap"] for row in baseline
        }
        feature_positions = np.arange(len(top_features), dtype=float)
        axis.bar(
            feature_positions - 0.15,
            [
                np.nan if baseline_values[feature] is None else baseline_values[feature]
                for feature in top_features
            ],
            width=0.3,
            label="Original",
        )
        for row in model_rows:
            values = {item["feature"]: item["mean_abs_shap"] for item in row[
                "perturbed_feature_importance"
            ]}
            axis.plot(
                feature_positions + 0.15,
                [
                    np.nan if values.get(feature) is None else values[feature]
                    for feature in top_features
                ],
                marker="o",
                label=f"delta={row['perturbation_level']:.2f}",
            )
        axis.set_title(model)
        axis.set_xticks(feature_positions, top_features, rotation=45, ha="right")
        axis.set_ylabel("Mean |SHAP|")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize="x-small")
    axes.flat[-1].set_visible(False)
    figure.suptitle("Global SHAP importance: original and perturbed held-out sample")
    figure.tight_layout()
    figure.savefig(output_dir / paths["global_importance"], dpi=150)
    plt.close(figure)

    comparison = result["shap_vs_permutation"]
    figure, axes = plt.subplots(3, 2, figsize=(12, 12), squeeze=False)
    for axis, model in zip(axes.flat, models, strict=False):
        row = next(item for item in comparison if item["model_name"] == model)
        shap_values = np.asarray(row["shap_mean_abs"], dtype=float)
        permutation_values = np.asarray(row["permutation_mean"], dtype=float)
        axis.scatter(permutation_values, shap_values, alpha=0.8)
        rho = row["spearman_rho"]
        label = "undefined" if rho is None else f"rho={rho:.3f}"
        axis.set_title(f"{model} ({label})")
        axis.set_xlabel("Permutation importance (ROC-AUC decrease)")
        axis.set_ylabel("Mean |SHAP| (same held-out sample)")
        axis.grid(alpha=0.2)
    axes.flat[-1].set_visible(False)
    figure.suptitle("SHAP versus permutation importance")
    figure.tight_layout()
    figure.savefig(output_dir / paths["shap_vs_permutation"], dpi=150)
    plt.close(figure)

    additivity = [row for row in result["additivity_summary"] if row["is_ensemble"]]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    names = [row["model_name"] for row in additivity]
    rates = [row["passed"] / row["total"] if row["total"] else 0.0 for row in additivity]
    axis.bar(names, rates, color=["#2f7f72", "#e29a38"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Validated explanation fraction")
    axis.set_title("Probability-space ensemble SHAP additivity")
    for index, row in enumerate(additivity):
        axis.text(index, rates[index] + 0.02, f"{row['passed']}/{row['total']}", ha="center")
    axis.axhline(1.0, color="#444444", linewidth=0.8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / paths["additivity"], dpi=150)
    plt.close(figure)
    return paths


def create_disease_comparison(artifacts_dir: Path) -> Path | None:
    diseases = ("diabetes", "heart", "ckd")
    grouped: dict[str, dict[float, list[float]]] = {}
    for disease in diseases:
        candidates = sorted((artifacts_dir / disease).glob("reliability_*.json"))
        current = [
            path
            for path in candidates
            if json.loads(path.read_text(encoding="utf-8")).get("schema_version") == 2
        ]
        if not current:
            return None
        result = json.loads(current[-1].read_text(encoding="utf-8"))
        disease_values: dict[float, list[float]] = {}
        for row in result["summary"]:
            value = row["spearman"]["mean"]
            if value is not None:
                disease_values.setdefault(float(row["perturbation_level"]), []).append(
                    float(value)
                )
        grouped[disease] = disease_values

    output_dir = artifacts_dir / "reliability_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "reliability_disease_comparison.png"
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for disease, values in grouped.items():
        levels = sorted(values)
        means = [float(np.mean(values[level])) for level in levels]
        deviations = [float(np.std(values[level], ddof=1)) if len(values[level]) > 1 else 0.0
                      for level in levels]
        axis.errorbar(levels, means, yerr=deviations, marker="o", capsize=3, label=disease)
    axis.set_ylim(-1, 1)
    axis.set_xticks(sorted({level for values in grouped.values() for level in values}))
    axis.set_xlabel("Perturbation level (standardized numeric units)")
    axis.set_ylabel("Mean Spearman rho across model configurations")
    axis.set_title("Disease-wise explanation stability")
    axis.legend(title="Disease")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path