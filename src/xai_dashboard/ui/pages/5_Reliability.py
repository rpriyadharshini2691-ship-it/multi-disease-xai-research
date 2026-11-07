"""Explanation reliability results and their interpretation limits."""

from typing import Any

import pandas as pd
import streamlit as st

from xai_dashboard.settings import get_settings
from xai_dashboard.ui.client import ApiError
from xai_dashboard.ui.components import load_diseases, require_login

st.set_page_config(page_title="Reliability Analysis", layout="wide")
client, _user = require_login()
st.title("Explanation Reliability")
st.warning(
    "Results measure explanation consistency under the tested conditions only. "
    "They do not establish medical correctness, causality, or clinical reliability."
)
diseases = load_diseases(client)
if diseases:
    disease = st.selectbox("Disease", diseases, format_func=lambda item: item["name"])
    job_key = f"reliability_job_id:{disease['id']}"
    controls = st.columns([1, 1, 1])
    with controls[0]:
        if st.button("Run 30-repeat evaluation", type="primary"):
            try:
                job = client.post(
                    f"/jobs/reliability/{disease['id']}",
                    {"repetitions": 30, "sample_size": 8},
                )
                st.session_state[job_key] = job["job_id"]
            except ApiError as exc:
                st.error(str(exc))
    job_id = st.session_state.get(job_key)
    with controls[1]:
        if job_id and st.button("Refresh job"):
            try:
                job = client.get(f"/jobs/{job_id}")
                st.session_state[f"{job_key}:status"] = job["status"]
                if job["status"] == "failed":
                    st.error(job.get("error", "Experiment failed."))
            except ApiError as exc:
                st.error(str(exc))
    with controls[2]:
        if job_id:
            st.caption(f"Job: {st.session_state.get(f'{job_key}:status', 'queued')}")

    try:
        response = client.get(f"/reliability/{disease['id']}")
        if response["status"] != "ready":
            st.info("Not yet generated.")
        else:
            artifact: dict[str, Any] = response["artifact"]
            if artifact.get("schema_version") != 2:
                st.info("Not yet generated for the current evaluation design.")
            else:
                experiment = artifact["experiment_config"]
                st.caption(
                    f"Held-out stratified sample: {experiment['sample_size_evaluated']} of "
                    f"{experiment['heldout_test_rows']} test rows. "
                    f"{experiment['repetitions_per_model_level']} perturbation repetitions per "
                    f"model and level; seeds {', '.join(map(str, experiment['random_seeds']))}. "
                    "Noise is applied only to standardized numeric features; one-hot category "
                    "blocks stay unchanged."
                )
                with st.expander("Metric definitions"):
                    st.markdown(
                        "- **Spearman rho** compares ranks of absolute grouped feature "
                        "attributions; constant or insufficient rankings are undefined.\n"
                        "- **Top-five Jaccard** is the intersection divided by the union of the "
                        "original and perturbed feature-name sets.\n"
                        "- **Global stability** compares mean absolute SHAP importance over the "
                        "same held-out sample. Confidence intervals resample repetitions, "
                        "not rows.\n"
                        "- **Additivity error** is the absolute difference between model "
                        "probability "
                        "and SHAP baseline plus contributions; failed checks are not validated."
                    )

                selectors = st.columns(2)
                models = artifact["model_configurations"]
                with selectors[0]:
                    model_name: str = st.selectbox("Model configuration", models)
                levels = experiment["perturbation_levels"]
                with selectors[1]:
                    level = float(
                        st.selectbox(
                            "Perturbation level",
                            levels,
                            format_func=lambda value: f"{float(value):.2f}",
                        )
                    )
                summary = next(
                    row
                    for row in artifact["summary"]
                    if row["model_name"] == model_name
                    and float(row["perturbation_level"]) == float(level)
                )
                score = summary["spearman"]
                overlap = summary["top5_jaccard"]
                additivity = summary["additivity_abs_error"]
                metric_columns = st.columns(3)
                with metric_columns[0]:
                    st.metric(
                        "Spearman mean (SD)",
                        "Undefined" if score["mean"] is None else f"{score['mean']:.3f}",
                        help="Mean of repetition-level averages across held-out samples.",
                    )
                    st.caption(
                        f"SD {score['sd'] if score['sd'] is not None else 'n/a'}; "
                        f"valid repetitions {score['n_repetitions']}/"
                        f"{experiment['repetitions_per_model_level']}"
                    )
                    spearman_ci = score.get("confidence_interval_95", [None, None])
                    if all(value is not None for value in spearman_ci):
                        st.caption(
                            f"95% repetition-bootstrap CI: [{spearman_ci[0]:.3f}, "
                            f"{spearman_ci[1]:.3f}]"
                        )
                with metric_columns[1]:
                    st.metric(
                        "Jaccard mean (SD)",
                        "n/a" if overlap["mean"] is None else f"{overlap['mean']:.3f}",
                        help="Overlap of original and perturbed top-five feature names.",
                    )
                    st.caption(
                        f"SD {overlap['sd'] if overlap['sd'] is not None else 'n/a'}; "
                        f"{summary['sample_comparisons']} sample comparisons"
                    )
                    jaccard_ci = overlap.get("confidence_interval_95", [None, None])
                    if all(value is not None for value in jaccard_ci):
                        st.caption(
                            f"95% repetition-bootstrap CI: [{jaccard_ci[0]:.3f}, "
                            f"{jaccard_ci[1]:.3f}]"
                        )
                with metric_columns[2]:
                    st.metric(
                        "Mean absolute additivity error",
                        "n/a" if additivity["mean"] is None else f"{additivity['mean']:.2e}",
                    )
                    st.caption(
                        f"SD {additivity['sd'] if additivity['sd'] is not None else 'n/a'}; "
                        f"passed {summary['additivity_passed']}, "
                        f"failed {summary['additivity_failed']}"
                    )

                rows = [
                    row
                    for row in artifact["sample_results"]
                    if row["model_name"] == model_name
                    and float(row["perturbation_level"]) == float(level)
                ]
                sample_ids = sorted({row["sample_id"] for row in rows})
                sample_id = st.selectbox("Held-out sample", sample_ids)
                repetitions = sorted(
                    row["repetition"] for row in rows if row["sample_id"] == sample_id
                )
                repetition = st.selectbox("Repetition", repetitions)
                detail = next(
                    row
                    for row in rows
                    if row["sample_id"] == sample_id and row["repetition"] == repetition
                )
                feature_columns = st.columns(2)
                with feature_columns[0]:
                    st.write("Original top five")
                    st.dataframe(
                        pd.DataFrame({"Feature": detail["original_top5_features"]}),
                        width="stretch",
                        hide_index=True,
                    )
                with feature_columns[1]:
                    st.write("Perturbed top five")
                    st.dataframe(
                        pd.DataFrame({"Feature": detail["perturbed_top5_features"]}),
                        width="stretch",
                        hide_index=True,
                    )
                st.write(f"Common features: {', '.join(detail['common_top5_features']) or 'none'}")
                rho_value = detail["spearman_rho"]
                rho_label = "undefined" if rho_value is None else f"{rho_value:.3f}"
                st.write(
                    f"Jaccard: {detail['top5_jaccard']:.3f} | Spearman: {rho_label} | "
                    f"Seed: {detail['random_seed']} | Repetition: {detail['repetition']}"
                )
                if detail["additivity_passed"]:
                    st.success(
                        f"Both original and perturbed SHAP checks passed; perturbed error "
                        f"{abs(detail['perturbed_additivity_error']):.2e}."
                    )
                else:
                    st.warning(
                        "At least one SHAP explanation failed additivity and is not validated."
                    )
                if artifact.get("warnings"):
                    for warning in artifact["warnings"]:
                        st.warning(warning)

                st.subheader("Model-wise stability at selected perturbation")
                model_rows = [
                    row
                    for row in artifact["summary"]
                    if float(row["perturbation_level"]) == float(level)
                ]
                model_table = [
                    {
                        "Model": row["model_name"],
                        "Spearman mean": row["spearman"]["mean"],
                        "Spearman SD": row["spearman"]["sd"],
                        "Valid repetitions": row["spearman"]["n_repetitions"],
                        "Jaccard mean": row["top5_jaccard"]["mean"],
                        "Jaccard SD": row["top5_jaccard"]["sd"],
                        "Additivity error mean": row["additivity_abs_error"]["mean"],
                        "Additivity passed/total": (
                            f"{row['additivity_passed']}/{row['sample_comparisons']}"
                        ),
                    }
                    for row in model_rows
                ]
                st.dataframe(pd.DataFrame(model_table), width="stretch", hide_index=True)

                global_rows = [
                    row
                    for row in artifact.get("global_stability", [])
                    if row["model_name"] == model_name
                ]
                if global_rows:
                    st.subheader("Global explanation stability")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Perturbation": row["perturbation_level"],
                                    "Global Spearman mean": row["spearman"]["mean"],
                                    "Global Spearman SD": row["spearman"]["sd"],
                                    "Valid repetitions": row["spearman"]["n_repetitions"],
                                    "Original top features": ", ".join(
                                        item["feature"]
                                        for item in row["baseline_feature_importance"][:5]
                                    ),
                                    "Perturbed top features": ", ".join(
                                        item["feature"]
                                        for item in row["perturbed_feature_importance"][:5]
                                    ),
                                }
                                for row in global_rows
                            ]
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("Not yet generated: global stability results are unavailable.")

                permutation_rows = [
                    row
                    for row in artifact.get("shap_vs_permutation", [])
                    if row["model_name"] == model_name
                ]
                if permutation_rows:
                    permutation = permutation_rows[0]
                    st.subheader("SHAP versus permutation importance")
                    rho = permutation["spearman_rho"]
                    st.metric(
                        "Spearman rho",
                        "Undefined" if rho is None else f"{rho:.3f}",
                        help=(
                            "Ranking agreement on the same held-out sample; undefined "
                            "for constant ranks."
                        ),
                    )
                    st.caption(
                        f"{permutation['evaluation_rows']} held-out rows | "
                        f"{permutation['permutation_repetitions']} grouped permutations | "
                        f"seed {permutation['seed']} | "
                        f"{permutation['feature_representation']}"
                    )
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Feature": permutation["features"],
                                "Mean |SHAP|": permutation["shap_mean_abs"],
                                "Permutation mean": permutation["permutation_mean"],
                                "Permutation SD": permutation["permutation_sd"],
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("Not yet generated: SHAP-versus-permutation results are unavailable.")

                st.subheader("Ensemble additivity validation")
                ensemble_checks = [
                    row
                    for row in artifact.get("additivity_summary", [])
                    if row.get("is_ensemble")
                ]
                if ensemble_checks:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Model": row["model_name"],
                                    "Score space": row["score_space"],
                                    "Passed": row["passed"],
                                    "Failed": row["failed"],
                                    "Maximum absolute error": row["max_abs_error"],
                                }
                                for row in ensemble_checks
                            ]
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("Not yet generated: ensemble additivity results are unavailable.")

                disease_rows: list[dict[str, Any]] = []
                for disease_item in diseases:
                    disease_artifact = artifact if disease_item["id"] == disease["id"] else None
                    if disease_artifact is None:
                        try:
                            disease_response = client.get(
                                f"/reliability/{disease_item['id']}"
                            )
                            if disease_response.get("status") == "ready":
                                disease_artifact = disease_response["artifact"]
                        except ApiError:
                            disease_artifact = None
                    if not disease_artifact or disease_artifact.get("schema_version") != 2:
                        continue
                    for perturbation in experiment["perturbation_levels"]:
                        entries = [
                            row
                            for row in disease_artifact["summary"]
                            if float(row["perturbation_level"]) == float(perturbation)
                            and row["spearman"]["mean"] is not None
                        ]
                        means = [float(row["spearman"]["mean"]) for row in entries]
                        deviations = [float(row["spearman"]["sd"]) for row in entries]
                        if means:
                            disease_rows.append(
                                {
                                    "Disease": disease_item["name"],
                                    "Perturbation": float(perturbation),
                                    "Mean across model summaries": sum(means) / len(means),
                                    "SD across model summaries": (
                                        pd.Series(means).std(ddof=1) if len(means) > 1 else 0.0
                                    ),
                                    "Mean within-repetition SD": sum(deviations) / len(deviations),
                                    "Models included": len(means),
                                }
                            )
                if disease_rows:
                    st.subheader("Disease-wise stability")
                    st.caption(
                        "Disease-wise mean/SD summarize model-level means at each perturbation; "
                        "within-repetition SD is reported separately."
                    )
                    st.dataframe(
                        pd.DataFrame(disease_rows),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("Not yet generated: no disease-wise results are available.")

                st.subheader("Experiment results")
                st.dataframe(
                    pd.DataFrame(rows)[
                        [
                            "random_seed",
                            "repetition",
                            "sample_id",
                            "spearman_rho",
                            "original_top5_features",
                            "perturbed_top5_features",
                            "common_top5_features",
                            "top5_jaccard",
                            "perturbed_additivity_error",
                            "additivity_passed",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

                settings = get_settings()
                artifact_dir = settings.artifacts_dir / disease["id"]
                chart_names = artifact.get("visualizations", {})
                for key in (
                    "spearman",
                    "jaccard",
                    "model_comparison",
                    "global_importance",
                    "shap_vs_permutation",
                    "additivity",
                ):
                    chart_path = artifact_dir / chart_names.get(key, "")
                    if chart_names.get(key) and chart_path.is_file():
                        st.image(str(chart_path), caption=key.replace("_", " ").title())
                disease_chart = (
                    settings.artifacts_dir
                    / "reliability_summary"
                    / "reliability_disease_comparison.png"
                )
                if disease_chart.is_file():
                    st.image(str(disease_chart), caption="Disease-wise stability comparison")
    except ApiError as exc:
        st.error(str(exc))
