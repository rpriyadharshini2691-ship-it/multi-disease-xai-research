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
                        use_container_width=True,
                        hide_index=True,
                    )
                with feature_columns[1]:
                    st.write("Perturbed top five")
                    st.dataframe(
                        pd.DataFrame({"Feature": detail["perturbed_top5_features"]}),
                        use_container_width=True,
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
                    use_container_width=True,
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
