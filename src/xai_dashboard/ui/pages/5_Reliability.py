"""Explanation reliability results and their interpretation limits."""

import pandas as pd
import streamlit as st

from xai_dashboard.ui.client import ApiError
from xai_dashboard.ui.components import load_diseases, require_login

st.set_page_config(page_title="Reliability Analysis", layout="wide")
client, _user = require_login()
st.title("Explanation Reliability")
st.warning(
    "Reliability results measure consistency under the tested conditions only. "
    "They do not establish medical correctness."
)
st.caption(
    "Perturbation deltas 0.01, 0.05, and 0.10 are experimental settings on "
    "standardized numeric features, not clinical tolerances."
)
diseases = load_diseases(client)
if diseases:
    disease = st.selectbox("Disease module", diseases, format_func=lambda item: item["name"])
    if st.button("Run reliability experiments", type="primary"):
        try:
            job = client.post(f"/jobs/reliability/{disease['id']}", {})
            st.session_state.reliability_job_id = job["job_id"]
        except ApiError as exc:
            st.error(str(exc))
    job_id = st.session_state.get("reliability_job_id")
    if job_id and st.button("Refresh experiment status"):
        try:
            job = client.get(f"/jobs/{job_id}")
            st.write(f"Experiment status: {job['status']}")
            if job["status"] == "failed":
                st.error(job.get("error", "Experiment failed."))
        except ApiError as exc:
            st.error(str(exc))
    try:
        response = client.get(f"/reliability/{disease['id']}")
        if response["status"] != "ready":
            st.info(
                "Reliability experiments are not yet generated. No values are "
                "displayed until experiment artifacts exist."
            )
        else:
            artifact = response["artifact"]
            stability = artifact.get("stability", {}).get("results", [])
            if stability:
                st.subheader("SHAP stability and top-5 overlap")
                st.dataframe(pd.DataFrame(stability), use_container_width=True, hide_index=True)
            permutation = artifact.get("shap_vs_permutation")
            if permutation:
                st.subheader("SHAP vs permutation importance")
                st.metric("Spearman rho", f"{permutation['spearman_rho']:.3f}")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Feature": permutation["features"],
                            "Mean |SHAP|": permutation["shap_mean_abs"],
                            "Permutation": permutation["permutation_mean"],
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
    except ApiError as exc:
        st.error(str(exc))
