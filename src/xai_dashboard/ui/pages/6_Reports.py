"""Download a report generated from stored evaluation results."""

import streamlit as st

from xai_dashboard.ui.client import ApiError
from xai_dashboard.ui.components import load_diseases, require_login

st.set_page_config(page_title="Reports", layout="wide")
client, _user = require_login()
st.title("Reports")
st.caption(
    "Research PDFs use stored evaluation/reliability artifacts. Individual prediction PDFs "
    "are generated in memory from the current prediction and are not persisted."
)
diseases = load_diseases(client)
if diseases:
    selected_id = st.query_params.get("disease")
    disease_ids = [item["id"] for item in diseases]
    selected_index = disease_ids.index(selected_id) if selected_id in disease_ids else 0
    disease = st.selectbox(
        "Disease module",
        diseases,
        index=selected_index,
        format_func=lambda item: item["name"],
    )
    st.subheader("Stored evaluation report")
    if st.button("Prepare evaluation PDF", icon=":material/picture_as_pdf:"):
        try:
            with st.spinner("Preparing evaluation report..."):
                pdf = client.download(f"/reports/{disease['id']}/evaluation.pdf")
            st.download_button(
                "Download evaluation PDF",
                data=pdf,
                file_name=f"{disease['id']}-evaluation.pdf",
                mime="application/pdf",
            )
        except ApiError as exc:
            if exc.status_code == 404:
                st.info("Not yet generated: a stored evaluation artifact is required.")
            else:
                st.error(f"Evaluation report unavailable: {exc}")

    st.subheader("Research report")
    if st.button("Prepare research report PDF", icon=":material/science:"):
        try:
            with st.spinner("Preparing research report..."):
                research_pdf = client.download(f"/reports/{disease['id']}/research.pdf")
            st.download_button(
                "Download research PDF",
                data=research_pdf,
                file_name=f"{disease['id']}-research.pdf",
                mime="application/pdf",
            )
        except ApiError as exc:
            if exc.status_code == 404:
                st.info("Not yet generated: evaluation artifacts are required before reporting.")
            else:
                st.error(f"Research report unavailable: {exc}")
    st.page_link(
        "pages/2_Prediction.py",
        label=f"Create an individual {disease['name']} prediction report",
        icon=":material/analytics:",
    )
st.caption(
    "Reliability findings, when generated, mean consistency under tested "
    "conditions only and never imply medical correctness."
)
