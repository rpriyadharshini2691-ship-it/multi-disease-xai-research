"""Download a report generated from stored evaluation results."""

import streamlit as st

from xai_dashboard.ui.client import ApiError
from xai_dashboard.ui.components import load_diseases, require_login

st.set_page_config(page_title="Reports", layout="wide")
client, _user = require_login()
st.title("Reports")
st.caption("Reports contain only metrics from stored evaluation artifacts.")
diseases = load_diseases(client)
if diseases:
    disease = st.selectbox("Disease module", diseases, format_func=lambda item: item["name"])
    if st.button("Prepare evaluation PDF", icon=":material/picture_as_pdf:"):
        try:
            pdf = client.download(f"/reports/{disease['id']}/evaluation.pdf")
            st.download_button(
                "Download PDF",
                data=pdf,
                file_name=f"{disease['id']}-evaluation.pdf",
                mime="application/pdf",
            )
        except ApiError as exc:
            st.info(str(exc))
st.caption(
    "Reliability findings, when generated, mean consistency under tested "
    "conditions only and never imply medical correctness."
)
