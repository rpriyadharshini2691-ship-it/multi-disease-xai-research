"""Local and global SHAP explanation views."""

import base64

import pandas as pd
import streamlit as st

from xai_dashboard.ui.client import ApiError
from xai_dashboard.ui.components import feature_form, load_diseases, require_login, show_score

st.set_page_config(page_title="Explainability", layout="wide")
client, _user = require_login()
st.title("Explainability")
st.caption("Local SHAP values explain the model output in positive-class score space.")
diseases = load_diseases(client)
if diseases:
    disease = st.selectbox("Disease module", diseases, format_func=lambda item: item["name"])
    with st.form("explain_form"):
        features = feature_form(disease, "explain")
        submitted = st.form_submit_button("Generate local explanation")
    if submitted:
        try:
            result = client.post(f"/explain/{disease['id']}/local", {"features": features})
            show_score(result["positive_class_score"], result["disclaimer"])
            st.metric("Additivity check", "Passed" if result["additivity_passed"] else "Failed")
            st.caption(f"Base value + SHAP contributions = {result['additivity_sum']:.6f}")
            st.image(
                base64.b64decode(result["waterfall_png_base64"]),
                caption="Local SHAP waterfall",
            )
            table = pd.DataFrame(result["features"]).sort_values(
                "shap_value", key=lambda column: column.abs(), ascending=False
            )
            st.dataframe(table, use_container_width=True, hide_index=True)
        except ApiError as exc:
            st.info(str(exc))

st.divider()
st.subheader("Global explanations")
if diseases:
    try:
        global_result = client.get(f"/explain/{disease['id']}/global")
        if global_result["status"] == "ready":
            st.image(
                base64.b64decode(global_result["beeswarm_png_base64"]),
                caption="Global SHAP beeswarm",
            )
            st.dataframe(
                pd.DataFrame(global_result["features"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(global_result["message"])
    except ApiError as exc:
        st.error(str(exc))
