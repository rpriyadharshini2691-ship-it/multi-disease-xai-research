"""Disease-specific authenticated prediction form."""

import streamlit as st

from xai_dashboard.ui.client import ApiError
from xai_dashboard.ui.components import feature_form, load_diseases, require_login, show_score

st.set_page_config(page_title="Prediction", layout="wide")
client, _user = require_login()
st.title("Prediction")
diseases = load_diseases(client)
if diseases:
    disease = st.selectbox("Disease module", diseases, format_func=lambda item: item["name"])
    st.caption("Research and education use only. Not a clinical device.")
    model_name = st.selectbox(
        "Model", ["weighted_ensemble", "logistic_regression", "random_forest", "xgboost"]
    )
    with st.form("prediction_form"):
        features = feature_form(disease, "predict")
        consent = st.checkbox("I consent to storing this prediction score in my account activity")
        submitted = st.form_submit_button("Estimate positive-class score", type="primary")
    if submitted:
        try:
            result = client.post(
                "/predict",
                {
                    "disease_id": disease["id"],
                    "features": features,
                    "model_name": model_name,
                    "consent": consent,
                },
            )
            show_score(result["positive_class_score"], result["disclaimer"])
            st.caption(
                "Score stored with your account."
                if result["stored"]
                else "This prediction was not stored."
            )
        except ApiError as exc:
            st.error(str(exc))
