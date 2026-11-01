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
        "Model",
        [
            "weighted_ensemble",
            "equal_weight_ensemble",
            "logistic_regression",
            "random_forest",
            "xgboost",
        ],
    )
    with st.form("prediction_form"):
        features = feature_form(disease, "predict")
        consent = st.checkbox("I consent to storing this prediction score in my account activity")
        submitted = st.form_submit_button("Estimate positive-class score", type="primary")
    if submitted:
        try:
            request_payload = {
                "disease_id": disease["id"],
                "features": features,
                "model_name": model_name,
                "consent": consent,
            }
            result = client.post("/predict", request_payload)
            st.session_state["last_prediction_report_request"] = {
                **request_payload,
                "consent": False,
            }
            show_score(result["positive_class_score"], result["disclaimer"])
            st.caption(
                "Score stored with your account."
                if result["stored"]
                else "This prediction was not stored."
            )
            st.caption(
                f"Model: {result['model_name']} ({result['model_version']}) | "
                f"Prediction reference: {result['prediction_reference']}"
            )
            if result["additivity_passed"]:
                st.success("The SHAP explanation passed probability-score validation.")
            else:
                st.warning("The SHAP explanation is not validated for this prediction.")
            st.dataframe(
                result["important_features"],
                use_container_width=True,
                hide_index=True,
            )
        except ApiError as exc:
            st.error(str(exc))

if st.session_state.get("last_prediction_report_request"):
    if st.button("Generate prediction PDF"):
        try:
            pdf, reference = client.post_download(
                "/reports/prediction.pdf",
                st.session_state["last_prediction_report_request"],
            )
            filename = f"prediction-{reference or 'report'}.pdf"
            st.download_button(
                "Download PDF",
                data=pdf,
                file_name=filename,
                mime="application/pdf",
            )
        except ApiError as exc:
            st.error(str(exc))
