"""Disease-specific authenticated prediction form."""

import streamlit as st

from xai_dashboard.ui.client import ApiError
from xai_dashboard.ui.components import feature_form, load_diseases, require_login, show_score

st.set_page_config(page_title="Prediction", layout="wide")
client, _user = require_login()
st.title("Prediction")
st.caption(
    "Choose one disease module and complete its configured inputs. Results are research "
    "estimates, not medical diagnoses or treatment recommendations."
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
    with st.form(f"prediction_form_{disease['id']}"):
        features = feature_form(disease, "predict")
        consent = st.checkbox("I consent to storing this prediction score in my account activity")
        submitted = st.form_submit_button("Estimate model score", type="primary")
    prediction_context = {
        "disease_id": disease["id"],
        "model_name": model_name,
        "features": features,
        "consent": consent,
    }
    prediction_result = None
    if submitted:
        st.session_state.pop("last_prediction_result", None)
        st.session_state.pop("last_prediction_context", None)
        st.session_state.pop("last_prediction_report_request", None)
        try:
            result = client.post("/predict", prediction_context)
            st.session_state["last_prediction_result"] = result
            st.session_state["last_prediction_context"] = prediction_context
            st.session_state["last_prediction_report_request"] = {
                **prediction_context,
                "consent": False,
            }
        except ApiError as exc:
            if exc.status_code == 422:
                st.error(f"Input validation failed: {exc}")
            elif exc.status_code >= 500:
                st.info(f"Prediction unavailable: {exc}")
            else:
                st.error(str(exc))
    elif st.session_state.get("last_prediction_context") == prediction_context:
        prediction_result = st.session_state.get("last_prediction_result")

    if submitted and "last_prediction_result" in st.session_state:
        prediction_result = st.session_state["last_prediction_result"]
    if prediction_result:
        st.subheader("Model estimate")
        st.metric("Model prediction", str(prediction_result["predicted_class"]).title())
        show_score(
            prediction_result["positive_class_score"], prediction_result["disclaimer"]
        )
        st.caption(
            "Score stored with your account."
            if prediction_result["stored"]
            else "This prediction was not stored."
        )
        st.caption(
            f"Model: {prediction_result['model_name']} ({prediction_result['model_version']}) | "
            f"Prediction reference: {prediction_result['prediction_reference']}"
        )
        if prediction_result["additivity_passed"]:
            st.success("The SHAP explanation passed probability-score validation.")
        else:
            st.warning("The SHAP explanation is not validated for this prediction.")
        st.subheader("Important feature contributions")
        st.caption(
            "Positive SHAP contributions increase the model-estimated positive-class score; "
            "negative contributions decrease it. These explain model behavior, not causation."
        )
        st.dataframe(
            prediction_result["important_features"],
            use_container_width=True,
            hide_index=True,
        )

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
