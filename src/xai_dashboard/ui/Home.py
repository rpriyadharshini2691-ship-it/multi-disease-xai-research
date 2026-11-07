"""Authentication gate and dashboard overview."""

import pandas as pd
import streamlit as st

from xai_dashboard.ui.client import ApiClient, ApiError
from xai_dashboard.ui.components import render_sidebar

st.set_page_config(page_title="Research Dashboard", layout="wide")
st.title("Multi-Disease XAI Research Dashboard")

if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.access_token:
    login_tab, register_tab = st.tabs(["Sign in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Username or email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")
        if submitted:
            try:
                response = ApiClient().login(email, password)
                st.session_state.access_token = response["access_token"]
                st.session_state.user = response["user"]
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))
    with register_tab:
        with st.form("register_form"):
            full_name = st.text_input("Full name")
            register_username = st.text_input("Username (3-50 chars: a-z, 0-9, _ . -)")
            register_email = st.text_input("Email address")
            register_password = st.text_input("Password (10+ characters)", type="password")
            role = st.selectbox("Role", ["researcher", "clinician"])
            register_submitted = st.form_submit_button("Create account")
        if register_submitted:
            try:
                ApiClient().post(
                    "/auth/register",
                    {
                        "username": register_username,
                        "full_name": full_name,
                        "email": register_email,
                        "password": register_password,
                        "role": role,
                    },
                )
                st.success("Account created. Sign in to continue.")
            except ApiError as exc:
                st.error(str(exc))
    st.info("Research and education use only. This dashboard is not a clinical device.")
    st.stop()

client = ApiClient(st.session_state.access_token)
user = st.session_state.user
render_sidebar(user)
left, right = st.columns([3, 1])
with left:
    st.subheader(f"Welcome, {user['full_name']}")
    st.write(f"Signed in as **{user['role']}**")
with right:
    if st.button("Sign out", icon=":material/logout:"):
        try:
            client.logout()
        except ApiError:
            pass  # token already expired or revoked; local session is cleared regardless
        st.session_state.access_token = None
        st.session_state.user = None
        st.rerun()

st.divider()
st.subheader("Disease evaluation overview")
try:
    diseases = client.get("/diseases")
    order = {"diabetes": 0, "heart": 1, "ckd": 2}
    for disease in sorted(diseases, key=lambda item: order.get(item["id"], 99)):
        st.markdown("---")
        st.subheader(disease["name"])
        st.caption(f"{len(disease['features'])} configured input features")
        with st.spinner(f"Loading {disease['name']} evaluation artifacts..."):
            try:
                result = client.get(f"/performance/{disease['id']}")
            except ApiError as exc:
                st.error(f"Performance results unavailable: {exc}")
                result = {"status": "unavailable"}
        if result["status"] == "ready":
            artifact = result["artifact"]
            metrics = artifact.get("holdout_metrics", {})
            if not metrics:
                st.info(
                    "Not yet generated: no model metrics are present in the evaluation artifact."
                )
            else:
                eligible = [
                    (name, values)
                    for name, values in metrics.items()
                    if values.get("roc_auc") is not None
                ]
                if eligible:
                    best_name, best_values = max(
                        eligible,
                        key=lambda item: (float(item[1]["roc_auc"]), float(item[1].get("f1", 0))),
                    )
                    st.caption(
                        f"Best holdout ROC-AUC: {best_name.replace('_', ' ').title()} "
                        f"({float(best_values['roc_auc']):.3f})"
                    )
                    metric_names = [
                        ("Accuracy", "accuracy"),
                        ("Precision", "precision"),
                        ("Recall", "sensitivity"),
                        ("Specificity", "specificity"),
                        ("F1", "f1"),
                        ("ROC-AUC", "roc_auc"),
                    ]
                    metric_columns = st.columns(6)
                    for column, (label, key) in zip(metric_columns, metric_names, strict=True):
                        value = best_values.get(key)
                        column.metric(label, "n/a" if value is None else f"{float(value):.3f}")

                table = pd.DataFrame(metrics).T.rename(
                    columns={
                        "sensitivity": "recall",
                        "brier_score": "brier score",
                        "roc_auc": "ROC-AUC",
                        "pr_auc": "PR-AUC",
                    }
                )
                table.index.name = "model"
                st.dataframe(table, width="stretch")
                cv = artifact.get("cv", {})
                experiment = cv.get("training_metrics", {})
                st.caption(
                    f"Dataset rows: {artifact.get('dataset_rows', 'n/a')} | "
                    f"Dataset SHA-256: {str(artifact.get('dataset_sha256', 'n/a'))[:16]}… | "
                    f"Holdout seed: {artifact.get('seed', 'n/a')} | "
                    f"Test fraction: {artifact.get('test_size', 'n/a')} | "
                    f"CV: {cv.get('repeats', 'n/a')} repeats × "
                    f"{cv.get('splits', 'n/a')} folds"
                )
                if experiment:
                    st.caption("CV confidence intervals are available on Performance Evaluation.")
        elif result["status"] == "not_generated":
            st.info("Not yet generated: model evaluation artifacts are unavailable.")
        prediction_link, explainability_link = st.columns(2)
        with prediction_link:
            st.page_link(
                "pages/2_Prediction.py",
                label=f"Open {disease['name']} prediction",
                icon=":material/analytics:",
            )
        with explainability_link:
            st.page_link(
                "pages/3_Explainability.py",
                label=f"Open {disease['name']} explainability",
                icon=":material/insights:",
            )
except ApiError as exc:
    st.error(str(exc))

try:
    activity = client.get("/users/me/activity")
    st.subheader("Your activity")
    st.metric("Consented prediction records", activity["consented_prediction_count"])
    if activity["by_disease"]:
        activity_rows = [
            {"Disease": disease_id, "Records": count}
            for disease_id, count in activity["by_disease"].items()
        ]
        st.dataframe(
            activity_rows,
            width="stretch",
            hide_index=True,
        )
except ApiError as exc:
    st.error(str(exc))

st.caption("Model scores are research outputs, not clinical probabilities or diagnoses.")
st.caption(
    "Reliability findings describe consistency under tested conditions only, "
    "not medical correctness."
)
