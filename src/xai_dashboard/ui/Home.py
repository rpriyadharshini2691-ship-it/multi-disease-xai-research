"""Authentication gate and dashboard overview."""

import streamlit as st

from xai_dashboard.ui.client import ApiClient, ApiError

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
            email = st.text_input("Email")
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
            register_email = st.text_input("Email address")
            register_password = st.text_input("Password (10+ characters)", type="password")
            role = st.selectbox("Role", ["researcher", "clinician"])
            register_submitted = st.form_submit_button("Create account")
        if register_submitted:
            try:
                ApiClient().post(
                    "/auth/register",
                    {
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
left, right = st.columns([3, 1])
with left:
    st.subheader(f"Welcome, {user['full_name']}")
    st.write(f"Signed in as **{user['role']}**")
with right:
    if st.button("Sign out", icon=":material/logout:"):
        st.session_state.access_token = None
        st.session_state.user = None
        st.rerun()

st.divider()
st.subheader("Disease modules")
try:
    diseases = client.get("/diseases")
    columns = st.columns(3)
    for column, disease in zip(columns, diseases, strict=False):
        with column:
            st.markdown(f"### {disease['name']}")
            result = client.get(f"/performance/{disease['id']}")
            if result["status"] == "ready":
                st.success("Evaluation artifacts available")
            else:
                st.caption("Evaluation not yet generated")
            st.caption(f"{len(disease['features'])} configured input features")
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
            use_container_width=True,
            hide_index=True,
        )
except ApiError as exc:
    st.error(str(exc))

st.caption("Model scores are research outputs, not clinical probabilities or diagnoses.")
st.caption(
    "Reliability findings describe consistency under tested conditions only, "
    "not medical correctness."
)
