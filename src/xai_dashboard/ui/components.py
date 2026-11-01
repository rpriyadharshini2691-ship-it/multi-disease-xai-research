"""Shared Streamlit components and authentication guards."""

from typing import Any, cast

import streamlit as st

from xai_dashboard.api.schemas import SCORE_DISCLAIMER
from xai_dashboard.ui.client import ApiClient, ApiError


def current_client() -> ApiClient:
    return ApiClient(st.session_state.get("access_token"))


def require_login() -> tuple[ApiClient, dict[str, Any]]:
    token = st.session_state.get("access_token")
    user = st.session_state.get("user")
    if not token or not user:
        st.warning("Sign in to continue.")
        st.switch_page("Home.py")
        st.stop()
    render_sidebar(user)
    return ApiClient(token), user


def render_sidebar(user: dict[str, Any]) -> None:
    with st.sidebar:
        st.markdown("### Research modules")
        st.page_link("Home.py", label="Dashboard", icon=":material/dashboard:")
        st.page_link("pages/2_Prediction.py", label="Prediction", icon=":material/analytics:")
        st.page_link(
            "pages/3_Explainability.py", label="Explainability", icon=":material/insights:"
        )
        st.page_link(
            "pages/4_Performance.py", label="Performance", icon=":material/query_stats:"
        )
        st.page_link(
            "pages/5_Reliability.py", label="Reliability", icon=":material/verified:"
        )
        st.page_link("pages/6_Reports.py", label="Reports", icon=":material/description:")
        if user.get("role") == "admin":
            st.page_link(
                "pages/7_User_Management.py",
                label="User management",
                icon=":material/manage_accounts:",
            )


def load_diseases(client: ApiClient) -> list[dict[str, Any]]:
    try:
        return cast(list[dict[str, Any]], client.get("/diseases"))
    except ApiError as exc:
        st.error(str(exc))
        return []


def feature_form(disease: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for feature in disease["features"]:
        name = feature["name"]
        feature_id = feature["id"]
        bounds = feature.get("valid_range")
        if feature["kind"] == "numeric":
            minimum, maximum = bounds if bounds else (0.0, 100.0)
            unit = feature.get("unit")
            label = f"{name} ({unit})" if unit else name
            values[feature_id] = st.number_input(
                label,
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(minimum),
                key=f"{key_prefix}_{feature_id}",
            )
        elif feature.get("options"):
            values[feature_id] = st.selectbox(
                name,
                feature["options"],
                key=f"{key_prefix}_{feature_id}",
            )
        else:
            values[feature_id] = st.text_input(name, key=f"{key_prefix}_{feature_id}")
    return values


def show_score(score: float, disclaimer: str = SCORE_DISCLAIMER) -> None:
    st.metric("Model-estimated positive-class score", f"{score:.3f}")
    st.caption(disclaimer)
