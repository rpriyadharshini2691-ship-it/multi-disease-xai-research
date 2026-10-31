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
    return ApiClient(token), user


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
        else:
            values[feature_id] = st.text_input(name, key=f"{key_prefix}_{feature_id}")
    return values


def show_score(score: float, disclaimer: str = SCORE_DISCLAIMER) -> None:
    st.metric("Model-estimated positive-class score", f"{score:.3f}")
    st.caption(disclaimer)
