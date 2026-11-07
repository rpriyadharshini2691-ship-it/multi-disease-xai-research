"""Local and global SHAP explanation views."""

import base64
from typing import Any

import pandas as pd
import streamlit as st

from xai_dashboard.ui.client import ApiError
from xai_dashboard.ui.components import feature_form, load_diseases, require_login, show_score


def _as_float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _group_contributions(
    rows: list[dict[str, Any]],
    disease: dict[str, Any],
    values: dict[str, Any] | None = None,
) -> pd.DataFrame:
    grouped: list[dict[str, Any]] = []
    feature_list: list[dict[str, Any]] = disease.get("features", [])
    for feature in feature_list:
        feature_id = str(feature["id"])
        if feature.get("kind") == "numeric":
            encoded_names = {f"numeric__{feature_id}"}
        else:
            encoded_names = {
                str(row["feature"])
                for row in rows
                if str(row["feature"]).startswith(f"categorical__{feature_id}_")
            }
        selected = [row for row in rows if str(row["feature"]) in encoded_names]
        if not selected:
            continue
        value = values.get(feature_id) if values is not None else selected[0].get("value")
        contributions: list[float] = []
        for row in selected:
            raw_value = row.get("shap_value", row.get("mean_abs_shap", 0.0))
            if raw_value is not None:
                contributions.append(float(raw_value))
        attribution = sum(contributions)
        grouped.append(
            {
                "Feature": feature.get("name", feature_id),
                "Value": value,
                "SHAP contribution": attribution,
                "Direction": (
                    "increases"
                    if attribution > 0
                    else "decreases" if attribution < 0 else "neutral"
                ),
            }
        )
    return pd.DataFrame(grouped).sort_values(
        "SHAP contribution", key=lambda column: column.abs(), ascending=False
    ) if grouped else pd.DataFrame(columns=["Feature", "Value", "SHAP contribution", "Direction"])


def _group_global_importance(
    rows: list[dict[str, Any]], disease: dict[str, Any]
) -> pd.DataFrame:
    grouped: list[dict[str, Any]] = []
    feature_list: list[dict[str, Any]] = disease.get("features", [])
    for feature in feature_list:
        feature_id = str(feature["id"])
        if feature.get("kind") == "numeric":
            encoded_names = {f"numeric__{feature_id}"}
        else:
            encoded_names = {
                str(row["feature"])
                for row in rows
                if str(row["feature"]).startswith(f"categorical__{feature_id}_")
            }
        importance_values = []
        for row in rows:
            if str(row["feature"]) in encoded_names:
                value = _as_float_or_none(row.get("mean_abs_shap"))
                if value is not None:
                    importance_values.append(value)
        importance = sum(importance_values)
        if encoded_names:
            grouped.append(
                {
                    "Feature": feature.get("name", feature_id),
                    "Mean absolute SHAP": importance,
                }
            )
    grouped.sort(key=lambda item: -float(item["Mean absolute SHAP"]))
    return pd.DataFrame(grouped)

st.set_page_config(page_title="Explainability", layout="wide")
client, _user = require_login()
st.title("Explainability")
st.caption(
    "SHAP values describe the selected model's contribution to its positive-class score; "
    "they do not establish causality or medical correctness."
)
diseases = load_diseases(client)
if diseases:
    disease = st.selectbox("Disease module", diseases, format_func=lambda item: item["name"])
    model_name = st.selectbox(
        "Model",
        [
            "weighted_ensemble",
            "equal_weight_ensemble",
            "logistic_regression",
            "random_forest",
            "xgboost",
        ],
        format_func=lambda item: item.replace("_", " ").title(),
    )
    with st.form("explain_form"):
        features = feature_form(disease, "explain")
        submitted = st.form_submit_button("Generate local explanation")
    if submitted:
        try:
            result = client.post(
                f"/explain/{disease['id']}/local",
                {"features": features, "model_name": model_name},
            )
            show_score(result["positive_class_score"], result["disclaimer"])
            validation_passed = result.get("additivity_passed", False) and result.get(
                "score_match_passed", True
            )
            if not validation_passed:
                st.error(
                    result.get(
                        "validation_warning",
                        "SHAP validation failed. This explanation is not validated.",
                    )
                )
                st.metric("Explanation validation", "Not validated")
                st.caption(f"Additivity error: {result.get('additivity_error', float('nan')):.6g}")
            else:
                st.success("SHAP additivity and model-score checks passed.")
                model_score = result.get(
                    "raw_model_score", result["positive_class_score"]
                )
                st.caption(
                    f"Base + contributions = {result['additivity_sum']:.6f}; "
                    f"model score = {model_score:.6f}; "
                    f"additivity error = {result.get('additivity_error', 0.0):.2e}"
                )
                st.image(
                    base64.b64decode(result["waterfall_png_base64"]),
                    caption=f"{result.get('model_name', model_name)} local SHAP waterfall",
                )
                table = _group_contributions(result["features"], disease, features)
                st.subheader("Top feature contributions")
                st.dataframe(table.head(5), width="stretch", hide_index=True)
        except ApiError as exc:
            if exc.status_code == 422:
                st.error(f"Input validation or explanation failed: {exc}")
            else:
                st.info(f"Local explanation unavailable: {exc}")

st.divider()
st.subheader("Global explanations")
if diseases:
    try:
        global_result = client.get(
            f"/explain/{disease['id']}/global?model_name={model_name}"
        )
        if global_result["status"] == "ready":
            st.image(
                base64.b64decode(global_result["beeswarm_png_base64"]),
                caption=(
                    f"{model_name.replace('_', ' ').title()} global SHAP beeswarm "
                    "(encoded feature representation)"
                ),
            )
            table = _group_global_importance(global_result["features"], disease)
            st.subheader("Global feature importance")
            st.dataframe(
                table,
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(global_result.get("message", "Not yet generated."))
    except ApiError as exc:
        st.error(f"Global explanation unavailable: {exc}")
