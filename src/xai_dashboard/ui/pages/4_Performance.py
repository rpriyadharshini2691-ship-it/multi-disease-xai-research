"""Metrics and plots produced by the training/evaluation pipeline."""

import base64

import pandas as pd
import streamlit as st

from xai_dashboard.ui.client import ApiError
from xai_dashboard.ui.components import load_diseases, require_login

st.set_page_config(page_title="Performance Evaluation", layout="wide")
client, _user = require_login()
st.title("Performance Evaluation")
diseases = load_diseases(client)
if diseases:
    disease = st.selectbox("Disease module", diseases, format_func=lambda item: item["name"])
    try:
        response = client.get(f"/performance/{disease['id']}")
        if response["status"] != "ready":
            st.info(
                "Evaluation artifacts are not yet generated. Provide a dataset "
                "and run the training job first."
            )
        else:
            artifact = response["artifact"]
            dataset_name = artifact["dataset_path"].replace("\\", "/").rsplit("/", 1)[-1]
            st.caption(
                f"Generated {artifact['generated_at']} from {dataset_name}"
            )
            st.dataframe(
                pd.DataFrame(artifact["holdout_metrics"]).T, use_container_width=True
            )
            st.subheader("Repeated stratified CV on training data")
            cv = artifact.get("cv", {}).get("training_metrics", {})
            if cv:
                st.json(cv)
            for image in response.get("plots", []):
                st.image(
                    base64.b64decode(image["png_base64"]),
                    caption=image["filename"].removesuffix(".png").replace("_", " "),
                )
            st.caption(
                "All displayed metrics and plots are loaded from stored "
                "training/evaluation artifacts."
            )
    except ApiError as exc:
        st.error(str(exc))
