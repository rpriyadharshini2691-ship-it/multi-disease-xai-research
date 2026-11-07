"""Metrics and plots produced by the training/evaluation pipeline."""

import base64

import pandas as pd
import streamlit as st

from xai_dashboard.ui.client import ApiError
from xai_dashboard.ui.components import load_diseases, require_login

st.set_page_config(page_title="Performance Evaluation", layout="wide")
client, _user = require_login()
st.title("Performance Evaluation")
st.caption(
    "Holdout metrics use the stored test split. CV confidence intervals summarize repeated "
    "stratified folds on training data and are not confidence intervals over independent patients."
)
diseases = load_diseases(client)
if diseases:
    disease = st.selectbox("Disease module", diseases, format_func=lambda item: item["name"])
    try:
        with st.spinner(f"Loading {disease['name']} evaluation artifacts..."):
            response = client.get(f"/performance/{disease['id']}")
        if response["status"] != "ready":
            st.info("Not yet generated: evaluation metrics and plots are unavailable.")
        else:
            artifact = response["artifact"]
            dataset_name = artifact["dataset_path"].replace("\\", "/").rsplit("/", 1)[-1]
            st.caption(
                f"Generated {artifact['generated_at']} from {dataset_name} | "
                f"Rows: {artifact.get('dataset_rows', 'n/a')} | "
                f"Seed: {artifact.get('seed', 'n/a')} | "
                f"Test fraction: {artifact.get('test_size', 'n/a')} | "
                f"Dataset SHA-256: {str(artifact.get('dataset_sha256', 'n/a'))[:16]}…"
            )
            holdout = artifact.get("holdout_metrics", {})
            if not holdout:
                st.info("Not yet generated: holdout metric values are missing.")
                st.stop()
            metric_labels = {
                "accuracy": "Accuracy",
                "precision": "Precision",
                "sensitivity": "Recall",
                "specificity": "Specificity",
                "f1": "F1",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
                "brier_score": "Brier score",
            }
            comparison = pd.DataFrame(holdout).T.rename(columns=metric_labels)
            comparison.index.name = "Model configuration"
            st.subheader("Held-out test comparison")
            st.dataframe(comparison, width="stretch")
            eligible = [
                (name, values)
                for name, values in holdout.items()
                if values.get("roc_auc") is not None
            ]
            if eligible:
                best_name, best_values = max(
                    eligible,
                    key=lambda item: (float(item[1]["roc_auc"]), float(item[1].get("f1", 0))),
                )
                st.success(
                    f"Best by held-out ROC-AUC: {best_name.replace('_', ' ').title()} "
                    f"({float(best_values['roc_auc']):.3f})."
                )
            st.subheader("Repeated stratified CV on training data")
            cv_artifact = artifact.get("cv", {})
            cv = cv_artifact.get("training_metrics", {})
            if cv:
                rows: list[dict[str, object]] = []
                for model, model_metrics in cv.items():
                    for metric, statistics in model_metrics.items():
                        if metric not in metric_labels:
                            continue
                        interval = statistics.get("bootstrap_ci_95", [None, None])
                        lower, upper = interval
                        rows.append(
                            {
                                "Model": model.replace("_", " ").title(),
                                "Metric": metric_labels[metric],
                                "Mean": statistics.get("mean"),
                                "SD": statistics.get("sd"),
                                "95% bootstrap CI": (
                                    "n/a"
                                    if lower is None or upper is None
                                    else f"[{float(lower):.3f}, {float(upper):.3f}]"
                                ),
                                "Folds": statistics.get("folds"),
                            }
                        )
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                st.caption(
                    f"{cv_artifact.get('repeats', 'n/a')} repeats × "
                    f"{cv_artifact.get('splits', 'n/a')} folds; "
                    f"{cv_artifact.get('interval_method', 'interval method unavailable')}."
                )
            else:
                st.info("Not yet generated: cross-validation confidence intervals are unavailable.")

            plots = response.get("plots", [])
            comparison_plot = next(
                (image for image in plots if image["filename"].startswith("model_comparison_")),
                None,
            )
            if comparison_plot:
                st.subheader("Model comparison plot")
                st.image(
                    base64.b64decode(comparison_plot["png_base64"]),
                    caption="Held-out ROC-AUC by model configuration",
                )
            model_names = list(holdout)
            if model_names:
                selected_model = st.selectbox(
                    "Confusion matrix, ROC, precision-recall and calibration plots",
                    model_names,
                    format_func=lambda value: value.replace("_", " ").title(),
                )
                model_plot = next(
                    (
                        image
                        for image in plots
                        if image["filename"].startswith(f"evaluation_{selected_model}_")
                    ),
                    None,
                )
                if model_plot:
                    st.image(
                        base64.b64decode(model_plot["png_base64"]),
                        caption=(
                            f"{selected_model.replace('_', ' ').title()}: confusion matrix, "
                            "ROC, precision-recall and calibration panels"
                        ),
                    )
                else:
                    st.info("Not yet generated: diagnostic plots for this model are unavailable.")
            st.caption(
                "All metrics and charts are read from stored evaluation artifacts; this page "
                "does not retrain models."
            )
    except ApiError as exc:
        st.error(str(exc))
