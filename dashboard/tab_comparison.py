import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import json

st.subheader("Model Performance Comparison")

# Load experiment log
@st.cache_data
def load_experiment():
    with open("outputs/experiments.json", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_oof():
    return pd.read_parquet("outputs/oof_predictions.parquet")

exp = load_experiment()

# Model comparison table
st.markdown("### Overall SMAPE by Model")

comparison_data = {
    "Model": [
        "Seasonal Naive",
        "Moving Average (28d)",
        "SARIMA (avg 3 series)",
        "LightGBM Global"
    ],
    "SMAPE (%)": [81.94, 49.99, 54.40, 46.97],
    "Notes": [
        "Predict = same day last week",
        "Predict = 28-day rolling average",
        "Per-series, 229hrs for all series",
        "Global model, minutes to train ✓"
    ]
}
comparison_df = pd.DataFrame(comparison_data)

# Color best row
def highlight_best(row):
    if row["Model"] == "LightGBM Global":
        return ["background-color: #d4edda"] * len(row)
    return [""] * len(row)

# FIXED: format SMAPE column to 2 decimal places
st.dataframe(
    comparison_df.style
    .apply(highlight_best, axis=1)
    .format({"SMAPE (%)": "{:.2f}"}),
    use_container_width=True,
    hide_index=True
)

# Bar chart
fig_bar = px.bar(
    comparison_df,
    x="Model",
    y="SMAPE (%)",
    color="SMAPE (%)",
    color_continuous_scale="RdYlGn_r",
    title="SMAPE Comparison — Lower is Better"
)
fig_bar.add_hline(
    y=46.97,
    line_dash="dash",
    line_color="green",
    annotation_text="LightGBM Best"
)
fig_bar.update_layout(height=400, showlegend=False)
st.plotly_chart(fig_bar, use_container_width=True)

# CV fold results
st.markdown("### LightGBM Cross-Validation Results")
if "fold_results" in exp:
    fold_df = pd.DataFrame(exp["fold_results"])
    fig_fold = px.line(
        fold_df,
        x="fold",
        y="smape",
        markers=True,
        title="SMAPE per CV Fold — Walk-Forward Validation"
    )
    fig_fold.add_hline(
        y=49.99,
        line_dash="dash",
        line_color="red",
        annotation_text="Baseline"
    )
    fig_fold.update_layout(height=350)
    st.plotly_chart(fig_fold, use_container_width=True)

# Key metrics from experiment
st.markdown("### LightGBM Model Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("CV SMAPE",    f"{exp['smape_mean']}%")
c2.metric("CV MAE",      f"{exp['mae_mean']}")
c3.metric("Bias",        f"{exp['bias_mean']}")
c4.metric("vs Baseline", f"-{exp['improvement']}%",
          delta=f"-{exp['improvement']}%")