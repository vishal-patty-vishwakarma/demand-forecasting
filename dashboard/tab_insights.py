import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import math

st.subheader("Business Insights & Inventory Recommendations")

@st.cache_data
def load_forecast():
    return pd.read_parquet(
        "outputs/forecasts/forecast_table.parquet"
    )

@st.cache_data
def load_oof():
    return pd.read_parquet("outputs/oof_predictions.parquet")

forecast_df = load_forecast()
oof_df      = load_oof()

# Store selector
store = st.selectbox(
    "Select Store",
    options=sorted(forecast_df["store_nbr"].unique()),
    key="insights_store"
)

st.caption(
    "Safety stock assumes 7-day lead time and "
    "90% service level (z=1.28). "
    "Adjust lead time for your supply chain."
)

# Filter forecast for selected store
store_fc = forecast_df[
    forecast_df["store_nbr"] == store
].copy()

# Compute per-item metrics
def compute_item_metrics(grp):
    rmse = float(np.sqrt(
        np.mean((grp["predicted"] - grp["unit_sales"])**2)
    ))
    mean_pred = float(grp["predicted"].mean())
    bias_val  = float(
        np.mean(grp["predicted"] - grp["unit_sales"])
    )
    lead_time    = 7
    z            = 1.28
    safety_stock = z * rmse * math.sqrt(lead_time)
    reorder_pt   = mean_pred * lead_time + safety_stock
    return pd.Series({
        "mean_predicted": round(mean_pred,    2),
        "rmse":           round(rmse,         2),
        "bias":           round(bias_val,     3),
        "safety_stock":   round(safety_stock, 1),
        "reorder_point":  round(reorder_pt,   1)
    })

item_metrics = (
    store_fc.groupby(["item_nbr","family"])
    .apply(compute_item_metrics)
    .reset_index()
)

# Risk flag
def risk_flag(row):
    if row["safety_stock"] > 50:
        return "🔴 HIGH"
    elif row["safety_stock"] > 20:
        return "🟡 MEDIUM"
    else:
        return "🟢 LOW"

item_metrics["risk"] = item_metrics.apply(risk_flag, axis=1)

# Safety stock table
st.markdown("### Safety Stock Recommendations")
st.dataframe(
    item_metrics[[
        "item_nbr","family","mean_predicted",
        "safety_stock","reorder_point","risk"
    ]].sort_values("safety_stock", ascending=False)
    .head(20),
    use_container_width=True,
    hide_index=True
)

# Over and under forecasted
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### ⚠️ Stockout Risk")
    st.caption("Items we consistently under-forecast")
    under = (
        item_metrics.nsmallest(10, "bias")
        [["item_nbr","family","bias"]]
    )
    under["bias"] = under["bias"].round(3)
    st.dataframe(under, use_container_width=True,
                 hide_index=True)

with col_b:
    st.markdown("### 📦 Overstock Risk")
    st.caption("Items we consistently over-forecast")
    over = (
        item_metrics.nlargest(10, "bias")
        [["item_nbr","family","bias"]]
    )
    over["bias"] = over["bias"].round(3)
    st.dataframe(over, use_container_width=True,
                 hide_index=True)

# Family SMAPE chart
st.markdown("### Model Accuracy by Product Family")

family_smape = []
for fam in store_fc["family"].unique():
    mask = store_fc["family"] == fam
    if mask.sum() < 10:
        continue
    s = float(np.mean(
        2 * np.abs(
            store_fc.loc[mask,"unit_sales"].values -
            store_fc.loc[mask,"predicted"].values
        ) / (
            np.abs(store_fc.loc[mask,"unit_sales"].values) +
            np.abs(store_fc.loc[mask,"predicted"].values) + 1e-8
        )
    ) * 100)
    family_smape.append({"family": fam, "smape": round(s, 1)})

fam_df = (
    pd.DataFrame(family_smape)
    .sort_values("smape")
)

fig_fam = px.bar(
    fam_df,
    x="smape",
    y="family",
    orientation="h",
    color="smape",
    color_continuous_scale="RdYlGn_r",
    title=f"Store {store} — SMAPE by Product Family"
)
fig_fam.add_vline(
    x=49.99,
    line_dash="dash",
    line_color="red",
    annotation_text="Baseline"
)
fig_fam.update_layout(height=500, showlegend=False)
st.plotly_chart(fig_fam, use_container_width=True)