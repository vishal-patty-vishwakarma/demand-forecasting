import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

@st.cache_data
def load_forecast():
    return pd.read_parquet(
        "outputs/forecasts/forecast_table.parquet"
    )

df = load_forecast()

st.subheader("Forecast vs Actual Sales")
st.caption(
    "Select a store and product family to view "
    "the 28-day forecast with confidence intervals."
)

# Controls
col1, col2 = st.columns(2)
store = col1.selectbox(
    "Store",
    options=sorted(df["store_nbr"].unique()),
    key="forecast_store"
)
family = col2.selectbox(
    "Product Family",
    options=sorted(df["family"].unique()),
    key="forecast_family"
)

# Filter data
filtered = df[
    (df["store_nbr"] == store) &
    (df["family"]    == family)
].sort_values("date")

if len(filtered) == 0:
    st.warning("No data for this store-family combination.")
else:
    # Aggregate by date (sum across items)
    daily = filtered.groupby("date").agg(
        actual      =("unit_sales",  "sum"),
        predicted   =("predicted",   "sum"),
        lower_bound =("lower_bound", "sum"),
        upper_bound =("upper_bound", "sum")
    ).reset_index()

    # Plotly chart
    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([daily["date"],
                     daily["date"][::-1]]),
        y=pd.concat([daily["upper_bound"],
                     daily["lower_bound"][::-1]]),
        fill="toself",
        fillcolor="rgba(255, 100, 100, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="90% Confidence Band",
        showlegend=True
    ))

    # Actual
    fig.add_trace(go.Scatter(
        x=daily["date"],
        y=daily["actual"],
        name="Actual Sales",
        line=dict(color="steelblue", width=2)
    ))

    # Predicted
    fig.add_trace(go.Scatter(
        x=daily["date"],
        y=daily["predicted"],
        name="Predicted Sales",
        line=dict(color="red", width=2, dash="dash")
    ))

    fig.update_layout(
        title=f"Store {store} | {family} | 28-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        height=450,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Key metrics
    m1, m2, m3 = st.columns(3)
    total_actual    = daily["actual"].sum()
    total_predicted = daily["predicted"].sum()
    pct_diff = (
        (total_predicted - total_actual) /
        (total_actual + 1e-8) * 100
    )

    m1.metric("Total Actual Sales",    f"{total_actual:,.0f} units")
    m2.metric("Total Predicted Sales", f"{total_predicted:,.0f} units")
    m3.metric(
        "Forecast vs Actual",
        f"{pct_diff:+.1f}%",
        delta_color="inverse"
    )

    # Download button
    csv = daily.to_csv(index=False)
    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name=f"forecast_store{store}_{family}.csv",
        mime="text/csv"
    )

    # Note about confidence intervals
    st.caption(
        "Confidence intervals use overall model RMSE (±18.19 units). "
        "Wide intervals on low-volume items are a known limitation — "
        "per-item RMSE would improve this in production."
    )