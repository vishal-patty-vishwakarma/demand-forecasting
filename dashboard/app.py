import streamlit as st

st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Demand Forecasting Dashboard")
st.caption(
    "Corporacion Favorita Grocery Sales | "
    "LightGBM Global Model | 10 Stores | 2013-2017"
)

tab1, tab2, tab3 = st.tabs([
    "📈 Forecast View",
    "🏆 Model Comparison",
    "💡 Business Insights"
])

with tab1:
    exec(open("dashboard/tab_forecast.py",
              encoding="utf-8").read())

with tab2:
    exec(open("dashboard/tab_comparison.py",
              encoding="utf-8").read())

with tab3:
    exec(open("dashboard/tab_insights.py",
              encoding="utf-8").read())