```markdown
# End-to-End Demand Forecasting System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## What This Project Does

Predicts future product demand for a retail grocery chain
using 4.5 years of historical sales data across 10 stores
and 3,400+ products. Outputs are served through an
interactive dashboard with inventory recommendations.

---

## Problem Statement

Retail businesses face four compounding problems:
- Inaccurate demand estimation leads to poor purchasing decisions
- Overstocking increases holding costs and spoilage
- Stockouts directly result in lost revenue
- Manual planning cannot capture seasonality or
  external drivers like holidays and oil prices

---

## Architecture

```
Raw CSVs (5 files, 125M rows)
       ↓
Pandas ETL Pipeline
(merge, clean, validate)
       ↓
Feature Store (19 features)
(lags, rolling stats, external regressors)
       ↓
LightGBM Global Model
(Tweedie loss, walk-forward CV)
       ↓
Forecast Table (parquet)
       ↓
Streamlit Dashboard
(3 tabs: forecast, comparison, insights)
```

---

## Key Results

| Model | SMAPE | Notes |
|---|---|---|
| Seasonal Naive | 81.94% | Predict = same day last week |
| Moving Average (28d) | 49.99% | Strong baseline |
| SARIMA | 54.40% | 229 hrs to scale, not viable |
| **LightGBM Global** | **46.97%** | **Best — minutes to train** |

- Forecast bias: -0.097 (nearly zero — well calibrated)
- Best family: HARDWARE (39.11% SMAPE)
- Most important feature: rolling_mean_7 (by large margin)
- Holiday uplift confirmed: 11.4% average sales increase
- Oil price correlation: -0.600 with total sales

---

## Dataset

Corporacion Favorita Grocery Sales (Kaggle)
- 125M+ transactions from 54 stores across Ecuador
- Date range: January 2013 to August 2017
- 5 relational files: sales, stores, items, holidays, oil prices
- This project uses 10-store sample for development

---

## Tech Stack

| Category | Tool | Why |
|---|---|---|
| Data Processing | Pandas, NumPy | Core data manipulation |
| ML Model | LightGBM | Fast, handles tabular + zero-inflated data |
| Statistical Model | pmdarima (SARIMA demo) | Scaling comparison only |
| Validation | scikit-learn TimeSeriesSplit | No data leakage |
| Visualization | Plotly, Matplotlib, Seaborn | Interactive + static plots |
| Dashboard | Streamlit | Fastest DS dashboard framework |
| Tracking | JSON experiment log | Lightweight, no MLflow overhead |
| Version Control | Git + GitHub | Standard |

---

## Project Structure

```
demand-forecasting/
├── data/
│   ├── raw/           # 5 Kaggle CSVs (gitignored)
│   ├── processed/     # master_sample.parquet
│   └── features/      # feature_store.parquet
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_sarima_demo.ipynb
│   ├── 07_prophet.ipynb
│   └── 08_lightgbm.ipynb
│   └── 09_evaluation.ipynb
├── dashboard/
│   ├── app.py
│   ├── tab_forecast.py
│   ├── tab_comparison.py
│   └── tab_insights.py
├── outputs/
│   ├── models/        # lightgbm_final.pkl
│   ├── forecasts/     # forecast_table.parquet
│   ├── experiments.json
│   └── screenshots/
├── requirements.txt
├── DECISIONS.md
└── README.md
```

---

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/vishal-patty-vishwakarma/demand-forecasting.git
cd demand-forecasting
```

### 2. Create environment
```bash
conda create -n forecast python=3.11 -y
conda activate forecast
pip install -r requirements.txt
```

### 3. Download dataset
- Go to kaggle.com/competitions/favorita-grocery-sales-forecasting
- Download: train.csv, stores.csv, items.csv,
  holidays_events.csv, oil.csv
- Place all files in data/raw/

### 4. Run notebooks in order
```bash
jupyter notebook
```
Run notebooks 01 through 09 in sequence.

### 5. Launch dashboard
```bash
streamlit run dashboard/app.py
```
Open http://localhost:8501

---

## Dashboard Screenshots

### Tab 1 — Forecast View
![Forecast View](outputs/screenshots/tab1_forecast_grocery1.png)

### Tab 2 — Model Comparison
![Model Comparison](outputs/screenshots/tab2_model_comparison.png)

### Tab 3 — Business Insights
![Business Insights](outputs/screenshots/tab3_business_insights.png)

---

## Key Engineering Decisions

See DECISIONS.md for full decision log.

- **Global model over per-series**: SARIMA needs 229 hours
  for all series. LightGBM trains once in minutes.
- **Tweedie loss**: Retail demand is zero-inflated.
  Tweedie handles zeros + right skew better than MSE.
- **Walk-forward CV**: Random split leaks future data
  in time series. Date-based folds prevent this.
- **Forward-fill oil only**: Backward fill uses future
  prices — temporal leakage.
- **Groupby lags**: Computing lags without groupby
  mixes data across store-item pairs — data leakage.

---

## Business Impact

- Safety stock recommendations per store-item
  (7-day lead time, 90% service level)
- Stockout risk flags for consistently
  under-forecasted items (e.g. FROZEN FOODS: -0.475 bias)
- Overstock warnings for over-forecasted items
- 46.97% SMAPE vs 49.99% baseline — model beats
  simple average across all product families

---

## Future Improvements

- Add lag_365 (same day last year — annual seasonality)
- Per-item confidence intervals instead of global RMSE
- Prophet with regressors
  (blocked by Windows CmdStan issue, documented)
- Automated retraining pipeline
- Docker containerization
- Cloud deployment (AWS/GCP)

---

## Author

4th Year BTech Student — Data Science Portfolio Project
Built: May 2026
```
