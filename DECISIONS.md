# Modeling & Engineering Decision Log

Every decision made in this project with rationale.
This document exists so I can explain every choice
in interviews without hesitation.

---

## Data Engineering Decisions

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Oil price fill | Forward fill | Backward fill | Backward fill uses future prices = temporal leakage |
| Negative sales | Clip to 0 | Keep as-is | Returns are not demand signals |
| onpromotion NaN | Fill with 0 | Drop rows | Unknown = assume no promotion |
| Transferred holidays | Ignore | Include | Moved to different date — original date is not a holiday |
| Local holidays | Match on city | Apply to all | Local holidays only affect stores in that city |
| Date reindex | Date-offset merge | Full reindex | Full reindex required 6GB RAM — not viable |
| Sample size | 10 stores | All 54 | Development speed. Final model can use all 54. |

---

## Feature Engineering Decisions

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Lag computation | groupby(store, item) | Full dataframe | Global computation mixes series = data leakage |
| Rolling computation | shift(1) then roll | Direct roll | Direct roll includes current day = leakage |
| Lag windows | 1, 7, 14, 28 days | 30 days | 7/14/28 capture weekly cycles. 30 has no weekly meaning |
| Target encoding | Full-sample mean | CV-aware mean | Small leakage risk accepted, documented |
| Oil feature | price_change_pct | Raw price only | Momentum signal more informative than absolute level |

---

## Modeling Decisions

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Model type | Global LightGBM | 34,656 x SARIMA | SARIMA = 229hrs. LightGBM = minutes. |
| Loss function | Tweedie (p=1.1) | MSE / MAE | Retail demand is zero-inflated. Tweedie handles zeros + right skew |
| Validation | Walk-forward CV | Random split | Random split leaks future into training |
| CV folds | 5 date-based folds | K-fold | K-fold ignores temporal order |
| num_leaves | 128 | 64 | More complex patterns captured. No overfitting observed. |
| Learning rate | 0.03 | 0.05 | Slower learning, better generalization |
| Confidence intervals | ±1.28 x RMSE | Quantile regression | Simpler, explainable. Known limitation: uses global RMSE |

---

## Tool Decisions

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Dashboard | Streamlit | FastAPI + Vanilla JS | 7-10 days saved. Same visual output for DS interviews |
| Experiment tracking | JSON file | MLflow | MLflow is scope creep for a portfolio project |
| Prophet | Skipped | Used | Windows CmdStan build failure. External regressors covered by LightGBM features |
| Data engine | Pandas | Polars | Already familiar. Polars adds learning overhead without DS value |

---

## Known Limitations

1. Confidence intervals use global RMSE (±18.19 units)
   Wide for low-volume items. Fix: per-item RMSE.

2. Target encoding computed on full dataset
   Small leakage risk. Fix: compute within each CV fold.

3. Prophet skipped due to Windows CmdStan issue
   Fix: conda install -c conda-forge prophet

4. Only 10 of 54 stores used
   Fix: run final model on all stores.

5. Future Prophet regressor values assumed constant
   Fix: use forecasted oil prices as future values.