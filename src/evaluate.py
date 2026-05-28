"""
Evaluation metrics for demand forecasting models.
"""

import numpy as np
import pandas as pd


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    Handles zero values. Range: 0% (perfect) to 200% (worst).
    Preferred over MAPE for zero-inflated retail demand.
    """
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    return float(100 * np.mean(
        2 * np.abs(y_true - y_pred) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error — interpretable in units sold."""
    return float(np.mean(np.abs(
        np.array(y_true) - np.array(y_pred)
    )))


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Forecast bias = mean(predicted - actual).
    Positive = over-forecasting (overstock risk).
    Negative = under-forecasting (stockout risk).
    """
    return float(np.mean(
        np.array(y_pred) - np.array(y_true)
    ))


def segment_accuracy(
    df: pd.DataFrame,
    segment_col: str = "family"
) -> pd.DataFrame:
    """
    Compute SMAPE and bias per segment (e.g. product family).
    Reveals where model works and where it fails.
    """
    results = []
    for segment in df[segment_col].unique():
        mask = df[segment_col] == segment
        if mask.sum() < 50:
            continue
        results.append({
            segment_col: segment,
            "smape": round(smape(
                df.loc[mask, "unit_sales"].values,
                df.loc[mask, "predicted"].values
            ), 2),
            "bias": round(forecast_bias(
                df.loc[mask, "unit_sales"].values,
                df.loc[mask, "predicted"].values
            ), 3),
            "rows": int(mask.sum())
        })
    return (
        pd.DataFrame(results)
        .sort_values("smape")
        .reset_index(drop=True)
    )