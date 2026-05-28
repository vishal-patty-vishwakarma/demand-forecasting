"""
Feature engineering functions for demand forecasting.
All lag and rolling features computed within store-item groups
to prevent data leakage.
"""

import pandas as pd
import numpy as np


def add_date_lag(df: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """
    Compute lag feature using date-offset merge.
    Correctly handles missing dates without reindexing.

    Args:
        df: DataFrame with store_nbr, item_nbr, date, unit_sales
        lag_days: number of days to lag

    Returns:
        DataFrame with new lag_{lag_days} column
    """
    lag_df = df[["store_nbr", "item_nbr",
                 "date", "unit_sales"]].copy()
    lag_df["date"] = lag_df["date"] + pd.Timedelta(days=lag_days)
    lag_df = lag_df.rename(
        columns={"unit_sales": f"lag_{lag_days}"}
    )
    return df.merge(
        lag_df, on=["store_nbr", "item_nbr", "date"], how="left"
    )


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all lag features: 1, 7, 14, 28 days.
    Uses date-offset merge to handle missing dates correctly.
    """
    for lag in [1, 7, 14, 28]:
        df = add_date_lag(df, lag)
        df[f"lag_{lag}"] = df[f"lag_{lag}"].fillna(0).astype("float32")
    return df


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rolling mean and std features.
    Uses shift(1) before rolling to exclude current day — prevents leakage.
    """
    df = df.sort_values(
        ["store_nbr", "item_nbr", "date"]
    ).reset_index(drop=True)

    grp = df.groupby(["store_nbr", "item_nbr"])["unit_sales"]
    shifted = grp.shift(1)

    df["rolling_mean_7"] = (
        shifted.transform(lambda x: x.rolling(7, min_periods=1).mean())
    ).astype("float32")

    df["rolling_mean_28"] = (
        shifted.transform(lambda x: x.rolling(28, min_periods=1).mean())
    ).astype("float32")

    df["rolling_std_7"] = (
        shifted.transform(
            lambda x: x.rolling(7, min_periods=1).std().fillna(0)
        )
    ).astype("float32")

    return df


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build calendar-based features from date column.
    """
    df["day_of_week"]  = df["date"].dt.dayofweek.astype("int8")
    df["month"]        = df["date"].dt.month.astype("int8")
    df["is_weekend"]   = (
        df["date"].dt.dayofweek.isin([5, 6])
    ).astype("int8")
    df["day_of_month"] = df["date"].dt.day.astype("int8")
    df["week_of_year"] = (
        df["date"].dt.isocalendar().week.astype("int8")
    )
    return df


def build_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build oil price momentum and promotion features.
    """
    df["oil_lag7"] = df["oil_price"].shift(7).astype("float32")
    df["oil_price_change_pct"] = (
        (df["oil_price"] - df["oil_lag7"]) /
        (df["oil_lag7"] + 1e-8)
    ).astype("float32")
    df["discount_depth"] = df["onpromotion"].astype("int8")
    df.drop(columns=["oil_lag7"], inplace=True)
    return df