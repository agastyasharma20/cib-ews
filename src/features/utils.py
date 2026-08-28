"""
Shared, leakage-safe rolling-window helpers used by every feature group.

"Leakage-safe" here means: every function only ever looks at data at or
before the row's own month (trailing windows, backward shifts). None of
these functions look forward — that discipline is what makes it valid to
later attach a label built from FUTURE months without the features
secretly already knowing the answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_month_index(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Add a 0-based `month_idx` per customer (months are chronologically
    sortable as 'YYYY-MM' strings). Mirrors src/labeling/deterioration_index
    .py's helper of the same name — duplicated deliberately (four lines) to
    keep src/features/ independently runnable without importing the
    labeling package."""
    df = panel_df.sort_values(["customer_id", "month"]).copy()
    df["month_idx"] = df.groupby("customer_id").cumcount()
    return df


def _ols_slope(y: np.ndarray) -> float:
    """Closed-form OLS slope of y against x=0..n-1 (equally spaced months).
    Used instead of np.polyfit for speed under rolling.apply — same result,
    cheaper per call across ~90k rows x several slope features."""
    n = len(y)
    if np.isnan(y).any():
        return np.nan
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    var_x = (x**2).mean() - x_mean**2
    if var_x == 0:
        return 0.0
    cov_xy = (x * y).mean() - x_mean * y_mean
    return cov_xy / var_x


def rolling_slope(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    """Trailing OLS trend slope of `col` over the last `window` months
    (units: col-per-month). NaN until a full window is observed."""
    return (
        df.groupby("customer_id")[col]
        .rolling(window=window, min_periods=window)
        .apply(_ols_slope, raw=True)
        .reset_index(level=0, drop=True)
    )


def rolling_mean(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return (
        df.groupby("customer_id")[col]
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )


def rolling_std(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return (
        df.groupby("customer_id")[col]
        .rolling(window=window, min_periods=window)
        .std()
        .reset_index(level=0, drop=True)
    )


def rolling_sum(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return (
        df.groupby("customer_id")[col]
        .rolling(window=window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
    )


def pct_change_lag(df: pd.DataFrame, col: str, periods: int) -> pd.Series:
    """Trailing % change vs. `periods` months ago — e.g. periods=1 is the
    30-day (month-over-month) change, periods=3 the 90-day change."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = df.groupby("customer_id")[col].pct_change(periods=periods)
    return out.replace([np.inf, -np.inf], np.nan)


def momentum_ratio(short_series: pd.Series, long_series: pd.Series) -> pd.Series:
    """(short-window average / long-window average) - 1. Positive means
    the recent window is running above the longer baseline (improving);
    negative means it's running below (declining)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = short_series / long_series - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def expanding_ever_true(df: pd.DataFrame, boolcol: str) -> pd.Series:
    """True from the first month this became True onward (e.g. 'ever had a
    trade-finance facility as of this month'). Implemented as a cumulative
    max on a boolean series, which is leakage-safe since it only looks
    backward within each customer."""
    return df.groupby("customer_id")[boolcol].cummax()
