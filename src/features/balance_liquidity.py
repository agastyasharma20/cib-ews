"""
Feature Group 1: Balance & Liquidity

The account balance is the ultimate business outcome for a CA/CIB
relationship, but it's also the laggiest and most seasonality-prone signal
(quarter-end window dressing, one-off large payments). These features
capture direction (trend), noisiness (volatility), short-term momentum, and
how far off a customer is from their OWN normal level (seasonality-adjusted
deviation) — rather than a single raw balance number.
"""

from __future__ import annotations

import pandas as pd

from src.features.config import (
    AMB_TREND_WINDOW,
    AMB_VOLATILITY_WINDOW,
    AMB_SEASONAL_BASELINE_WINDOW,
    register,
)
from src.features.utils import rolling_slope, rolling_mean, rolling_std, pct_change_lag, momentum_ratio

GROUP = "Balance & Liquidity"

FEATURE_COLUMNS = [
    "amb_trend_slope_3m",
    "amb_volatility_3m",
    "amb_pct_change_30d",
    "amb_pct_change_60d",
    "amb_pct_change_90d",
    "amb_seasonal_adjusted_deviation",
]
register(GROUP, FEATURE_COLUMNS)


def build(df: pd.DataFrame) -> pd.DataFrame:
    """`df` must already be sorted by customer_id, month_idx and contain
    `average_monthly_balance`. Returns df with the group's columns added."""
    df = df.copy()
    col = "average_monthly_balance"

    # Trend: is the balance sloping up or down over the last 3 months?
    df["amb_trend_slope_3m"] = rolling_slope(df, col, AMB_TREND_WINDOW)

    # Volatility: coefficient of variation over the last 3 months — a
    # noisy balance is harder to read than a smoothly declining one, and
    # itself can be an early sign of irregular cash management.
    trailing_mean_3m = rolling_mean(df, col, AMB_VOLATILITY_WINDOW)
    trailing_std_3m = rolling_std(df, col, AMB_VOLATILITY_WINDOW)
    df["amb_volatility_3m"] = trailing_std_3m / trailing_mean_3m.replace(0, pd.NA)

    # Momentum at three horizons, mapped from the brief's 30/60/90 day
    # windows onto the closest honest equivalent on monthly data.
    df["amb_pct_change_30d"] = pct_change_lag(df, col, periods=1)
    df["amb_pct_change_60d"] = pct_change_lag(df, col, periods=2)
    df["amb_pct_change_90d"] = pct_change_lag(df, col, periods=3)

    # Seasonality-adjusted deviation: how far is THIS month's balance from
    # the customer's own trailing 6-month baseline? A window long enough to
    # span a full quarter-end cycle twice nets out recurring seasonal bumps
    # without needing an external calendar-effects table.
    baseline_6m = rolling_mean(df, col, AMB_SEASONAL_BASELINE_WINDOW)
    df["amb_seasonal_adjusted_deviation"] = momentum_ratio(df[col], baseline_6m)

    return df
