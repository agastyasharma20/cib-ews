"""
Feature Group 2: Transaction & Digital Activity

Money movement and digital engagement typically shift BEFORE the account
balance visibly drops — someone quietly routing payments elsewhere still
carries a balance for a while. These features capture transaction momentum
and how a customer's digital-channel usage is trending, independent of
account size.
"""

from __future__ import annotations

import pandas as pd

from src.features.config import TXN_TREND_WINDOW, DIGITAL_TREND_WINDOW, register
from src.features.utils import rolling_slope, rolling_mean, momentum_ratio

GROUP = "Transaction & Digital Activity"

FEATURE_COLUMNS = [
    "txn_count_trend_slope_3m",
    "txn_value_trend_slope_3m",
    "digital_channel_share",
    "digital_channel_share_trend_3m",
    "login_frequency_trend_3m",
]
register(GROUP, FEATURE_COLUMNS)


def build(df: pd.DataFrame) -> pd.DataFrame:
    """`df` must be sorted by customer_id, month_idx and contain
    transaction_count_debit/credit, transaction_value_debit/credit,
    mobile_banking_txn_pct, digital_login_count."""
    df = df.copy()

    df["txn_count_total"] = df["transaction_count_debit"] + df["transaction_count_credit"]
    df["txn_value_total"] = df["transaction_value_debit"] + df["transaction_value_credit"]

    # Trend slopes: is overall transaction activity rising or falling?
    df["txn_count_trend_slope_3m"] = rolling_slope(df, "txn_count_total", TXN_TREND_WINDOW)
    df["txn_value_trend_slope_3m"] = rolling_slope(df, "txn_value_total", TXN_TREND_WINDOW)

    # Digital vs. branch/other-channel activity: the share of transactions
    # done via mobile banking IS the digital-vs-branch split in this
    # dataset (a customer who used to transact mostly on mobile and now
    # skews back toward other channels is disengaging from the digital
    # relationship, often before disengaging from the bank overall).
    df["digital_channel_share"] = df["mobile_banking_txn_pct"]
    short_mean = rolling_mean(df, "mobile_banking_txn_pct", DIGITAL_TREND_WINDOW)
    long_mean = rolling_mean(df, "mobile_banking_txn_pct", DIGITAL_TREND_WINDOW * 2)
    df["digital_channel_share_trend_3m"] = momentum_ratio(short_mean, long_mean)

    # Login frequency trend: same short-vs-long momentum treatment applied
    # to raw login counts (a distinct signal from the % done via mobile —
    # a customer can log in less often while still doing most of what they
    # DO transact on mobile).
    login_short = rolling_mean(df, "digital_login_count", DIGITAL_TREND_WINDOW)
    login_long = rolling_mean(df, "digital_login_count", DIGITAL_TREND_WINDOW * 2)
    df["login_frequency_trend_3m"] = momentum_ratio(login_short, login_long)

    return df
