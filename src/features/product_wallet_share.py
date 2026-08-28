"""
Feature Group 3: Product & Wallet-Share

Payroll processing and trade-finance facilities are "sticky" products — a
customer rarely moves them lightly, so when they DO start slipping it's a
strong tell, distinct from the more easily-fluctuating transaction/balance
signals. This group also tracks whether the customer holds these products
at all (has_payroll_book / has_trade_finance), so a customer for whom a
product never applied isn't penalized for a metric with no meaning for
them.
"""

from __future__ import annotations

import pandas as pd

from src.features.config import PAYROLL_REGULARITY_WINDOW, TRADE_TREND_WINDOW, register
from src.features.utils import rolling_mean, rolling_slope, expanding_ever_true

GROUP = "Product & Wallet-Share"

FEATURE_COLUMNS = [
    "payroll_regularity_score",
    "has_payroll_book",
    "trade_utilization_trend_3m",
    "has_trade_finance",
]
register(GROUP, FEATURE_COLUMNS)


def build(df: pd.DataFrame) -> pd.DataFrame:
    """`df` must be sorted by customer_id, month_idx and contain
    payroll_credit_flag, payroll_credit_amount, trade_finance_utilization_pct."""
    df = df.copy()

    # Ever held the product, as of this month (leakage-safe: only looks
    # backward). Kept separate from the trend feature so a model can learn
    # "no facility at all" differently from "has one, currently declining."
    df["_flag_payroll"] = df["payroll_credit_amount"] > 0
    df["has_payroll_book"] = expanding_ever_true(df, "_flag_payroll")
    df["_flag_trade"] = df["trade_finance_utilization_pct"].notna()
    df["has_trade_finance"] = expanding_ever_true(df, "_flag_trade")
    df = df.drop(columns=["_flag_payroll", "_flag_trade"])

    # Regularity: fraction of the trailing 6 months where payroll was
    # actually credited. A customer sliding from 6/6 to 2/6 is quietly
    # migrating payroll processing elsewhere well before it hits zero.
    df["payroll_regularity_score"] = rolling_mean(df, "payroll_credit_flag", PAYROLL_REGULARITY_WINDOW)

    # Trade-finance utilization trend (NaN for customers with no facility —
    # intentional; has_trade_finance disambiguates "no facility" from
    # "facility, flat utilization").
    df["trade_utilization_trend_3m"] = rolling_slope(df, "trade_finance_utilization_pct", TRADE_TREND_WINDOW)

    return df
