"""
Composite Deterioration Index (DI) construction.

WHY a composite index instead of a single metric:
No single behavioral signal reliably identifies "silent" CA attrition on its
own — balance alone is noisy (quarter-end window dressing, one-off tax
payments) and lags real behavior change; digital activity alone is too soft
a signal. The approach here blends several TRAILING, PERCENTILE-RANKED
decline signals into one 0-1 index, so a customer is only flagged when
several independent signals agree they're deteriorating worse than peers —
not because one noisy number happened to dip this month.

WHY percentile rank within (segment, month), not a flat rupee/percent
threshold:
An SME's balance moves by different absolute amounts — and different
PERCENTAGES — than a Large Corporate's as a matter of course, and whole
segments/industries share seasonal effects (quarter-end bumps, festive-
season turnover swings) that move everyone in the same direction at once.
Ranking each customer's trailing-vs-prior change against same-segment,
same-month peers nets out both effects automatically: a customer whose
balance fell 8% in a month when the whole segment fell 8% is not
deteriorating RELATIVE TO PEERS, but a customer who fell 8% while peers were
flat clearly is. This is exactly the kind of relative, cohort-aware scoring
real credit/behavioral risk models use instead of hardcoded cutoffs.

Everything here uses only TRAILING (past-and-current) data relative to the
month being scored — no forward-looking values are read — so this index is
safe to compute for any month and use as the basis for a genuinely
forward-looking label in labels.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.labeling.config import TRAILING_WINDOW_MONTHS, COMPONENT_WEIGHTS


def add_month_index(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Add a 0-based `month_idx` per customer (months are already
    chronologically sortable as 'YYYY-MM' strings)."""
    df = panel_df.sort_values(["customer_id", "month"]).copy()
    df["month_idx"] = df.groupby("customer_id").cumcount()
    return df


def _trailing_prior_pct_change(df: pd.DataFrame, col: str, window: int = TRAILING_WINDOW_MONTHS) -> pd.Series:
    """
    For each customer-month, compute:
        pct_change = (trailing `window`-month mean) / (prior `window`-month mean) - 1
    trailing = the `window` months ending at (and including) this month.
    prior    = the `window` months immediately before the trailing window.

    Both windows only use data at or before the current month — this is a
    strictly backward-looking (trailing) feature, never a forward one.
    Returns NaN wherever either window isn't fully observed yet (early
    months) or the prior mean is 0/undefined (e.g. a metric that's always 0
    for this customer, such as payroll for a customer with no payroll book —
    correctly signals "not applicable" rather than an infinite % change).
    """
    grouped = df.groupby("customer_id")[col]
    trailing_mean = grouped.rolling(window=window, min_periods=window).mean().reset_index(level=0, drop=True)

    shifted = grouped.shift(window)
    prior_mean = shifted.groupby(df["customer_id"]).rolling(window=window, min_periods=window).mean().reset_index(level=0, drop=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_change = (trailing_mean - prior_mean) / prior_mean
    pct_change = pct_change.replace([np.inf, -np.inf], np.nan)
    return pct_change


def _percentile_decline_score(df: pd.DataFrame, pct_change_col: str) -> pd.Series:
    """
    Within each (segment, month_idx) group, rank this pct_change ascending
    (most negative = most decline = lowest rank), then invert so that the
    worst decline scores near 1.0 and the best growth scores near 0.0.
    Rows with NaN pct_change stay NaN (component not applicable that month).
    """
    rank_pct = df.groupby(["segment", "month_idx"])[pct_change_col].rank(pct=True, ascending=True)
    return 1.0 - rank_pct


def build_component_scores(panel_df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the four DI components (each a 0-1 decline-percentile score,
    NaN if not applicable) for every customer-month.
    """
    df = add_month_index(panel_df)
    df = df.merge(customers_df[["customer_id", "segment"]], on="customer_id", how="left")

    # Convenience combined columns
    df["txn_count_total"] = df["transaction_count_debit"] + df["transaction_count_credit"]
    df["txn_value_total"] = df["transaction_value_debit"] + df["transaction_value_credit"]

    # --- 1. AMB decline ---
    df["amb_pct_change"] = _trailing_prior_pct_change(df, "average_monthly_balance")
    df["amb_decline_score"] = _percentile_decline_score(df, "amb_pct_change")

    # --- 2. Transaction activity decline (count + value, averaged) ---
    df["txn_count_pct_change"] = _trailing_prior_pct_change(df, "txn_count_total")
    df["txn_value_pct_change"] = _trailing_prior_pct_change(df, "txn_value_total")
    df["txn_count_decline_score"] = _percentile_decline_score(df, "txn_count_pct_change")
    df["txn_value_decline_score"] = _percentile_decline_score(df, "txn_value_pct_change")
    df["txn_decline_score"] = df[["txn_count_decline_score", "txn_value_decline_score"]].mean(axis=1, skipna=True)

    # --- 3. Digital activity decline (logins + mobile txn share, averaged) ---
    df["digital_login_pct_change"] = _trailing_prior_pct_change(df, "digital_login_count")
    df["mobile_pct_pct_change"] = _trailing_prior_pct_change(df, "mobile_banking_txn_pct")
    df["digital_login_decline_score"] = _percentile_decline_score(df, "digital_login_pct_change")
    df["mobile_decline_score"] = _percentile_decline_score(df, "mobile_pct_pct_change")
    df["digital_decline_score"] = df[["digital_login_decline_score", "mobile_decline_score"]].mean(axis=1, skipna=True)

    # --- 4. Payroll / trade-finance decline (sticky products, averaged) ---
    # Both source columns are already NaN/0 for customers where the product
    # doesn't apply, so pct_change naturally comes out NaN for them and the
    # component is excluded from their DI rather than penalizing them.
    df["payroll_pct_change"] = _trailing_prior_pct_change(df, "payroll_credit_amount")
    df["trade_pct_change"] = _trailing_prior_pct_change(df, "trade_finance_utilization_pct")
    df["payroll_decline_score"] = _percentile_decline_score(df, "payroll_pct_change")
    df["trade_decline_score"] = _percentile_decline_score(df, "trade_pct_change")
    df["payroll_trade_decline_score"] = df[["payroll_decline_score", "trade_decline_score"]].mean(axis=1, skipna=True)

    return df


def compute_composite_index(df: pd.DataFrame, weights: dict = COMPONENT_WEIGHTS) -> pd.DataFrame:
    """
    Weighted average of the 4 component scores, renormalized per-row over
    whichever components are non-NaN for that customer-month. DI is NaN
    only when ALL components are unavailable (e.g. the first
    2*TRAILING_WINDOW_MONTHS-1 months of a customer's history, before any
    trailing/prior window is fully observed).
    """
    # weight keys are e.g. "amb_decline"; the actual columns built above are
    # named "amb_decline_score" — map between the two here.
    keys = list(weights.keys())
    cols = [f"{key}_score" for key in keys]
    values = df[cols].to_numpy(dtype=float)
    w = np.array([weights[key] for key in keys])

    valid = ~np.isnan(values)
    weighted_sum = np.nansum(values * w, axis=1)
    weight_total = (valid * w).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        di = np.where(weight_total > 0, weighted_sum / weight_total, np.nan)

    df = df.copy()
    df["deterioration_index"] = di
    return df
