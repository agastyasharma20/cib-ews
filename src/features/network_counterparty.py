"""
Feature Group 4: Network / Counterparty

This is a PLACEHOLDER version using only per-customer aggregates of their
own counterparty transactions — the full graph version (shared-entity /
linked-customer contagion risk, built with networkx over relationships.csv-
style structures) comes in Phase 7. Even without the graph, the raw
counterparty mix already carries a strong signal: what share of a
customer's outward money movement goes to a small fixed list of competitor
banks, and how concentrated their counterparty relationships are.
"""

from __future__ import annotations

import pandas as pd

from src.features.config import COMPETITOR_SHARE_SHORT_WINDOW, COMPETITOR_SHARE_LONG_WINDOW, register
from src.features.utils import rolling_mean, momentum_ratio

GROUP = "Network & Counterparty"

FEATURE_COLUMNS = [
    "competitor_txn_share",
    "competitor_txn_share_trend",
    "counterparty_concentration_hhi",
]
register(GROUP, FEATURE_COLUMNS)


def _monthly_counterparty_aggregates(counterparty_df: pd.DataFrame) -> pd.DataFrame:
    """One row per customer-month: competitor-bank share of counterparty
    transactions, and the Herfindahl-Hirschman Index (HHI) of how
    concentrated those counterparties are across banks. HHI ranges from
    ~1/n_banks (evenly spread across many banks) to 1.0 (all transactions
    go to a single bank) — a rising HHI toward one specific competitor is a
    sharper signal than the raw competitor share alone, since it shows the
    money isn't just diversifying, it's consolidating elsewhere."""
    competitor_share = (
        counterparty_df.groupby(["customer_id", "month"])["is_competitor_bank"].mean().rename("competitor_txn_share")
    )

    bank_counts = (
        counterparty_df.groupby(["customer_id", "month", "counterparty_bank_ifsc_prefix"]).size().rename("n").reset_index()
    )
    bank_totals = bank_counts.groupby(["customer_id", "month"])["n"].transform("sum")
    bank_counts["share_sq"] = (bank_counts["n"] / bank_totals) ** 2
    hhi = bank_counts.groupby(["customer_id", "month"])["share_sq"].sum().rename("counterparty_concentration_hhi")

    return pd.concat([competitor_share, hhi], axis=1).reset_index()


def build(df: pd.DataFrame, counterparty_df: pd.DataFrame) -> pd.DataFrame:
    """`df` must be sorted by customer_id, month_idx. `counterparty_df` is
    the raw per-transaction counterparty detail table from Phase 1."""
    df = df.copy()
    monthly_cp = _monthly_counterparty_aggregates(counterparty_df)
    df = df.merge(monthly_cp, on=["customer_id", "month"], how="left")

    # Trend: recent (3m) competitor share vs. a longer (6m) baseline. Rising
    # above 0 means the customer is leaking wallet share faster than their
    # own recent history, not just running at a persistently high level.
    short_mean = rolling_mean(df, "competitor_txn_share", COMPETITOR_SHARE_SHORT_WINDOW)
    long_mean = rolling_mean(df, "competitor_txn_share", COMPETITOR_SHARE_LONG_WINDOW)
    df["competitor_txn_share_trend"] = momentum_ratio(short_mean, long_mean)

    return df
