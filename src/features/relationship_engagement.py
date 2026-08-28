"""
Feature Group 5: Relationship & Engagement

Softer, structural signals about the overall relationship: how long the
customer has banked here, how much dissatisfaction is building up
(complaints/service tickets), and how many distinct product lines they
actually use — a broader relationship is generally stickier and gives more
signal surface area for the other groups to draw on.

NOTE: "RM contact recency" from the original brief is intentionally
omitted — the synthetic panel (Phase 1) does not simulate RM visit/contact
events, so fabricating one here would just be noise dressed as a feature.
Documented as a known gap; a real deployment would source this from CRM
contact logs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.config import COMPLAINT_ROLLING_WINDOW, register
from src.features.utils import rolling_sum

GROUP = "Relationship & Engagement"

FEATURE_COLUMNS = [
    "relationship_tenure_months",
    "complaint_rolling_3m_sum",
    "service_ticket_rolling_3m_sum",
    "product_holding_breadth",
]
register(GROUP, FEATURE_COLUMNS)


def _tenure_months_asof(account_open_date: pd.Series, month: pd.Series) -> pd.Series:
    """Recompute tenure AS OF each observed month, rather than reusing the
    single static tenure value snapshotted at generation time (which only
    reflects tenure as of the LAST simulated month, not month-varying) —
    this way the feature genuinely evolves month to month like a real
    tenure figure would."""
    open_period = pd.PeriodIndex(pd.to_datetime(account_open_date), freq="M")
    month_period = pd.PeriodIndex(month, freq="M")
    # For monthly periods, the integer ordinal (.asi8) difference IS the
    # month count between them — simpler and more robust across pandas
    # versions than subtracting PeriodIndexes directly (which returns
    # DateOffset objects, not plain ints).
    return pd.array(month_period.asi8 - open_period.asi8, dtype="int64")


def build(df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    """`df` must be sorted by customer_id, month_idx and contain
    complaint_count, service_ticket_count. `customers_df` supplies
    account_open_date. Requires has_payroll_book / has_trade_finance to
    already be present (built by product_wallet_share.build first)."""
    df = df.copy()
    df = df.merge(customers_df[["customer_id", "account_open_date"]], on="customer_id", how="left")

    df["relationship_tenure_months"] = _tenure_months_asof(df["account_open_date"], df["month"])
    df = df.drop(columns=["account_open_date"])

    df["complaint_rolling_3m_sum"] = rolling_sum(df, "complaint_count", COMPLAINT_ROLLING_WINDOW)
    df["service_ticket_rolling_3m_sum"] = rolling_sum(df, "service_ticket_count", COMPLAINT_ROLLING_WINDOW)

    # Product holding breadth: the current account itself (always 1) plus
    # whichever sticky products (Group 3) have ever been observed for this
    # customer as of this month.
    df["product_holding_breadth"] = (
        1 + df["has_payroll_book"].astype(int) + df["has_trade_finance"].astype(int)
    )

    return df
