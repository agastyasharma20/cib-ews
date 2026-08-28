"""
Seasonal false-positive filter.

The raw index-breach rule (DI > threshold) will fire for ANY customer whose
blended decline signal is worse than peers that month — including a
customer whose balance dipped for a real but harmless reason (advance tax
payment, dividend payout, a seasonal working-capital cycle) and recovers a
month or two later. `seasonal_false_positive` in the Phase 1 ground truth
exists specifically to model this. Flagging every one of those would make
RMs distrust the system fast — the whole point of an EWS is to be worth
acting on.

Rule implemented (matches the labeling brief):
    A breach month is downgraded from "confirmed deterioration" to a
    harmless dip if BOTH:
      1. it REVERTS — DI falls back below `recovery_threshold` within
         `reversal_window_months`, AND
      2. it is NOT corroborated — neither the digital-activity nor the
         payroll/trade component was also elevated at the time of breach.
    A breach that persists, OR that reverts but was accompanied by real
    digital/payroll/trade decline, is kept as genuine deterioration —
    because two independent signals moving together is not a coincidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.labeling.config import DI_THRESHOLD, SEASONAL_FILTER


def flag_confirmed_deterioration(
    df: pd.DataFrame,
    threshold: float = DI_THRESHOLD,
    filter_cfg: dict = SEASONAL_FILTER,
) -> pd.DataFrame:
    """
    Adds two columns:
      - `breach`: DI > threshold this month (raw signal, before filtering)
      - `confirmed_deterioration`: breach survives the seasonal filter
    """
    # reset_index(drop=True) is important here: the loop below writes into a
    # positional numpy array using group.index values, which only lines up
    # with row position if the DataFrame index is a plain 0..n-1 RangeIndex.
    df = df.sort_values(["customer_id", "month_idx"]).reset_index(drop=True).copy()
    df["breach"] = df["deterioration_index"] > threshold

    reversal_window = filter_cfg["reversal_window_months"]
    recovery_threshold = filter_cfg["recovery_threshold"]
    accompany_threshold = filter_cfg["accompanying_decline_threshold"]

    confirmed = np.zeros(len(df), dtype=bool)

    # Per-customer sequential logic — cheap at 5,000 customers x 18 months.
    for _, group in df.groupby("customer_id", sort=False):
        idx = group.index.to_numpy()
        di = group["deterioration_index"].to_numpy()
        breach = group["breach"].to_numpy()
        digital = group["digital_decline_score"].to_numpy()
        payroll_trade = group["payroll_trade_decline_score"].to_numpy()
        n = len(group)

        for i in range(n):
            if not breach[i]:
                continue

            # Does DI fall back below recovery_threshold within the window?
            look_ahead = di[i + 1 : i + 1 + reversal_window]
            reverted = bool(np.any(look_ahead < recovery_threshold)) if len(look_ahead) > 0 else False

            # Was the dip corroborated by digital or payroll/trade decline?
            corroborated = (
                (not np.isnan(digital[i]) and digital[i] >= accompany_threshold)
                or (not np.isnan(payroll_trade[i]) and payroll_trade[i] >= accompany_threshold)
            )

            # Suppress only if it reverted AND was NOT corroborated.
            confirmed[idx[i]] = not (reverted and not corroborated)

    df["confirmed_deterioration"] = confirmed & df["breach"]
    return df
