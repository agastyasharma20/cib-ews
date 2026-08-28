"""
Forward-looking binary labels.

Given `confirmed_deterioration` (a per-month, trailing-data-only flag from
seasonal_filter.py), the label for month t and horizon h is:

    deteriorates_in_{h}(t) = 1 if confirmed_deterioration is True in ANY of
                              months (t+1 .. t+h), else 0.

This is the actual EWS output shape: "given everything known up to and
including month t, will this customer cross into confirmed deterioration
within the next h months?" It is deliberately built from FUTURE
confirmed-deterioration flags relative to t — that lookahead is exactly
what makes it a usable supervised-learning target (the features used to
predict it, built in the next phase, must only ever use data up to month
t — that leakage boundary is the whole point of the label/feature split).

Labels are left as NaN wherever month t+h falls beyond the last observed
month for that customer (there isn't enough future history to know the
right answer yet) — these rows must be dropped before training, not treated
as negatives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.labeling.config import HORIZONS_MONTHS


def build_forward_labels(df: pd.DataFrame, horizons: dict = HORIZONS_MONTHS) -> pd.DataFrame:
    df = df.sort_values(["customer_id", "month_idx"]).reset_index(drop=True).copy()
    max_month_idx = df.groupby("customer_id")["month_idx"].transform("max")

    for label_col, h in horizons.items():
        col = np.full(len(df), np.nan)
        for _, group in df.groupby("customer_id", sort=False):
            idx = group.index.to_numpy()
            confirmed = group["confirmed_deterioration"].to_numpy()
            n = len(group)
            for i in range(n):
                if i + h > n - 1:
                    continue  # not enough forward history yet — leave NaN (censored)
                col[idx[i]] = bool(np.any(confirmed[i + 1 : i + 1 + h]))
        df[label_col] = col

    return df
