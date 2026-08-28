"""
Turns per-row SHAP values into a "top 3 reason codes" list of
plain-language driver labels.

WHY aggregate by driver label before ranking, rather than just taking the
top 3 individual SHAP features:
Several raw features often belong to the same underlying story (e.g.
amb_pct_change_30d/60d/90d are all "Falling balances") — showing an RM
"Falling balances, Falling balances, Falling balances" as their 3 reason
codes would be redundant and less useful than "Falling balances, Rising
competitor-transfer share, Reduced payroll activity". Grouping by label
first and ranking the GROUP's total contribution surfaces 3 genuinely
distinct storylines instead of 3 views of the same one.

WHY rank by signed SHAP sum, not absolute value:
A reason code should explain why risk is HIGH, not list whatever moved the
score most in either direction. Ranking by the (positive-biased) signed sum
means a strongly protective factor (e.g. a big drop in complaints) won't
crowd out the actual risk drivers just because its magnitude is large.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.explainability.config import FEATURE_TO_DRIVER_LABEL, TOP_N_REASON_CODES


def assert_all_features_mapped(feature_names: list[str]) -> None:
    unmapped = [f for f in feature_names if f not in FEATURE_TO_DRIVER_LABEL]
    if unmapped:
        raise ValueError(
            f"Feature(s) {unmapped} have no entry in FEATURE_TO_DRIVER_LABEL "
            f"(src/explainability/config.py) — every model feature must map to a "
            f"plain-language driver label before reason codes can be trusted."
        )


def top_n_reason_codes_for_row(shap_row: pd.Series, n: int = TOP_N_REASON_CODES) -> list[str]:
    """`shap_row` is indexed by feature name, valued by that feature's SHAP
    contribution for one customer-month. Returns up to `n` distinct driver
    labels, ranked by their total (summed) SHAP contribution, highest
    (most risk-increasing) first. Kept for single-row/interactive use (e.g.
    explaining one customer on demand); `build_reason_codes_table` below
    does the equivalent computation vectorized for the full table."""
    labeled = shap_row.rename(index=FEATURE_TO_DRIVER_LABEL)
    grouped = labeled.groupby(level=0).sum().sort_values(ascending=False)
    return grouped.head(n).index.tolist()


def build_reason_codes_table(shap_df: pd.DataFrame, n: int = TOP_N_REASON_CODES) -> pd.Series:
    """
    `shap_df` has one row per customer-month, one column per feature (SHAP
    values). Returns a Series of `n`-element lists, one per row.

    Vectorized rather than a per-row `.apply(...)`: a first pass at this
    (grouping and sorting inside a Python-level loop over ~90,000 rows) took
    well over 10 minutes — most of it Python/pandas call overhead, not the
    actual arithmetic. Instead, the column-wise group-by-label sum is done
    ONCE for the whole table (23 feature columns -> ~13 label columns), then
    the top-n selection per row is a single numpy argsort over that much
    smaller matrix. Same result, seconds instead of minutes.
    """
    assert_all_features_mapped(list(shap_df.columns))

    label_of = pd.Series(FEATURE_TO_DRIVER_LABEL)
    # Group FEATURE columns by their driver label and sum once across all
    # rows (transposing a ~90k x 23 frame is cheap; the groupby then runs
    # over ~13 label groups instead of once per row).
    grouped = shap_df.T.groupby(label_of.reindex(shap_df.columns).to_numpy()).sum().T

    labels = grouped.columns.to_numpy()
    values = grouped.to_numpy()
    top_idx = np.argsort(-values, axis=1)[:, :n]
    top_labels = labels[top_idx]

    return pd.Series(list(top_labels), index=shap_df.index)
