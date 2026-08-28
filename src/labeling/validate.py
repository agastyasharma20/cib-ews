"""
Validation of the labeling framework against the Phase 1 ground truth.

`ground_truth_cohort` is NEVER used to build the index or labels — it is
read here purely to check, after the fact, whether a labeling framework
built only from behavioral data actually recovers the deterioration
episodes we know we planted. This is the closest thing to "does this
approach work" we can check before real data exists.

Two views are reported:
  1. Customer-level "ever confirmed" vs cohort — the simple, intuitive
     confusion matrix the brief asks for (gradual/sudden should mostly be
     flagged at least once; stable/seasonal mostly never).
  2. Row-level (customer-month) precision/recall per horizon — the more
     rigorous check, since the actual model in later phases is trained on
     individual customer-month rows, not one verdict per customer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.labeling.deterioration_index import build_component_scores, compute_composite_index
from src.labeling.seasonal_filter import flag_confirmed_deterioration
from src.labeling.labels import build_forward_labels
from src.labeling.config import HORIZONS_MONTHS

DETERIORATING_COHORTS = {"gradual_deterioration", "sudden_deterioration"}
HEALTHY_COHORTS = {"stable", "seasonal_false_positive"}


def customer_level_confusion(labels_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of true cohort vs. whether the customer was EVER confirmed-
    deteriorating at any point in their 18-month history."""
    ever_confirmed = labels_df.groupby("customer_id")["confirmed_deterioration"].any().rename("ever_flagged")
    merged = ground_truth_df[["customer_id", "ground_truth_cohort"]].merge(ever_confirmed, on="customer_id")

    table = pd.crosstab(merged["ground_truth_cohort"], merged["ever_flagged"])
    table["flag_rate"] = (table.get(True, 0) / table.sum(axis=1)).round(3)
    return table, merged


def precision_recall_from_confusion(merged: pd.DataFrame) -> dict:
    """Treat {gradual_deterioration, sudden_deterioration} as the positive
    class and {stable, seasonal_false_positive} as negative, at the
    customer level."""
    y_true = merged["ground_truth_cohort"].isin(DETERIORATING_COHORTS)
    y_pred = merged["ever_flagged"]

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "fpr": fpr}


def row_level_confusion(labels_df: pd.DataFrame, ground_truth_df: pd.DataFrame, horizon_col: str, horizon_months: int) -> dict:
    """
    Rigorous row-level check: was this customer-month's TRUE forward state
    (based on the known injected onset month, not the index) correctly
    predicted by the engineered label?

    true_row_deteriorating(t) = customer is gradual/sudden AND t is at/after
                                 the true injected onset month.
    true_forward(t, h)        = true_row_deteriorating in any of (t+1..t+h)
                                 — built the same way as the engineered
                                 label, so the comparison is apples-to-apples.
    """
    df = labels_df.merge(
        ground_truth_df[["customer_id", "ground_truth_cohort", "deterioration_start_month"]],
        on="customer_id",
    )
    is_deteriorating_cohort = df["ground_truth_cohort"].isin(DETERIORATING_COHORTS)
    df["true_row_deteriorating"] = is_deteriorating_cohort & (df["month_idx"] >= df["deterioration_start_month"])

    true_forward = np.full(len(df), np.nan)
    for _, group in df.groupby("customer_id", sort=False):
        idx = group.index.to_numpy()
        truth = group["true_row_deteriorating"].to_numpy()
        n = len(group)
        for i in range(n):
            if i + horizon_months > n - 1:
                continue
            true_forward[idx[i]] = bool(np.any(truth[i + 1 : i + 1 + horizon_months]))
    df["true_forward"] = true_forward

    valid = df[horizon_col].notna() & df["true_forward"].notna()
    y_true = df.loc[valid, "true_forward"].astype(bool)
    y_pred = df.loc[valid, horizon_col].astype(bool)

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    return {"horizon": horizon_col, "n_rows": int(valid.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "fpr": fpr}


def lead_time_report(labels_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    For gradual/sudden customers who WERE flagged: how many months after
    the true injected onset did our index first confirm deterioration?
    (Positive = detected after onset i.e. some lag; this is expected since
    the index needs a trailing window to build up evidence. What matters
    for the business case is that this lag is much shorter than "wait until
    the balance visibly craters".)
    """
    first_confirmed = (
        labels_df[labels_df["confirmed_deterioration"]]
        .groupby("customer_id")["month_idx"]
        .min()
        .rename("first_confirmed_month")
    )
    gt = ground_truth_df[ground_truth_df["ground_truth_cohort"].isin(DETERIORATING_COHORTS)][
        ["customer_id", "ground_truth_cohort", "deterioration_start_month"]
    ]
    merged = gt.merge(first_confirmed, on="customer_id", how="left")
    merged["detected"] = merged["first_confirmed_month"].notna()
    merged["lag_months"] = merged["first_confirmed_month"] - merged["deterioration_start_month"]
    return merged


def grid_search_threshold(component_df: pd.DataFrame, ground_truth_df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """
    For each candidate DI threshold, rebuild confirmed_deterioration (with
    the seasonal filter applied) and report the customer-level FPR on
    stable+seasonal vs. recall on gradual+sudden, so a threshold can be
    picked with the trade-off visible rather than guessed.
    """
    rows = []
    for t in thresholds:
        flagged_df = flag_confirmed_deterioration(component_df, threshold=t)
        _, merged = customer_level_confusion(flagged_df, ground_truth_df)
        metrics = precision_recall_from_confusion(merged)
        rows.append({"threshold": t, **metrics})
    return pd.DataFrame(rows)
