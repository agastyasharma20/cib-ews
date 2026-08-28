"""
PFaR — Probability-weighted Funds/Flows at Risk.

    PFaR = probability_of_deterioration x estimated_balance_at_risk

WHY this metric, on top of the raw risk score:
A risk score alone (e.g. ews_score_90d) tells an RM WHO is likely to
deteriorate, but not how much is actually at stake — a 90%-risk SME with a
0.5 Cr balance is a very different priority than a 40%-risk Large Corporate
with a 50 Cr balance. PFaR turns "likelihood" into an expected-rupee-impact
figure, which is what portfolio prioritization should actually rank on.

  - probability_of_deterioration: the core model's ews_score_90d (Phase 5).
    90 days is used (not 30/60) because PFaR is meant to size an
    at-risk-over-the-full-warning-window exposure, matching the "30-90 day
    advance warning" framing of the whole project.

  - estimated_balance_at_risk = current AMB x expected_balance_decline_pct.
    expected_balance_decline_pct comes from an EMPIRICAL, HISTORICAL
    average (see compute_expected_decline_pct_by_segment below) — the
    average peak-to-trough balance drop actually observed among customers
    our OWN labeling framework confirmed as deteriorating in the past
    (data/processed/deterioration_labels.parquet's confirmed_deterioration
    flag). This deliberately does NOT read Phase 1's hidden
    ground_truth_cohort/deterioration_floor — using the planted "answer key"
    to size a production-facing risk metric would be exactly the kind of
    leakage a real deployment could never get away with (real deployments
    don't have a ground truth file). Calibrating off our own historical
    confirmed cases is the realistic equivalent of "what usually happens
    once we've called it."

PFaR is then decomposed into a driver TYPE (liquidity / relationship /
competitor) from the customer's top reason code, and customers are put into
priority tiers by their PFaR rank across the portfolio.

Run with: python -m src.models.pfar
(requires data/processed/customer_scores.parquet from
 src.explainability.score_customers, and data/processed/
 deterioration_labels.parquet from src.labeling.run_labeling)
"""

from __future__ import annotations

import pandas as pd

from src.config import DATA_PROCESSED_DIR, CUSTOMERS_FILE, MONTHLY_PANEL_FILE
from src.models.action_engine import RuleBasedActionRecommender, recommend_actions

SCORES_FILE = DATA_PROCESSED_DIR / "customer_scores.parquet"
LABELS_FILE = DATA_PROCESSED_DIR / "deterioration_labels.parquet"
PFAR_TABLE_FILE = DATA_PROCESSED_DIR / "pfar_risk_segmentation.parquet"
PFAR_HISTORY_FILE = DATA_PROCESSED_DIR / "pfar_history.parquet"

# --- Which broad driver TYPE each reason-code label rolls up into ---
# Three business-meaningful buckets: cash/balance mechanics (liquidity),
# engagement/service health (relationship), and money actively moving to a
# named competitor or a competitor-adjacent product move (competitor).
DRIVER_TYPE_MAP: dict[str, str] = {
    "Falling balances": "liquidity",
    "Unstable balances": "liquidity",
    "Declining credits": "liquidity",
    "Declining transaction activity": "liquidity",
    "Lower digital activity": "relationship",
    "Rising service complaints": "relationship",
    "Narrowing product relationship": "relationship",
    "Relationship tenure profile": "relationship",
    "Segment-level risk profile": "relationship",
    "Rising competitor-transfer share": "competitor",
    "Concentrating money flow to fewer counterparties": "competitor",
    "Reduced payroll activity": "competitor",   # payroll migrating elsewhere is a concrete wallet-share loss
    "Reduced trade utilization": "competitor",  # same logic for trade-finance business moving to another bank
}
DEFAULT_DRIVER_TYPE = "relationship"

# Priority tiers by PFaR rank across the scored population — tunable, not
# a business-given constant. Top 10% High, next 20% Medium, rest Low: a
# shape meant to keep the "High" list short enough for RMs to actually act
# on every name in it.
PRIORITY_TIER_QUANTILES = {"High": 0.90, "Medium": 0.70}


def compute_expected_decline_pct_by_segment(labels_df: pd.DataFrame, panel_df: pd.DataFrame, customers_df: pd.DataFrame) -> tuple[dict, float]:
    """
    For every customer OUR OWN labeling framework ever confirmed as
    deteriorating (confirmed_deterioration == True in any month), compute
    their peak-to-trough AMB decline: (baseline - trough) / baseline, where
    baseline is their own first-3-month average AMB. Returns the average of
    that decline, by segment, plus an overall fallback average.
    """
    merged = labels_df.merge(
        panel_df[["customer_id", "month", "average_monthly_balance"]], on=["customer_id", "month"]
    ).merge(customers_df[["customer_id", "segment"]], on="customer_id")
    merged = merged.sort_values(["customer_id", "month_idx"])

    ever_confirmed = merged.groupby("customer_id")["confirmed_deterioration"].any()
    deteriorating_ids = ever_confirmed[ever_confirmed].index
    sub = merged[merged["customer_id"].isin(deteriorating_ids)]

    baseline = sub.groupby("customer_id").head(3).groupby("customer_id")["average_monthly_balance"].mean()
    trough = sub.groupby("customer_id")["average_monthly_balance"].min()
    pct_decline = ((baseline - trough) / baseline).clip(lower=0, upper=1).rename("pct_decline")

    decline_df = pct_decline.to_frame().join(customers_df.set_index("customer_id")[["segment"]])
    by_segment = decline_df.groupby("segment")["pct_decline"].mean().to_dict()
    overall = float(decline_df["pct_decline"].mean())
    return by_segment, overall


def _expected_decline_for_segment(segment: str, by_segment: dict, overall: float) -> float:
    return by_segment.get(segment, overall)


def _driver_type_for_top_reason(reason_codes) -> str:
    if reason_codes is None or len(reason_codes) == 0:
        return DEFAULT_DRIVER_TYPE
    return DRIVER_TYPE_MAP.get(reason_codes[0], DEFAULT_DRIVER_TYPE)


def compute_pfar_table(scores_df: pd.DataFrame, panel_df: pd.DataFrame, customers_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    by_segment, overall = compute_expected_decline_pct_by_segment(labels_df, panel_df, customers_df)
    print(f"Expected balance decline %, by segment (from historically confirmed deterioration): {by_segment}")
    print(f"Overall fallback: {overall:.3f}")

    df = scores_df.merge(
        panel_df[["customer_id", "month", "average_monthly_balance"]], on=["customer_id", "month"], how="left"
    )

    df["expected_decline_pct"] = df["segment"].apply(lambda s: _expected_decline_for_segment(s, by_segment, overall))
    df["estimated_balance_at_risk"] = df["average_monthly_balance"] * df["expected_decline_pct"]
    df["PFaR"] = df["ews_score_90d"] * df["estimated_balance_at_risk"]
    df["PFaR_driver_type"] = df["top_3_reason_codes"].apply(_driver_type_for_top_reason)

    return df


def assign_priority_tiers(pfar_snapshot: pd.DataFrame) -> pd.DataFrame:
    pfar_snapshot = pfar_snapshot.copy()
    high_cut, medium_cut = pfar_snapshot["PFaR"].quantile(
        [PRIORITY_TIER_QUANTILES["High"], PRIORITY_TIER_QUANTILES["Medium"]]
    )

    def _tier(pfar_value: float) -> str:
        if pfar_value >= high_cut:
            return "High"
        elif pfar_value >= medium_cut:
            return "Medium"
        return "Low"

    pfar_snapshot["priority_tier"] = pfar_snapshot["PFaR"].apply(_tier)
    return pfar_snapshot


def build_latest_snapshot(pfar_df: pd.DataFrame) -> pd.DataFrame:
    """The final risk segmentation table is ONE row per customer — their
    most recent scored month, which is what an RM cares about today."""
    latest = pfar_df.sort_values("month_idx").groupby("customer_id").tail(1).copy()
    latest = assign_priority_tiers(latest)
    latest["recommended_action"] = recommend_actions(latest["top_3_reason_codes"], RuleBasedActionRecommender())
    return latest


def main() -> None:
    print("Loading scored table, panel, customers, and labels...")
    scores_df = pd.read_parquet(SCORES_FILE)
    panel_df = pd.read_parquet(MONTHLY_PANEL_FILE)
    customers_df = pd.read_parquet(CUSTOMERS_FILE)
    labels_df = pd.read_parquet(LABELS_FILE)

    print("Computing PFaR for every customer-month...")
    pfar_df = compute_pfar_table(scores_df, panel_df, customers_df, labels_df)

    # Full customer-month history (not just the latest snapshot) — the
    # dashboard (Phase 9) uses this for a portfolio-wide PFaR trend chart.
    pfar_df[["customer_id", "month", "month_idx", "PFaR"]].to_parquet(PFAR_HISTORY_FILE, index=False)
    print(f"Saved {PFAR_HISTORY_FILE} ({len(pfar_df):,} customer-months)")

    print("Building latest-month risk segmentation snapshot...")
    snapshot = build_latest_snapshot(pfar_df)

    output_cols = ["customer_id", "month", "segment", "PFaR", "PFaR_driver_type", "priority_tier", "recommended_action"]
    final_table = snapshot[output_cols].sort_values("PFaR", ascending=False).reset_index(drop=True)
    final_table.to_parquet(PFAR_TABLE_FILE, index=False)
    print(f"\nSaved {PFAR_TABLE_FILE} ({len(final_table):,} customers)")

    print(f"\nPriority tier distribution:\n{final_table['priority_tier'].value_counts()}")
    print(f"\nPFaR driver type distribution:\n{final_table['PFaR_driver_type'].value_counts()}")

    print(f"\n{'=' * 100}\nTop 20 customers by PFaR\n{'=' * 100}")
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 55)
    print(final_table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
