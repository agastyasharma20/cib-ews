"""
CIB Early Warning System — RM/Portfolio dashboard prototype (Phase 9).

This is a UX PROTOTYPE, not a production app: it reads the parquet
artifacts every earlier phase already produced (no live scoring, no
database) and exists to react to the shape of the output, not to be
hardened for real use. All data is synthetic (see docs/methodology.md —
Phase 10 — once written).

Run with:
    streamlit run app/dashboard.py

Two views, toggled from the sidebar:
  - RM Cockpit:   one row per customer, ranked by PFaR, filterable by
                  branch/segment/priority tier, with an inline survival-
                  curve sparkline per row (st.column_config.LineChartColumn
                  — no separate detail panel needed for that).
  - Portfolio View: aggregate charts — risk by branch/segment, driver-mix
                  breakdown, and a month-over-month PFaR trend (the
                  underlying data is monthly, not weekly — see the caption
                  on that chart for why "week-over-week" became "month-
                  over-month").
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/dashboard.py` sets sys.path[0] to app/, not the project
# root, so the `src` package (everything this dashboard reads was built by)
# isn't importable without this — a real bug caught by actually running the
# app, not something to skip.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DATA_PROCESSED_DIR, CUSTOMERS_FILE

PFAR_TABLE_FILE = DATA_PROCESSED_DIR / "pfar_risk_segmentation.parquet"
PFAR_HISTORY_FILE = DATA_PROCESSED_DIR / "pfar_history.parquet"
SCORES_FILE = DATA_PROCESSED_DIR / "customer_scores.parquet"
SURVIVAL_SCORES_FILE = DATA_PROCESSED_DIR / "survival_scores.parquet"
SURVIVAL_CURVES_FILE = DATA_PROCESSED_DIR / "survival_curves.parquet"

st.set_page_config(page_title="CIB Early Warning System", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached — these are static files for this prototype; a real
# deployment would replace this layer with a live query, nothing downstream
# would need to change — see docs/methodology.md's "path to production").
# ---------------------------------------------------------------------------
@st.cache_data
def load_master_table() -> pd.DataFrame:
    customers = pd.read_parquet(CUSTOMERS_FILE)[["customer_id", "branch_id", "rm_id"]]
    pfar = pd.read_parquet(PFAR_TABLE_FILE)

    scores = pd.read_parquet(SCORES_FILE)
    latest_scores = scores.sort_values("month_idx").groupby("customer_id").tail(1)
    latest_scores = latest_scores[["customer_id", "ews_score_30d", "ews_score_60d", "ews_score_90d", "top_3_reason_codes"]]

    survival = pd.read_parquet(SURVIVAL_SCORES_FILE)[
        ["customer_id", "risk_score", "predicted_median_days_to_deterioration", "median_beyond_observed_horizon"]
    ].rename(columns={"risk_score": "survival_risk_score"})

    curves = pd.read_parquet(SURVIVAL_CURVES_FILE).sort_values(["customer_id", "month_offset"])
    curve_lists = curves.groupby("customer_id")["survival_probability"].apply(list).rename("survival_curve")

    df = (
        pfar.merge(customers, on="customer_id", how="left")
        .merge(latest_scores, on="customer_id", how="left")
        .merge(survival, on="customer_id", how="left")
        .merge(curve_lists, on="customer_id", how="left")
    )
    df["top_3_reason_codes_str"] = df["top_3_reason_codes"].apply(
        lambda codes: " → ".join(codes) if isinstance(codes, (list, np.ndarray)) else ""
    )
    return df


@st.cache_data
def load_pfar_history() -> pd.DataFrame:
    return pd.read_parquet(PFAR_HISTORY_FILE)


master_df = load_master_table()

# ---------------------------------------------------------------------------
# Sidebar: view toggle + filters
# ---------------------------------------------------------------------------
st.sidebar.title("CIB Early Warning System")
view = st.sidebar.radio("View", ["RM Cockpit", "Portfolio View"])

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
segment_filter = st.sidebar.multiselect("Segment", sorted(master_df["segment"].unique()), default=[])
branch_filter = st.sidebar.multiselect("Branch", sorted(master_df["branch_id"].unique()), default=[])
tier_filter = st.sidebar.multiselect("Priority tier", ["High", "Medium", "Low"], default=[])

filtered_df = master_df.copy()
if segment_filter:
    filtered_df = filtered_df[filtered_df["segment"].isin(segment_filter)]
if branch_filter:
    filtered_df = filtered_df[filtered_df["branch_id"].isin(branch_filter)]
if tier_filter:
    filtered_df = filtered_df[filtered_df["priority_tier"].isin(tier_filter)]

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(filtered_df):,} of {len(master_df):,} customers shown")


# ---------------------------------------------------------------------------
# View 1: RM Cockpit
# ---------------------------------------------------------------------------
def render_rm_cockpit(df: pd.DataFrame) -> None:
    st.title("RM Cockpit")
    st.caption("Customers ranked by PFaR (Probability-weighted Funds at Risk). Click a column header to sort.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers in view", f"{len(df):,}")
    col2.metric("High priority", f"{(df['priority_tier'] == 'High').sum():,}")
    col3.metric("Total PFaR at risk (Cr)", f"{df['PFaR'].sum():,.1f}")
    col4.metric("Avg 90d EWS score", f"{df['ews_score_90d'].mean():.2f}" if len(df) else "—")

    display_df = df.sort_values("PFaR", ascending=False).copy()
    # NaN here means the survival model's curve never crossed 50% within its
    # observed 12-month horizon (see src/models/survival.py) — a real,
    # explainable outcome (that model anchors on an early snapshot, month 5,
    # while PFaR/EWS use the LATEST month, so the two can legitimately
    # disagree for a customer whose risk emerged later). Showing a bare
    # "None" for that case reads as missing data, not as an outcome, so it's
    # spelled out as text instead of left as a blank number.
    display_df["Est. days to deterioration"] = np.where(
        display_df["median_beyond_observed_horizon"],
        "> 360d (beyond model horizon)",
        display_df["predicted_median_days_to_deterioration"].round(0).astype("Int64").astype(str) + "d",
    )

    display_df = display_df[
        [
            "customer_id", "segment", "branch_id", "priority_tier", "PFaR", "PFaR_driver_type",
            "ews_score_90d", "top_3_reason_codes_str", "recommended_action",
            "Est. days to deterioration", "survival_curve",
        ]
    ].rename(columns={
        "top_3_reason_codes_str": "Top reason codes",
        "ews_score_90d": "EWS score (90d)",
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "PFaR": st.column_config.NumberColumn(format="%.1f Cr"),
            "EWS score (90d)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
            "survival_curve": st.column_config.LineChartColumn(
                "Survival curve (12mo)", y_min=0, y_max=1, width="small"
            ),
        },
    )
    st.caption(
        "Survival curve: estimated probability the customer is NOT YET confirmed-deteriorating, "
        "month 0-12 from the earliest scoreable month (src/models/survival.py). A curve that drops "
        "early is a customer already close to (or past) the model's risk threshold."
    )


# ---------------------------------------------------------------------------
# View 2: Portfolio View
# ---------------------------------------------------------------------------
def render_portfolio_view(df: pd.DataFrame) -> None:
    st.title("Portfolio View")

    left, right = st.columns(2)

    with left:
        st.subheader("Risk by branch & segment")
        st.caption("Mean PFaR per branch/segment — showing the top 20 branches by total PFaR for readability "
                   "(150 branches exist in the full synthetic book).")
        top_branches = df.groupby("branch_id")["PFaR"].sum().nlargest(20).index
        heatmap_data = (
            df[df["branch_id"].isin(top_branches)]
            .pivot_table(index="branch_id", columns="segment", values="PFaR", aggfunc="mean")
            .reindex(top_branches)
        )
        fig = px.imshow(
            heatmap_data, color_continuous_scale="Reds", aspect="auto",
            labels=dict(color="Mean PFaR (Cr)"),
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Driver-mix breakdown")
        st.caption("What's driving PFaR across the (filtered) portfolio, by top reason-code category.")
        driver_counts = df["PFaR_driver_type"].value_counts().reset_index()
        driver_counts.columns = ["driver_type", "count"]
        fig = px.pie(driver_counts, names="driver_type", values="count", hole=0.4,
                     color="driver_type",
                     color_discrete_map={"liquidity": "#d62728", "relationship": "#1f77b4", "competitor": "#ff7f0e"})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Portfolio PFaR trend")
    st.caption(
        "The underlying data is MONTHLY (18 simulated months), not weekly — this shows month-over-month "
        "total PFaR instead of the week-over-week view a higher-frequency data feed would support."
    )
    history = load_pfar_history()
    filtered_ids = set(df["customer_id"])
    history = history[history["customer_id"].isin(filtered_ids)]
    trend = history.groupby("month", as_index=False)["PFaR"].sum().sort_values("month")
    fig = px.line(trend, x="month", y="PFaR", markers=True)
    fig.update_layout(height=350, yaxis_title="Total portfolio PFaR (Cr)")
    st.plotly_chart(fig, use_container_width=True)


if view == "RM Cockpit":
    render_rm_cockpit(filtered_df)
else:
    render_portfolio_view(filtered_df)
