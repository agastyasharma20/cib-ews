"""
CIB Early Warning System — RM/Portfolio dashboard prototype (Phase 9).

This is a UX PROTOTYPE, not a production app: it reads the parquet
artifacts every earlier phase already produced (no live scoring, no
database) and exists to react to the shape of the output, not to be
hardened for real use. All data is synthetic (see docs/methodology.md).

Run with:
    streamlit run app/dashboard.py

Three views, toggled from the sidebar:
  - RM Cockpit:     one row per customer, ranked by PFaR, filterable by
                    branch/segment/priority tier, with an inline survival-
                    curve sparkline per row (st.column_config.LineChartColumn
                    — no separate detail panel needed for that).
  - Portfolio View: aggregate charts — risk by branch/segment, driver-mix
                    breakdown, and a month-over-month PFaR trend (the
                    underlying data is monthly, not weekly — see the
                    caption on that chart for why "week-over-week" became
                    "month-over-month").
  - About This Project: a plain-language walkthrough of the whole project
                    for anyone landing on the dashboard cold, plus credit
                    and the synthetic-data / non-affiliation disclaimer.

BRANDING NOTE (why there's no real HDFC Bank logo here): this repository
is public, under the author's own name, and not an HDFC Bank work product
— embedding the bank's actual trademarked logo, or a "confidential /
internal use" banner, would misrepresent an independent portfolio project
as an authorized or affiliated one. The header below uses a generic bank
glyph and an accurate disclaimer instead; the visual effect asked for
(logo mark top-left, scrolling banner) is the same, the claim it makes is
just true.
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

AUTHOR_NAME = "Agastya Sharma"
AUTHOR_EMAIL = "work.agastya20@gmail.com"
REPO_URL = "https://github.com/agastyasharma20/cib-ews"
DISCLAIMER_TEXT = (
    "⚠ Independent portfolio project — built on 100% SYNTHETIC data for a Data Science "
    "internship application. Not affiliated with, endorsed by, or built using data from HDFC Bank "
    "or any financial institution.  •  All figures shown are simulated.  •  "
    f"Built by {AUTHOR_NAME}."
)

CHART_COLORS = {"liquidity": "#155E75", "relationship": "#0891B2", "competitor": "#EA580C"}

st.set_page_config(page_title="CIB Early Warning System", page_icon="🏦", layout="wide")


# ---------------------------------------------------------------------------
# Styling: header, scrolling disclaimer banner, badge helpers
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Streamlit's own top toolbar (Deploy/Stop/menu) is a fixed,
        transparent-background bar that floats OVER the page rather than
        reserving layout space — 1.2rem of top padding wasn't enough to
        clear it, so the toolbar's own buttons visually overlapped our
        header's title and credit text. Reported by the user with a
        screenshot; fixed by pushing content down far enough to clear it. */
        .block-container { padding-top: 4.5rem; }

        .ews-header {
            display: flex; align-items: center; justify-content: space-between;
            padding: 0.6rem 1rem; margin-bottom: 0.6rem;
            background: linear-gradient(90deg, #0F172A 0%, #155E75 100%);
            border-radius: 10px; color: white;
        }
        .ews-header .brand { display: flex; align-items: center; gap: 0.7rem; }
        .ews-header .logo-mark {
            font-size: 1.9rem; background: white; color: #155E75;
            border-radius: 8px; width: 46px; height: 46px;
            display: flex; align-items: center; justify-content: center;
        }
        .ews-header .title { font-size: 1.25rem; font-weight: 700; line-height: 1.1; }
        .ews-header .subtitle { font-size: 0.78rem; opacity: 0.85; }
        .ews-header .credit { font-size: 0.8rem; text-align: right; opacity: 0.9; }
        .ews-header .credit a { color: #7DD3FC; text-decoration: none; }

        .ews-marquee-wrap {
            overflow: hidden; white-space: nowrap; background: #FEF3C7;
            border: 1px solid #FDE68A; border-radius: 6px; padding: 0.35rem 0;
            margin-bottom: 1rem;
        }
        .ews-marquee-track {
            display: inline-block; padding-left: 100%;
            animation: ews-scroll 22s linear infinite;
            font-size: 0.82rem; color: #92400E; font-weight: 600;
        }
        @keyframes ews-scroll {
            0%   { transform: translate(0, 0); }
            100% { transform: translate(-100%, 0); }
        }

        .ews-footer {
            margin-top: 2rem; padding-top: 0.8rem; border-top: 1px solid #E2E8F0;
            font-size: 0.75rem; color: #64748B; text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        f"""
        <div class="ews-header">
            <div class="brand">
                <div class="logo-mark">🏦</div>
                <div>
                    <div class="title">CIB Early Warning System</div>
                    <div class="subtitle">30&ndash;90 day silent-deterioration risk &bull; prototype on synthetic data</div>
                </div>
            </div>
            <div class="credit">
                Built by <b>{AUTHOR_NAME}</b><br>
                <a href="{REPO_URL}" target="_blank">GitHub ↗</a>
            </div>
        </div>
        <div class="ews-marquee-wrap">
            <div class="ews-marquee-track">{DISCLAIMER_TEXT}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="ews-footer">
            CIB Early Warning System — portfolio project by {AUTHOR_NAME} ({AUTHOR_EMAIL}) &bull;
            <a href="{REPO_URL}" target="_blank">{REPO_URL}</a> &bull;
            All data synthetic &bull; Not affiliated with HDFC Bank
        </div>
        """,
        unsafe_allow_html=True,
    )


TIER_BADGE = {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}
DRIVER_BADGE = {"liquidity": "💧 Liquidity", "relationship": "🤝 Relationship", "competitor": "🏦 Competitor"}


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
# Page chrome + sidebar: view toggle + filters
# ---------------------------------------------------------------------------
inject_css()
render_header()

st.sidebar.markdown("### Navigation")
view = st.sidebar.radio(
    "View", ["🎯 RM Cockpit", "📊 Portfolio View", "ℹ️ About This Project"], label_visibility="collapsed"
)

filtered_df = master_df.copy()
if view != "ℹ️ About This Project":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    segment_filter = st.sidebar.multiselect("Segment", sorted(master_df["segment"].unique()), default=[])
    branch_filter = st.sidebar.multiselect("Branch", sorted(master_df["branch_id"].unique()), default=[])
    tier_filter = st.sidebar.multiselect("Priority tier", ["High", "Medium", "Low"], default=[])

    if segment_filter:
        filtered_df = filtered_df[filtered_df["segment"].isin(segment_filter)]
    if branch_filter:
        filtered_df = filtered_df[filtered_df["branch_id"].isin(branch_filter)]
    if tier_filter:
        filtered_df = filtered_df[filtered_df["priority_tier"].isin(tier_filter)]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"{len(filtered_df):,} of {len(master_df):,} customers shown")

st.sidebar.markdown("---")
st.sidebar.caption(f"Built by **{AUTHOR_NAME}**\n\n[{REPO_URL.replace('https://', '')}]({REPO_URL})")


# ---------------------------------------------------------------------------
# View 1: RM Cockpit
# ---------------------------------------------------------------------------
def render_rm_cockpit(df: pd.DataFrame) -> None:
    st.title("🎯 RM Cockpit")
    st.caption("Customers ranked by PFaR (Probability-weighted Funds at Risk). Click a column header to sort.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers in view", f"{len(df):,}")
    col2.metric("🔴 High priority", f"{(df['priority_tier'] == 'High').sum():,}")
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
    display_df["priority_tier"] = display_df["priority_tier"].map(TIER_BADGE)
    display_df["PFaR_driver_type"] = display_df["PFaR_driver_type"].map(DRIVER_BADGE)

    display_df = display_df[
        [
            "customer_id", "segment", "branch_id", "priority_tier", "PFaR", "PFaR_driver_type",
            "ews_score_90d", "top_3_reason_codes_str", "recommended_action",
            "Est. days to deterioration", "survival_curve",
        ]
    ].rename(columns={
        "top_3_reason_codes_str": "Top reason codes",
        "ews_score_90d": "EWS score (90d)",
        "priority_tier": "Priority",
        "PFaR_driver_type": "Driver type",
        "branch_id": "Branch",
        "customer_id": "Customer",
        "segment": "Segment",
        "recommended_action": "Recommended action",
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=560,
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
    st.title("📊 Portfolio View")

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers in view", f"{len(df):,}")
    col2.metric("Total PFaR at risk (Cr)", f"{df['PFaR'].sum():,.1f}")
    col3.metric("🔴 High priority share", f"{(df['priority_tier'] == 'High').mean():.0%}" if len(df) else "—")

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
            heatmap_data, color_continuous_scale="Blues", aspect="auto",
            labels=dict(color="Mean PFaR (Cr)"),
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Driver-mix breakdown")
        st.caption("What's driving PFaR across the (filtered) portfolio, by top reason-code category.")
        driver_counts = df["PFaR_driver_type"].value_counts().reset_index()
        driver_counts.columns = ["driver_type", "count"]
        fig = px.pie(driver_counts, names="driver_type", values="count", hole=0.45,
                     color="driver_type", color_discrete_map=CHART_COLORS)
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
    fig = px.line(trend, x="month", y="PFaR", markers=True, color_discrete_sequence=["#155E75"])
    fig.update_layout(height=350, yaxis_title="Total portfolio PFaR (Cr)")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# View 3: About This Project
# ---------------------------------------------------------------------------
def render_about_page() -> None:
    st.title("ℹ️ About This Project")

    st.markdown(
        """
Large corporate customers rarely close a Current Account outright when a
relationship sours — they quietly shift balances, transaction flow,
payroll, and trade activity to a competing bank while the account stays
technically open. By the time that shows up as a visibly declining
monthly balance, most of the wallet share is usually already gone, and
today's process is reactive: RMs act only once the erosion is obvious.

**This project builds an Early Warning System (EWS) that flags that
silent deterioration 30–90 days in advance, explains *why* in plain
language, estimates roughly *when*, sizes the exposure in rupee terms, and
recommends a specific RM action** — for a Data Science internship
application, built end-to-end on realistic synthetic data since real core-
banking data isn't available outside a bank's own systems.
        """
    )

    st.markdown("### How it works, phase by phase")
    phases = [
        ("1. Synthetic data", "5,000 CA/CIB customers, 18 months of behavioral history, 4 planted deterioration cohorts with known ground truth for validation."),
        ("2. Labeling & features", "A composite Deterioration Index (percentile-ranked vs. peers, not a flat threshold) builds the forward-looking label; 22 features across 5 groups (Balance, Transactions, Product, Network, Relationship) feed the model."),
        ("3-4. Modeling", "Logistic regression baseline → tuned XGBoost core model, compared on identical metrics and a strict time-based train/test split."),
        ("5. Explainability", "SHAP values translated into plain-language reason codes (e.g. \"Falling balances\", \"Rising competitor-transfer share\") — not just a risk number."),
        ("6. Risk sizing & actions", "PFaR (Probability-weighted Funds at Risk) sizes each customer's exposure in ₹ Cr; a rules-based engine maps the top reason code to a concrete RM action."),
        ("7. Survival analysis", "A time-to-deterioration model (Random Survival Forest, chosen after Cox's assumptions failed a real check) answers roughly *when*, not just *if*."),
        ("8. Graph feature", "A wallet-leakage graph feature was tested and HONESTLY found not to improve the model — reported as a negative result, not hidden."),
        ("9. This dashboard", "RM Cockpit + Portfolio View, reading directly from the pipeline's output artifacts."),
    ]
    for name, desc in phases:
        st.markdown(f"**{name}** — {desc}")

    st.markdown("### Key results")
    results_df = pd.DataFrame(
        [
            ["Core model (XGBoost, 90d)", "ROC-AUC 0.958 · PR-AUC 0.903 · 6.3x top-decile lift"],
            ["Label validation vs. hidden ground truth", "98.9% precision / 98.3% recall on planted cohorts"],
            ["Survival model", "Concordance index 0.659 (Random Survival Forest)"],
            ["Graph feature experiment", "Honest negative result — no AUC lift, redundant with an existing feature"],
        ],
        columns=["Metric", "Result"],
    )
    st.table(results_df)

    st.markdown("### Tech stack")
    st.markdown(
        "`pandas` `numpy` `scikit-learn` `XGBoost` `LightGBM` `SHAP` `lifelines` "
        "`scikit-survival` `networkx` `Streamlit` `Plotly`"
    )

    st.markdown("### Full write-up")
    st.markdown(
        f"The complete methodology — every modeling choice, result, limitation, and the path to "
        f"production — is in [`docs/methodology.md`]({REPO_URL}/blob/master/docs/methodology.md) "
        f"in the repository: **[{REPO_URL}]({REPO_URL})**"
    )

    st.markdown("---")
    st.markdown(
        f"""
**Built by {AUTHOR_NAME}**
📧 [{AUTHOR_EMAIL}](mailto:{AUTHOR_EMAIL}) &nbsp;·&nbsp; 🔗 [GitHub]({REPO_URL})

*This is an independent portfolio project created for a Data Science internship application.
It is not affiliated with, sponsored by, or built using any real data from HDFC Bank or any
other financial institution — every number on this dashboard is synthetic.*
        """
    )


if view == "🎯 RM Cockpit":
    render_rm_cockpit(filtered_df)
elif view == "📊 Portfolio View":
    render_portfolio_view(filtered_df)
else:
    render_about_page()

render_footer()
