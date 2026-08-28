"""
Phase 8 — Wallet-leakage graph feature (stretch).

WHAT GRAPH THIS ACTUALLY IS, AND WHY (an honest scoping note):
"Customer-counterparty graph" suggests entity-level links (two customers
paying the same vendor, shared suppliers, etc.), but the synthetic data
only records the COUNTERPARTY'S BANK, not a counterparty entity identity
(src/data_generation/generate_synthetic_data.py never simulated individual
counterparty accounts — see counterparty_transactions.parquet's columns).
So the graph built here is a customer <-> bank bipartite graph (~5,000
customer nodes, 11 bank nodes), not a customer <-> customer one. This is a
real scoping limitation worth being upfront about — a production system
with actual counterparty account identifiers could build the richer
customer-to-customer graph (shared-vendor / shared-payroll-processor
contagion) this phase's name evokes; what's built here is the version the
available data actually supports.

WHY THIS IS STILL A GENUINE NETWORKX-NATIVE COMPUTATION, NOT JUST A
RELABELED PANDAS AGGREGATION:
Two of the three features below only make sense on a graph:
  - `graph_bank_diversity_degree` is a literal node degree — how many
    distinct bank nodes a customer connects to that month.
  - `graph_competitor_centrality_exposure` requires computing eigenvector
    centrality across the FULL bipartite graph (all ~5,000 customers and
    11 banks at once, so a bank's centrality reflects how much of the
    ENTIRE portfolio's money is flowing to it that month) and then reading
    off, for each customer, a value-weighted average of the centrality of
    the specific banks they personally transact with. That requires the
    graph structure — it cannot be computed from one customer's rows in
    isolation the way the Phase 2 pandas features were.

THE THIRD FEATURE, competitor share, IS conceptually similar to Phase 2's
`competitor_txn_share` but is recomputed here directly from graph edge
weights (transaction VALUE, not count) as the base signal the other two
graph features build on, and to keep this module runnable standalone.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, COUNTERPARTY_TXNS_FILE
from src.features.utils import add_month_index, rolling_mean, momentum_ratio

GRAPH_FEATURES_FILE = DATA_PROCESSED_DIR / "graph_features.parquet"

GRAPH_FEATURE_COLUMNS = [
    "graph_competitor_share_value_weighted",
    "graph_bank_diversity_degree",
    "graph_competitor_centrality_exposure",
    "graph_competitor_share_trend_3m6m",
]


def _build_monthly_bipartite_graph(month_txns: pd.DataFrame) -> nx.Graph:
    """One bipartite graph for a single month: customer nodes (bipartite=0)
    and bank nodes (bipartite=1), edge weight = total transaction value
    between that customer and that bank this month."""
    g = nx.Graph()
    edge_weights = month_txns.groupby(["customer_id", "counterparty_bank_ifsc_prefix"])["txn_value"].sum()

    customers = month_txns["customer_id"].unique()
    banks = month_txns["counterparty_bank_ifsc_prefix"].unique()
    g.add_nodes_from(customers, bipartite=0)
    g.add_nodes_from(banks, bipartite=1)

    for (customer_id, bank), value in edge_weights.items():
        g.add_edge(customer_id, bank, weight=float(value))

    return g


def _monthly_graph_features(month: str, month_txns: pd.DataFrame, competitor_banks: set[str]) -> pd.DataFrame:
    g = _build_monthly_bipartite_graph(month_txns)

    # Eigenvector centrality across the WHOLE portfolio's bipartite graph
    # this month — a bank's centrality rises when more customers, and more
    # value, flow to it relative to other banks. weight="weight" makes this
    # value-weighted, not just count-weighted.
    try:
        centrality = nx.eigenvector_centrality_numpy(g, weight="weight")
    except Exception:
        # Degenerate/disconnected graphs (shouldn't happen at this scale,
        # but eigenvector centrality can fail to converge on pathological
        # inputs) — fall back to weighted degree centrality, a reasonable
        # and still graph-native substitute.
        centrality = {n: d for n, d in g.degree(weight="weight")}
        max_c = max(centrality.values()) or 1.0
        centrality = {n: c / max_c for n, c in centrality.items()}

    rows = []
    customers = [n for n, d in g.nodes(data=True) if d.get("bipartite") == 0]
    for customer_id in customers:
        neighbors = list(g.neighbors(customer_id))
        weights = np.array([g[customer_id][bank]["weight"] for bank in neighbors])
        total_weight = weights.sum()

        is_competitor = np.array([bank in competitor_banks for bank in neighbors])
        competitor_share = float(weights[is_competitor].sum() / total_weight) if total_weight > 0 else np.nan

        bank_centralities = np.array([centrality.get(bank, 0.0) for bank in neighbors])
        centrality_exposure = float(np.average(bank_centralities, weights=weights)) if total_weight > 0 else np.nan

        rows.append({
            "customer_id": customer_id,
            "month": month,
            "graph_competitor_share_value_weighted": competitor_share,
            "graph_bank_diversity_degree": g.degree(customer_id),
            "graph_competitor_centrality_exposure": centrality_exposure,
        })

    return pd.DataFrame(rows)


def build_graph_features(counterparty_df: pd.DataFrame) -> pd.DataFrame:
    """Builds one bipartite graph PER MONTH (a customer's set of
    counterparties and the portfolio-wide centrality landscape both change
    month to month) and stacks the resulting per-customer features."""
    competitor_banks = set(counterparty_df.loc[counterparty_df["is_competitor_bank"], "counterparty_bank_ifsc_prefix"].unique())
    print(f"Competitor bank set: {sorted(competitor_banks)}")

    monthly_frames = []
    for month, month_txns in counterparty_df.groupby("month"):
        monthly_frames.append(_monthly_graph_features(month, month_txns, competitor_banks))
    df = pd.concat(monthly_frames, ignore_index=True)

    # Trend: recent (3m) vs longer (6m) value-weighted competitor share —
    # same momentum methodology as src/features/network_counterparty.py,
    # applied to the graph-derived share so the two are comparable.
    df = add_month_index(df)
    short_mean = rolling_mean(df, "graph_competitor_share_value_weighted", 3)
    long_mean = rolling_mean(df, "graph_competitor_share_value_weighted", 6)
    df["graph_competitor_share_trend_3m6m"] = momentum_ratio(short_mean, long_mean)

    return df.drop(columns=["month_idx"])


def main() -> None:
    print("Loading counterparty transactions...")
    counterparty_df = pd.read_parquet(COUNTERPARTY_TXNS_FILE)

    print("Building monthly bipartite customer<->bank graphs and computing features...")
    graph_features_df = build_graph_features(counterparty_df)

    graph_features_df.to_parquet(GRAPH_FEATURES_FILE, index=False)
    print(f"\nSaved {GRAPH_FEATURES_FILE} ({len(graph_features_df):,} rows)")
    print("\nSummary statistics:")
    print(graph_features_df[GRAPH_FEATURE_COLUMNS].describe().round(4))


if __name__ == "__main__":
    main()
