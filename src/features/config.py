"""
Tunable knobs for feature engineering, organized by the same 5 feature
groups used throughout src/features/. Grouping is deliberate: it's the same
grouping the reason-code / RM-action mapping (Phase 5+) will use to turn
"which features drove this score" into a human-readable explanation, e.g.
"flagged mainly on Network/Counterparty signals -> recommend a wallet-share
retention conversation" vs. "flagged on Relationship/Engagement -> recommend
a service recovery call."

FEATURE_GROUPS below is the single source of truth for that mapping -
build_features.py asserts every engineered column is registered here.
"""

# --- 1. Balance & Liquidity ---
AMB_TREND_WINDOW = 3          # months, for the trend-slope feature
AMB_VOLATILITY_WINDOW = 3     # months, for rolling coefficient of variation
AMB_SEASONAL_BASELINE_WINDOW = 6  # trailing months used as "own typical level"
# NOTE on the seasonality baseline: with only 18 months of history, a
# trailing 6-month window is used as each customer's "normal" level (long
# enough to span a full quarter-end cycle twice). A longer trailing-12m
# baseline would be more robust but would leave the first 11 months of every
# customer's history without this feature at all - a real deployment with
# multi-year history should lengthen this window.

# --- 2. Transaction & Digital Activity ---
TXN_TREND_WINDOW = 3          # months, for count/value trend slopes
DIGITAL_TREND_WINDOW = 3      # months, for login-frequency trend slope

# --- 3. Product & Wallet-Share ---
PAYROLL_REGULARITY_WINDOW = 6  # months, fraction of months payroll credited
TRADE_TREND_WINDOW = 3         # months, for trade-utilization trend slope

# --- 4. Network / Counterparty (placeholder — full graph version in Phase 7) ---
COMPETITOR_SHARE_SHORT_WINDOW = 3   # months, "recent" competitor-share level
COMPETITOR_SHARE_LONG_WINDOW = 6    # months, "baseline" competitor-share level

# --- 5. Relationship & Engagement ---
COMPLAINT_ROLLING_WINDOW = 3   # months, complaint/service-ticket build-up

# --- Train/test split ---
# Time-based (not random): all customer-months with month_idx <= this
# cutoff are TRAIN, the rest TEST. This mirrors real deployment (train on
# history, score forward) and avoids the subtler leakage of a random split
# letting the model see a customer's later months while training on their
# earlier ones.
TRAIN_MAX_MONTH_IDX = 11  # months 0-11 (12 months) train, 12-17 (6 months) test

# --- Feature group registry ---
# Populated incrementally by each group module calling `register()` below,
# so the mapping can never silently drift out of sync with what's actually
# built. Phase 3+ can do FEATURE_GROUPS["Balance & Liquidity"] to get that
# group's column list, or invert it into {column: group} for reason codes.
FEATURE_GROUPS: dict[str, list[str]] = {
    "Balance & Liquidity": [],
    "Transaction & Digital Activity": [],
    "Product & Wallet-Share": [],
    "Network & Counterparty": [],
    "Relationship & Engagement": [],
}


def register(group: str, columns: list[str]) -> None:
    """Called once by each group's build_* module at import time."""
    FEATURE_GROUPS[group].extend(c for c in columns if c not in FEATURE_GROUPS[group])


def _ensure_groups_registered() -> None:
    """
    FEATURE_GROUPS is populated by IMPORTING each group module (register()
    runs as an import-time side effect in balance_liquidity.py etc.) — so
    any caller that imports only this config module, without also having
    imported the group modules first, would silently see an EMPTY registry
    instead of an error. That exact bug produced a model trained on zero
    real features early in Phase 3. Every accessor below calls this first
    so the registry is always populated regardless of import order.
    Imported lazily (inside the function, not at module load time) because
    the group modules import FROM this config module — a top-level import
    here would be circular.
    """
    if all(len(cols) == 0 for cols in FEATURE_GROUPS.values()):
        from src.features import (  # noqa: F401
            balance_liquidity,
            transaction_digital,
            product_wallet_share,
            network_counterparty,
            relationship_engagement,
        )


def all_feature_columns() -> list[str]:
    _ensure_groups_registered()
    return [c for cols in FEATURE_GROUPS.values() for c in cols]


def feature_to_group_map() -> dict[str, str]:
    _ensure_groups_registered()
    return {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}
