"""
Feature -> plain-language "reason code" driver label.

WHY a hand-authored mapping instead of just showing raw feature names:
"amb_seasonal_adjusted_deviation = -0.42" means nothing to an RM. "Falling
balances" does. This is the translation layer between what the model
actually computes (src/features/, 5 groups) and what a human reads on the
dashboard. Every feature in src/features/config.py:FEATURE_GROUPS must have
an entry here — build_customer_reason_codes() in reason_codes.py asserts
this so a newly added feature can never silently show up unexplained.

Multiple features are deliberately mapped to the SAME label (e.g. all three
amb_pct_change_* windows -> "Falling balances") — a reason code should name
a DRIVER category an RM can act on, not enumerate every underlying column
that moved.
"""

FEATURE_TO_DRIVER_LABEL: dict[str, str] = {
    # --- 1. Balance & Liquidity ---
    "amb_trend_slope_3m": "Falling balances",
    "amb_volatility_3m": "Unstable balances",
    "amb_pct_change_30d": "Falling balances",
    "amb_pct_change_60d": "Falling balances",
    "amb_pct_change_90d": "Falling balances",
    "amb_seasonal_adjusted_deviation": "Falling balances",

    # --- 2. Transaction & Digital Activity ---
    "txn_count_trend_slope_3m": "Declining transaction activity",
    "txn_value_trend_slope_3m": "Declining credits",
    "digital_channel_share": "Lower digital activity",
    "digital_channel_share_trend_3m": "Lower digital activity",
    "login_frequency_trend_3m": "Lower digital activity",

    # --- 3. Product & Wallet-Share ---
    "payroll_regularity_score": "Reduced payroll activity",
    "has_payroll_book": "Reduced payroll activity",
    "trade_utilization_trend_3m": "Reduced trade utilization",
    "has_trade_finance": "Reduced trade utilization",

    # --- 4. Network & Counterparty ---
    "competitor_txn_share": "Rising competitor-transfer share",
    "competitor_txn_share_trend": "Rising competitor-transfer share",
    "counterparty_concentration_hhi": "Concentrating money flow to fewer counterparties",

    # --- 5. Relationship & Engagement ---
    "relationship_tenure_months": "Relationship tenure profile",
    "complaint_rolling_3m_sum": "Rising service complaints",
    "service_ticket_rolling_3m_sum": "Rising service complaints",
    "product_holding_breadth": "Narrowing product relationship",

    # --- Customer master (passed through as a native categorical feature) ---
    "segment": "Segment-level risk profile",
}

# Which horizon's core model drives the reason codes in the final scored
# table. All 3 horizon scores are still reported numerically — this only
# picks which model's SHAP values back the single top_3_reason_codes
# column, matching the brief's business framing ("30-90 day advance
# warning"): 90 days is the primary decision window an RM is meant to act
# within, so its drivers are the ones explained.
CORE_REASON_HORIZON = "deteriorates_in_90d"

TOP_N_REASON_CODES = 3
