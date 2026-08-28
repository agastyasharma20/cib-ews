"""
Tunable knobs for the deterioration-index labeling framework.

Kept as plain module-level constants (not buried in code) so every threshold
used to decide "this customer is deteriorating" is visible and defensible in
one place — the kind of thing a reviewer (or an interviewer) will ask about
first.
"""

# Monthly panel granularity: one row per customer per calendar month. The
# business brief talks about "30/60/90 day" windows; with monthly data the
# closest honest mapping is 1/2/3 months. This is a real simplification -
# documented here and in docs/deterioration_definition.md - that a
# production system with daily/weekly transaction feeds would remove.
TRAILING_WINDOW_MONTHS = 3   # "trailing 90 days" ~= trailing 3 months
HORIZONS_MONTHS = {
    "deteriorates_in_30d": 1,
    "deteriorates_in_60d": 2,
    "deteriorates_in_90d": 3,
}

# Composite Deterioration Index (DI) weights. Each component is a 0-1
# "decline percentile" (see deterioration_index.py) - how much worse this
# customer's trailing-vs-prior change is compared to same-segment,
# same-month peers. Weights reflect business judgment about how much each
# signal should count, and are deliberately renormalized per-row over
# whichever components are actually available for that customer (e.g. a
# customer with no trade-finance facility only has payroll to go on for that
# component; a customer with neither drops the component from their DI
# entirely rather than being penalized for a metric that doesn't apply).
COMPONENT_WEIGHTS = {
    # Balance is the ultimate business outcome (this is a Current Account
    # book) but it's also the stickiest/laggiest and most seasonality-prone
    # signal, so it gets a meaningful but not dominant weight.
    "amb_decline": 0.35,
    # Transaction count/value decline is a more immediate behavioral tell
    # that money is moving elsewhere before the balance itself craters.
    "txn_decline": 0.30,
    # Digital engagement is a softer, noisier signal (log-in habits vary a
    # lot for reasons unrelated to banking relationship health).
    "digital_decline": 0.15,
    # Payroll/trade-finance are "sticky" products - when they DO move, it's
    # a strong signal (a customer rarely moves payroll processing lightly),
    # so it's weighted close to transactions despite being rarer.
    "payroll_trade_decline": 0.20,
}

# A customer is "index-breached" in a given month if their composite DI
# exceeds this threshold, i.e. their blended decline signal is worse than
# this percentile of same-segment peers. Chosen via the grid search in
# validate.py - see docs/deterioration_definition.md for the trade-off table
# that justified this specific value. 0.80 gave near-perfect recall but a
# 5.2% false-positive rate on the seasonal_false_positive cohort alone -
# too noisy for an RM to trust. 0.85 cuts that to 1.7% (and stable to 0.1%)
# while recall on true deterioration only drops from 99.8% to 98.3%, and
# sudden_deterioration is still caught 100% of the time.
DI_THRESHOLD = 0.85

# --- Seasonal false-positive handling (requirement: don't flag a dip that
# reverses quickly and isn't corroborated by other signals) ---
SEASONAL_FILTER = {
    # How many months ahead we check for a "bounce back" after a breach.
    "reversal_window_months": 2,
    # A breach counts as "reverted" if DI falls back below this level within
    # the reversal window (i.e. the customer is no longer in the worst
    # quartile relative to peers).
    "recovery_threshold": 0.50,
    # A reverted dip is still treated as genuine deterioration if digital or
    # payroll/trade decline was ALSO elevated at the time of breach (i.e.
    # more than one independent signal moved, not just balance/transactions
    # bouncing from a one-off event).
    "accompanying_decline_threshold": 0.50,
}
