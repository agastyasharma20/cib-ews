# Explainability: From SHAP Values to Reason Codes

This doc explains how a raw model score turns into the 3 plain-language
"reason codes" an RM sees for a flagged customer — the answer to "why is
this account being called out, in terms I can act on?"

## Why SHAP, and specifically per-customer, not just global feature importance

Phase 4's gain-based feature importance (`results/figures/*_importance.png`)
answers "which features did the model lean on overall, across the whole
portfolio." That's useful for sanity-checking the model, but it's the wrong
tool for the actual product requirement: an RM doesn't need to know the
portfolio's average driver, they need to know **why this specific
customer, this month** was flagged. SHAP (`src/explainability/shap_explain.py`)
computes exactly that — a per-row attribution of how much each feature
pushed *that customer's* predicted risk up or down, relative to the
model's average prediction. `TreeExplainer` is used because it's exact
(not an approximation) for gradient-boosted trees and fast enough to score
the full ~90,000-row panel in one pass.

## From SHAP values to reason codes

Three design choices turn a row of 23 SHAP numbers into 3 readable labels
(`src/explainability/reason_codes.py`):

1. **Group by driver label before ranking, not after.** Several raw
   features often tell the same story — `amb_pct_change_30d`,
   `_60d`, and `_90d` are all "Falling balances." Taking the top 3
   individual SHAP features first could return "Falling balances" three
   times over. Instead, every feature's SHAP value is relabeled to its
   driver category (`src/explainability/config.py:FEATURE_TO_DRIVER_LABEL`)
   and SUMMED within each category first — so the ranking is over distinct
   storylines, not distinct columns.

2. **Rank by signed sum, not absolute value.** A reason code should
   explain why risk is HIGH. Ranking by absolute SHAP magnitude could let a
   strongly *protective* factor (e.g., a big improvement in digital
   activity) crowd out the actual risk drivers just because its magnitude
   happens to be large. Ranking by the signed sum keeps the reason codes
   pointed at what's pushing risk up.

3. **One horizon's model backs the explanation, all three still report
   scores.** The scored table carries `ews_score_30d`, `_60d`, and `_90d`
   from their own respective models, but `top_3_reason_codes` is always
   computed from the **90-day model's** SHAP values
   (`CORE_REASON_HORIZON` in `src/explainability/config.py`). Three
   horizons could in principle disagree on what's driving risk; picking
   one keeps the story an RM sees consistent, and 90 days matches the
   project's primary "advance warning" framing.

## The driver label mapping

Every one of the 22 engineered features (plus `segment`) has a
hand-authored entry in `FEATURE_TO_DRIVER_LABEL`, grouped along the same 5
feature groups as `docs/feature_dictionary.md`:

| Driver label | Feature group | Example underlying features |
|---|---|---|
| Falling balances | Balance & Liquidity | `amb_pct_change_30/60/90d`, `amb_trend_slope_3m`, `amb_seasonal_adjusted_deviation` |
| Unstable balances | Balance & Liquidity | `amb_volatility_3m` |
| Declining transaction activity | Transaction & Digital | `txn_count_trend_slope_3m` |
| Declining credits | Transaction & Digital | `txn_value_trend_slope_3m` |
| Lower digital activity | Transaction & Digital | `digital_channel_share(_trend_3m)`, `login_frequency_trend_3m` |
| Reduced payroll activity | Product & Wallet-Share | `payroll_regularity_score`, `has_payroll_book` |
| Reduced trade utilization | Product & Wallet-Share | `trade_utilization_trend_3m`, `has_trade_finance` |
| Rising competitor-transfer share | Network & Counterparty | `competitor_txn_share(_trend)` |
| Concentrating money flow to fewer counterparties | Network & Counterparty | `counterparty_concentration_hhi` |
| Rising service complaints | Relationship & Engagement | `complaint_rolling_3m_sum`, `service_ticket_rolling_3m_sum` |
| Narrowing product relationship | Relationship & Engagement | `product_holding_breadth` |
| Relationship tenure profile | Relationship & Engagement | `relationship_tenure_months` |
| Segment-level risk profile | Customer master | `segment` |

`src/explainability/reason_codes.py:assert_all_features_mapped` fails loudly
if a feature is ever added to `src/features/` without a corresponding entry
here — a new signal can never silently reach the dashboard unexplained.

## What this doesn't do (yet)

Reason codes here are per-customer-month, independent snapshots — they
don't yet track HOW a customer's driver mix evolved over the months
leading up to a flag (e.g. "complaints, then falling balances, then
competitor-transfer share rising" as a sequence). That richer narrative is
a natural extension once the RM-action mapping (a later phase) needs it,
but isn't required for the current "why is this customer flagged right
now" use case.
