# Feature Dictionary

Every feature in `data/processed/model_dataset.parquet`, grouped exactly as
in `src/features/` — this grouping is the same one the reason-code / RM-
action mapping (Phase 5+) will reuse, so "why was this customer flagged"
can be answered at the group level (e.g. "mostly Network & Counterparty
signals → wallet-share retention conversation") before drilling into
individual features.

All features are **leakage-safe**: every rolling/trend calculation uses
only data at or before the row's own month. None of them read the labels
or `ground_truth_cohort`.

## 1. Balance & Liquidity

| Feature | Business rationale |
|---|---|
| `amb_trend_slope_3m` | Direction and speed of balance change over the last 3 months — a smooth decline is a stronger tell than a single bad month. |
| `amb_volatility_3m` | Coefficient of variation of balance over 3 months — erratic cash management can itself precede a relationship problem. |
| `amb_pct_change_30d` | Month-over-month balance change — the most immediate liquidity signal. |
| `amb_pct_change_60d` | 2-month balance change — filters out single-month noise. |
| `amb_pct_change_90d` | 3-month balance change — the closest monthly-data equivalent to the brief's 90-day window. |
| `amb_seasonal_adjusted_deviation` | How far this month's balance sits from the customer's own trailing 6-month baseline — nets out recurring quarter-end/festive bumps so a genuine dip isn't confused with a seasonal one. |

## 2. Transaction & Digital Activity

| Feature | Business rationale |
|---|---|
| `txn_count_trend_slope_3m` | Trend in total transaction count — money movement usually slows before the balance visibly drops. |
| `txn_value_trend_slope_3m` | Trend in total transaction value — captures large-ticket activity leaving even if transaction count holds steady. |
| `digital_channel_share` | Share of transactions done via mobile banking this month — a proxy for digital vs. branch/other-channel engagement. |
| `digital_channel_share_trend_3m` | Whether that digital share is rising or falling vs. its own 6-month baseline — declining digital engagement often precedes disengagement from the bank overall. |
| `login_frequency_trend_3m` | Trend in raw login counts, independent of the mobile-txn-share feature — a customer can log in less while still doing most of what they DO transact on mobile. |

## 3. Product & Wallet-Share

| Feature | Business rationale |
|---|---|
| `payroll_regularity_score` | Fraction of the trailing 6 months payroll was actually credited — a customer sliding from 6/6 to 2/6 is quietly migrating payroll elsewhere well before it hits zero. |
| `has_payroll_book` | Whether the customer has ever run payroll through this account — disambiguates "no facility" from "facility, declining." |
| `trade_utilization_trend_3m` | Trend in trade-finance (LC/BG) limit utilization — a sticky, high-value product; movement here is a strong signal. |
| `has_trade_finance` | Whether the customer has ever held a trade-finance facility — same disambiguation purpose as `has_payroll_book`. |

## 4. Network & Counterparty *(placeholder — full graph version in Phase 7)*

| Feature | Business rationale |
|---|---|
| `competitor_txn_share` | Share of this month's counterparty transactions going to a fixed list of competitor banks — the most direct "wallet leaving the bank" signal available. |
| `competitor_txn_share_trend` | Recent (3m) competitor share vs. a 6-month baseline — rising above 0 means the leak is accelerating, not just persistently present. |
| `counterparty_concentration_hhi` | Herfindahl-Hirschman Index of the customer's counterparty-bank mix — money consolidating toward ONE other bank is a sharper signal than diversifying spend. |

*Not yet included: cross-customer contagion via shared-promoter/director
relationships (`relationships.csv`-style linkage) — that requires the
networkx-based graph propagation built in Phase 7.*

## 5. Relationship & Engagement

| Feature | Business rationale |
|---|---|
| `relationship_tenure_months` | How long the customer has banked here, recomputed as of each observed month (not a static snapshot) — longer relationships are typically stickier, changing how the same warning signal should be weighted. |
| `complaint_rolling_3m_sum` | Complaints logged over the trailing 3 months — building dissatisfaction is a leading indicator, especially for service-issue-driven attrition. |
| `service_ticket_rolling_3m_sum` | Same idea for service tickets — a broader net than complaints alone. |
| `product_holding_breadth` | Count of distinct product lines held (CA + payroll + trade finance) — a broader relationship generally has more signal surface and is stickier to fully unwind. |

**Known gap:** "RM contact recency" from the original brief is omitted —
the Phase 1 synthetic panel does not simulate RM visit/contact events, so
adding this column would be fabricated noise, not a feature. A real
deployment would source it from CRM contact logs.

## Labels (not features — the modeling targets)

| Column | Meaning |
|---|---|
| `deteriorates_in_30d` / `_60d` / `_90d` | Whether the customer crosses into confirmed deterioration within that forward window. See `docs/deterioration_definition.md` for how these are built. `NaN` where there isn't enough future history yet — must be dropped, not treated as 0, before training. |

## Split column

`split` is `"train"` for `month_idx <= 11` (first 12 months) and `"test"`
for the remaining 6 months — a **time-based** split, not random, so the
model is evaluated the way it will actually be used: trained on history,
scored going forward. See `src/features/config.py` for the cutoff.
