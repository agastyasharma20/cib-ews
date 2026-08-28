# Methodology

The single master document tying together every phase of this project —
written to be read end-to-end as an interview-prep narrative, or dipped
into section by section. Where a phase has its own deep-dive doc
(`deterioration_definition.md`, `feature_dictionary.md`,
`explainability.md`), this document gives the full story plus the key
numbers, and links out for the complete detail.

---

## 1. Business Problem

HDFC Bank's Current Account / CIB ("Customers in Base") portfolio faces a
quieter risk than account closure: large corporate customers keep their
HDFC current account technically open while gradually shifting balances,
transaction flow, payroll processing, and trade activity to competing
banks. By the time this shows up as a visibly declining monthly balance,
most of the wallet share is usually already gone, and today's process is
reactive — RMs act only once the erosion is obvious.

This project builds an Early Warning System (EWS) that:

1. **Scores** every CIB customer for 30/60/90-day advance risk of silent
   deterioration (not closure),
2. **Explains** *why* via reason codes derived from their own trailing
   behavior,
3. **Estimates** roughly *when* via a time-to-event survival model,
4. **Sizes** the exposure in rupee terms via a risk metric (PFaR), and
5. **Recommends** a concrete RM action per customer.

No real HDFC data was available, so the entire pipeline runs on a
realistic **synthetic** CA/CIB dataset with known ground-truth
deterioration cohorts for validation — built deliberately so that a real
core-banking data source could be substituted later without reworking the
feature, model, or dashboard code (see §10, Path to Production).

---

## 2. Data

`src/data_generation/generate_synthetic_data.py` (Phase 1) simulates 5,000
CA/CIB customers over 18 months across three tables:

- **`customers.parquet`** — static master (segment, branch, RM, tenure).
- **`monthly_panel.parquet`** — the core behavioral time series: balance,
  transactions, digital activity, payroll, trade finance, complaints.
- **`counterparty_transactions.parquet`** — per-transaction detail with
  the counterparty's bank, including a fixed set of "competitor bank" IFSC
  prefixes, used for the wallet-leakage signal.
- **`ground_truth_cohorts.parquet`** — the HIDDEN true cohort per
  customer, used ONLY for validation, never as a model input, anywhere in
  this project.

Four cohorts were planted with distinct, realistic dynamics: `stable`
(55%), `gradual_deterioration` (20%, slow multi-month bleed), `sudden_
deterioration` (10%, sharp 1-2 month drop from a simulated payroll-
migration event), and `seasonal_false_positive` (15%, periodic balance
dips that recover — a deliberate trap for the labeling framework). The
EDA notebook (`notebooks/01_eda_synthetic_data.ipynb`) visually confirms
each cohort's trajectory looks like real, messy behavior rather than a
clean synthetic template.

---

## 3. Deterioration Definition & Label Validation

Full detail: [`deterioration_definition.md`](deterioration_definition.md).

**The problem with an obvious approach.** A flat balance-drop threshold
fails for three reasons: scale (an SME's normal swings are a Large
Corporate's crisis), shared seasonality (quarter-end window dressing moves
everyone at once), and fragility (balance is the LAST thing to move in a
quietly eroding relationship — a threshold on it alone would be a lagging
indicator, not an early one).

**The approach.** A composite **Deterioration Index (DI)** blends four
trailing, percentile-ranked signals — computed relative to same-segment,
same-month peers, not an absolute cutoff:

| Component | Weight | Rationale |
|---|---|---|
| AMB decline | 0.35 | The business outcome, but laggy/seasonal |
| Transaction activity decline | 0.30 | Moves earlier than balance |
| Digital activity decline | 0.15 | Real but noisier signal |
| Payroll/trade-finance decline | 0.20 | Rare to move, strong when it does |

A customer "breaches" when DI exceeds a threshold — **0.85**, chosen by
grid search (0.80 gave near-perfect recall but a 5.2% false-positive rate
on `seasonal_false_positive` alone; 0.85 cut that to 1.7% for a 1.5-point
recall cost). A breach is downgraded to "harmless dip" only if it BOTH
reverts within 2 months AND isn't corroborated by digital/payroll/trade
decline — this is what stops the framework from flagging the
`seasonal_false_positive` cohort.

**Validation against the hidden ground truth** (customer-level: was this
customer ever confirmed-deteriorating?):

| True cohort | Flag rate |
|---|---|
| sudden_deterioration | 100.0% |
| gradual_deterioration | 97.5% |
| seasonal_false_positive | 1.7% |
| stable | 0.1% |

Precision 98.9%, recall 98.3%. Row-level (stricter, per forward-horizon)
precision stays 99%+, but recall is lower (37-51%) — expected, since the
index needs a few months of trailing evidence to accumulate before firing.
Median detection lag versus the true (hidden) onset: **2 months for
sudden deterioration, 4 months for gradual** — both still well ahead of
when balance-only monitoring would typically catch it.

---

## 4. Feature Groups

Full detail: [`feature_dictionary.md`](feature_dictionary.md). Every
feature is leakage-safe (trailing/current data only) and belongs to one of
5 groups — the same grouping the reason-code mapping (§6) reuses:

1. **Balance & Liquidity** (6 features) — trend slope, volatility,
   30/60/90-day % change, seasonality-adjusted deviation from a 6-month
   baseline.
2. **Transaction & Digital Activity** (5) — transaction count/value trend,
   digital channel share and its trend, login frequency trend.
3. **Product & Wallet-Share** (4) — payroll regularity score, trade
   utilization trend, plus `has_payroll_book`/`has_trade_finance` flags so
   "no facility" is never confused with "facility, declining."
4. **Network & Counterparty** (3) — competitor-bank transaction share and
   trend, counterparty concentration (HHI). A graph-native version was
   trialed in Phase 8 (§9).
5. **Relationship & Engagement** (4) — time-varying tenure, rolling
   complaint/service-ticket counts, product holding breadth.

`relationship_tenure_months` is deliberately recomputed AS OF each
observed month (not a static snapshot), and `segment` is passed through as
a native category rather than engineered.

---

## 5. Model Comparison Results (All Phases)

### 5.1 Baseline: Logistic Regression (Phase 3)

Explainable-by-construction benchmark: median imputation + standardization
+ one-hot encoding, `class_weight="balanced"`, evaluated on a **time-based**
split (train months 0-11, test 12-17 — never random, since this is a
forecasting problem).

| Horizon | ROC-AUC | PR-AUC | Precision@0.5 | Recall@0.5 | Top-decile lift |
|---|---|---|---|---|---|
| 30d | 0.984 | 0.925 | 0.525 | 0.982 | 8.25 |
| 60d | 0.960 | 0.888 | 0.454 | 0.945 | 6.90 |
| 90d | 0.935 | 0.841 | 0.429 | 0.910 | 5.85 |

Top protective coefficients (standardized): `amb_pct_change_90d`,
`amb_seasonal_adjusted_deviation`, `amb_volatility_3m` — all directionally
sensible (balance stability/growth lowers risk). Top risk-increasing:
`has_payroll_book`, `amb_pct_change_30d`, `complaint_rolling_3m_sum`.
**Caveat found and reported, not hidden:** `has_payroll_book` being
risk-*increasing* is a synthetic-data artifact — `sudden_deterioration` is
specifically defined as a payroll-migration event, so the model correctly
recovered the mechanism that was built into the simulator, not a general
real-world truth. **Calibration is poor** (see
`results/figures/baseline_logreg_*_calibration.png`): `class_weight=
"balanced"` reweights the loss, which biases raw predicted probabilities
away from the true base rate. Good for ranking/recall, not for showing an
RM a number that means what it says without recalibration.

### 5.2 Core Model: XGBoost vs. LightGBM, Tuned (Phases 4-5)

A small randomized hyperparameter search (8 draws each, scored on a
time-based validation slice — never K-fold CV, which would reintroduce
random-split leakage) was run for both algorithms; the winner was chosen
by **mean test PR-AUC across all 3 horizons** (not ROC-AUC — see §7 for
why).

| Horizon | Model | ROC-AUC | PR-AUC | Precision@0.5 | Recall@0.5 | Top-decile lift |
|---|---|---|---|---|---|---|
| 30d | lightgbm (tuned) | 0.990 | 0.931 | 0.787 | 0.929 | 8.33 |
| 30d | **xgboost (tuned)** | **0.992** | **0.952** | 0.625 | 0.986 | **8.51** |
| 60d | lightgbm (tuned) | 0.976 | 0.899 | 0.974 | 0.015 | 6.94 |
| 60d | **xgboost (tuned)** | **0.978** | **0.930** | 0.620 | 0.956 | **7.20** |
| 90d | lightgbm (tuned) | 0.951 | 0.854 | 0.000 | 0.000 | 6.16 |
| 90d | **xgboost (tuned)** | **0.958** | **0.903** | 0.544 | 0.925 | **6.29** |

**XGBoost chosen as the core model** (mean PR-AUC 0.928 vs LightGBM's
0.895). A finding worth keeping: LightGBM's PR-AUC/lift are close to
XGBoost's, but at the default 0.5 threshold its precision/recall
**collapse to 0** at the 90d horizon — its probabilities skew low even
with `scale_pos_weight`. This is exactly why the selection metric is
threshold-free PR-AUC, not a fixed-cutoff score.

An earlier, un-tuned LightGBM pass (default-ish leaf-wise growth
parameters) was materially worse (PR-AUC 0.815 at 90d, stopping after only
~8 boosting rounds) — traced to overfitting the validation slice almost
immediately; `num_leaves=31`, `min_child_samples=100`, lower learning
rate, and longer early-stopping patience closed most of the gap. Kept in
the repo's history as an example of not accepting a first pass at face
value.

### 5.3 Survival Analysis: Cox → Random Survival Forest (Phase 7)

Reframes "will this customer deteriorate in a window" as "how long until
this customer deteriorates" — the event is the labeling framework's own
`confirmed_deterioration` flag, snapshotted at each customer's earliest
fully-defined month (month_idx=5).

**CoxPHFitter was tried first**, as the standard approach, and explicitly
checked via `check_assumptions` — which flagged **11 of 24 covariates
(46%)**, including nearly every balance-trend feature, as violating the
proportional-hazards assumption. That's pervasive, not a couple of
borderline cases, so the model actually used is a **Random Survival
Forest** (scikit-survival), which makes no such assumption.

| Model | Concordance index |
|---|---|
| CoxPHFitter (violated assumption, kept only for comparison) | 0.591 |
| **Random Survival Forest (used)** | **0.659** |

Modest next to the classification models' 0.90+ AUC — expected, not a red
flag: this model gets exactly ONE early snapshot per customer rather than
fresh trailing features every month. The train/test split here is
customer-level, not row-level time-based (a one-row-per-customer survival
dataset doesn't have "months" to split on the same way), but the same
"never train on what wouldn't be known yet" discipline is reproduced by
administratively censoring the training fit at month 11 and evaluating
concordance against full follow-up (month 17) on held-out customers.

### 5.4 Graph Feature: Wallet-Leakage Bipartite Graph (Phase 8, stretch)

A customer↔bank bipartite graph (not customer↔customer — the synthetic
data only records the counterparty's *bank*, not an entity ID) was built
per month via networkx, yielding a value-weighted competitor-share
feature, its trend, bank-diversity degree, and a portfolio-wide
eigenvector-centrality exposure measure. Fed into the already-tuned
XGBoost core model (same hyperparameters, only the feature set changed):

| Horizon | Variant | ROC-AUC | PR-AUC |
|---|---|---|---|
| 90d | before (no graph) | 0.9585 | 0.9027 |
| 90d | after (+ graph) | 0.9580 | 0.9019 |

**No meaningful improvement** — flat to very slightly worse at every
horizon, and the graph features rank 14th-25th of 27 by mean \|SHAP\|.
Root cause, confirmed by direct correlation: the graph's value-weighted
competitor share correlates **0.88-0.91** with Phase 2's existing
pandas-derived `competitor_txn_share`/`_trend`, computed from the same
underlying transactions. Methodologically distinct computation, largely
redundant information. Kept as an honestly-negative result — the natural
extension point once real counterparty-entity data supports a genuine
customer↔customer contagion graph.

---

## 6. Explainability Approach

Full detail: [`explainability.md`](explainability.md).

SHAP `TreeExplainer` (exact for tree ensembles) computes per-customer
attributions from the core XGBoost model — global feature importance
answers "what does the model lean on overall"; SHAP answers "why did THIS
customer, THIS month, get this score," which is the actual product
requirement.

Turning SHAP values into 3 reason codes involves two deliberate choices:
(1) **group raw features by driver label BEFORE ranking** (e.g.
`amb_pct_change_30/60/90d` are all "Falling balances") so the 3 codes tell
3 distinct stories, not one story 3 times; (2) **rank by signed SHAP sum,
not absolute value**, so a strongly protective factor never crowds out the
actual risk drivers. `top_3_reason_codes` is always computed from the
**90-day model** specifically, to keep one consistent story per customer
even though all 3 horizon scores are reported.

Every engineered feature has a hand-authored entry in
`FEATURE_TO_DRIVER_LABEL` (`src/explainability/config.py`), and
`assert_all_features_mapped` fails loudly if a new feature is ever added
without one — a signal can never silently reach the dashboard unexplained.

---

## 7. PFaR Methodology

```
PFaR = probability_of_deterioration x estimated_balance_at_risk
     = ews_score_90d x (current AMB x expected_balance_decline_pct)
```

**Why 90 days:** matches the project's "30-90 day advance warning" framing
— PFaR sizes the exposure over the full warning window, not a shorter one.

**Why the decline % is empirically calibrated, not read from ground
truth:** `expected_balance_decline_pct` comes from the average
peak-to-trough AMB drop actually observed among customers OUR OWN labeling
framework has historically confirmed as deteriorating (`confirmed_
deterioration` in `deterioration_labels.parquet`) — never from Phase 1's
hidden `ground_truth_cohort`/`deterioration_floor`. Using the planted
answer key to size a production-facing risk metric would be exactly the
kind of leakage a real deployment could never replicate. The resulting
calibration is fairly uniform across segments (~61-62%).

**Driver decomposition.** PFaR is tagged `liquidity` / `relationship` /
`competitor` based on the customer's TOP reason code (e.g. "Falling
balances" → liquidity; "Rising competitor-transfer share" → competitor;
"Lower digital activity" → relationship) — see `DRIVER_TYPE_MAP` in
`src/models/pfar.py`.

**Priority tiers** are assigned by portfolio-relative PFaR rank — top 10%
High, next 20% Medium, rest Low — deliberately tunable, not a business
constant.

**A known, documented gap:** PFaR weights by rupee exposure, so a "top 20
by PFaR" list is dominated by Large Corporate customers even at moderate
risk scores — a 95%-risk SME can rank far lower in absolute PFaR. A
probability-only or within-segment ranking would be the natural
complement; not yet built.

**RM action mapping** is a deliberately simple, swappable rules-based
lookup (`src/models/action_engine.py`) from top reason code to a concrete
action, using the exact business-specified table (extended with
additional coverage for driver labels beyond that table, kept in a
separate dict so the extension is visibly distinct from the specified
rules). The interface (`ActionRecommender.recommend(reason_codes)`) is
shaped so a future **contextual bandit** — learning from real RM-outcome
feedback which action actually works for which customer context — could
implement the same interface without any caller changing. That upgrade is
explicitly NOT built now; there's no outcome data yet to learn from.

---

## 8. Validation Metrics — What Was Used, and Why

| Metric | Used for | Why this metric |
|---|---|---|
| ROC-AUC | All classifiers | Standard ranking quality, but see PR-AUC note below |
| **PR-AUC** | Model selection (core model choice) | At a ~6-15% positive rate, ROC-AUC can look deceptively high while precision in the actionable top slice is mediocre. PR-AUC (and lift) are sensitive to exactly that top-of-list quality. |
| Precision/Recall @ 0.5 | All classifiers | The operational "if we acted on everyone scored ≥ threshold" view — and where LightGBM's miscalibration surfaced (§5.2). |
| Lift-by-decile | All classifiers | "If an RM only has bandwidth for the top decile, how concentrated is the real risk there?" (8.5x at 30d for XGBoost.) |
| Calibration curve | Baseline + tuned models | Checks whether a predicted probability MEANS what it says — distinct from ranking quality, and found to be poor under `class_weight="balanced"` (§5.1). |
| Concordance index (c-index) | Survival model | The time-to-event analogue of AUC — do higher-risk customers get correctly predicted to fail sooner, across all valid time-pairs. |
| `check_assumptions` (PH test) | Survival model selection | The reason Cox was rejected in favor of RSF (§5.3) — a real diagnostic outcome, not a formality. |
| Before/after comparison, same metrics/splits | Graph feature (Phase 8) | Isolates whether a NEW feature earns its place, holding the model and hyperparameters fixed. |
| Customer-level + row-level confusion matrices | Label validation | Two views of the same question at different grains — see §3. |

---

## 9. Known Limitations of Using Synthetic Data

Being direct about these matters more than the numbers above — a real
deployment should expect the following to change:

1. **Performance will very likely drop on real data.** The classification
   models' 0.90+ AUCs are unusually strong because the engineered features
   (trends/momentum on trailing signals) are naturally correlated with how
   the synthetic labels themselves were constructed — the simulator's
   attrition mechanism and the labeling framework's detection logic are
   looking at the same kind of signal by design. Real attrition is driven
   by a messier, more heterogeneous mix of causes the model has never seen
   simulated.
2. **The cohort mechanism is cleaner than reality.** Four discrete cohorts
   with smooth (smoothstep) decay curves is a simplification; real
   deterioration doesn't arrive in four flavors with parametrized ramps.
3. **The competitor-bank set is small and fixed** (4 codes). Real wallet
   leakage spreads across many more institutions with shifting relative
   attractiveness over time.
4. **The graph feature (Phase 8) is scope-limited by the data itself** —
   no counterparty entity IDs were simulated, so only a customer↔bank
   graph was possible, not the richer customer↔customer contagion graph
   that would likely show a real effect.
5. **Missingness is simplistic** (random ~3-5% MCAR on non-critical
   fields). Real missingness is often systematic and itself informative
   (e.g. a customer who stops reporting is often already leaving).
6. **18 months of history is short.** Several features (e.g. the
   seasonality baseline) explicitly compromise on window length because of
   this; a production system with years of history should lengthen them.
7. **5,000 customers is small** for a real HDFC-scale CIB book — segment
   cardinality effects, rare-event tail behavior, and branch-level
   aggregates (e.g. the dashboard's branch heatmap) would look different
   at real scale.
8. **The survival model's covariates are frozen at one early snapshot**
   (month 5) — a real deployment would likely use a time-varying-covariate
   survival model (or repeated landmarking) instead, at the cost of
   complexity this project's scope didn't require.

---

## 10. Path to Production

The pipeline was built, from Phase 1 onward, so that swapping in real
HDFC data touches ONE seam, not every module:

### What changes: the data source boundary

`src/config.py` defines every raw/processed file path as a constant.
`src/data_generation/` is the ONLY place that currently populates
`data/raw/*.parquet`. A real deployment replaces that module with a
connector to core-banking/CRM extracts that writes the SAME schema —
`customers`, `monthly_panel`, `counterparty_transactions` with identical
column names — to the same paths (or a database the loader layer queries
instead of reading parquet). **Nothing downstream — features, labeling,
models, explainability, PFaR, the dashboard — needs to change**, because
none of it imports from `src/data_generation/`; it all reads from the
paths in `src/config.py`.

### What must be recalibrated, not reused blindly

- **`DI_THRESHOLD` (0.85)** and the seasonal-filter thresholds
  (`src/labeling/config.py`) were tuned against this synthetic book's
  cohort mix and would need re-tuning against real deterioration/
  false-positive rates.
- **`expected_balance_decline_pct`** (PFaR) is empirically calibrated —
  this recalibrates itself automatically as real `confirmed_
  deterioration` history accumulates, but will need a cold-start estimate
  (e.g. from expert judgment or a pilot period) before enough real history
  exists.
- **Model hyperparameters** (`core_gbm.py`'s tuned XGBoost params) were
  tuned on this synthetic distribution and should be re-tuned on real
  data, not assumed to transfer.
- **The PH-assumption check and Cox-vs-RSF decision** (§5.3) should be
  re-run on real data — a different data-generating process could easily
  satisfy proportional hazards even if this synthetic one didn't, or vice
  versa.
- **Calibration.** `class_weight="balanced"` makes raw probabilities
  unreliable (§5.1); a production dashboard showing an actual probability
  number to an RM should add `CalibratedClassifierCV` (Platt/isotonic) on
  top of the chosen model first.

### What becomes possible for the first time

- **The full wallet-leakage graph.** Real counterparty account/entity
  identifiers would enable the customer↔customer graph Phase 8's name
  evoked but the data couldn't support — shared-vendor or shared-payroll-
  processor contagion between customers is a plausible real signal this
  version structurally cannot see.
- **The contextual-bandit action recommender.** Once real RM outcomes
  exist (did the recommended action correlate with retained balance/
  activity?), `ActionRecommender`'s interface (§7) is ready for a bandit
  implementation to slot in without touching any call site.
- **Model monitoring and drift detection** — feature distributions,
  calibration, and PR-AUC should be tracked over time in a way that isn't
  meaningful to build against a fixed synthetic snapshot.

### Governance and compliance (out of this project's scope, but real)

A production RM-facing risk score for a live corporate banking book would
need: model risk management sign-off, fair-treatment review of the
reason-code/action mapping, data privacy controls far stricter than
synthetic data requires, and an audit trail from score → reason code →
recommended action → RM outcome. None of this changes the technical
pipeline described above, but all of it gates whether that pipeline is
allowed to reach a real RM's screen.
