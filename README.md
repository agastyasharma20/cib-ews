# CIB Early Warning System (EWS)

## Business Problem

HDFC Bank's Current Account / CIB ("Customers in Base") portfolio faces a
quieter risk than account closure: large corporate customers keep their
HDFC current account technically open while gradually shifting balances,
transaction flow, payroll processing, and trade activity to competing
banks. Today, relationship managers act reactively — only after the
balance erosion is already visible in monthly reports, by which point most
of the wallet share is already gone. This project builds an Early Warning
System that scores each CIB customer for **30–90 day advance risk of
silent deterioration**, explains *why* via reason codes derived from their
transaction behavior, and recommends the specific RM action to take —
turning a lagging, reactive process into a forward-looking one. Since real
HDFC data isn't available, the entire pipeline runs on a realistic
**synthetic** CA/CIB dataset (with known ground-truth deterioration
cohorts for validation), built so a real core-banking data source could be
substituted later without reworking the feature, model, or dashboard code.

## Project Structure

```
cib-ews/
├── data/
│   ├── raw/            # synthetic raw data (generated, not hand-edited)
│   └── processed/      # engineered feature tables
├── src/
│   ├── config.py            # shared paths, random seed
│   ├── data_generation/     # synthetic CA/CIB data generator
│   ├── features/            # feature engineering
│   ├── labeling/            # forward-looking deterioration label construction
│   ├── models/               # baseline + core (XGBoost/LightGBM) models
│   ├── explainability/        # SHAP reason codes
│   ├── graph/                 # networkx wallet-leakage / linked-entity features
│   └── dashboard/              # shared logic used by the Streamlit app
├── notebooks/           # exploratory analysis
├── app/                  # Streamlit dashboard entry point
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

## How to Run

*(This section is filled in incrementally as each phase is built.)*

### 1. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate the synthetic dataset

```bash
python -m src.data_generation.generate_synthetic_data
```

*(Command will be finalized once the generator is built — Phase 1.)*

### 3. Build the deterioration labels

```bash
python -m src.labeling.run_labeling
```

Reads `data/raw/{customers,monthly_panel,ground_truth_cohorts}.parquet`,
computes the composite Deterioration Index, applies the seasonal
false-positive filter, builds `deteriorates_in_{30,60,90}d`, validates
against the known ground-truth cohorts, and saves
`data/processed/deterioration_labels.parquet`. See
[`docs/deterioration_definition.md`](docs/deterioration_definition.md) for
what the index means and how the threshold was chosen.

### 4. Build features

```bash
python -m src.features.build_features
```

Builds all 5 feature groups (Balance & Liquidity, Transaction & Digital
Activity, Product & Wallet-Share, Network & Counterparty, Relationship &
Engagement), merges in the Phase 2 labels, applies a time-based train/test
split, and saves `data/processed/model_dataset.parquet`. See
[`docs/feature_dictionary.md`](docs/feature_dictionary.md) for every
feature's definition and business rationale.

### 5. Train the baseline model

```bash
python -m src.models.baseline_logistic
```

Trains the explainable benchmark — logistic regression (median imputation +
standardization + one-hot encoding, `class_weight="balanced"`) — for each
of the 3 label horizons, on a **time-based** train/test split (train on
months 0-11, test on months 12-17, never random — this is a forecasting
problem). Reports ROC-AUC, PR-AUC, precision/recall, and lift-by-decile;
plots calibration and the top risk-increasing/decreasing coefficients for
interpretability. Saves everything to `results/{models,metrics,figures}/`.

### 6. Train the core model (XGBoost / LightGBM, lightly tuned)

```bash
python -m src.models.core_gbm
```

Runs a small randomized hyperparameter search (8 draws per model per
horizon, scored on a time-based validation slice — never K-fold CV, which
would reintroduce random-split leakage) for both XGBoost and LightGBM,
evaluates the tuned models on the held-out test set with the same metrics
as the baseline (ROC-AUC, PR-AUC, precision/recall, calibration), and picks
**one algorithm as "the core model"** by average test PR-AUC across all 3
horizons. Requires `results/metrics/baseline_logreg_summary.csv` to exist
first (run step 5 above).

**Result: XGBoost chosen as the core model** (mean PR-AUC 0.928 vs
LightGBM's 0.895 across horizons) — the winning models are saved as
`results/models/core_model_{30,60,90}d.joblib`:

| Horizon | Model | ROC-AUC | PR-AUC | Precision@0.5 | Recall@0.5 | Top-decile lift |
|---|---|---|---|---|---|---|
| 90d | logistic_regression | 0.935 | 0.841 | 0.429 | 0.910 | 5.85 |
| 90d | lightgbm (tuned) | 0.951 | 0.854 | 0.000 | 0.000 | 6.16 |
| 90d | xgboost (tuned) | 0.958 | 0.903 | 0.544 | 0.925 | 6.29 |

(See `results/metrics/core_gbm_comparison.csv` for all 3 horizons.)

Worth noting: LightGBM's PR-AUC/lift are close to XGBoost's, but at the
default 0.5 threshold its precision/recall collapse to 0 — its predicted
probabilities skew low even with `scale_pos_weight`. This is exactly why
the core-model choice is made on PR-AUC (threshold-free) rather than a
fixed cutoff — see `src/models/core_gbm.py` for the reasoning.

### 7. Explainability — SHAP reason codes

```bash
python -m src.explainability.score_customers
```

Computes SHAP values from the core XGBoost model (90-day horizon) for
every customer-month, maps the top contributing features to plain-language
driver labels (`src/explainability/config.py:FEATURE_TO_DRIVER_LABEL` —
e.g. "Falling balances", "Rising competitor-transfer share", "Reduced
payroll activity"), and saves a customer-level scored table —
`customer_id, month, ews_score_30d/60d/90d, top_3_reason_codes` — to
`data/processed/customer_scores.parquet`. Requires the core models from
step 6 above. See [`docs/explainability.md`](docs/explainability.md) for
the full reasoning behind the driver-label mapping and how reason codes
are ranked.

### 8. RM action mapping + PFaR risk segmentation

```bash
python -m src.models.pfar
```

`src/models/action_engine.py` maps a customer's top reason code to a
concrete recommended action via a swappable rules-based lookup (the exact
table is business-specified — see the module for the extended coverage of
labels beyond that table, and the documented interface a future
contextual-bandit recommender could implement in its place once real
RM-outcome feedback exists).

`src/models/pfar.py` computes **PFaR** (Probability-weighted Funds at
Risk) = `ews_score_90d x estimated_balance_at_risk`, where the expected
balance-decline % is calibrated empirically from customers our OWN
labeling framework has historically confirmed as deteriorating (never from
Phase 1's hidden ground truth — that would be leakage a real deployment
could never replicate). Decomposes PFaR into a driver type
(liquidity/relationship/competitor) from the top reason code, assigns
High/Medium/Low priority tiers by portfolio-relative PFaR rank, and saves
`customer_id, month, segment, PFaR, PFaR_driver_type, priority_tier,
recommended_action` to `data/processed/pfar_risk_segmentation.parquet`
(one row per customer — their latest scored month). Requires steps 6-7
above.

**Note:** PFaR intentionally weights by rupee exposure, so a "top 20 by
PFaR" table is dominated by Large Corporate customers even at moderate
risk scores — a very-high-probability SME can rank far lower in absolute
PFaR. A probability-only or within-segment view is a natural complement
for surfacing those cases, not yet built.

### 9. Survival analysis (time-to-erosion)

```bash
python -m src.models.survival
```

Frames "will this customer deteriorate" as "how long until this customer
deteriorates" — the event is the labeling framework's own
`confirmed_deterioration` flag (never the hidden ground truth), snapshotted
at each customer's earliest fully-defined month (month_idx=5). A
CoxPHFitter is fit first, as the standard baseline, and explicitly checked
via `check_assumptions` — it flagged **11 of 24 covariates (46%)** as
violating the proportional-hazards assumption, including nearly every
balance-trend feature. That's a pervasive violation, not a couple of
borderline cases, so the model actually used is a **Random Survival
Forest** (scikit-survival) instead, which makes no such assumption. The
train/test split is customer-level (not row-level time-based — see the
module docstring for why that doesn't translate to one-row-per-customer
survival data), but the same "never train on what wouldn't be known yet"
discipline is reproduced by administratively censoring the training fit at
month 11 and evaluating concordance against full follow-up through month
17 on held-out customers, before refitting a deployment model on
everyone's full follow-up for the actual scored table.

**Result: concordance index = 0.659** (Cox's, for comparison, was 0.591 —
also worse, consistent with the assumption violation). Modest compared to
the classification models' 0.90+ AUC — expected, since this model gets
only ONE early snapshot per customer rather than fresh trailing features
every month. Saves `customer_id, segment, risk_score,
predicted_median_days_to_deterioration` to
`data/processed/survival_scores.parquet` (joinable with
`customer_scores.parquet` by `customer_id`), plus 3 example survival
curves in `results/figures/survival_example_curves.png`.

### 10. Graph features (wallet-leakage, stretch)

```bash
python -m src.graph.wallet_leakage
python -m src.graph.reevaluate_core_model
```

**Scoping note:** the synthetic data only records each transaction's
*counterparty bank*, not a counterparty entity identifier, so the graph
built here is a customer↔bank bipartite graph (~5,000 customer nodes, 11
bank nodes) rather than the richer customer↔customer graph the phase name
evokes — that would need real counterparty account IDs. `wallet_leakage.py`
builds one bipartite graph per month and computes, per customer: a
value-weighted competitor-bank share and its 3m/6m trend (both read
directly off graph edge weights), plus two genuinely graph-native
measures — bank-diversity degree, and a value-weighted exposure to bank
eigenvector centrality computed across the *whole portfolio's* graph each
month (captures whether a customer is disproportionately connected to a
bank that's gaining share portfolio-wide, not just relative to their own
history).

`reevaluate_core_model.py` refits the tuned XGBoost core model (Phase 5's
already-chosen hyperparameters, held fixed, so only the feature set
changes) with and without the 4 graph features, on identical splits.

**Result: the graph features did NOT meaningfully improve the model.**
ROC-AUC/PR-AUC are flat to very slightly worse at every horizon (e.g. 90d
PR-AUC: 0.9027 → 0.9019), and they rank 14th-25th of 27 features by mean
|SHAP| — nowhere near top drivers. Traced the cause: the value-weighted
graph features correlate **0.88-0.91** with Phase 2's existing
`competitor_txn_share`/`_trend` (pandas-derived, from the same underlying
transactions). The graph computation is methodologically distinct
(bipartite eigenvector centrality vs. direct aggregation), but on data
this size it mostly re-derives information XGBoost already had. The one
partial validation: mean |SHAP| for the competitor-share graph feature
*is* higher for `sudden_deterioration` (0.133) and `gradual_deterioration`
(0.096) than `stable` (0.040) — the right directional signal, just not an
incremental one. Full numbers in
`results/metrics/graph_feature_before_after.csv` and
`graph_feature_shap_check.txt`. Kept in the codebase as an honestly-negative
result and the natural extension point once real counterparty-entity data
exists (the customer↔customer graph would likely show a real effect via
shared-counterparty contagion, which this bank-level version cannot see).

### 11. Run the dashboard

```bash
streamlit run app/main.py
```

*(To be added — Phase 9.)*

## Project Phases

1. Synthetic data generation (customer master + monthly panel + counterparty
   transactions + ground-truth cohorts)
2. Feature engineering + forward-looking label construction
3. Baseline model (logistic regression) + evaluation framework
4. Core model (XGBoost / LightGBM) tuned and compared to baseline
5. Explainability (SHAP) → reason codes per customer
6. Survival analysis (lifelines) → time-to-erosion estimate
7. Graph features (networkx) → correlated risk across linked entities /
   wallet leakage to competitor banks
8. RM action recommendation engine
9. Streamlit dashboard
10. Tests, documentation, polish
