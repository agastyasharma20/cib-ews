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

### 7. Train the core model

*(To be added — Phase 4.)*

### 8. Explainability (reason codes)

*(To be added — Phase 5.)*

### 9. Survival analysis (time-to-erosion)

*(To be added — Phase 6.)*

### 10. Graph features (linked-entity risk)

*(To be added — Phase 7.)*

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
