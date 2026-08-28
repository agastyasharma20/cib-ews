"""
Phase 5 (part 2) — produces the customer-level scored table:
customer_id, month, ews_score_30d, ews_score_60d, ews_score_90d,
top_3_reason_codes.

Scores EVERY customer-month in the dataset (not just the test split) —
unlike model evaluation, a live EWS table is meant to cover the whole
portfolio, including rows with a censored (NaN) label, since scoring never
actually needs to know the future outcome, only the trailing features.

Reason codes are computed from the 90-day core model's SHAP values (see
src/explainability/config.py:CORE_REASON_HORIZON) and apply to every row —
the 30d/60d scores are still separate model outputs, just not the ones
driving the explanation text, to keep one clear, consistent story per
customer rather than 3 potentially-conflicting explanations.

Run with: python -m src.explainability.score_customers
(requires results/models/core_model_{30,60,90}d.joblib from
 src.models.core_gbm to already exist)
"""

from __future__ import annotations

import joblib
import pandas as pd

from src.config import DATA_PROCESSED_DIR, PROJECT_ROOT
from src.features.config import all_feature_columns
from src.models.config import CATEGORICAL_FEATURES, TARGET_HORIZONS
from src.models.data_prep import load_model_dataset
from src.explainability.shap_explain import compute_shap_values, shap_values_to_frame
from src.explainability.reason_codes import build_reason_codes_table
from src.explainability.config import CORE_REASON_HORIZON

MODELS_DIR = PROJECT_ROOT / "results" / "models"
SCORED_TABLE_FILE = DATA_PROCESSED_DIR / "customer_scores.parquet"


def load_core_models() -> dict:
    models = {}
    for horizon in TARGET_HORIZONS:
        path = MODELS_DIR / f"core_model_{horizon}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run `python -m src.models.core_gbm` first.")
        models[horizon] = joblib.load(path)
    return models


def main() -> None:
    print("Loading model dataset and core models...")
    df = load_model_dataset()
    models = load_core_models()

    numeric_cols = all_feature_columns()
    feature_cols = numeric_cols + CATEGORICAL_FEATURES

    X = df[feature_cols].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    scores = df[["customer_id", "month", "month_idx", "segment"]].copy()

    print("Scoring every customer-month for all 3 horizons...")
    for horizon in TARGET_HORIZONS:
        proba = models[horizon].predict_proba(X)[:, 1]
        score_col = "ews_score_" + horizon.replace("deteriorates_in_", "")
        scores[score_col] = proba

    print(f"Computing SHAP values from the {CORE_REASON_HORIZON} core model (for reason codes)...")
    reason_model = models[CORE_REASON_HORIZON]
    shap_values = compute_shap_values(reason_model, X)
    shap_df = shap_values_to_frame(shap_values, feature_cols)

    print("Building top-3 reason codes per customer-month...")
    scores["top_3_reason_codes"] = build_reason_codes_table(shap_df).values

    scores.to_parquet(SCORED_TABLE_FILE, index=False)
    print(f"\nSaved {SCORED_TABLE_FILE} ({len(scores):,} rows)")
    print(scores.head())

    # --- 3 example customers, high risk, for a sanity-check readout ---
    print(f"\n{'=' * 70}\n3 example high-risk customers (ranked by ews_score_90d)\n{'=' * 70}")
    examples = scores.sort_values("ews_score_90d", ascending=False).drop_duplicates("customer_id").head(3)
    for _, row in examples.iterrows():
        print(f"\ncustomer_id={row['customer_id']}  month={row['month']}  segment={row['segment']}")
        print(f"  ews_score_30d={row['ews_score_30d']:.3f}  ews_score_60d={row['ews_score_60d']:.3f}  ews_score_90d={row['ews_score_90d']:.3f}")
        print(f"  top_3_reason_codes: {row['top_3_reason_codes']}")


if __name__ == "__main__":
    main()
