"""
Baseline Logistic Regression — the explainable benchmark.

Purpose: before reaching for XGBoost/LightGBM (Phase 4), establish the
simplest model that could plausibly work, evaluated rigorously, and fully
interpretable out of the box (a logistic regression's coefficients ARE its
explanation — no SHAP required). Any later model has to beat this by enough
to justify its extra complexity and reduced interpretability.

Key design choices (the "why" an interviewer will ask about):

1. TIME-based train/test split, not random.
   This is a forecasting problem: at deployment time, the model only ever
   sees the past to predict the future. A random split would let the model
   train on a customer's month-14 data and get evaluated on that same
   customer's month-6 — leaking information a real deployment would never
   have. Splitting by month_idx (see src/features/config.py:
   TRAIN_MAX_MONTH_IDX) mirrors how the model is actually used.

2. class_weight="balanced" instead of resampling.
   The positive rate is ~6-15% depending on horizon. Balanced class
   weighting reweights the loss function inversely to class frequency
   without discarding rows (undersampling) or fabricating synthetic ones
   (SMOTE) — simplest correct option for a benchmark model.

3. Standardization + median imputation.
   Logistic regression's coefficients (and its L2 penalty) are scale-
   sensitive, and it can't handle NaN at all — both handled by the shared
   pipeline in baseline_model.py.

4. One model PER horizon (30d/60d/90d), not one multi-output model.
   Each horizon has a different, valid label (see docs/deterioration_
   definition.md) built from a different forward window — training three
   independent binary classifiers keeps each one's calibration and
   coefficients honestly tied to its own horizon.

Run with: python -m src.models.baseline_logistic
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import PROJECT_ROOT
from src.models.config import TARGET_HORIZONS
from src.models.data_prep import load_model_dataset, prepare_xy
from src.models.baseline_model import build_baseline_pipeline
from src.models.evaluate import compute_metrics, lift_table, plot_evaluation, plot_calibration, evaluate_at_threshold

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
for _d in (MODELS_DIR, METRICS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def get_feature_names(pipeline) -> list[str]:
    """Human-readable names for every column the model actually sees,
    after the ColumnTransformer's imputation/scaling/one-hot encoding."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


def plot_top_coefficients(pipeline, feature_names: list[str], title: str, save_path, top_n: int = 10) -> None:
    """
    Logistic regression's whole appeal as a benchmark: the model IS its own
    explanation. Positive coefficients push risk up, negative pull it down
    — plotted together so it's immediately clear which signals the model
    actually leaned on, which doubles as a sanity check (do the top
    features match the domain intuition baked into the labeling framework?
    e.g. competitor_txn_share and amb decline should show up here).
    """
    coefs = pipeline.named_steps["model"].coef_[0]
    coef_series = pd.Series(coefs, index=feature_names).sort_values()

    top_negative = coef_series.head(top_n)   # most protective (pushes risk down)
    top_positive = coef_series.tail(top_n)   # most risk-increasing
    combined = pd.concat([top_negative, top_positive])

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in combined.values]
    ax.barh(combined.index, combined.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic regression coefficient (standardized features)")
    ax.set_title(f"Top {top_n} risk-increasing (red) and risk-decreasing (blue) features — {title}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return top_positive.sort_values(ascending=False), top_negative


def run_for_horizon(df: pd.DataFrame, horizon_col: str) -> dict:
    print(f"\n{'=' * 70}\n{horizon_col}\n{'=' * 70}")
    data = prepare_xy(df, horizon_col)
    print(f"Train: {len(data['X_train']):,} rows (month_idx 0-11, base rate {data['y_train'].mean():.3f})")
    print(f"Test:  {len(data['X_test']):,} rows (month_idx 12-17, base rate {data['y_test'].mean():.3f})")

    pipeline = build_baseline_pipeline(data["numeric_cols"], data["categorical_cols"])
    pipeline.fit(data["X_train"], data["y_train"])

    y_test = data["y_test"].to_numpy()
    y_proba = pipeline.predict_proba(data["X_test"])[:, 1]

    # --- 1. Ranking metrics (AUC-ROC, PR-AUC, lift) ---
    metrics = compute_metrics(y_test, y_proba)
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}   PR-AUC: {metrics['pr_auc']:.3f}")

    # --- 2. Precision/recall at a fixed operating threshold ---
    threshold_metrics = evaluate_at_threshold(y_test, y_proba, threshold=0.5)
    print(f"At threshold=0.5 -> precision={threshold_metrics['precision']:.3f}  "
          f"recall={threshold_metrics['recall']:.3f}  f1={threshold_metrics['f1']:.3f}")

    lift = lift_table(y_test, y_proba)
    print("\nLift table (test set):")
    print(lift.to_string(index=False))

    # --- 3. Calibration ---
    calib_path = FIGURES_DIR / f"baseline_logreg_{horizon_col}_calibration.png"
    plot_calibration(y_test, y_proba, horizon_col, calib_path, model_label="Logistic regression")
    print(f"Saved calibration plot -> {calib_path}")

    # --- 4. ROC / PR / lift plot (reused from Phase 3 evaluation framework) ---
    roc_path = FIGURES_DIR / f"baseline_logreg_{horizon_col}_roc_pr_lift.png"
    plot_evaluation(y_test, y_proba, f"Baseline Logistic Regression — {horizon_col}", roc_path)

    # --- 5. Interpretability: top coefficients ---
    feature_names = get_feature_names(pipeline)
    coef_path = FIGURES_DIR / f"baseline_logreg_{horizon_col}_coefficients.png"
    top_positive, top_negative = plot_top_coefficients(pipeline, feature_names, horizon_col, coef_path)
    print(f"\nTop risk-INCREASING features:\n{top_positive.round(3).to_string()}")
    print(f"\nTop risk-DECREASING (protective) features:\n{top_negative.round(3).to_string()}")

    # --- 6. Save model + metrics ---
    model_path = MODELS_DIR / f"baseline_logreg_{horizon_col}.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved model -> {model_path}")

    result = {
        "horizon": horizon_col,
        "n_train": len(data["X_train"]),
        "n_test": len(data["X_test"]),
        "train_base_rate": float(data["y_train"].mean()),
        "test_base_rate": float(data["y_test"].mean()),
        **metrics,
        **threshold_metrics,
        "top_decile_lift": float(lift.iloc[0]["lift"]),
        "top_risk_increasing_features": top_positive.round(4).to_dict(),
        "top_risk_decreasing_features": top_negative.round(4).to_dict(),
    }
    return result


def main() -> None:
    df = load_model_dataset()
    all_results = []

    for horizon_col in TARGET_HORIZONS:
        result = run_for_horizon(df, horizon_col)
        all_results.append(result)

    print(f"\n{'=' * 70}\nSummary across horizons\n{'=' * 70}")
    summary = pd.DataFrame(
        [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in all_results]
    )
    print(summary.to_string(index=False))

    summary.to_csv(METRICS_DIR / "baseline_logreg_summary.csv", index=False)
    with open(METRICS_DIR / "baseline_logreg_full_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved metrics -> {METRICS_DIR / 'baseline_logreg_summary.csv'}")
    print(f"Saved full results (incl. coefficients) -> {METRICS_DIR / 'baseline_logreg_full_results.json'}")


if __name__ == "__main__":
    main()
