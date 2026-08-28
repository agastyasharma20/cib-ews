"""
Trains and evaluates the logistic regression baseline for all 3 label
horizons, saves each trained pipeline + evaluation plot, and prints a
summary table.

Run with: python -m src.models.train_baseline
"""

from __future__ import annotations

import joblib
import pandas as pd

from src.config import PROJECT_ROOT
from src.models.config import TARGET_HORIZONS
from src.models.data_prep import load_model_dataset, prepare_xy
from src.models.baseline_model import build_baseline_pipeline
from src.models.evaluate import compute_metrics, lift_table, plot_evaluation

MODELS_DIR = PROJECT_ROOT / "models_store"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = load_model_dataset()
    summary_rows = []

    for horizon_col in TARGET_HORIZONS:
        print(f"\n{'=' * 70}\n{horizon_col}\n{'=' * 70}")
        data = prepare_xy(df, horizon_col)
        print(f"Train: {len(data['X_train']):,} rows (base rate {data['y_train'].mean():.3f})")
        print(f"Test:  {len(data['X_test']):,} rows (base rate {data['y_test'].mean():.3f})")
        print(f"Dropped (censored / missing label): {data['n_dropped_censored']:,} rows total in dataset")

        pipeline = build_baseline_pipeline(data["numeric_cols"], data["categorical_cols"])
        pipeline.fit(data["X_train"], data["y_train"])

        y_proba_test = pipeline.predict_proba(data["X_test"])[:, 1]
        metrics = compute_metrics(data["y_test"].to_numpy(), y_proba_test)
        print(f"Test ROC-AUC: {metrics['roc_auc']:.3f}   Test PR-AUC: {metrics['pr_auc']:.3f}")

        lift = lift_table(data["y_test"].to_numpy(), y_proba_test)
        print("\nLift table (test set):")
        print(lift.to_string(index=False))

        plot_path = FIGURES_DIR / f"baseline_logreg_{horizon_col}.png"
        plot_evaluation(data["y_test"].to_numpy(), y_proba_test, f"Baseline Logistic Regression — {horizon_col}", plot_path)
        print(f"Saved evaluation plot to {plot_path}")

        model_path = MODELS_DIR / f"baseline_logreg_{horizon_col}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"Saved model to {model_path}")

        summary_rows.append({"horizon": horizon_col, **metrics, "top_decile_lift": lift.iloc[0]["lift"]})

    print(f"\n{'=' * 70}\nSummary across horizons\n{'=' * 70}")
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
