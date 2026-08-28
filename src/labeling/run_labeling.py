"""
End-to-end labeling pipeline:
  1. Load synthetic panel + customer master + ground truth (Phase 1 output).
  2. Compute the composite Deterioration Index and its components.
  3. Grid-search the DI threshold against the ground truth to pick a value
     that keeps the seasonal_false_positive flag-rate low without gutting
     recall on the real deterioration cohorts.
  4. Apply the seasonal false-positive filter and build the three forward
     labels (deteriorates_in_30d / 60d / 90d).
  5. Validate against ground truth (customer-level + row-level) and print a
     report.
  6. Save the final label table to data/processed/deterioration_labels.parquet.

Run with: python -m src.labeling.run_labeling
"""

from __future__ import annotations

import pandas as pd

from src.config import CUSTOMERS_FILE, MONTHLY_PANEL_FILE, GROUND_TRUTH_FILE, DATA_PROCESSED_DIR
from src.labeling.deterioration_index import build_component_scores, compute_composite_index
from src.labeling.seasonal_filter import flag_confirmed_deterioration
from src.labeling.labels import build_forward_labels
from src.labeling import config as label_cfg
from src.labeling.validate import (
    customer_level_confusion,
    precision_recall_from_confusion,
    row_level_confusion,
    lead_time_report,
    grid_search_threshold,
)

LABELS_FILE = DATA_PROCESSED_DIR / "deterioration_labels.parquet"


def main() -> None:
    print("Loading Phase 1 synthetic data...")
    customers_df = pd.read_parquet(CUSTOMERS_FILE)
    panel_df = pd.read_parquet(MONTHLY_PANEL_FILE)
    ground_truth_df = pd.read_parquet(GROUND_TRUTH_FILE)

    print("Computing deterioration index components...")
    component_df = build_component_scores(panel_df, customers_df)
    component_df = compute_composite_index(component_df)

    print("\n=== Threshold grid search (customer-level, positive = gradual+sudden) ===")
    candidate_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    grid = grid_search_threshold(component_df, ground_truth_df, candidate_thresholds)
    print(grid.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    threshold = label_cfg.DI_THRESHOLD
    print(f"\nUsing DI_THRESHOLD = {threshold} (see src/labeling/config.py + docs/deterioration_definition.md)")

    print("\nApplying seasonal false-positive filter + building forward labels...")
    flagged_df = flag_confirmed_deterioration(component_df, threshold=threshold)
    labels_df = build_forward_labels(flagged_df)

    print("\n=== Customer-level validation: true cohort vs. ever flagged ===")
    table, merged = customer_level_confusion(labels_df, ground_truth_df)
    print(table)
    metrics = precision_recall_from_confusion(merged)
    print(f"\nPrecision (positive=gradual+sudden): {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"FPR on stable+seasonal: {metrics['fpr']:.3f}")

    print("\n=== Row-level validation per horizon (rigorous check) ===")
    for label_col, h in label_cfg.HORIZONS_MONTHS.items():
        result = row_level_confusion(labels_df, ground_truth_df, label_col, h)
        print(
            f"{label_col:22s} n={result['n_rows']:>6d}  "
            f"precision={result['precision']:.3f}  recall={result['recall']:.3f}  fpr={result['fpr']:.3f}"
        )

    print("\n=== Lead-time report (months after true onset before first confirmed) ===")
    lead = lead_time_report(labels_df, ground_truth_df)
    summary = lead.groupby("ground_truth_cohort").agg(
        n=("customer_id", "count"),
        detected_rate=("detected", "mean"),
        median_lag_months=("lag_months", "median"),
        mean_lag_months=("lag_months", "mean"),
    )
    print(summary.round(2))

    # Persist only the columns a downstream feature/model step needs, plus
    # the diagnostic columns useful for debugging — never ground_truth_cohort.
    keep_cols = [
        "customer_id", "month", "month_idx", "segment",
        "amb_decline_score", "txn_decline_score", "digital_decline_score", "payroll_trade_decline_score",
        "deterioration_index", "breach", "confirmed_deterioration",
        *label_cfg.HORIZONS_MONTHS.keys(),
    ]
    labels_df[keep_cols].to_parquet(LABELS_FILE, index=False)
    print(f"\nSaved labels to {LABELS_FILE} ({len(labels_df):,} rows)")


if __name__ == "__main__":
    main()
