"""
Orchestrates all 5 feature groups into one customer-month feature table,
merges in the labels from Phase 2 labeling, and writes the model-ready
dataset.

Run with: python -m src.features.build_features
"""

from __future__ import annotations

import pandas as pd

from src.config import CUSTOMERS_FILE, MONTHLY_PANEL_FILE, COUNTERPARTY_TXNS_FILE, DATA_PROCESSED_DIR
from src.features.utils import add_month_index
from src.features import balance_liquidity, transaction_digital, product_wallet_share, network_counterparty, relationship_engagement
from src.features.config import TRAIN_MAX_MONTH_IDX, all_feature_columns, FEATURE_GROUPS
from src.labeling.config import HORIZONS_MONTHS

LABELS_FILE = DATA_PROCESSED_DIR / "deterioration_labels.parquet"
MODEL_DATASET_FILE = DATA_PROCESSED_DIR / "model_dataset.parquet"


def build_feature_table(panel_df: pd.DataFrame, customers_df: pd.DataFrame, counterparty_df: pd.DataFrame) -> pd.DataFrame:
    """Runs all 5 feature groups in sequence and returns one wide table:
    customer_id, month, month_idx, segment, + every registered feature
    column. Order matters only where one group's output feeds another
    (product_wallet_share's has_payroll_book/has_trade_finance are inputs
    to relationship_engagement's product_holding_breadth)."""
    df = add_month_index(panel_df)
    df = df.merge(customers_df[["customer_id", "segment"]], on="customer_id", how="left")

    df = balance_liquidity.build(df)
    df = transaction_digital.build(df)
    df = product_wallet_share.build(df)
    df = network_counterparty.build(df, counterparty_df)
    df = relationship_engagement.build(df, customers_df)

    return df


def main() -> None:
    print("Loading Phase 1 synthetic data...")
    customers_df = pd.read_parquet(CUSTOMERS_FILE)
    panel_df = pd.read_parquet(MONTHLY_PANEL_FILE)
    counterparty_df = pd.read_parquet(COUNTERPARTY_TXNS_FILE)

    print("Building features across 5 groups...")
    feature_df = build_feature_table(panel_df, customers_df, counterparty_df)

    print("\nFeature groups registered:")
    for group, cols in FEATURE_GROUPS.items():
        print(f"  {group}: {len(cols)} features -> {cols}")

    print("\nMerging Phase 2 labels (deteriorates_in_30/60/90d)...")
    labels_df = pd.read_parquet(LABELS_FILE)
    label_cols = ["customer_id", "month", *HORIZONS_MONTHS.keys()]
    model_df = feature_df.merge(labels_df[label_cols], on=["customer_id", "month"], how="left")

    print("Applying time-based train/test split...")
    model_df["split"] = model_df["month_idx"].apply(lambda m: "train" if m <= TRAIN_MAX_MONTH_IDX else "test")

    keep_cols = [
        "customer_id", "month", "month_idx", "segment", "split",
        *all_feature_columns(),
        *HORIZONS_MONTHS.keys(),
    ]
    model_df = model_df[keep_cols]
    model_df.to_parquet(MODEL_DATASET_FILE, index=False)

    print(f"\nSaved {MODEL_DATASET_FILE} ({len(model_df):,} rows, {len(all_feature_columns())} features)")
    print(model_df["split"].value_counts())
    print("\nLabel availability (non-null) by split:")
    for h in HORIZONS_MONTHS:
        print(f"  {h}: train={model_df.loc[model_df['split']=='train', h].notna().sum():,}  "
              f"test={model_df.loc[model_df['split']=='test', h].notna().sum():,}")

    print("\nMissing-value rate per feature (train split):")
    train_missing = model_df.loc[model_df["split"] == "train", all_feature_columns()].isna().mean().round(3)
    print(train_missing.sort_values(ascending=False))


if __name__ == "__main__":
    main()
