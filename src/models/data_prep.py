"""
Loads the Phase 2 model dataset and prepares (X, y) splits for a given
label horizon.

WHY drop NaN-label rows here rather than upstream: `model_dataset.parquet`
deliberately keeps NaN labels for customer-months too close to the end of
their history to know the true forward outcome (see
docs/deterioration_definition.md). Silently treating those as 0 would teach
the model "no evidence yet" looks the same as "confirmed not deteriorating"
— this function makes dropping them an explicit, visible step instead.
"""

from __future__ import annotations

import pandas as pd

from src.config import DATA_PROCESSED_DIR
from src.features.config import all_feature_columns
from src.models.config import CATEGORICAL_FEATURES

MODEL_DATASET_FILE = DATA_PROCESSED_DIR / "model_dataset.parquet"


def load_model_dataset() -> pd.DataFrame:
    return pd.read_parquet(MODEL_DATASET_FILE)


def get_feature_columns() -> tuple[list[str], list[str]]:
    """Returns (numeric_feature_cols, categorical_feature_cols)."""
    numeric_cols = all_feature_columns()
    return numeric_cols, CATEGORICAL_FEATURES


def prepare_xy(df: pd.DataFrame, horizon_col: str) -> dict:
    """
    Returns a dict with X_train, y_train, X_test, y_test for the given
    label horizon, using the pre-computed time-based `split` column and
    dropping rows where the label is NaN (censored — not enough forward
    history yet to know the answer).
    """
    numeric_cols, categorical_cols = get_feature_columns()
    feature_cols = numeric_cols + categorical_cols

    valid = df[df[horizon_col].notna()].copy()
    valid[horizon_col] = valid[horizon_col].astype(int)

    train = valid[valid["split"] == "train"]
    test = valid[valid["split"] == "test"]

    return {
        "X_train": train[feature_cols],
        "y_train": train[horizon_col],
        "X_test": test[feature_cols],
        "y_test": test[horizon_col],
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "n_dropped_censored": len(df) - len(valid),
    }
