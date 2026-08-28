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
from src.features.config import TRAIN_MAX_MONTH_IDX
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


def prepare_xy_tree(df: pd.DataFrame, horizon_col: str, val_start_month_idx: int = 10) -> dict:
    """
    Same idea as prepare_xy, but for tree models (XGBoost/LightGBM) that
    natively handle missing values and categorical dtypes — so, unlike the
    logistic regression baseline, NO imputation, scaling, or one-hot
    encoding happens here. `segment` is cast to pandas 'category' dtype and
    passed through as-is.

    Also carves out a VALIDATION slice from the tail of the train period
    (month_idx >= val_start_month_idx, still within TRAIN_MAX_MONTH_IDX) for
    early stopping. This is still a time-based split — the validation
    months are chronologically after the sub-train months and before the
    held-out test months — so early stopping never peeks at genuinely
    future data either.
    """
    numeric_cols, categorical_cols = get_feature_columns()
    feature_cols = numeric_cols + categorical_cols

    valid = df[df[horizon_col].notna()].copy()
    valid[horizon_col] = valid[horizon_col].astype(int)
    for col in categorical_cols:
        valid[col] = valid[col].astype("category")

    train_full = valid[valid["split"] == "train"]
    train_sub = train_full[train_full["month_idx"] < val_start_month_idx]
    val = train_full[
        (train_full["month_idx"] >= val_start_month_idx) & (train_full["month_idx"] <= TRAIN_MAX_MONTH_IDX)
    ]
    test = valid[valid["split"] == "test"]

    return {
        "X_train": train_sub[feature_cols],
        "y_train": train_sub[horizon_col],
        "X_val": val[feature_cols],
        "y_val": val[horizon_col],
        "X_test": test[feature_cols],
        "y_test": test[horizon_col],
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "n_dropped_censored": len(df) - len(valid),
    }
