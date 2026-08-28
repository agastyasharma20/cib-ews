"""
Baseline model: logistic regression on top of a standard preprocessing
pipeline. This exists to answer "does the feature set carry signal at all,
with the simplest possible model" before reaching for XGBoost/LightGBM in
Phase 4 — if a tree ensemble barely beats a well-regularized linear
baseline, that's a finding worth reporting, not a reason to skip the
baseline.

Preprocessing choices, and why they're needed here but NOT for the Phase 4
tree models:
  - Median imputation: logistic regression can't handle NaN at all, unlike
    XGBoost/LightGBM which split around missing values natively. Median is
    a defensible default for skewed financial/behavioral features (robust
    to outliers, unlike mean).
  - Standard scaling: logistic regression's coefficients (and its
    regularization penalty) are scale-sensitive — an unscaled feature with
    large raw magnitude would dominate the L2 penalty for no principled
    reason. Tree models split on raw thresholds and don't need this.
  - One-hot encoding of `segment`: logistic regression needs an explicit
    numeric encoding for categoricals; tree models can take a native
    category dtype.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.config import LOGREG_PARAMS


def build_preprocessing_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )


def build_baseline_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocessing = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    model = LogisticRegression(**LOGREG_PARAMS)
    return Pipeline(steps=[("preprocess", preprocessing), ("model", model)])
