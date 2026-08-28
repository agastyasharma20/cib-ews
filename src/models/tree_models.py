"""
Core models: gradient-boosted trees (XGBoost, LightGBM).

WHY these on top of the logistic regression baseline:
Tree ensembles can capture non-linear thresholds and feature INTERACTIONS
(e.g. "high competitor share matters much more when payroll regularity is
ALSO dropping") that a linear model structurally cannot, without needing to
hand-engineer every interaction term. If they don't meaningfully beat the
baseline, that itself is a useful, reportable finding — not a foregone
conclusion.

WHY no imputation/scaling/one-hot encoding here (unlike the baseline):
  - Missing values: both libraries learn, at each split, which branch a
    missing value should default to — a NaN in `trade_finance_utilization_
    pct` for a customer with no trade-finance facility is itself
    information ("this customer doesn't have this product"), and forcing a
    median value in would destroy that signal.
  - Scaling: tree splits are threshold-based and invariant to monotonic
    transformations of a feature — standardizing changes nothing about
    what the model can express, so skipping it isn't a shortcut, it's
    correct.
  - Categoricals: both libraries accept a native pandas 'category' dtype
    directly, without one-hot encoding, and can find splits smarter than
    "treat every level as a fully independent dummy."

WHY class imbalance is still handled explicitly:
Native NaN/categorical handling doesn't fix class imbalance — that's a
separate problem, addressed the same way as the baseline
(scale_pos_weight ~= class_weight="balanced", i.e. reweight the minority
class by the inverse of its frequency) so both models are compared on a
level footing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_SEED

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,           # shallow trees — this dataset has ~90k rows and 22 features; deep trees would overfit fast
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "enable_categorical": True,
    "eval_metric": "aucpr",   # optimize/monitor the metric that matters most given class imbalance
    "random_state": RANDOM_SEED,
}

LGBM_PARAMS = {
    # A first pass at these params (max_depth=4, num_leaves=15, lr=0.05,
    # default min_child_samples) made LightGBM's leaf-wise growth overfit
    # the validation slice within ~8-10 boosting rounds and stop early,
    # well short of XGBoost's ~130-270 rounds, at a visibly worse PR-AUC
    # (0.815 vs 0.890 on the 90d horizon). A small manual sweep over
    # learning_rate/num_leaves/min_child_samples (see git history / model
    # comparison notes) found this combination gives LightGBM enough room
    # to actually build out a comparable number of trees before its
    # validation metric plateaus, closing most of that gap.
    "n_estimators": 1000,
    "max_depth": 5,
    "num_leaves": 31,
    "learning_rate": 0.02,
    "min_child_samples": 100,  # counteracts leaf-wise growth's tendency to carve out tiny, overfit leaves
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbosity": -1,
}


def _scale_pos_weight(y: pd.Series) -> float:
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return float(n_neg / max(n_pos, 1))


def build_xgboost_model(y_train: pd.Series) -> XGBClassifier:
    spw = _scale_pos_weight(y_train)
    return XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw, early_stopping_rounds=50)


def build_lightgbm_model(y_train: pd.Series) -> LGBMClassifier:
    spw = _scale_pos_weight(y_train)
    return LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw)


def fit_xgboost(model: XGBClassifier, X_train, y_train, X_val, y_val) -> XGBClassifier:
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def fit_lightgbm(model: LGBMClassifier, X_train, y_train, X_val, y_val, categorical_cols: list[str]) -> LGBMClassifier:
    import lightgbm as lgb

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        categorical_feature=categorical_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return model


def get_gain_importance(model, feature_names: list[str]) -> pd.Series:
    """Gain-based feature importance — how much each feature's splits
    reduced the loss on average, summed across all trees. Not a
    per-customer explanation (that's SHAP, Phase 5) but a fast global
    sanity check on which signals the model actually leaned on."""
    if isinstance(model, XGBClassifier):
        raw = model.get_booster().get_score(importance_type="gain")
        # XGBoost names features f0, f1, ... internally unless a DataFrame
        # with real column names was used for training (which it was here),
        # in which case get_score already returns the real names.
        importance = pd.Series(raw)
        importance = importance.reindex(feature_names).fillna(0.0)
    else:
        importance = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=feature_names)
    return importance.sort_values(ascending=False)
