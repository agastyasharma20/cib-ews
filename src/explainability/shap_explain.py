"""
SHAP explainability for the core model (XGBoost — see
src/models/core_model_choice.json for which algorithm won and why).

WHY SHAP (and specifically TreeExplainer) instead of the model's own
gain-based feature importance (already plotted in Phase 4/5):
Gain importance is GLOBAL — "which features did the model lean on overall."
SHAP values are PER-PREDICTION — "for THIS customer, in THIS month, which
features pushed their score up or down, and by how much." An early warning
system's whole value proposition is explaining an individual customer's
risk to their RM, not the portfolio in aggregate, so per-customer
attribution is the actual requirement here.

TreeExplainer is exact (not an approximation) for tree ensembles and is
fast enough to run on the full ~90k-row panel in one pass, unlike the
model-agnostic KernelExplainer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


def compute_shap_values(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Returns a (n_rows, n_features) array of SHAP values, same column
    order as X. A positive value means that feature pushed THIS row's
    predicted risk up; negative means it pushed risk down."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return np.asarray(shap_values)


def shap_values_to_frame(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(shap_values, columns=feature_names)
