"""
Shared evaluation framework — used by the Phase 3 baseline AND (unchanged)
by the Phase 4 XGBoost/LightGBM model, so the two are compared on identical
metrics rather than each phase inventing its own scoring.

WHY ROC-AUC isn't enough on its own:
With a ~6-15% positive rate, a model that's very good at ranking healthy
customers correctly can still post a deceptively high ROC-AUC while being
mediocre at the thing that actually matters for an EWS: precision in the
TOP slice of risk scores an RM has bandwidth to act on. PR-AUC and the lift
table are reported alongside ROC-AUC specifically because they're sensitive
to that top-of-list quality in a way ROC-AUC is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "base_rate": float(np.mean(y_true)),
        "n": len(y_true),
    }


def lift_table(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Ranks every scored customer-month into `n_bins` equal-sized buckets by
    predicted risk (bin 1 = highest risk) and reports the actual positive
    rate in each — this is the practical question an RM/ops team asks:
    "if I only act on the top decile the model flags, how much of the real
    deterioration am I actually catching, and how concentrated is it?"
    """
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df["risk_decile"] = pd.qcut(df["y_proba"].rank(method="first"), n_bins, labels=False)
    df["risk_decile"] = n_bins - df["risk_decile"]  # 1 = highest risk

    overall_rate = df["y_true"].mean()
    table = (
        df.groupby("risk_decile")
        .agg(n=("y_true", "size"), n_positive=("y_true", "sum"), actual_rate=("y_true", "mean"))
        .sort_index()
    )
    table["lift"] = (table["actual_rate"] / overall_rate).round(2)
    table["pct_of_all_positives_captured"] = (table["n_positive"].cumsum() / table["n_positive"].sum()).round(3)
    return table.reset_index()


def plot_evaluation(y_true: np.ndarray, y_proba: np.ndarray, title: str, save_path) -> None:
    """Three-panel figure: ROC curve, Precision-Recall curve, lift chart."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_proba):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle=":", color="grey")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    axes[1].plot(recall, precision, label=f"PR-AUC={average_precision_score(y_true, y_proba):.3f}")
    axes[1].axhline(np.mean(y_true), linestyle=":", color="grey", label="base rate")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    lift = lift_table(y_true, y_proba)
    axes[2].bar(lift["risk_decile"], lift["lift"])
    axes[2].axhline(1.0, linestyle=":", color="grey")
    axes[2].set_xlabel("Risk decile (1 = highest predicted risk)")
    axes[2].set_ylabel("Lift vs. base rate")
    axes[2].set_title("Lift by Risk Decile")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
