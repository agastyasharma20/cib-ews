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
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
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


def evaluate_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Precision/recall/F1 at a fixed operating threshold — complements the
    threshold-free ranking metrics above with the "if we acted on everyone
    scored >= threshold" operational view. Shared by the baseline and
    core-model scripts so every model is judged at the same cutoff."""
    y_pred = (y_proba >= threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "threshold": threshold,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "support_positive": report["1"]["support"],
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


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray, title: str, save_path, model_label: str = "Model") -> None:
    """
    A calibration plot answers a different question than AUC: "when the
    model says 70% risk, do ~70% of those customers actually deteriorate?"
    A well-ranked model (high AUC) can still be poorly CALIBRATED (e.g.
    systematically over- or under-confident) — this matters a lot for an
    EWS, since an RM's trust in the tool depends on the risk score meaning
    what it says, not just ranking customers correctly relative to each
    other. Shared by the baseline and core-model scripts so calibration is
    judged the same way for every model.
    """
    import matplotlib.pyplot as plt

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_predicted_value, fraction_of_positives, marker="o", label=model_label)
    ax.plot([0, 1], [0, 1], linestyle=":", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted risk (within bin)")
    ax.set_ylabel("Actual fraction that deteriorated")
    ax.set_title(f"Calibration — {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
