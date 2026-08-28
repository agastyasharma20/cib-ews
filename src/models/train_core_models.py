"""
Phase 4 — core models: trains XGBoost and LightGBM for each of the 3 label
horizons, evaluates them with the same metrics as the Phase 3 baseline
(src/models/evaluate.py — identical ROC-AUC/PR-AUC/lift definitions, so the
comparison is apples-to-apples), and reports whether the added complexity
of a tree ensemble is actually earning its keep over plain logistic
regression.

Run with: python -m src.models.train_core_models
(requires results/metrics/baseline_logreg_summary.csv from
 src.models.baseline_logistic to already exist, for the comparison table)
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROJECT_ROOT
from src.models.config import TARGET_HORIZONS
from src.models.data_prep import load_model_dataset, prepare_xy_tree
from src.models.tree_models import (
    build_xgboost_model,
    build_lightgbm_model,
    fit_xgboost,
    fit_lightgbm,
    get_gain_importance,
)
from src.models.evaluate import compute_metrics, lift_table, plot_evaluation

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
for _d in (MODELS_DIR, METRICS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BASELINE_SUMMARY_FILE = METRICS_DIR / "baseline_logreg_summary.csv"


def plot_importance(importance: pd.Series, title: str, save_path, top_n: int = 15) -> None:
    top = importance.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index, top.values, color="#2ca02c")
    ax.set_xlabel("Gain-based feature importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_model_for_horizon(model_name: str, df: pd.DataFrame, horizon_col: str) -> dict:
    data = prepare_xy_tree(df, horizon_col)
    print(f"\n--- {model_name} | {horizon_col} ---")
    print(f"Train(sub): {len(data['X_train']):,}  Val: {len(data['X_val']):,}  Test: {len(data['X_test']):,}")
    print(f"Train base rate: {data['y_train'].mean():.3f}  scale_pos_weight target ratio applied internally")

    feature_names = data["numeric_cols"] + data["categorical_cols"]

    if model_name == "xgboost":
        model = build_xgboost_model(data["y_train"])
        model = fit_xgboost(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"])
        print(f"Best iteration (early stopping): {model.best_iteration}")
    elif model_name == "lightgbm":
        model = build_lightgbm_model(data["y_train"])
        model = fit_lightgbm(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"], data["categorical_cols"])
        print(f"Best iteration (early stopping): {model.best_iteration_}")
    else:
        raise ValueError(model_name)

    y_test = data["y_test"].to_numpy()
    y_proba = model.predict_proba(data["X_test"])[:, 1]
    metrics = compute_metrics(y_test, y_proba)
    print(f"Test ROC-AUC: {metrics['roc_auc']:.3f}   Test PR-AUC: {metrics['pr_auc']:.3f}")

    lift = lift_table(y_test, y_proba)
    top_decile_lift = float(lift.iloc[0]["lift"])
    print(f"Top-decile lift: {top_decile_lift:.2f}")

    plot_path = FIGURES_DIR / f"{model_name}_{horizon_col}_roc_pr_lift.png"
    plot_evaluation(y_test, y_proba, f"{model_name} — {horizon_col}", plot_path)

    importance = get_gain_importance(model, feature_names)
    importance_path = FIGURES_DIR / f"{model_name}_{horizon_col}_importance.png"
    plot_importance(importance, f"{model_name} gain-based feature importance — {horizon_col}", importance_path)
    print(f"Top 5 features: {importance.head(5).round(1).to_dict()}")

    model_path = MODELS_DIR / f"{model_name}_{horizon_col}.joblib"
    joblib.dump(model, model_path)

    return {
        "model": model_name,
        "horizon": horizon_col,
        "n_train": len(data["X_train"]),
        "n_val": len(data["X_val"]),
        "n_test": len(data["X_test"]),
        **metrics,
        "top_decile_lift": top_decile_lift,
        "top_features": importance.head(10).round(2).to_dict(),
    }


def main() -> None:
    df = load_model_dataset()
    all_results = []

    for model_name in ["xgboost", "lightgbm"]:
        for horizon_col in TARGET_HORIZONS:
            result = run_model_for_horizon(model_name, df, horizon_col)
            all_results.append(result)

    core_summary = pd.DataFrame([{k: v for k, v in r.items() if k != "top_features"} for r in all_results])

    print(f"\n{'=' * 80}\nCore model summary\n{'=' * 80}")
    print(core_summary.to_string(index=False))

    # --- Compare against the Phase 3 baseline, same metrics, same horizons ---
    if BASELINE_SUMMARY_FILE.exists():
        baseline_summary = pd.read_csv(BASELINE_SUMMARY_FILE)
        baseline_summary = baseline_summary.assign(model="logistic_regression")[
            ["model", "horizon", "roc_auc", "pr_auc", "top_decile_lift"]
        ]
        comparison = pd.concat(
            [baseline_summary, core_summary[["model", "horizon", "roc_auc", "pr_auc", "top_decile_lift"]]],
            ignore_index=True,
        ).sort_values(["horizon", "model"])
        print(f"\n{'=' * 80}\nBaseline vs. core model comparison (same metrics, same test set)\n{'=' * 80}")
        print(comparison.to_string(index=False))
        comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
        print(f"\nSaved comparison table -> {METRICS_DIR / 'model_comparison.csv'}")
    else:
        print(f"\n(Skipping baseline comparison — {BASELINE_SUMMARY_FILE} not found. "
              f"Run `python -m src.models.baseline_logistic` first.)")

    core_summary.to_csv(METRICS_DIR / "core_model_summary.csv", index=False)
    with open(METRICS_DIR / "core_model_full_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved core model metrics -> {METRICS_DIR / 'core_model_summary.csv'}")


if __name__ == "__main__":
    main()
