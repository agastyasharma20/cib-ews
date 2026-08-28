"""
Phase 8 (part 2) — feeds the graph features from wallet_leakage.py into the
core XGBoost model and checks whether they actually earn their place:
does test AUC/PR-AUC improve, and do the graph features show up as
meaningful SHAP drivers specifically for the customers whose ground-truth
cohort (gradual/sudden deterioration) the wallet-leakage story is supposed
to explain?

METHODOLOGY: the SAME tuned hyperparameters found in Phase 5
(results/metrics/core_gbm_full_results.json) are reused for both the
"before" (5 feature groups only) and "after" (+ 4 graph features) models,
per horizon — deliberately NOT re-run through another random search. This
isolates the effect of ADDING the graph features as the only thing that
changed between the two runs; re-tuning each variant separately would
muddy whether an improvement came from the new features or from a luckier
hyperparameter draw.

`ground_truth_cohort` is read ONLY for the final "does this show up as a
driver for the right cohort" validation step below — never as a model
input, same discipline as every other use of it in this project.

Run with: python -m src.graph.reevaluate_core_model
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from src.config import PROJECT_ROOT, RANDOM_SEED, GROUND_TRUTH_FILE
from src.models.config import TARGET_HORIZONS
from src.models.data_prep import load_model_dataset, prepare_xy_tree
from src.models.tree_models import _scale_pos_weight, fit_xgboost
from src.models.evaluate import compute_metrics, lift_table
from src.graph.wallet_leakage import GRAPH_FEATURES_FILE, GRAPH_FEATURE_COLUMNS

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BEST_PARAMS_FILE = METRICS_DIR / "core_gbm_full_results.json"


def load_best_xgb_params() -> dict:
    with open(BEST_PARAMS_FILE) as f:
        results = json.load(f)
    return {r["horizon"]: r["best_params"] for r in results if r["model"] == "xgboost"}


def fit_and_evaluate(df: pd.DataFrame, horizon_col: str, params: dict, extra_numeric_cols: list[str] | None) -> dict:
    data = prepare_xy_tree(df, horizon_col, extra_numeric_cols=extra_numeric_cols)
    model = XGBClassifier(
        n_estimators=1000, tree_method="hist", enable_categorical=True, eval_metric="aucpr",
        random_state=RANDOM_SEED, scale_pos_weight=_scale_pos_weight(data["y_train"]),
        early_stopping_rounds=50, **params,
    )
    model = fit_xgboost(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"])

    y_test = data["y_test"].to_numpy()
    y_proba = model.predict_proba(data["X_test"])[:, 1]
    metrics = compute_metrics(y_test, y_proba)
    lift = lift_table(y_test, y_proba)

    return {"model": model, "data": data, "metrics": metrics, "top_decile_lift": float(lift.iloc[0]["lift"])}


def main() -> None:
    print("Loading model dataset and graph features...")
    model_df = load_model_dataset()
    graph_df = pd.read_parquet(GRAPH_FEATURES_FILE)
    merged_df = model_df.merge(graph_df, on=["customer_id", "month"], how="left")

    best_params_by_horizon = load_best_xgb_params()

    comparison_rows = []
    after_models = {}
    for horizon_col in TARGET_HORIZONS:
        params = best_params_by_horizon[horizon_col]
        print(f"\n{'=' * 70}\n{horizon_col}  (reusing tuned params: {params})\n{'=' * 70}")

        print("Fitting BEFORE model (5 feature groups only)...")
        before = fit_and_evaluate(merged_df, horizon_col, params, extra_numeric_cols=None)
        print(f"  BEFORE  ROC-AUC={before['metrics']['roc_auc']:.4f}  PR-AUC={before['metrics']['pr_auc']:.4f}  "
              f"top-decile lift={before['top_decile_lift']:.2f}")

        print("Fitting AFTER model (5 feature groups + 4 graph features)...")
        after = fit_and_evaluate(merged_df, horizon_col, params, extra_numeric_cols=GRAPH_FEATURE_COLUMNS)
        print(f"  AFTER   ROC-AUC={after['metrics']['roc_auc']:.4f}  PR-AUC={after['metrics']['pr_auc']:.4f}  "
              f"top-decile lift={after['top_decile_lift']:.2f}")

        after_models[horizon_col] = after

        comparison_rows.append({"horizon": horizon_col, "variant": "before (no graph features)", **before["metrics"], "top_decile_lift": before["top_decile_lift"]})
        comparison_rows.append({"horizon": horizon_col, "variant": "after (+ graph features)", **after["metrics"], "top_decile_lift": after["top_decile_lift"]})

    comparison = pd.DataFrame(comparison_rows)
    print(f"\n{'=' * 90}\nBefore / after comparison — does the graph feature earn its place?\n{'=' * 90}")
    print(comparison[["horizon", "variant", "roc_auc", "pr_auc", "top_decile_lift"]].to_string(index=False))
    comparison.to_csv(METRICS_DIR / "graph_feature_before_after.csv", index=False)

    # --- Does it show up as a top SHAP driver for the RIGHT cohort? ---
    print(f"\n{'=' * 90}\nSHAP check: do graph features drive risk for gradual/sudden deterioration customers?\n{'=' * 90}")
    ground_truth = pd.read_parquet(GROUND_TRUTH_FILE)[["customer_id", "ground_truth_cohort"]]

    horizon = "deteriorates_in_90d"
    after = after_models[horizon]
    X_test = after["data"]["X_test"]
    feature_names = list(X_test.columns)

    explainer = shap.TreeExplainer(after["model"])
    shap_values = explainer.shap_values(X_test)
    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_test.index)
    shap_df["customer_id"] = merged_df.loc[X_test.index, "customer_id"].values
    shap_df = shap_df.merge(ground_truth, on="customer_id", how="left")

    mean_abs_shap_by_cohort = (
        shap_df.groupby("ground_truth_cohort")[GRAPH_FEATURE_COLUMNS + feature_names[:3]]
        .apply(lambda g: g.abs().mean())
    )
    print("\nMean |SHAP| for graph features + top-3 other features, by ground-truth cohort (validation only):")
    print(mean_abs_shap_by_cohort.round(3).to_string())

    # Overall global rank of each graph feature among all features (by mean |SHAP|)
    global_mean_abs_shap = shap_df[feature_names].abs().mean().sort_values(ascending=False)
    global_rank = {col: int(global_mean_abs_shap.index.get_loc(col)) + 1 for col in GRAPH_FEATURE_COLUMNS}
    print(f"\nGlobal importance rank (1 = most important) out of {len(feature_names)} features:")
    for col, rank in global_rank.items():
        print(f"  {col}: rank {rank}")

    with open(METRICS_DIR / "graph_feature_shap_check.txt", "w") as f:
        f.write("Mean |SHAP| by ground-truth cohort:\n")
        f.write(mean_abs_shap_by_cohort.round(4).to_string())
        f.write(f"\n\nGlobal importance rank (1=most important) out of {len(feature_names)} features:\n")
        for col, rank in global_rank.items():
            f.write(f"  {col}: rank {rank}\n")
    print(f"\nSaved -> {METRICS_DIR / 'graph_feature_shap_check.txt'}")


if __name__ == "__main__":
    main()
