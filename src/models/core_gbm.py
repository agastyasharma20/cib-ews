"""
Phase 5 (part 1) — lightly-tuned core GBM: trains XGBoost and LightGBM for
each label horizon with a small randomized hyperparameter search, compares
both against the Phase 3 logistic regression baseline on ROC-AUC,
precision/recall, and calibration, and picks ONE algorithm as "the core
model" going forward (used by src/explainability/ next).

WHY a manual randomized search over train_sub/val, not sklearn's
RandomizedSearchCV with K-fold CV:
K-fold CV shuffles rows across folds, which would silently reintroduce the
exact random-split leakage the whole project has been careful to avoid
(src/features/config.py:TRAIN_MAX_MONTH_IDX, docs/deterioration_definition
.md). Instead, each candidate hyperparameter set is fit on the SAME
time-ordered train_sub (month_idx 0-9) and scored on the SAME time-ordered
validation slice (month_idx 10-11) already used for early stopping in
Phase 4 (src/models/data_prep.py: prepare_xy_tree) — one consistent,
leakage-safe evaluation surface instead of a different (and here, invalid)
one just for tuning.

WHY "light" tuning (a handful of random draws, not an exhaustive grid):
This is a benchmark/portfolio project on synthetic data with a clear,
strong signal (see Phase 3/4 results) — the marginal value of an
exhaustive search is low, and the brief explicitly asks to keep this fast.
A small random search over the parameters most likely to matter
(tree depth/leaves, learning rate, row/column subsampling, minimum leaf
size) is enough to meaningfully improve on the Phase 4 defaults without
turning this into a multi-hour tuning exercise.

Run with: python -m src.models.core_gbm
(requires results/metrics/baseline_logreg_summary.csv to exist already)
"""

from __future__ import annotations

import json
import random

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.config import PROJECT_ROOT, RANDOM_SEED
from src.models.config import TARGET_HORIZONS
from src.models.data_prep import load_model_dataset, prepare_xy_tree
from src.models.tree_models import _scale_pos_weight, fit_xgboost, fit_lightgbm, get_gain_importance
from src.models.evaluate import compute_metrics, lift_table, plot_evaluation, plot_calibration, evaluate_at_threshold

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
for _d in (MODELS_DIR, METRICS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BASELINE_SUMMARY_FILE = METRICS_DIR / "baseline_logreg_summary.csv"
N_SEARCH_ITER = 8  # random hyperparameter draws per model per horizon — kept small on purpose (see module docstring)

XGB_SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 5, 10],
}

LGBM_SEARCH_SPACE = {
    "num_leaves": [15, 31, 63],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.02, 0.03, 0.05],
    "min_child_samples": [50, 100, 150],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
}


def _sample_params(space: dict, rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in space.items()}


def _fit_and_score_xgb(params: dict, data: dict) -> tuple[XGBClassifier, float]:
    spw = _scale_pos_weight(data["y_train"])
    model = XGBClassifier(
        n_estimators=1000, tree_method="hist", enable_categorical=True, eval_metric="aucpr",
        random_state=RANDOM_SEED, scale_pos_weight=spw, early_stopping_rounds=50, **params,
    )
    model = fit_xgboost(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    val_proba = model.predict_proba(data["X_val"])[:, 1]
    return model, compute_metrics(data["y_val"].to_numpy(), val_proba)["pr_auc"]


def _fit_and_score_lgbm(params: dict, data: dict) -> tuple[LGBMClassifier, float]:
    spw = _scale_pos_weight(data["y_train"])
    model = LGBMClassifier(
        n_estimators=1000, random_state=RANDOM_SEED, verbosity=-1, scale_pos_weight=spw, **params,
    )
    model = fit_lightgbm(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"], data["categorical_cols"])
    val_proba = model.predict_proba(data["X_val"])[:, 1]
    return model, compute_metrics(data["y_val"].to_numpy(), val_proba)["pr_auc"]


def tune_model(model_name: str, data: dict, rng: random.Random) -> tuple:
    """Randomized search selecting on VALIDATION PR-AUC (the metric most
    sensitive to top-of-list quality, which is what an EWS actually needs —
    see src/models/evaluate.py docstring). Returns (best_model, best_params,
    best_val_pr_auc)."""
    space = XGB_SEARCH_SPACE if model_name == "xgboost" else LGBM_SEARCH_SPACE
    fit_fn = _fit_and_score_xgb if model_name == "xgboost" else _fit_and_score_lgbm

    best_model, best_params, best_score = None, None, -1.0
    for i in range(N_SEARCH_ITER):
        params = _sample_params(space, rng)
        model, val_pr_auc = fit_fn(params, data)
        print(f"  [{model_name}] iter {i+1}/{N_SEARCH_ITER} params={params} val_pr_auc={val_pr_auc:.4f}")
        if val_pr_auc > best_score:
            best_model, best_params, best_score = model, params, val_pr_auc

    return best_model, best_params, best_score


def run_for_horizon(df: pd.DataFrame, horizon_col: str, rng: random.Random) -> list[dict]:
    print(f"\n{'=' * 70}\n{horizon_col}\n{'=' * 70}")
    data = prepare_xy_tree(df, horizon_col)
    feature_names = data["numeric_cols"] + data["categorical_cols"]
    results = []

    for model_name in ["xgboost", "lightgbm"]:
        print(f"\nTuning {model_name}...")
        best_model, best_params, best_val_pr_auc = tune_model(model_name, data, rng)
        print(f"Best {model_name} params: {best_params} (val PR-AUC={best_val_pr_auc:.4f})")

        y_test = data["y_test"].to_numpy()
        y_proba = best_model.predict_proba(data["X_test"])[:, 1]
        metrics = compute_metrics(y_test, y_proba)
        threshold_metrics = evaluate_at_threshold(y_test, y_proba, threshold=0.5)
        lift = lift_table(y_test, y_proba)
        print(f"Test ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
              f"precision={threshold_metrics['precision']:.3f}  recall={threshold_metrics['recall']:.3f}  "
              f"top-decile lift={lift.iloc[0]['lift']:.2f}")

        calib_path = FIGURES_DIR / f"core_gbm_{model_name}_{horizon_col}_calibration.png"
        plot_calibration(y_test, y_proba, f"{model_name} (tuned) — {horizon_col}", calib_path, model_label=model_name)

        eval_path = FIGURES_DIR / f"core_gbm_{model_name}_{horizon_col}_roc_pr_lift.png"
        plot_evaluation(y_test, y_proba, f"{model_name} (tuned) — {horizon_col}", eval_path)

        importance = get_gain_importance(best_model, feature_names)

        results.append({
            "model": model_name,
            "horizon": horizon_col,
            "best_params": best_params,
            "val_pr_auc": best_val_pr_auc,
            "model_object": best_model,
            "top_features": importance.head(10).round(2).to_dict(),
            **metrics,
            **threshold_metrics,
            "top_decile_lift": float(lift.iloc[0]["lift"]),
        })

    return results


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    df = load_model_dataset()
    all_results = []

    for horizon_col in TARGET_HORIZONS:
        all_results.extend(run_for_horizon(df, horizon_col, rng))

    tuned_summary = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("model_object", "top_features", "best_params")} for r in all_results]
    )
    print(f"\n{'=' * 80}\nTuned XGBoost vs. LightGBM (test set)\n{'=' * 80}")
    print(tuned_summary.to_string(index=False))

    # --- Pick ONE algorithm as the core model, by average test PR-AUC across all 3 horizons ---
    avg_by_model = tuned_summary.groupby("model")["pr_auc"].mean().sort_values(ascending=False)
    core_model_name = avg_by_model.index[0]
    print(f"\nAverage test PR-AUC across horizons: {avg_by_model.to_dict()}")
    print(f"\n>>> CORE MODEL CHOICE: {core_model_name} <<<")
    print(
        f"Chosen because it has the higher mean PR-AUC across all 3 horizons "
        f"({avg_by_model.iloc[0]:.4f} vs {avg_by_model.iloc[1]:.4f}). PR-AUC (not ROC-AUC) is the "
        f"deciding metric because an EWS lives or dies on precision within the small top-risk slice "
        f"an RM can actually act on — see src/models/evaluate.py for why ROC-AUC alone is misleading "
        f"at this class balance."
    )

    # --- Baseline comparison ---
    if BASELINE_SUMMARY_FILE.exists():
        baseline_summary = pd.read_csv(BASELINE_SUMMARY_FILE).assign(model="logistic_regression")
        comparison = pd.concat(
            [baseline_summary[["model", "horizon", "roc_auc", "pr_auc", "precision", "recall", "top_decile_lift"]],
             tuned_summary[["model", "horizon", "roc_auc", "pr_auc", "precision", "recall", "top_decile_lift"]]],
            ignore_index=True,
        ).sort_values(["horizon", "model"])
        print(f"\n{'=' * 80}\nFull comparison: baseline vs. tuned core models\n{'=' * 80}")
        print(comparison.to_string(index=False))
        comparison.to_csv(METRICS_DIR / "core_gbm_comparison.csv", index=False)

    # --- Save the winning algorithm's model for each horizon as "the core model" ---
    for r in all_results:
        if r["model"] == core_model_name:
            path = MODELS_DIR / f"core_model_{r['horizon']}.joblib"
            joblib.dump(r["model_object"], path)
            print(f"Saved core model -> {path}")

    tuned_summary.to_csv(METRICS_DIR / "core_gbm_summary.csv", index=False)
    with open(METRICS_DIR / "core_gbm_full_results.json", "w") as f:
        json.dump(
            [{k: v for k, v in r.items() if k != "model_object"} for r in all_results],
            f, indent=2, default=str,
        )
    with open(METRICS_DIR / "core_model_choice.json", "w") as f:
        json.dump({"core_model_name": core_model_name, "avg_test_pr_auc_by_model": avg_by_model.to_dict()}, f, indent=2)
    print(f"\nSaved tuning results -> {METRICS_DIR / 'core_gbm_full_results.json'}")
    print(f"Saved core model choice -> {METRICS_DIR / 'core_model_choice.json'}")


if __name__ == "__main__":
    main()
