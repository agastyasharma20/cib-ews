"""
Phase 7 — Time-to-deterioration survival analysis.

WHY this is a genuinely different question than the classification models:
Phases 3-5 answer "will this customer deteriorate within a fixed 30/60/90
day window?" — a series of yes/no questions. Survival analysis instead
asks "how much time is left before this customer deteriorates?", which
lets the output read as "likely to deteriorate in ~45 days" instead of
three separate probabilities that don't directly say WHEN.

THE "EVENT": the labeling framework's own confirmed_deterioration flag
(src/labeling/seasonal_filter.py) — the same trailing, seasonal-filtered
signal used to build the classification labels, NOT the hidden
ground_truth_cohort. A customer "has the event" the first month their
Deterioration Index is confirmed-breached.

WHY A FIXED INDEX MONTH (month_idx=5), NOT ONE ROW PER CUSTOMER-MONTH:
Cox-style models (and concordance) are built around "at time zero, given
these covariates, how long until the event" — a single snapshot per
subject. month_idx=5 is the earliest month every customer's trailing
features are simultaneously defined (the labeling framework's DI itself is
undefined before month 5 for the same reason — see
docs/deterioration_definition.md) so nobody's "clock" starts before we can
actually see them.

WHY THE TRAIN/TEST SPLIT ISN'T THE SAME MECHANIC AS PHASES 3-5:
Row-level time-based splitting (train on early months, test on late
months) doesn't translate directly to one-row-per-customer survival data —
every customer's outcome is observed through the same final month
regardless of who they are. Instead, the SAME discipline is reproduced a
different way: the model is FIT only on outcomes as they would be known by
month 11 (src.features.config.TRAIN_MAX_MONTH_IDX) — customers not yet
confirmed-deteriorating by then are administratively censored at 11 for
training, exactly as a real deployment training "as of today" would have
to. Concordance is then evaluated on a HELD-OUT set of customers, scored
against their FULL follow-up through month 17 — "if we'd trained this in
month 11 and waited to see what really happened by month 17, how good was
the ranking?"

WHY COXPH WAS TRIED FIRST, AND WHY IT WAS REJECTED HERE:
Cox proportional hazards is the standard first model for this kind of
problem — directly interpretable hazard ratios — but it assumes each
covariate's effect on the hazard is constant over time. `cph.check_
assumptions` is run below and the result is NOT a formality: it flagged
11 of 22 covariates (p<0.05, several p<5e-05) as violating proportional
hazards — including almost every balance-trend feature, the single most
important signal group in this whole project (see full output in
results/metrics/survival_ph_check.txt). That is a pervasive violation, not
a couple of borderline covariates, so trusting Cox's hazard ratios here
would be reporting numbers the model's own diagnostic says aren't reliable.
Per the brief's own instruction, this is exactly the trigger to switch to
a Random Survival Forest (scikit-survival) instead — which makes no
proportional-hazards assumption at all. The Cox fit and its PH check are
still run and reported (useful, interpretable diagnostic context) but the
RSF is the model actually used for the headline concordance index, hazard
curves, and the saved scored table.

Run with: python -m src.models.survival
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from src.config import PROJECT_ROOT, RANDOM_SEED, DATA_PROCESSED_DIR
from src.features.config import all_feature_columns, TRAIN_MAX_MONTH_IDX
from src.models.data_prep import load_model_dataset

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
for _d in (METRICS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LABELS_FILE = DATA_PROCESSED_DIR / "deterioration_labels.parquet"
SURVIVAL_SCORES_FILE = DATA_PROCESSED_DIR / "survival_scores.parquet"
PH_CHECK_FILE = METRICS_DIR / "survival_ph_check.txt"

INDEX_MONTH_IDX = 5          # earliest month every customer's trailing features are defined
TEST_CUSTOMER_FRACTION = 0.25
COX_PENALIZER = 0.1          # small L2 penalty — several covariates are correlated (e.g. amb_pct_change_30/60/90d)
PH_VIOLATION_FRACTION_TO_SWITCH = 0.30  # switch to RSF if more than this share of covariates fail the PH test


def build_survival_dataset(model_df: pd.DataFrame, labels_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One row per customer: covariates at INDEX_MONTH_IDX, plus duration/
    event pairs for both the training-time administrative censor (month 11)
    and the full follow-up (last observed month)."""
    feature_cols = all_feature_columns()
    full_followup_month_idx = int(model_df["month_idx"].max())

    snapshot = model_df.loc[model_df["month_idx"] == INDEX_MONTH_IDX, ["customer_id", "segment", *feature_cols]].copy()

    first_confirmed = (
        labels_df[labels_df["confirmed_deterioration"]]
        .groupby("customer_id")["month_idx"]
        .min()
        .rename("first_confirmed_month")
    )
    df = snapshot.merge(first_confirmed, on="customer_id", how="left")

    def _duration_event(admin_censor_month_idx: int) -> tuple[pd.Series, pd.Series]:
        occurred_in_window = df["first_confirmed_month"].notna() & (df["first_confirmed_month"] <= admin_censor_month_idx)
        event = occurred_in_window.astype(int)
        event_month = df["first_confirmed_month"].where(occurred_in_window, admin_censor_month_idx)
        duration = event_month - INDEX_MONTH_IDX
        return duration, event

    df["duration_train"], df["event_train"] = _duration_event(TRAIN_MAX_MONTH_IDX)
    df["duration_full"], df["event_full"] = _duration_event(full_followup_month_idx)

    # One-hot segment now (deterministic, no outcome information) so train
    # and test share identical dummy columns regardless of the split.
    segment_dummies = pd.get_dummies(df["segment"], prefix="segment", drop_first=True).astype(float)
    df = pd.concat([df, segment_dummies], axis=1)
    covariate_cols = feature_cols + list(segment_dummies.columns)

    return df, covariate_cols


def split_and_impute(df: pd.DataFrame, covariate_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Customer-level train/test split (see module docstring for why this
    replaces the row-level time split here), with the imputer FIT ONLY on
    train customers and applied to both — the same leakage discipline as
    the Phase 3 baseline's preprocessing pipeline, just spelled out
    explicitly instead of hidden inside a sklearn Pipeline."""
    train_df, test_df = train_test_split(
        df, test_size=TEST_CUSTOMER_FRACTION, random_state=RANDOM_SEED, stratify=df["segment"]
    )
    train_df, test_df = train_df.copy(), test_df.copy()

    imputer = SimpleImputer(strategy="median")
    train_df[covariate_cols] = imputer.fit_transform(train_df[covariate_cols])
    test_df[covariate_cols] = imputer.transform(test_df[covariate_cols])

    return train_df, test_df


def check_ph_assumptions(cph: CoxPHFitter, cox_train_df: pd.DataFrame, n_covariates: int) -> tuple[str, int]:
    """Runs lifelines' built-in proportional-hazards check and captures its
    printed output (it reports via print, not a return value). Returns the
    raw text plus a count of flagged covariates, parsed from lifelines'
    own wording ("failed the non-proportional test") — counted carefully
    since a first pass at this used the wrong case ("Failed...") and
    silently reported 0 flags when there were actually 11."""
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            cph.check_assumptions(cox_train_df, p_value_threshold=0.05, show_plots=False)
    except Exception as e:  # lifelines raises if it can't compute some statistics — log and move on
        buffer.write(f"\n[check_assumptions raised: {e}]\n")
    text = buffer.getvalue()
    n_flagged = text.count("failed the non-proportional test")
    return text, n_flagged


def _median_survival_time(surv_fn, time_points: np.ndarray) -> float:
    """First time point where the step function's survival probability
    drops to <= 0.5; np.inf if it never does within the observed horizon
    (sksurv step functions can be queried at arbitrary points)."""
    probs = surv_fn(time_points)
    below_half = np.where(probs <= 0.5)[0]
    return float(time_points[below_half[0]]) if len(below_half) > 0 else np.inf


def main() -> None:
    print("Loading model dataset and labels...")
    model_df = load_model_dataset()
    labels_df = pd.read_parquet(LABELS_FILE)

    print(f"Building survival dataset at index month_idx={INDEX_MONTH_IDX}...")
    df, covariate_cols = build_survival_dataset(model_df, labels_df)
    print(f"{len(df):,} customers. Event rate (full follow-up): {df['event_full'].mean():.3f}  "
          f"(training-window event rate, admin-censored at month {TRAIN_MAX_MONTH_IDX}: {df['event_train'].mean():.3f})")

    train_df, test_df = split_and_impute(df, covariate_cols)
    print(f"Train customers: {len(train_df):,}  Test customers: {len(test_df):,}")

    cox_train_df = train_df[covariate_cols + ["duration_train", "event_train"]].rename(
        columns={"duration_train": "duration", "event_train": "event"}
    )

    # --- Step 1: fit Cox as the standard first model, and CHECK it ---
    print(f"\nFitting CoxPHFitter (penalizer={COX_PENALIZER}) as the first candidate model...")
    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    cph.fit(cox_train_df, duration_col="duration", event_col="event")

    print("Checking proportional-hazards assumption...")
    ph_report, n_flagged = check_ph_assumptions(cph, cox_train_df, len(covariate_cols))
    with open(PH_CHECK_FILE, "w", encoding="utf-8") as f:
        f.write(ph_report)
    violation_fraction = n_flagged / len(covariate_cols)
    print(f"PH check: {n_flagged}/{len(covariate_cols)} covariates ({violation_fraction:.0%}) flagged as violating "
          f"proportional hazards (p<0.05). Full detail -> {PH_CHECK_FILE}")

    use_rsf = violation_fraction > PH_VIOLATION_FRACTION_TO_SWITCH
    print(f"\n>>> {'Switching to Random Survival Forest' if use_rsf else 'Keeping CoxPHFitter'} "
          f"(threshold: >{PH_VIOLATION_FRACTION_TO_SWITCH:.0%} of covariates flagged) <<<")

    print("\nCox hazard ratios (kept for interpretability, despite the PH violation — treat with caution):")
    summary = cph.summary[["coef", "exp(coef)", "p"]].sort_values("exp(coef)", ascending=False)
    print(summary.head(10).round(3).to_string())

    # --- Step 2: fit the model actually used (RSF, given the violation above) ---
    X_train, X_test = train_df[covariate_cols], test_df[covariate_cols]
    y_train_admin = Surv.from_arrays(event=train_df["event_train"].astype(bool), time=train_df["duration_train"].clip(lower=0.01))
    y_test_full = Surv.from_arrays(event=test_df["event_full"].astype(bool), time=test_df["duration_full"].clip(lower=0.01))

    if use_rsf:
        print(f"\nFitting RandomSurvivalForest...")
        model = RandomSurvivalForest(n_estimators=300, min_samples_leaf=15, max_depth=6, n_jobs=-1, random_state=RANDOM_SEED)
        model.fit(X_train, y_train_admin)
        c_index = model.score(X_test, y_test_full)
        model_name = "RandomSurvivalForest"
    else:
        cox_test_full_df = test_df[covariate_cols + ["duration_full", "event_full"]].rename(
            columns={"duration_full": "duration", "event_full": "event"}
        )
        model = cph
        c_index = cph.score(cox_test_full_df, scoring_method="concordance_index")
        model_name = "CoxPHFitter"

    print(f"\n>>> Concordance index ({model_name}, held-out customers, full follow-up): {c_index:.3f} <<<")
    print(
        "(This c-index is the honest, leakage-safe number: the model that produced it only ever saw "
        f"outcomes admin-censored at month {TRAIN_MAX_MONTH_IDX}, exactly as a real deployment training "
        "'as of today' would have. Its own survival-function domain is therefore capped at "
        f"{TRAIN_MAX_MONTH_IDX - INDEX_MONTH_IDX} months, too short to speak to the full 90-day+ horizon "
        "the scored table needs.)"
    )

    # --- Step 3: refit a DEPLOYMENT model on ALL customers with FULL
    # follow-up (standard practice: evaluate rigorously on a held-out,
    # time-disciplined split first, then retrain on everything available
    # for the model that actually gets served) ---
    print("\nRefitting on all customers with full follow-up for the deployed scored table...")
    full_covariates = pd.concat([train_df, test_df]).sort_index()
    X_all = full_covariates[covariate_cols]
    y_all_full = Surv.from_arrays(event=full_covariates["event_full"].astype(bool), time=full_covariates["duration_full"].clip(lower=0.01))

    full_followup_month_idx = int(model_df["month_idx"].max())
    max_observable_months = full_followup_month_idx - INDEX_MONTH_IDX
    time_points = np.arange(0, max_observable_months + 1, dtype=float)

    if use_rsf:
        deployed_model = RandomSurvivalForest(n_estimators=300, min_samples_leaf=15, max_depth=6, n_jobs=-1, random_state=RANDOM_SEED)
        deployed_model.fit(X_all, y_all_full)
        risk_score = deployed_model.predict(X_all)  # higher = more likely to deteriorate sooner
        surv_fns = deployed_model.predict_survival_function(X_all)
        median_duration_months = np.array([_median_survival_time(fn, time_points) for fn in surv_fns])
    else:
        deployed_model = CoxPHFitter(penalizer=COX_PENALIZER)
        deployed_model.fit(
            full_covariates[covariate_cols + ["duration_full", "event_full"]].rename(columns={"duration_full": "duration", "event_full": "event"}),
            duration_col="duration", event_col="event",
        )
        risk_score = deployed_model.predict_partial_hazard(X_all).to_numpy()
        median_duration_months = deployed_model.predict_median(X_all).to_numpy()

    survival_scores = pd.DataFrame({
        "customer_id": full_covariates["customer_id"].values,
        "segment": full_covariates["segment"].values,
        "event_observed_full_followup": full_covariates["event_full"].values,
        "risk_score": risk_score,
        "predicted_median_months_to_deterioration": median_duration_months,
    })
    survival_scores["median_beyond_observed_horizon"] = ~np.isfinite(survival_scores["predicted_median_months_to_deterioration"])
    survival_scores["predicted_median_days_to_deterioration"] = np.where(
        survival_scores["median_beyond_observed_horizon"],
        np.nan,
        survival_scores["predicted_median_months_to_deterioration"] * 30,
    )
    survival_scores.to_parquet(SURVIVAL_SCORES_FILE, index=False)
    print(f"Saved {SURVIVAL_SCORES_FILE} ({len(survival_scores):,} customers)")

    # --- Full survival curve, every customer, in LONG format — the
    # dashboard (Phase 9) uses this to render a per-customer sparkline
    # without needing to load the model itself. ---
    print("Exporting full survival curves for all customers (for the dashboard)...")
    if use_rsf:
        curve_matrix = np.array([fn(time_points) for fn in surv_fns])  # (n_customers, n_time_points)
    else:
        curve_matrix = deployed_model.predict_survival_function(X_all, times=time_points).T.to_numpy()
    curves_long = pd.DataFrame(curve_matrix, columns=time_points.astype(int)).assign(
        customer_id=full_covariates["customer_id"].values
    ).melt(id_vars="customer_id", var_name="month_offset", value_name="survival_probability")
    curves_long.to_parquet(DATA_PROCESSED_DIR / "survival_curves.parquet", index=False)
    print(f"Saved {DATA_PROCESSED_DIR / 'survival_curves.parquet'} ({len(curves_long):,} rows)")

    # --- Step 4: 3 example hazard/survival curves spanning the risk spectrum ---
    print("\nGenerating 3 example hazard curves (high / medium / low risk)...")
    order = np.argsort(-risk_score)
    example_positions = [order[0], order[len(order) // 2], order[-1]]
    labels = ["High risk", "Medium risk", "Low risk"]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, pos in zip(labels, example_positions):
        row = full_covariates.iloc[pos]
        if use_rsf:
            surv_probs = surv_fns[pos](time_points)
        else:
            surv_probs = deployed_model.predict_survival_function(row[covariate_cols].to_frame().T, times=time_points).iloc[:, 0].values
        ax.plot(time_points * 30, surv_probs, marker="o", markersize=3, label=f"{label} ({row['customer_id']})")
    ax.axhline(0.5, color="grey", linestyle=":", label="50% survival (median)")
    ax.set_xlabel(f"Days since index month (month_idx={INDEX_MONTH_IDX})")
    ax.set_ylabel("Estimated probability of NOT yet being confirmed-deteriorating")
    ax.set_title(f"Example survival curves ({model_name}) — 3 customers spanning the risk spectrum")
    ax.legend()
    fig.tight_layout()
    curve_path = FIGURES_DIR / "survival_example_curves.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"Saved {curve_path}")

    print(f"\n{'=' * 70}\n3 example customers — hazard summary\n{'=' * 70}")
    for label, pos in zip(labels, example_positions):
        row = full_covariates.iloc[pos]
        cid = row["customer_id"]
        rec = survival_scores.loc[survival_scores["customer_id"] == cid].iloc[0]
        median_str = (
            f"~{rec['predicted_median_days_to_deterioration']:.0f} days"
            if not rec["median_beyond_observed_horizon"]
            else f"> {max_observable_months * 30} days (no crossing observed within horizon)"
        )
        print(f"{label:12s} customer_id={cid}  segment={row['segment']}  expected time-to-deterioration: {median_str}")


if __name__ == "__main__":
    main()
