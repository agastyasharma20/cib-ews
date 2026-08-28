"""
Synthetic data generator for the CIB (Customers in Base) Current Account
Early Warning System.

WHAT THIS SIMULATES
--------------------
Real HDFC data isn't available, so this module fabricates a dataset that
behaves like one, well enough to build and demonstrate a real EWS pipeline:

  1. customers.parquet               - static customer master
  2. monthly_panel.parquet           - monthly behavioral panel (18 months)
  3. counterparty_transactions.parquet - per-transaction detail with the
                                          counterparty's bank, used later to
                                          build a wallet-leakage graph feature
  4. ground_truth_cohorts.parquet    - the HIDDEN true cohort per customer.

THE CORE MODELING IDEA
------------------------
Four cohorts are simulated so we have an answer key to validate labeling
and models against later:

  - stable                 (~55%) - flat behavior, noise only
  - gradual_deterioration  (~20%) - a slow, multi-month bleed across balance,
                                     transactions, digital activity, and a
                                     rising share of money moving to
                                     competitor-bank counterparties
  - sudden_deterioration   (~10%) - a sharp 1-2 month drop (e.g. a payroll
                                     migration event), then a lower plateau
  - seasonal_false_positive(~15%) - balance dips periodically (tax cycles /
                                     dividend payouts) and recovers; nothing
                                     else about the relationship changes.
                                     This cohort exists specifically to be a
                                     TRAP: a good label/model must NOT flag
                                     it as deterioration.

For the two genuine deterioration cohorts, every affected metric is driven
off ONE shared per-customer "erosion curve" (see `_erosion_curve`), each
scaled by a metric-specific weight plus its own noise. That is what creates
realistic CORRELATION between signals (e.g. digital activity and
transaction volume decline together) without making them identical or
perfectly synchronized - exactly what a model needs to learn a genuine
pattern rather than one perfectly collinear feature.

`ground_truth_cohort` (and the timing/severity fields next to it) are
written to a SEPARATE file and must never be joined into a model's feature
set - they exist purely so later phases can check "did our behavior-only
label and model actually recover the cohorts we planted".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    RANDOM_SEED,
    CUSTOMERS_FILE,
    MONTHLY_PANEL_FILE,
    COUNTERPARTY_TXNS_FILE,
    GROUND_TRUTH_FILE,
)

# ---------------------------------------------------------------------------
# Simulation configuration (kept local to the generator - only this module
# needs to know these knobs).
# ---------------------------------------------------------------------------
SIM_CONFIG = {
    "n_customers": 5000,
    "n_months": 18,
    "start_period": "2024-03",  # first simulated month (YYYY-MM)

    "segment_mix": {"SME": 0.60, "Mid-Corporate": 0.30, "Large Corporate": 0.10},
    "n_branches": 150,
    "n_rms": 120,

    "cohort_mix": {
        "stable": 0.55,
        "gradual_deterioration": 0.20,
        "sudden_deterioration": 0.10,
        "seasonal_false_positive": 0.15,
    },

    # Guardrails so every deteriorating customer has enough "before" and
    # "after" history for later label construction.
    "min_month_before": 3,
    "min_month_after": 3,

    # Missingness only applied to non-critical fields (a core-banking feed
    # would rarely lose balance/transaction data, but engagement/service
    # metrics realistically have reporting gaps).
    "missing_rate": 0.04,  # 3-5% requested -> use the midpoint

    "counterparty_txns_per_month": (5, 15),
    "competitor_bank_prefixes": ["ICIC", "SBIN", "AXIS", "KKBK"],   # ICICI, SBI, Axis, Kotak
    "other_bank_prefixes": ["PUNB", "BARB", "IDIB", "UBIN", "CNRB", "YESB"],
    "own_bank_prefix": "HDFC",
    "baseline_competitor_share": 0.09,   # ~8-10% baseline, per spec
    "max_extra_competitor_share": 0.38,  # additional share leaked at full erosion
}

# Segment -> baseline AMB range (INR Crores) and transaction velocity range.
SEGMENT_PROFILE = {
    "SME": {
        "amb_cr": (0.3, 4.0),
        "velocity": (3.0, 6.0),          # monthly turnover / AMB
        "counterparty_txns": (5, 9),
        "trade_finance_prob": 0.15,
        "payroll_book_prob": 0.55,
        "digital_logins": (5, 25),
    },
    "Mid-Corporate": {
        "amb_cr": (4.0, 25.0),
        "velocity": (3.5, 7.0),
        "counterparty_txns": (7, 12),
        "trade_finance_prob": 0.45,
        "payroll_book_prob": 0.70,
        "digital_logins": (10, 40),
    },
    "Large Corporate": {
        "amb_cr": (25.0, 150.0),
        "velocity": (2.5, 6.0),
        "counterparty_txns": (10, 15),
        "trade_finance_prob": 0.75,
        "payroll_book_prob": 0.85,
        "digital_logins": (20, 60),
    },
}


def _sample_categorical(rng: np.random.Generator, mapping: dict, size: int) -> np.ndarray:
    cats = list(mapping.keys())
    probs = np.array(list(mapping.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(cats, size=size, p=probs)


def _smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    """0->1 S-curve ramp over x in [0,1], flat outside. Gives gradual decline
    curves instead of a straight line or a sudden cliff."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------------------
# 1. Customer master
# ---------------------------------------------------------------------------


def generate_customers(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """
    Static customer master - the kind of record that would live in
    CRM/core-banking and doesn't change month to month. Everything that
    evolves over time lives in the monthly panel instead.
    """
    n = cfg["n_customers"]
    customer_ids = [f"CIB{100000 + i}" for i in range(n)]

    segment = _sample_categorical(rng, cfg["segment_mix"], n)
    branch_id = [f"BR{b:03d}" for b in rng.integers(1, cfg["n_branches"] + 1, size=n)]
    rm_id = [f"RM{r:03d}" for r in rng.integers(1, cfg["n_rms"] + 1, size=n)]

    relationship_tenure_months = rng.integers(6, 181, size=n)
    end_period = pd.Period(cfg["start_period"], freq="M") + cfg["n_months"] - 1
    account_open_date = [
        (end_period.to_timestamp() - pd.DateOffset(months=int(v))).date()
        for v in relationship_tenure_months
    ]

    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "account_open_date": account_open_date,
            "segment": segment,
            "branch_id": branch_id,
            "relationship_tenure_months": relationship_tenure_months,
            "rm_id": rm_id,
        }
    )


# ---------------------------------------------------------------------------
# 2. Ground truth cohort assignment
# ---------------------------------------------------------------------------


def assign_ground_truth_cohorts(customers_df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """
    Assign each customer to one of the four cohorts and, for the two
    deterioration cohorts, pick when/how the decline unfolds. This table is
    the answer key - kept separate from every feature/model table.
    """
    n = len(customers_df)
    cohort = _sample_categorical(rng, cfg["cohort_mix"], n)

    n_months = cfg["n_months"]
    lo, hi = cfg["min_month_before"], n_months - cfg["min_month_after"]

    start_month = np.full(n, -1, dtype=int)
    ramp_months = np.full(n, -1, dtype=int)
    floor = np.full(n, np.nan)
    seasonal_dip_cycle = np.full(n, -1, dtype=int)
    seasonal_dip_frac = np.full(n, np.nan)

    grad_mask = cohort == "gradual_deterioration"
    n_grad = grad_mask.sum()
    start_month[grad_mask] = rng.integers(lo, hi - 4, size=n_grad)  # leave room for a 4-8mo ramp
    ramp_months[grad_mask] = rng.integers(4, 9, size=n_grad)
    floor[grad_mask] = rng.uniform(0.35, 0.65, size=n_grad)  # fraction of baseline retained

    sudden_mask = cohort == "sudden_deterioration"
    n_sudden = sudden_mask.sum()
    start_month[sudden_mask] = rng.integers(lo, hi, size=n_sudden)
    ramp_months[sudden_mask] = rng.integers(1, 3, size=n_sudden)  # 1-2 month sharp drop
    floor[sudden_mask] = rng.uniform(0.15, 0.35, size=n_sudden)  # deeper, plateaus fast

    seasonal_mask = cohort == "seasonal_false_positive"
    n_seasonal = seasonal_mask.sum()
    seasonal_dip_cycle[seasonal_mask] = rng.integers(3, 5, size=n_seasonal)  # every 3-4 months
    seasonal_dip_frac[seasonal_mask] = rng.uniform(0.25, 0.45, size=n_seasonal)  # dip depth, recovers next month

    return pd.DataFrame(
        {
            "customer_id": customers_df["customer_id"].to_numpy(),
            "ground_truth_cohort": cohort,
            "deterioration_start_month": start_month,
            "deterioration_ramp_months": ramp_months,
            "deterioration_floor": floor,
            "seasonal_dip_cycle_months": seasonal_dip_cycle,
            "seasonal_dip_fraction": seasonal_dip_frac,
        }
    )


def _erosion_curve(cohort: str, start: int, ramp: int, floor: float, n_months: int) -> np.ndarray:
    """
    Returns an array of length n_months: erosion_factor in [0, 1-floor],
    0 = perfectly healthy. Used as the single shared driver behind every
    metric affected by genuine deterioration, so declines across signals
    stay correlated instead of independent.
    """
    erosion = np.zeros(n_months)
    if cohort in ("gradual_deterioration", "sudden_deterioration"):
        months = np.arange(n_months)
        progress = (months - start) / max(ramp, 1e-6)
        erosion = (1.0 - floor) * _smoothstep(progress)
        erosion = np.where(months < start, 0.0, erosion)
    return erosion


# ---------------------------------------------------------------------------
# 3. Monthly behavioral panel
# ---------------------------------------------------------------------------


def generate_monthly_panel(
    customers_df: pd.DataFrame, ground_truth_df: pd.DataFrame, cfg: dict, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds the monthly panel AND the counterparty transaction detail table
    together (per customer-month) so both are driven by the same erosion
    curve and stay internally consistent (e.g. a customer whose balance is
    eroding also shows rising competitor-bank counterparty share that
    month, not an unrelated random one).
    """
    n_months = cfg["n_months"]
    start_period = pd.Period(cfg["start_period"], freq="M")
    months = [str(start_period + m) for m in range(n_months)]
    gt_idx = ground_truth_df.set_index("customer_id")

    panel_rows = []
    cp_rows = []
    txn_counter = 0

    for _, cust in customers_df.iterrows():
        cid = cust["customer_id"]
        segment = cust["segment"]
        profile = SEGMENT_PROFILE[segment]
        gt = gt_idx.loc[cid]
        cohort = gt["ground_truth_cohort"]

        # --- per-customer baseline ("healthy") levels ---
        base_amb = rng.uniform(*profile["amb_cr"])
        velocity = rng.uniform(*profile["velocity"])
        base_credit_value = base_amb * velocity
        base_debit_value = base_credit_value * rng.uniform(0.90, 1.02)
        base_digital = rng.uniform(*profile["digital_logins"])
        base_mobile_pct = rng.uniform(40, 85)
        has_trade_finance = rng.random() < profile["trade_finance_prob"]
        base_trade_util = rng.uniform(0.30, 0.75) if has_trade_finance else np.nan
        has_payroll_book = rng.random() < profile["payroll_book_prob"]
        base_payroll_amount = base_amb * rng.uniform(0.15, 0.45) if has_payroll_book else 0.0
        base_complaint_rate = rng.uniform(0.05, 0.20)
        base_ticket_rate = rng.uniform(0.05, 0.25)
        cp_txn_lo, cp_txn_hi = profile["counterparty_txns"]

        # Per-metric erosion weights: correlated with the shared erosion
        # curve, but not identical, via a small random jitter per customer
        # plus each metric's own weight (balance is most exposed, sticky
        # products like trade finance / payroll erode more slowly).
        w_balance = 1.0
        w_txn = rng.normal(0.90, 0.08)
        w_digital = rng.normal(0.80, 0.12)
        w_mobile = rng.normal(0.70, 0.12)
        w_payroll = rng.normal(0.55, 0.10)
        w_trade = rng.normal(0.45, 0.10)

        erosion = _erosion_curve(
            cohort, gt["deterioration_start_month"], gt["deterioration_ramp_months"], gt["deterioration_floor"], n_months
        )

        # Sudden-deterioration payroll migration event: at the start month,
        # if the customer runs payroll through HDFC, it abruptly stops -
        # this IS the triggering event for that cohort, not a side effect.
        payroll_migration_month = int(gt["deterioration_start_month"]) if cohort == "sudden_deterioration" else None

        for m_idx, month in enumerate(months):
            e = float(np.clip(erosion[m_idx], 0.0, 1.0))

            # --- balance, with quarter-end window-dressing seasonality ---
            period_month = (start_period + m_idx).month
            qtr_bump = 1.08 if period_month in (3, 6, 9, 12) else 1.0
            amb = base_amb * (1 - w_balance * e) * qtr_bump * rng.lognormal(0.0, 0.05)

            # seasonal_false_positive: a periodic, self-recovering dip (tax
            # payment / dividend payout) - NOT genuine deterioration.
            if cohort == "seasonal_false_positive":
                cycle = int(gt["seasonal_dip_cycle_months"])
                if (m_idx % cycle) == cycle - 1:  # the dip month in each cycle
                    amb = amb * (1 - float(gt["seasonal_dip_fraction"]))

            # --- transactions (count + value, split debit/credit) ---
            txn_mult = max(0.05, 1 - w_txn * e) * rng.lognormal(0.0, 0.10)
            credit_value = base_credit_value * txn_mult
            debit_value = base_debit_value * txn_mult
            # the seasonal dip month reflects a real, large one-off payment,
            # not reduced activity - debit value actually spikes that month.
            if cohort == "seasonal_false_positive" and (m_idx % int(gt["seasonal_dip_cycle_months"])) == int(gt["seasonal_dip_cycle_months"]) - 1:
                debit_value = debit_value + base_amb * float(gt["seasonal_dip_fraction"])
            credit_count = max(1, int(rng.poisson(lam=max(credit_value / max(base_amb, 0.1), 1) * 6)))
            debit_count = max(1, int(rng.poisson(lam=max(debit_value / max(base_amb, 0.1), 1) * 6)))

            # --- digital engagement (correlated with, not identical to, txn decline) ---
            digital_logins = max(0, int(base_digital * max(0.05, 1 - w_digital * e) * rng.normal(1.0, 0.12)))
            mobile_pct = float(np.clip(base_mobile_pct * max(0.10, 1 - w_mobile * e) * rng.normal(1.0, 0.08), 0, 100))

            # --- payroll ---
            if payroll_migration_month is not None and has_payroll_book and m_idx >= payroll_migration_month:
                payroll_flag, payroll_amount = 0, 0.0
            elif has_payroll_book:
                payroll_amount = base_payroll_amount * max(0.0, 1 - w_payroll * e) * rng.lognormal(0.0, 0.04)
                payroll_flag = int(payroll_amount > 0.01)
            else:
                payroll_flag, payroll_amount = 0, 0.0

            # --- trade finance utilization (sticky; NaN if no facility) ---
            if has_trade_finance:
                trade_util = float(np.clip(base_trade_util * max(0.0, 1 - w_trade * e) * rng.normal(1.0, 0.05), 0, 1)) * 100
            else:
                trade_util = np.nan

            # --- complaints / service tickets: rise mildly with erosion ---
            complaint_count = int(rng.poisson(lam=base_complaint_rate * (1 + 2.5 * e)))
            service_ticket_count = int(rng.poisson(lam=base_ticket_rate * (1 + 2.0 * e)))

            panel_rows.append(
                {
                    "customer_id": cid,
                    "month": month,
                    "average_monthly_balance": round(amb, 3),
                    "transaction_count_debit": debit_count,
                    "transaction_count_credit": credit_count,
                    "transaction_value_debit": round(debit_value, 3),
                    "transaction_value_credit": round(credit_value, 3),
                    "digital_login_count": digital_logins,
                    "mobile_banking_txn_pct": round(mobile_pct, 2),
                    "payroll_credit_flag": payroll_flag,
                    "payroll_credit_amount": round(payroll_amount, 3),
                    "trade_finance_utilization_pct": round(trade_util, 2) if not np.isnan(trade_util) else np.nan,
                    "complaint_count": complaint_count,
                    "service_ticket_count": service_ticket_count,
                }
            )

            # --- counterparty transactions for this customer-month ---
            n_cp = rng.integers(cp_txn_lo, cp_txn_hi + 1)
            competitor_share = min(
                0.95,
                cfg["baseline_competitor_share"] + cfg["max_extra_competitor_share"] * e,
            )
            for _ in range(n_cp):
                txn_counter += 1
                roll = rng.random()
                if roll < competitor_share:
                    bank_prefix = rng.choice(cfg["competitor_bank_prefixes"])
                    is_competitor = True
                elif roll < competitor_share + 0.35:
                    bank_prefix = cfg["own_bank_prefix"]
                    is_competitor = False
                else:
                    bank_prefix = rng.choice(cfg["other_bank_prefixes"])
                    is_competitor = False
                cp_rows.append(
                    {
                        "customer_id": cid,
                        "month": month,
                        "txn_id": f"TXN{txn_counter:08d}",
                        "counterparty_bank_ifsc_prefix": bank_prefix,
                        "is_competitor_bank": is_competitor,
                        "txn_type": rng.choice(["debit", "credit"], p=[0.55, 0.45]),
                        "txn_value": round(float(base_amb * rng.lognormal(-1.5, 0.7)), 3),
                    }
                )

    panel_df = pd.DataFrame(panel_rows)
    cp_df = pd.DataFrame(cp_rows)

    # --- Realistic missingness on NON-CRITICAL fields only (3-5%) ---
    non_critical_cols = [
        "digital_login_count",
        "mobile_banking_txn_pct",
        "trade_finance_utilization_pct",
        "service_ticket_count",
    ]
    for col in non_critical_cols:
        mask = rng.random(len(panel_df)) < cfg["missing_rate"]
        panel_df.loc[mask, col] = np.nan

    return panel_df, cp_df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    cfg = SIM_CONFIG

    print("Generating customer master...")
    customers_df = generate_customers(cfg, rng)

    print("Assigning ground-truth cohorts...")
    ground_truth_df = assign_ground_truth_cohorts(customers_df, cfg, rng)

    print("Generating monthly panel + counterparty transactions (slow step)...")
    panel_df, cp_df = generate_monthly_panel(customers_df, ground_truth_df, cfg, rng)

    customers_df.to_parquet(CUSTOMERS_FILE, index=False)
    panel_df.to_parquet(MONTHLY_PANEL_FILE, index=False)
    cp_df.to_parquet(COUNTERPARTY_TXNS_FILE, index=False)
    ground_truth_df.to_parquet(GROUND_TRUTH_FILE, index=False)

    print("\nSaved:")
    print(f"  {CUSTOMERS_FILE}         ({len(customers_df):,} customers)")
    print(f"  {MONTHLY_PANEL_FILE}     ({len(panel_df):,} customer-months)")
    print(f"  {COUNTERPARTY_TXNS_FILE} ({len(cp_df):,} counterparty transactions)")
    print(f"  {GROUND_TRUTH_FILE}      ({len(ground_truth_df):,} customers)")

    print("\nCohort distribution:")
    print(ground_truth_df["ground_truth_cohort"].value_counts())
    print("\nCompetitor-bank share of counterparty transactions, overall: "
          f"{cp_df['is_competitor_bank'].mean():.1%}")


if __name__ == "__main__":
    main()
