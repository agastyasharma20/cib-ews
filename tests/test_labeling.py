"""
Tests for the deterioration-labeling framework (src/labeling/).

These are the tests that matter most for this project: if the percentile
ranking direction, the seasonal false-positive filter, or the forward-label
censoring were ever silently broken by a refactor, every downstream model
and every number in docs/methodology.md would quietly become wrong without
any error being raised. That's exactly the kind of bug a test suite exists
to catch that manual inspection of results won't.
"""

import numpy as np
import pandas as pd

from src.labeling.deterioration_index import _percentile_decline_score
from src.labeling.seasonal_filter import flag_confirmed_deterioration
from src.labeling.labels import build_forward_labels
from src.labeling.config import SEASONAL_FILTER


# ---------------------------------------------------------------------------
# Percentile ranking direction
# ---------------------------------------------------------------------------
def test_percentile_decline_score_ranks_worst_decline_highest():
    """The customer with the most NEGATIVE pct_change (worst decline) must
    get a decline_score near 1.0; the one with the most positive change
    (growth) must get a score near 0.0 — this is the entire premise of
    'percentile-ranked, not a flat threshold' from docs/deterioration_
    definition.md, so getting the direction backwards would silently invert
    every risk signal in the project."""
    df = pd.DataFrame({
        "segment": ["SME"] * 5,
        "month_idx": [0] * 5,
        "pct_change": [-0.50, -0.10, 0.0, 0.10, 0.50],  # worst decline -> best growth
    })
    scores = _percentile_decline_score(df, "pct_change")
    # Most negative pct_change -> highest decline score
    assert scores.iloc[0] == scores.max()
    # Most positive pct_change -> lowest decline score
    assert scores.iloc[-1] == scores.min()
    assert scores.is_monotonic_decreasing


def test_percentile_decline_score_is_relative_to_segment_and_month():
    """Two customers with IDENTICAL pct_change but in different segments (or
    different months) must be ranked independently — this is what lets the
    index cancel out segment-scale differences and shared seasonality."""
    df = pd.DataFrame({
        "segment": ["SME", "SME", "Large Corporate", "Large Corporate"],
        "month_idx": [0, 0, 0, 0],
        "pct_change": [-0.30, 0.10, -0.30, 0.10],
    })
    scores = _percentile_decline_score(df, "pct_change")
    # Within each segment, the -0.30 customer should score higher than the +0.10 one
    assert scores.iloc[0] > scores.iloc[1]
    assert scores.iloc[2] > scores.iloc[3]
    # The two segments' -0.30 customers should be ranked identically (each
    # is the worst within their own 2-person segment group)
    assert scores.iloc[0] == scores.iloc[2]


# ---------------------------------------------------------------------------
# Seasonal false-positive filter
# ---------------------------------------------------------------------------
def _make_customer_months(di_values, digital_scores, payroll_trade_scores, threshold_col_extra=None):
    n = len(di_values)
    df = pd.DataFrame({
        "customer_id": ["C1"] * n,
        "month_idx": range(n),
        "deterioration_index": di_values,
        "digital_decline_score": digital_scores,
        "payroll_trade_decline_score": payroll_trade_scores,
    })
    if threshold_col_extra:
        for k, v in threshold_col_extra.items():
            df[k] = v
    return df


def test_reverting_uncorroborated_breach_is_suppressed():
    """A one-month balance/DI spike that reverts within the filter's
    reversal window, with NO accompanying digital or payroll/trade decline,
    is exactly the seasonal_false_positive scenario (e.g. a tax payment) —
    it must NOT be confirmed as deterioration."""
    di = [0.3, 0.3, 0.90, 0.3, 0.3, 0.3]  # breach at index 2, reverts immediately after
    digital = [0.1] * 6   # low — not corroborating
    payroll = [0.1] * 6   # low — not corroborating
    df = _make_customer_months(di, digital, payroll)

    result = flag_confirmed_deterioration(df, threshold=0.85)
    assert result.loc[2, "breach"] == True  # noqa: E712 — raw breach did fire
    assert result.loc[2, "confirmed_deterioration"] == False  # but was filtered out


def test_reverting_but_corroborated_breach_is_kept():
    """The same reverting DI spike, but WITH accompanying digital decline,
    is a genuine (if brief) deterioration signal — two independent signals
    moving together is not a coincidence, so it should NOT be suppressed."""
    di = [0.3, 0.3, 0.90, 0.3, 0.3, 0.3]
    digital = [0.1, 0.1, 0.80, 0.1, 0.1, 0.1]  # elevated at the breach month
    payroll = [0.1] * 6
    df = _make_customer_months(di, digital, payroll)

    result = flag_confirmed_deterioration(df, threshold=0.85)
    assert result.loc[2, "confirmed_deterioration"] == True  # noqa: E712


def test_persistent_breach_is_kept_even_if_uncorroborated():
    """A breach that does NOT revert within the window (genuine ongoing
    deterioration) must be confirmed regardless of the other signals —
    the filter only suppresses REVERTING dips, never persistent ones."""
    di = [0.3, 0.3, 0.90, 0.92, 0.91, 0.93]  # breaches and stays high
    digital = [0.1] * 6
    payroll = [0.1] * 6
    df = _make_customer_months(di, digital, payroll)

    result = flag_confirmed_deterioration(df, threshold=0.85)
    assert result.loc[2, "confirmed_deterioration"] == True  # noqa: E712


# ---------------------------------------------------------------------------
# Forward label construction: censoring
# ---------------------------------------------------------------------------
def test_forward_labels_are_nan_when_insufficient_future_history():
    """A customer-month too close to the end of their observed history must
    get NaN (not False/0) for a horizon that would need future months
    beyond what's observed — silently treating 'we don't know yet' as
    'confirmed not deteriorating' would bias every model trained on it."""
    n_months = 6
    df = pd.DataFrame({
        "customer_id": ["C1"] * n_months,
        "month_idx": range(n_months),
        "confirmed_deterioration": [False] * n_months,
    })
    labeled = build_forward_labels(df, horizons={"deteriorates_in_1m": 1, "deteriorates_in_3m": 3})

    # Last month (idx 5) can't look 1 month ahead -> NaN
    assert pd.isna(labeled.loc[labeled["month_idx"] == 5, "deteriorates_in_1m"].iloc[0])
    # Month 4 CAN look 1 month ahead (to month 5) -> should be a real 0/1, not NaN
    assert not pd.isna(labeled.loc[labeled["month_idx"] == 4, "deteriorates_in_1m"].iloc[0])
    # Month 3 cannot look 3 months ahead (would need month 6, doesn't exist) -> NaN
    assert pd.isna(labeled.loc[labeled["month_idx"] == 3, "deteriorates_in_3m"].iloc[0])


def test_forward_label_true_if_any_confirmed_event_in_window():
    """deteriorates_in_3m(t) must be True if confirmed_deterioration is True
    in ANY of months t+1..t+3, not just t+3 specifically."""
    df = pd.DataFrame({
        "customer_id": ["C1"] * 6,
        "month_idx": range(6),
        "confirmed_deterioration": [False, False, True, False, False, False],
    })
    labeled = build_forward_labels(df, horizons={"deteriorates_in_2m": 2})
    # Month 0: looks ahead to months 1, 2 -> month 2 is True -> label True
    assert labeled.loc[labeled["month_idx"] == 0, "deteriorates_in_2m"].iloc[0] == True  # noqa: E712
    # Month 3: looks ahead to months 4, 5 -> both False -> label False
    assert labeled.loc[labeled["month_idx"] == 3, "deteriorates_in_2m"].iloc[0] == False  # noqa: E712
