"""
Leakage-safety tests for src/features/utils.py.

The single most important property of every feature in this project is
that it never uses data from a month AFTER the one being scored — that's
what makes the forward-looking labels (src/labeling/labels.py) valid
targets to train on at all. These tests prove it directly: change only
FUTURE rows of a synthetic panel and assert that a trailing feature's value
at an earlier month is completely unaffected. A leakage bug (e.g. an
off-by-one in a rolling window) would fail these tests immediately, rather
than silently inflating every model's reported accuracy.
"""

import numpy as np
import pandas as pd

from src.features.utils import rolling_mean, rolling_slope, pct_change_lag, momentum_ratio


def _make_panel(values: list[float]) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame({
        "customer_id": ["C1"] * n,
        "month": [f"2024-{i+1:02d}" for i in range(n)],
        "value": values,
    })


def test_rolling_mean_unaffected_by_future_values():
    base_values = [10, 12, 11, 13, 14, 15, 16, 17]
    df_original = _make_panel(base_values)
    result_original = rolling_mean(df_original, "value", window=3)

    # Mutate ONLY the values at month_idx >= 5 (the future, relative to month_idx=4)
    mutated_values = base_values.copy()
    for i in range(5, len(mutated_values)):
        mutated_values[i] = 999999.0
    df_mutated = _make_panel(mutated_values)
    result_mutated = rolling_mean(df_mutated, "value", window=3)

    # The rolling mean AT month_idx=4 (which only looks at months 2,3,4) must
    # be identical whether or not months 5+ were changed.
    assert result_original.iloc[4] == result_mutated.iloc[4]
    assert result_original.iloc[3] == result_mutated.iloc[3]
    # Sanity check the mutation actually changed something later on, so this
    # test would have failed if leakage were present.
    assert result_original.iloc[6] != result_mutated.iloc[6]


def test_rolling_slope_unaffected_by_future_values():
    base_values = [5, 6, 7, 8, 9, 10, 11]
    df_original = _make_panel(base_values)
    result_original = rolling_slope(df_original, "value", window=3)

    mutated_values = base_values.copy()
    mutated_values[-1] = -500.0  # change only the LAST (future-most) month
    df_mutated = _make_panel(mutated_values)
    result_mutated = rolling_slope(df_mutated, "value", window=3)

    # Every month except the mutated one and windows that include it should
    # be unaffected. month_idx=3 (window = months 1,2,3) is unaffected.
    assert result_original.iloc[3] == result_mutated.iloc[3]


def test_pct_change_lag_unaffected_by_future_values():
    base_values = [100, 110, 105, 120, 130]
    df_original = _make_panel(base_values)
    result_original = pct_change_lag(df_original, "value", periods=2)

    mutated_values = base_values.copy()
    mutated_values[-1] = 0.0
    df_mutated = _make_panel(mutated_values)
    result_mutated = pct_change_lag(df_mutated, "value", periods=2)

    # month_idx=2 (compares month 2 to month 0) doesn't depend on month 4 at all
    assert result_original.iloc[2] == result_mutated.iloc[2]


def test_momentum_ratio_direction():
    """momentum_ratio should be positive when the recent window is running
    ABOVE the longer baseline (improving) and negative when below
    (declining) — getting this backwards would flip the sign of several
    features' business meaning (e.g. 'declining' would read as 'growing')."""
    short = pd.Series([120.0])
    long = pd.Series([100.0])
    assert momentum_ratio(short, long).iloc[0] > 0  # running above baseline

    short_declining = pd.Series([80.0])
    assert momentum_ratio(short_declining, long).iloc[0] < 0  # running below baseline
