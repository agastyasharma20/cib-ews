"""
Tests for src/models/action_engine.py and src/models/pfar.py.

Covers the parts of the RM-facing output an RM would actually notice if
broken: does the right action get recommended for a given reason code, and
does the priority-tier / driver-type logic behave the way
docs/methodology.md §7 claims it does.
"""

import pandas as pd

from src.models.action_engine import (
    RuleBasedActionRecommender,
    REASON_CODE_TO_ACTION,
    DEFAULT_ACTION,
)
from src.models.pfar import _driver_type_for_top_reason, assign_priority_tiers, DRIVER_TYPE_MAP


# ---------------------------------------------------------------------------
# Action engine
# ---------------------------------------------------------------------------
def test_recommender_uses_only_the_top_reason_code():
    """By design, the recommender looks at reason_codes[0] only — the 2nd
    and 3rd codes must never change the recommended action."""
    recommender = RuleBasedActionRecommender()
    action_a = recommender.recommend(["Falling balances", "Rising service complaints"])
    action_b = recommender.recommend(["Falling balances", "Reduced trade utilization"])
    assert action_a == action_b == REASON_CODE_TO_ACTION["Falling balances"]


def test_recommender_matches_every_business_specified_mapping():
    """Verbatim check against the brief's exact table — a typo in the
    lookup dict would silently recommend the wrong action to an RM."""
    recommender = RuleBasedActionRecommender()
    expected = {
        "Declining credits": "Review receivables and cash-management setup",
        "Falling balances": "Discuss liquidity requirements",
        "Reduced payroll activity": "Assess payroll migration risk",
        "Lower digital activity": "Re-engage through digital solutions",
        "Reduced trade utilization": "Explore trade finance opportunities",
        "Rising competitor-transfer share": "Proactive wallet-share conversation / pricing review",
    }
    for reason_code, expected_action in expected.items():
        assert recommender.recommend([reason_code]) == expected_action


def test_recommender_falls_back_to_default_for_unknown_or_empty():
    recommender = RuleBasedActionRecommender()
    assert recommender.recommend(["Some Made Up Reason Code"]) == DEFAULT_ACTION
    assert recommender.recommend([]) == DEFAULT_ACTION
    assert recommender.recommend(None) == DEFAULT_ACTION


# ---------------------------------------------------------------------------
# PFaR driver-type decomposition
# ---------------------------------------------------------------------------
def test_driver_type_liquidity_relationship_competitor_examples():
    assert _driver_type_for_top_reason(["Falling balances"]) == "liquidity"
    assert _driver_type_for_top_reason(["Rising service complaints"]) == "relationship"
    assert _driver_type_for_top_reason(["Rising competitor-transfer share"]) == "competitor"


def test_driver_type_falls_back_for_unmapped_reason():
    assert _driver_type_for_top_reason(["Totally Unmapped Label"]) == "relationship"  # DEFAULT_DRIVER_TYPE
    assert _driver_type_for_top_reason([]) == "relationship"


def test_every_driver_type_is_one_of_the_three_business_categories():
    assert set(DRIVER_TYPE_MAP.values()) <= {"liquidity", "relationship", "competitor"}


# ---------------------------------------------------------------------------
# Priority tiers
# ---------------------------------------------------------------------------
def test_priority_tiers_are_top_10_next_20_rest_70():
    df = pd.DataFrame({"PFaR": list(range(1, 101))})  # 1..100, evenly spread
    result = assign_priority_tiers(df)
    counts = result["priority_tier"].value_counts()

    assert counts["High"] == 10     # top 10% (values 91-100)
    assert counts["Medium"] == 20   # next 20% (values 71-90)
    assert counts["Low"] == 70      # remaining 70%

    # The single highest PFaR value must be High, the lowest must be Low.
    assert result.loc[result["PFaR"] == 100, "priority_tier"].iloc[0] == "High"
    assert result.loc[result["PFaR"] == 1, "priority_tier"].iloc[0] == "Low"
