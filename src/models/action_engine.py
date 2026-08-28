"""
RM Action Mapping — turns a customer's top reason code into a concrete,
recommended relationship-manager action.

WHY a plain rules-based lookup for now, and how it's meant to evolve:
This is deliberately the simplest thing that could work: reason code in,
action out, one row per label, editable by a business owner without
touching model code. It is NOT meant to be the final word — once the
system has been live long enough to observe RM-OUTCOME feedback (did the
RM take the recommended action, and did the account's balance/activity
actually recover afterward?), the natural upgrade is a CONTEXTUAL BANDIT
(e.g. LinUCB or Thompson sampling) that:
  - treats each reason code (or full customer context) as the "context",
  - treats the set of possible actions as "arms",
  - learns from the observed reward (retention / balance recovery) which
    action actually works best for which kind of customer, instead of a
    fixed 1:1 mapping, and
  - can explore alternative actions for a reason code rather than always
    recommending the same one.

That upgrade is NOT built here (there is no RM-outcome data yet to learn
from) — but the interface is shaped for it now: `ActionRecommender.recommend`
takes the customer's full ranked reason-code list (not just a string), so a
future `BanditActionRecommender` can use richer context (e.g. segment,
PFaR tier) without changing any call site that already calls `.recommend(...)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# --- The business-specified mapping (verbatim from the brief) ---
REASON_CODE_TO_ACTION: dict[str, str] = {
    "Declining credits": "Review receivables and cash-management setup",
    "Falling balances": "Discuss liquidity requirements",
    "Reduced payroll activity": "Assess payroll migration risk",
    "Lower digital activity": "Re-engage through digital solutions",
    "Reduced trade utilization": "Explore trade finance opportunities",
    "Rising competitor-transfer share": "Proactive wallet-share conversation / pricing review",
}

# --- Extended coverage for driver labels that exist in the feature-group
# taxonomy (src/explainability/config.py:FEATURE_TO_DRIVER_LABEL) but
# weren't in the brief's table. Kept in a SEPARATE dict so it's obvious
# which entries were explicitly specified vs. added here for completeness
# (a real business owner should review/override these, not assume they
# carry the same authority as the table above). ---
EXTENDED_REASON_CODE_TO_ACTION: dict[str, str] = {
    "Unstable balances": "Discuss liquidity requirements",
    "Declining transaction activity": "Review receivables and cash-management setup",
    "Concentrating money flow to fewer counterparties": "Proactive wallet-share conversation / pricing review",
    "Rising service complaints": "Service recovery outreach",
    "Narrowing product relationship": "Cross-sell / relationship deepening conversation",
    "Relationship tenure profile": "Standard relationship review (no acute driver)",
    "Segment-level risk profile": "Standard relationship review (no acute driver)",
}

DEFAULT_ACTION = "Standard relationship review (no acute driver)"

# Single merged lookup used at runtime — the brief's table takes precedence
# on any (currently nonexistent) key overlap.
ACTION_MAP: dict[str, str] = {**EXTENDED_REASON_CODE_TO_ACTION, **REASON_CODE_TO_ACTION}


class ActionRecommender(ABC):
    """Interface a future learned recommender (e.g. a contextual bandit)
    would also implement, so the rest of the codebase never needs to know
    which implementation is behind `.recommend(...)`."""

    @abstractmethod
    def recommend(self, reason_codes: list[str]) -> str:
        """`reason_codes` is a customer's ranked list of driver labels
        (e.g. top_3_reason_codes, highest-impact first). Returns one
        recommended action string."""
        raise NotImplementedError


class RuleBasedActionRecommender(ActionRecommender):
    """Current implementation: looks up the action for the SINGLE top-
    ranked reason code. Ignores reason codes 2 and 3 by design — the top
    driver is taken as the dominant story worth acting on; a bandit-based
    successor could use the full list as richer context instead."""

    def __init__(self, action_map: dict[str, str] = ACTION_MAP, default_action: str = DEFAULT_ACTION):
        self.action_map = action_map
        self.default_action = default_action

    def recommend(self, reason_codes: list[str]) -> str:
        if reason_codes is None or len(reason_codes) == 0:
            return self.default_action
        top_reason = reason_codes[0]
        return self.action_map.get(top_reason, self.default_action)


def recommend_actions(reason_codes_series, recommender: ActionRecommender | None = None):
    """Vectorized convenience wrapper: apply a recommender over a pandas
    Series of reason-code lists (e.g. customer_scores['top_3_reason_codes'])."""
    recommender = recommender or RuleBasedActionRecommender()
    return reason_codes_series.apply(recommender.recommend)
