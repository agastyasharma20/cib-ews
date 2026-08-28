"""Tunable knobs for modeling."""

from src.config import RANDOM_SEED

TARGET_HORIZONS = ["deteriorates_in_30d", "deteriorates_in_60d", "deteriorates_in_90d"]

CATEGORICAL_FEATURES = ["segment"]

# class_weight="balanced" instead of oversampling/undersampling: reweights
# the loss inversely to class frequency without throwing away any rows or
# fabricating synthetic ones — a reasonable default for a ~6-15% positive
# rate baseline, and it keeps the comparison with the Phase 4 tree model
# (which will use scale_pos_weight the same way) apples-to-apples.
LOGREG_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 2000,
    "C": 1.0,
    "random_state": RANDOM_SEED,
}

LIFT_N_BINS = 10
