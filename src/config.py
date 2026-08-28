"""
Central configuration for the CIB Early Warning System project.

WHY a single config module:
Every path and random seed lives here instead of being hard-coded inside
individual scripts, for two reasons that matter for this project specifically:
  1. Reproducibility - one RANDOM_SEED drives every stage, so re-running the
     pipeline end-to-end always yields the same synthetic dataset and model.
  2. Swappability - Phase 1 builds this on synthetic data, but as long as a
     real core-banking extract is mapped onto the same file/column names
     defined here, nothing downstream (features, models, dashboard) needs
     to change.

Simulation-specific parameters (cohort mix, sample size, etc.) live next to
the generator in src/data_generation/, since they're only meaningful there.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Raw synthetic data (Phase 1 output). Parquet keeps monthly-panel I/O fast
# and preserves dtypes (important once we have NaNs mixed with counterparty
# list columns) better than CSV.
CUSTOMERS_FILE = DATA_RAW_DIR / "customers.parquet"
MONTHLY_PANEL_FILE = DATA_RAW_DIR / "monthly_panel.parquet"
COUNTERPARTY_TXNS_FILE = DATA_RAW_DIR / "counterparty_transactions.parquet"
GROUND_TRUTH_FILE = DATA_RAW_DIR / "ground_truth_cohorts.parquet"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
