"""Ensures the project root (not just tests/) is importable as `src.*`
when running `pytest` from any working directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
