"""filterclog — corrected, leakage-audited XGBoost pipeline for gas filter
clogging detection (master's thesis / journal paper).

The legacy monolith (archived at ``legacy/refactored_filter_system.py``) is
kept frozen as the "before" artifact; every defect it contains and the
corresponding fix in this package is recorded in ``docs/modification_log.md``.
"""

__version__ = "1.0.0"

RANDOM_SEED = 42
