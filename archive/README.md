# Archive Directory

This directory contains experimental scripts, deprecated documentation, and legacy code that are not part of the main project workflow but are preserved for reference.

## Contents

### `experimental/`
Experimental and alternative implementations that were used during development:

- **`run_anomaly_only.py`** - Standalone script for running only anomaly detection (useful for extreme imbalance cases)
- **`run_extreme_imbalance.py`** - Experimental script for handling extreme data imbalance
- **`main_backup.py`** - Earlier version of main.py with enhanced visualizations
- **`legacy_monolithic_implementation.py`** - Original monolithic implementation (~2500 lines) before modular refactoring

### `deprecated_docs/`
Documentation files from development phases that are no longer actively maintained:

- **`COX_CONVERGENCE_FIX.md`** - Notes on fixing Cox model convergence issues
- **`CRITICAL_FIXES_NEEDED.md`** - Historical list of fixes (now completed)
- **`IMPLEMENTATION_COMPLETE.md`** - Milestone documentation
- **`INTERPRETABILITY_BUGFIXES.md`** - Bug fix notes for interpretability features
- **`RESTRUCTURING_SUMMARY.md`** - Notes from code restructuring process
- **`README_MODULAR.md`** - Earlier version of modular README

## Why These Files Are Archived

These files served important purposes during development but are not needed for the main project:

1. **Experimental scripts** - Alternative approaches that were tested but not adopted as the primary workflow
2. **Legacy code** - Older implementations superseded by the current modular architecture
3. **Development notes** - Historical documentation of issues that have been resolved
4. **Duplicate docs** - Earlier versions of documentation files that have been updated

## Should You Use These Files?

**For normal usage:** No, use the files in the main project directories (`src/`, `scripts/`, `docs/`)

**For reference:** Yes, these files may be helpful if you want to:
- Understand the project's evolution
- Run alternative experimental approaches
- See how specific issues were resolved
- Compare the monolithic vs modular implementations

## Active Project Files

For the current, maintained project files, see:
- **Source code:** `src/`
- **Main script:** `main.py`
- **Current documentation:** `docs/`
- **Utility scripts:** `scripts/`
