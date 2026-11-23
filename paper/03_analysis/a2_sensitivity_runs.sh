#!/usr/bin/env bash
# Paper analysis A2: sensitivity of the corrected pipeline to (i) the random
# seed (HPO sampler, model, bootstrap), (ii) the monotone constraints,
# (iii) Platt calibration. Each run is the unmodified run_experiment.py with
# a config that changes exactly one key relative to configs/main.yaml
# (seed 42 = the main run). Usage: a2_sensitivity_runs.sh <python>
set -u
PY="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
for c in seed0 seed1 seed2 seed3 seed4 no_monotone no_calibration; do
  "$PY" run_experiment.py --config paper/03_analysis/configs/$c.yaml \
      > paper/03_analysis/runs/$c.log 2>&1
  echo "$c exit $?"
done
