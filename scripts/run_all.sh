#!/usr/bin/env bash
# Full experiment schedule (resumable; finished runs are skipped). Run under nohup.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-../tasr-venv/bin/python}
MAIN=$(PYTHONPATH=. $PY -c "from tasr.configs import MAIN; print(' '.join(MAIN))")
AMZ=$(PYTHONPATH=. $PY -c "from tasr.configs import AMAZON; print(' '.join(AMAZON))")
NEG=$(PYTHONPATH=. $PY -c "from tasr.configs import CONFIGS; print(' '.join(c for c in CONFIGS if c.startswith('neg:')))")
ml() { DATA=data/processed/ml1m.npz MAXLEN=100 "$@"; }
amz() { DATA=data/processed/amz_Industrial_and_Scientific.npz MAXLEN=50 "$@"; }
# ML-1M: main grid x 3 seeds, then the negative-sampling heatmap (seed 0), then 3 seeds for the planned neg comparisons
( ONLY="$MAIN" SEEDS="0 1 2" JOBS=${ML_JOBS:-4} ml scripts/run_grid.sh
  ONLY="$NEG" SEEDS="0" JOBS=${ML_JOBS:-4} ml scripts/run_grid.sh
  ONLY="neg:S512-H16 neg:S512-H0 neg:S512-H16-nopa" SEEDS="1 2" JOBS=${ML_JOBS:-4} ml scripts/run_grid.sh ) &
# Amazon Industrial_and_Scientific: subset x 3 seeds
( while [[ ! -f data/processed/amz_Industrial_and_Scientific_incr_flags.npz ]]; do sleep 60; done
  ONLY="$AMZ" SEEDS="0 1 2" JOBS=${AMZ_JOBS:-2} amz scripts/run_grid.sh ) &
wait
echo "$(date '+%F %T') ALL DONE" >> logs/grid.log
