#!/usr/bin/env bash
# Full experiment schedule (resumable; finished or in-flight runs are skipped). Run under nohup.
#   ML-1M: CORE x seeds 0-2, SECONDARY x seed 0, negatives heatmap x seed 0   (ML_JOBS parallel)
#   Amazon'23 Industrial_and_Scientific, 10k-user subsample: AMAZON x seeds 0-2   (AMZ_JOBS parallel)
set -u
cd "$(dirname "$0")/.."
# interpreter: $PY if set, else the in-repo .venv, else python3 on PATH
PY=${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python3)}
cfg() { PYTHONPATH=. $PY -c "import tasr.configs as c; print(' '.join($1))"; }
ml() { DATA=data/processed/ml1m.npz MAXLEN=100 "$@"; }
amz() { DATA=data/processed/amz_Industrial_and_Scientific_u10k.npz MAXLEN=50 "$@"; }
( ONLY="$(cfg c.CORE)" SEEDS="0 1 2" JOBS=${ML_JOBS:-4} ml scripts/run_grid.sh
  ONLY="$(cfg c.SECONDARY)" SEEDS="0" JOBS=${ML_JOBS:-4} ml scripts/run_grid.sh
  ONLY="$(cfg "[k for k in c.CONFIGS if k.startswith('neg:')]")" SEEDS="0" JOBS=${ML_JOBS:-4} ml scripts/run_grid.sh ) &
( ONLY="$(cfg c.AMAZON)" SEEDS="0 1 2" JOBS=${AMZ_JOBS:-2} amz scripts/run_grid.sh ) &
wait
echo "$(date '+%F %T') ALL DONE" >> logs/grid.log
