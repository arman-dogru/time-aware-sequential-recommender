#!/usr/bin/env bash
# Runs every (config, seed) in tasr/configs.py that has no runs/<cfg>/seed<k>/metrics.json yet.
# Resumable; JOBS parallel workers x THREADS torch threads each.  Usage: SEEDS="0 1 2" JOBS=4 scripts/run_grid.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-../tasr-venv/bin/python}
SEEDS=${SEEDS:-"0 1 2"}; JOBS=${JOBS:-4}; THREADS=${THREADS:-2}; ONLY=${ONLY:-}
mkdir -p logs
list=$(PYTHONPATH=. $PY -c "from tasr.configs import CONFIGS; print('\n'.join(CONFIGS))")
for s in $SEEDS; do for c in $list; do
  [[ -n "$ONLY" && ! " $ONLY " =~ " $c " ]] && continue
  [[ -f "runs/$c/seed$s/metrics.json" ]] && continue
  echo "$c $s"
done; done | while read -r c s; do
  while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do sleep 5; done
  args=$(PYTHONPATH=. $PY -c "from tasr.configs import CONFIGS, COMMON; print(' '.join(COMMON + CONFIGS['$c']))")
  echo "$(date '+%F %T') start $c seed$s" >> logs/grid.log
  ( PYTHONPATH=. $PY -m tasr.train --name "$c" --seed "$s" --threads "$THREADS" $args > "logs/${c}_s$s.log" 2>&1 \
      && echo "$(date '+%F %T') done $c seed$s" >> logs/grid.log \
      || echo "$(date '+%F %T') FAIL $c seed$s" >> logs/grid.log ) &
done
wait
