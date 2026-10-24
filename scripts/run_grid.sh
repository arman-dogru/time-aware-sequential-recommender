#!/usr/bin/env bash
# Runs every (config, seed) in tasr/configs.py that has no runs/<cfg>/seed<k>/metrics.json yet.
# Resumable; JOBS parallel workers x THREADS torch threads each.  Usage: SEEDS="0 1 2" JOBS=4 scripts/run_grid.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-../tasr-venv/bin/python}
DATA=${DATA:-data/processed/ml1m.npz}; MAXLEN=${MAXLEN:-100}; DS=$(basename "$DATA" .npz)
SEEDS=${SEEDS:-"0 1 2"}; JOBS=${JOBS:-4}; THREADS=${THREADS:-2}; ONLY=${ONLY:-}
mkdir -p logs
list=$(PYTHONPATH=. $PY -c "from tasr.configs import CONFIGS; print(chr(10).join(CONFIGS))")
for s in $SEEDS; do for c in $list; do
  [[ -n "$ONLY" && ! " $ONLY " =~ " $c " ]] && continue
  [[ -f "runs/$DS/$c/seed$s/metrics.json" ]] && continue
  [[ -f "runs/$DS/$c/seed$s/RUNNING" ]] && kill -0 "$(cat "runs/$DS/$c/seed$s/RUNNING")" 2>/dev/null && continue  # another runner has it
  echo "$c $s"
done; done | while read -r c s; do
  while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do sleep 5; done
  args=$(PYTHONPATH=. $PY -c "from tasr.configs import CONFIGS, COMMON; print(' '.join(COMMON + CONFIGS['$c']))")
  [[ -f "runs/$DS/$c/seed$s/metrics.json" ]] && continue
  echo "$(date '+%F %T') start $DS $c seed$s" >> logs/grid.log
  ( mkdir -p "runs/$DS/$c/seed$s"; echo $BASHPID > "runs/$DS/$c/seed$s/RUNNING"; PYTHONPATH=. $PY -m tasr.train --data "$DATA" --max-len "$MAXLEN" --name "$c" --seed "$s" --threads "$THREADS" $args > "logs/${DS}_${c}_s$s.log" 2>&1 \
      && echo "$(date '+%F %T') done $DS $c seed$s" >> logs/grid.log \
      || echo "$(date '+%F %T') FAIL $DS $c seed$s" >> logs/grid.log ) &
done
wait
