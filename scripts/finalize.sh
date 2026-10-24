#!/usr/bin/env bash
# After the grid: CPU bench, aggregation, figures, README tables; commit + push to the feature branch.
# WAIT=1 blocks until no trainer / grid runner is alive.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-../tasr-venv/bin/python}; export PYTHONPATH=.
if [[ "${WAIT:-0}" == 1 ]]; then
  while pgrep -f "tasr.train" >/dev/null || pgrep -f "scripts/run_grid.sh" >/dev/null; do sleep 120; done
fi
echo "$(date '+%F %T') finalize start" >> logs/grid.log
$PY scripts/bench.py --reps 20 > logs/bench_final.log 2>&1
$PY scripts/plot_data.py > /dev/null 2>&1
$PY scripts/aggregate.py --ds ml1m > logs/agg_ml1m.out 2>&1
$PY scripts/aggregate.py --ds amz_Industrial_and_Scientific_u10k > logs/agg_amz.out 2>&1
$PY scripts/make_figures.py > logs/figs.out 2>&1
$PY scripts/readme_tables.py > /dev/null 2>&1
git add -A results docs README.md
git commit -qm "Final grid: aggregate, figures, README tables, CPU bench" && env -u GITHUB_TOKEN git push -q
echo "$(date '+%F %T') finalize done $(git rev-parse --short HEAD)" >> logs/grid.log
