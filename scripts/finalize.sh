#!/usr/bin/env bash
# After the grid: assert completeness, CPU bench, aggregation, figures, query-time ablation, README tables;
# fast-forward to the remote branch, commit, push (never force). WAIT=1 blocks until no trainer/runner is alive.
set -u
cd "$(dirname "$0")/.."
# interpreter: $PY if set, else the in-repo .venv, else python3 on PATH
PY=${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python3)}; export PYTHONPATH=.
BRANCH=microbench-v1
log() { echo "$(date '+%F %T') $*" | tee -a logs/grid.log; }
fail() { log "FINALIZE FAILED: $*"; exit 1; }
busy() { pgrep -f "tasr.train" >/dev/null || pgrep -f "scripts/run_grid.sh" >/dev/null || pgrep -f "scripts/run_all.sh" >/dev/null; }
if [[ "${WAIT:-0}" == 1 ]]; then
  while true; do  # require two consecutive idle polls (runner phases hand over within milliseconds)
    while busy; do sleep 120; done
    sleep 120; busy || break
  done
fi
log "finalize start (PY=$PY)"
$PY scripts/check_grid.py >> logs/grid.log 2>&1 || fail "grid incomplete, see check_grid output above; nothing committed"
$PY -m pytest -q tests > logs/pytest_final.log 2>&1 || fail "tests failed (logs/pytest_final.log)"
$PY scripts/bench.py --reps 20 > logs/bench_final.log 2>&1 || fail "bench"
$PY scripts/plot_data.py > /dev/null 2>&1 || fail "plot_data"
$PY scripts/aggregate.py --ds ml1m > logs/agg_ml1m.out 2>&1 || fail "aggregate ml1m"
$PY scripts/aggregate.py --ds amz_Industrial_and_Scientific_u10k > logs/agg_amz.out 2>&1 || fail "aggregate amz"
$PY scripts/eval_query_time.py > logs/query_time.out 2>&1 || fail "query-time ablation"
$PY scripts/make_figures.py > logs/figs.out 2>&1 || fail "make_figures"
$PY scripts/readme_tables.py > /dev/null 2>&1 || fail "readme_tables"
$PY scripts/audit.py > logs/audit.out 2>&1 || fail "audit"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$BRANCH" ]] || fail "not on $BRANCH"
git stash push -q -- results docs README.md || fail "stash"
env -u GITHUB_TOKEN git pull -q --ff-only origin "$BRANCH" || { git stash pop -q; fail "remote $BRANCH is not a fast-forward of local; resolve by hand"; }
git stash pop -q || fail "stash pop conflicted after fast-forward; resolve by hand"
git add -A results docs README.md
git commit -qm "Final grid: aggregate, figures, query-time ablation, README tables, CPU bench" || fail "nothing to commit?"
env -u GITHUB_TOKEN git push -q origin "$BRANCH" || fail "push rejected (no force-push); resolve by hand"
log "finalize done $(git rev-parse --short HEAD)"
