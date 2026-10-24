"""Assert the scheduled grid (tasr.configs.SCHEDULE) is complete: every (dataset, config, seed) has
runs/<ds>/<cfg>/seed<k>/test_ranks.npy. Placeholder metrics.json files with a "skipped" key (intentional
CPU-budget skips) are listed but are never part of the schedule. Exit 1 with the missing list otherwise."""
import glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasr.configs import SCHEDULE

missing, ok = [], 0
for ds, blocks in SCHEDULE.items():
    for cfgs, seeds in blocks:
        for c in cfgs:
            for s in seeds:
                d = f"runs/{ds}/{c}/seed{s}"
                if os.path.exists(f"{d}/test_ranks.npy"):
                    ok += 1
                    continue
                m = f"{d}/metrics.json"
                skipped = os.path.exists(m) and "skipped" in json.load(open(m))
                missing.append(f"{d}{' (marked skipped but scheduled!)' if skipped else ''}")
skips = sorted(p for p in glob.glob("runs/*/*/seed*/metrics.json") if "skipped" in json.load(open(p)))
print(f"complete runs: {ok}; intentional skips: {len(skips)}")
for p in skips:
    print("  skip", os.path.dirname(p))
if missing:
    print(f"GRID INCOMPLETE: {len(missing)} scheduled run(s) missing:", *missing, sep="\n  ", file=sys.stderr)
    sys.exit(1)
print("GRID COMPLETE")
