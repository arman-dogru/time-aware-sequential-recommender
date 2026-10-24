"""Dataset timeline figures: weekly event volume and inter-event gap distribution for both datasets.
Writes docs/figures/data_timeline.png and results/data_stats.json."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DS = {"ml1m": "MovieLens-1M", "amz_Industrial_and_Scientific": "Amazon'23 Industrial & Scientific"}
os.makedirs("docs/figures", exist_ok=True)
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
stats = {}
for col, (ds, title) in enumerate(DS.items()):
    z = np.load(f"data/processed/{ds}.npz")
    ts, off = z["ts"], z["offsets"]
    week = (ts - ts.min()) // (7 * 86400)
    cnt = np.bincount(week)
    dates = np.datetime64(int(ts.min()), "s") + np.arange(len(cnt)) * np.timedelta64(7, "D")
    ax[0, col].plot(dates, cnt, lw=1)
    ax[0, col].set_yscale("log"); ax[0, col].set_title(f"{title}: events per week"); ax[0, col].set_ylabel("events / week")
    same_user = np.ones(len(ts) - 1, bool); same_user[off[1:-1] - 1] = False
    gaps = np.diff(ts)[same_user]
    # gap from last history event to the test target (what the query-time features see at inference)
    last_gap = ts[off[1:] - 1] - ts[off[1:] - 2]
    ax[1, col].hist(np.maximum(gaps, 0.6), bins=np.logspace(-0.3, 9, 60), alpha=0.6, label="all consecutive gaps")
    ax[1, col].hist(np.maximum(last_gap, 0.6), bins=np.logspace(-0.3, 9, 60), alpha=0.6, label="last history event -> test target")
    for s, lab in ((60, "1 min"), (3600, "1 h"), (86400, "1 day"), (30 * 86400, "30 d"), (365 * 86400, "1 y")):
        ax[1, col].axvline(s, color="grey", lw=0.6, ls=":"); ax[1, col].text(s, ax[1, col].get_ylim()[1] * 0.5, lab, fontsize=7, rotation=90)
    ax[1, col].set_xscale("log"); ax[1, col].set_yscale("log"); ax[1, col].set_xlabel("gap (seconds, same-second gaps drawn at 0.6 s)")
    ax[1, col].set_title(f"{title}: inter-event gaps"); ax[1, col].legend(fontsize=8)
    stats[ds] = dict(users=int(len(off) - 1), items=int(z["n_items"]), events=int(len(ts)),
                     first=str(np.datetime64(int(ts.min()), "s")), last=str(np.datetime64(int(ts.max()), "s")),
                     gap_same_second=float(np.mean(gaps == 0)), gap_lt_1min=float(np.mean(gaps < 60)),
                     gap_gt_1day=float(np.mean(gaps > 86400)), test_gap_lt_1min=float(np.mean(last_gap < 60)),
                     test_gap_gt_1day=float(np.mean(last_gap > 86400)), median_len=float(np.median(np.diff(off))))
fig.tight_layout(); fig.savefig("docs/figures/data_timeline.png", dpi=110)
json.dump(stats, open("results/data_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
