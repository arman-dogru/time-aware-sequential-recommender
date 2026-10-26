"""Tune (on validation) and evaluate the non-neural baselines; writes results/baselines.json and
per-user test ranks (results/ranks/<name>.npy) plus ensemble-union hit flags for incremental recall."""
import json, os
import numpy as np
import torch

from tasr.data import SeqData
from tasr.ensemble import Ensemble
from tasr.metrics import summarize, KS

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed/ml1m.npz")
ap.add_argument("--max-len", type=int, default=100)
args = ap.parse_args()
torch.set_num_threads(2)
d = SeqData(args.data, args.max_len)
R = os.path.join("results", os.path.basename(args.data).replace(".npz", ""))
os.makedirs(f"{R}/ranks", exist_ok=True)
tune = {}
# tune on (at most) 6,040 validation users so the 51k-user Amazon set stays cheap; ML-1M uses all users
tune_users = np.random.default_rng(0).permutation(d.n_users)[:6040]

for m in (1, 3, 10, 20):
    for dec in (0.5, 0.8, 1.0):
        for win in ((7, 30, 90) if (m, dec) == (10, 0.8) else (30,)):  # members are separable: tune window once
            e = Ensemble(d, m, dec, win)
            r = e.ranks("valid", users=tune_users, rrf=False)
            tune[f"m={m},decay={dec},win={win}"] = {k: summarize(v)["NDCG@10"] for k, v in r.items()}
            print(f"m={m} decay={dec} win={win}", tune[f"m={m},decay={dec},win={win}"], flush=True)
knn_key = max(tune, key=lambda k: tune[k]["ItemKNN"])
win_key = max(tune, key=lambda k: tune[k]["RecentPop"])
m = int(knn_key.split(",")[0][2:]); dec = float(knn_key.split(",")[1][6:]); win = int(win_key.split(",")[2][4:])
print("chosen", m, dec, win)
e = Ensemble(d, m, dec, win)
test = e.ranks("test")
res = {"max_len": args.max_len, "chosen": {"knn_m": m, "knn_decay": dec, "recentpop_window_days": win}, "valid_tuning_ndcg10": tune, "n_tune_users": int(len(tune_users)),
       "test": {k: summarize(v) for k, v in test.items()}}
for k, v in test.items():
    np.save(f"{R}/ranks/{k}.npy", v.astype(np.int32))
union = {f"K{K}": (np.minimum.reduce([test["Pop"], test["RecentPop"], test["ItemKNN"]]) <= K) for K in KS}
res["test"]["Ensemble-union"] = {f"R@{K}": float(union[f"K{K}"].mean()) for K in KS}
np.savez_compressed(f"{R}/ranks/ensemble_union_hits.npz", **union)
json.dump(res, open(f"{R}/baselines.json", "w"), indent=1)
print(json.dumps(res["test"], indent=1))
