"""Tune (on validation) and evaluate the non-neural baselines; writes results/baselines.json and
per-user test ranks (results/ranks/<name>.npy) plus ensemble-union hit flags for incremental recall."""
import json, os
import numpy as np
import torch

from tasr.data import SeqData
from tasr.ensemble import Ensemble
from tasr.metrics import summarize, KS

torch.set_num_threads(2)
d = SeqData("data/processed/ml1m.npz", 100)
os.makedirs("results/ranks", exist_ok=True)
tune = {}
for m in (1, 3, 10, 20):
    for dec in (0.5, 0.8, 1.0):
        for win in ((7, 30, 90) if (m, dec) == (10, 0.8) else (30,)):  # members are separable: tune window once
            e = Ensemble(d, m, dec, win)
            r = e.ranks("valid")
            tune[f"m={m},decay={dec},win={win}"] = {k: summarize(v)["NDCG@10"] for k, v in r.items()}
            print(f"m={m} decay={dec} win={win}", tune[f"m={m},decay={dec},win={win}"], flush=True)
knn_key = max(tune, key=lambda k: tune[k]["ItemKNN"])
win_key = max(tune, key=lambda k: tune[k]["RecentPop"])
m = int(knn_key.split(",")[0][2:]); dec = float(knn_key.split(",")[1][6:]); win = int(win_key.split(",")[2][4:])
print("chosen", m, dec, win)
e = Ensemble(d, m, dec, win)
test = e.ranks("test")
res = {"chosen": {"knn_m": m, "knn_decay": dec, "recentpop_window_days": win}, "valid_tuning_ndcg10": tune,
       "test": {k: summarize(v) for k, v in test.items()}}
for k, v in test.items():
    np.save(f"results/ranks/{k}.npy", v.astype(np.int32))
union = {f"K{K}": (np.minimum.reduce([test["Pop"], test["RecentPop"], test["ItemKNN"]]) <= K) for K in KS}
res["test"]["Ensemble-union"] = {f"R@{K}": float(union[f"K{K}"].mean()) for K in KS}
np.savez_compressed("results/ranks/ensemble_union_hits.npz", **union)
json.dump(res, open("results/baselines.json", "w"), indent=1)
print(json.dumps(res["test"], indent=1))
