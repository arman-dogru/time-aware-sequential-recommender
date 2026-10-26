"""For every training target in the train window, flag whether the ensemble (union of top-K of Pop,
RecentPop and ItemKNN computed from that prefix, already-seen items excluded) MISSES it.
Used by --incr-beta (boosting-style upweighting of the ensemble's blind spots, cf. the Shopify post)."""
import argparse
import numpy as np
import torch

from tasr.data import SeqData, window_positions
from tasr.ensemble import Ensemble

ap = argparse.ArgumentParser()
ap.add_argument("--max-len", type=int, default=100)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--data", default="data/processed/ml1m.npz")
a = ap.parse_args()
torch.set_num_threads(2)
d = SeqData(a.data, a.max_len)
ens = Ensemble(d)
flags, off = [], [0]
m = ens.knn.m
for u in range(d.n_users):
    it, ts = d.seq(u)
    pos = window_positions(d, u)  # target indices
    itt = torch.from_numpy(it.astype(np.int64))
    # last-m items before each target
    idx = pos[:, None] - np.arange(m, 0, -1)[None]
    last = torch.where(torch.from_numpy(idx) >= 0, itt[np.clip(idx, 0, None)], torch.zeros(1, dtype=torch.long))
    sc_knn = ens.knn.scores_last(last) + 1e-6 * ens.pop[None]
    sc_rp = ens.rpop.scores(ts[pos])
    first = np.full(d.n_items + 1, 10**9); first[it[::-1]] = np.arange(len(it))[::-1]
    seen = torch.from_numpy(first[None, :] < pos[:, None]); seen[:, 0] = True
    hit = torch.zeros(len(pos), dtype=torch.bool)
    tgt = itt[pos]
    for sc in (ens.pop[None].expand(len(pos), -1), sc_knn, sc_rp):
        top = torch.topk(sc.masked_fill(seen, float("-inf")), a.k, 1).indices
        hit |= (top == tgt[:, None]).any(1)
    flags.append((~hit).numpy()); off.append(off[-1] + len(pos))
f = np.concatenate(flags)
out = a.data.replace(".npz", "_incr_flags.npz")
np.savez_compressed(out, flags=f, off=np.array(off), k=a.k, max_len=a.max_len)
print(f"{out}: {len(f)} train targets, ensemble miss rate @{a.k} = {f.mean():.3f}")
