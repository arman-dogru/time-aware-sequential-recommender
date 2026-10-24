"""Download and prepare an Amazon Reviews 2023 5-core category as chronological sequences
(same .npz format as scripts/prepare_ml1m.py).

Source: Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, Julian McAuley. 2024.
"Bridging Language and Items for Retrieval and Recommendation" (arXiv:2403.03952),
dataset card https://amazon-reviews-2023.github.io/ , files on the Hugging Face Hub:
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/benchmark/5core/rating_only
Default category: Industrial_and_Scientific (5-core: 50,985 users, 25,848 items, 412,947 reviews).
A review is used as a proxy for a purchase event; timestamps are in ms and converted to seconds.
Raw data is not committed.
"""
import argparse, csv, hashlib, os, sys, urllib.request
import numpy as np

BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/5core/rating_only/"
SHA256 = {"Industrial_and_Scientific": "c0cb90d6c99f5c13a298565f02743588d590e2cfa4568dfb0bd88fec1a523bb6"}

ap = argparse.ArgumentParser()
ap.add_argument("--category", default="Industrial_and_Scientific")
ap.add_argument("--raw-dir", default="data/raw")
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or f"data/processed/amz_{a.category}.npz"
os.makedirs(a.raw_dir, exist_ok=True); os.makedirs(os.path.dirname(out), exist_ok=True)
path = os.path.join(a.raw_dir, f"{a.category}.csv")
if not os.path.exists(path):
    print("downloading", BASE + a.category + ".csv", file=sys.stderr)
    urllib.request.urlretrieve(BASE + a.category + ".csv", path)
h = hashlib.sha256(open(path, "rb").read()).hexdigest()
if a.category in SHA256 and h != SHA256[a.category]:
    sys.exit(f"sha256 mismatch: {h}")
U, I, T = [], [], []
for r in csv.DictReader(open(path)):
    U.append(r["user_id"]); I.append(r["parent_asin"]); T.append(int(r["timestamp"]) // 1000)
uid = {u: k for k, u in enumerate(sorted(set(U)))}
users = np.array([uid[u] for u in U]); ts = np.array(T, dtype=np.int64)
order = np.lexsort((np.arange(len(users)), ts, users))
users, ts = users[order], ts[order]
I = [I[k] for k in order]
remap, items = {}, np.empty(len(I), dtype=np.int32)
for k, it in enumerate(I):
    items[k] = remap.setdefault(it, len(remap) + 1)
_, starts = np.unique(users, return_index=True)
offsets = np.append(starts, len(users)).astype(np.int64)
lens = np.diff(offsets)
assert lens.min() >= 3
np.savez_compressed(out, items=items, ts=ts, offsets=offsets, n_items=np.int64(len(remap)),
                    raw_movie_ids=np.array(sorted(remap, key=remap.get)))
dt = np.diff(ts)[np.diff(users) == 0]
print(f"{out}: users={len(lens)} items={len(remap)} events={len(items)} len median={int(np.median(lens))} "
      f"mean={lens.mean():.1f} max={lens.max()} | same-second gaps={np.mean(dt == 0):.3f} "
      f"gap<1min={np.mean(dt < 60):.3f} gap>1day={np.mean(dt > 86400):.3f} gap>30d={np.mean(dt > 30*86400):.3f}")
