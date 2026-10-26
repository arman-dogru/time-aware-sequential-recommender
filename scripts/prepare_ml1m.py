"""Download and prepare MovieLens-1M as chronological implicit-feedback sequences.

Source: GroupLens Research, "MovieLens 1M Dataset" (released 2/2003),
https://grouplens.org/datasets/movielens/1m/  (file: https://files.grouplens.org/datasets/movielens/ml-1m.zip)
License: research use per the GroupLens README (no redistribution), so raw data is NOT committed.
Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
ACM TiiS 5, 4, Article 19. https://doi.org/10.1145/2827872

Processing (standard SASRec/HSTU protocol):
  * every rating (1-5) is treated as an implicit interaction;
  * per user, events are sorted by (timestamp, original file order) -- ML-1M has many same-second ties;
  * items are re-indexed 1..N (0 = padding) in order of first appearance;
  * leave-one-out split: last event = test target, second-to-last = validation target.
Output: data/processed/ml1m.npz with ragged arrays (items, timestamps, user offsets).
"""
import argparse, hashlib, io, os, sys, urllib.request, zipfile
import numpy as np

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed/ml1m.npz")
    a = ap.parse_args()
    os.makedirs(a.raw_dir, exist_ok=True); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    zpath = os.path.join(a.raw_dir, "ml-1m.zip")
    if not os.path.exists(zpath):
        print("downloading", URL, file=sys.stderr)
        urllib.request.urlretrieve(URL, zpath)
    h = hashlib.sha256(open(zpath, "rb").read()).hexdigest()
    if h != SHA256:
        sys.exit(f"sha256 mismatch: {h}")
    with zipfile.ZipFile(zpath) as z:
        raw = z.read("ml-1m/ratings.dat").decode("latin-1")
    arr = np.loadtxt(io.StringIO(raw.replace("::", " ")), dtype=np.int64)
    users, movies, ts = arr[:, 0], arr[:, 1], arr[:, 3]
    order = np.lexsort((np.arange(len(arr)), ts, users))  # user, then ts, then file order
    users, movies, ts = users[order], movies[order], ts[order]
    remap, items = {}, np.empty_like(movies)
    for i, m in enumerate(movies):
        items[i] = remap.setdefault(int(m), len(remap) + 1)
    _, starts = np.unique(users, return_index=True)
    offsets = np.append(starts, len(users)).astype(np.int64)
    lens = np.diff(offsets)
    assert lens.min() >= 5
    np.savez_compressed(a.out, items=items.astype(np.int32), ts=ts.astype(np.int64), offsets=offsets,
                        n_items=np.int64(len(remap)), raw_movie_ids=np.array(sorted(remap, key=remap.get)))
    dt = np.diff(ts)[np.diff(users) == 0]
    print(f"users={len(lens)} items={len(remap)} events={len(items)} len median={int(np.median(lens))} "
          f"mean={lens.mean():.1f} max={lens.max()} | same-second gaps={np.mean(dt == 0):.3f} "
          f"gap<1min={np.mean(dt < 60):.3f} gap>1day={np.mean(dt > 86400):.3f}")


if __name__ == "__main__":
    main()
