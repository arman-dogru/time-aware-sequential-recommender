"""Aggregate runs/ into results/summary.json + results/summary.md.

Uncertainty: hierarchical bootstrap (B=2000) that resamples test users AND training seeds, so the
CI reflects both finite-test-set noise and seed-to-seed training variance.
Paired tests (planned comparisons in tasr/configs.py): exact-in-distribution Monte-Carlo sign-flip
permutation test (10k flips) on per-user seed-averaged differences, plus a Welch t-test on per-seed
means (seed-level, low power with 3 seeds). Holm correction over the planned family, per metric.
Primary metric: Recall@10; NDCG@10 secondary.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasr.configs import CONFIGS, COMPARISONS

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ds", default="ml1m")
DS = ap.parse_args().ds
R = f"results/{DS}"
B, FLIPS = 2000, 10000
rng = np.random.default_rng(12345)
union = dict(np.load(f"{R}/ranks/ensemble_union_hits.npz"))
base = json.load(open(f"{R}/baselines.json"))


def per_user(r):
    r = np.asarray(r, float)
    return {"R@2": r <= 2, "R@10": r <= 10, "R@50": r <= 50,
            "NDCG@10": np.where(r <= 10, 1 / np.log2(r + 1), 0.0),
            "Incr@2": (r <= 2) & ~union["K2"], "Incr@10": (r <= 10) & ~union["K10"]}


METRICS = ["R@2", "R@10", "R@50", "NDCG@10", "Incr@2", "Incr@10"]
runs = {}
for c in CONFIGS:
    seeds = sorted(glob.glob(f"runs/{DS}/{c}/seed*/test_ranks.npy"))
    if seeds:
        runs[c] = {"ranks": np.stack([np.load(s) for s in seeds]),
                   "meta": [json.load(open(s.replace("test_ranks.npy", "metrics.json"))) for s in seeds]}
for k in ("Pop", "RecentPop", "ItemKNN", "Ensemble-RRF"):
    runs[k] = {"ranks": np.load(f"{R}/ranks/{k}.npy")[None], "meta": None}

n_users = next(iter(runs.values()))["ranks"].shape[1]
# bootstrap user multiplicities, shared by all configs (=> differences are paired over users)
W = np.stack([np.bincount(rng.integers(0, n_users, n_users), minlength=n_users) for _ in range(B)]).astype(np.float32)


def hboot(M):  # M: (S, U) per-seed per-user values -> bootstrap means (B,), resampling users and seeds
    per_seed = (M.astype(np.float32) @ W.T) / n_users  # (S, B)
    S = M.shape[0]
    sidx = rng.integers(0, S, (B, S))
    return per_seed[sidx, np.arange(B)[:, None]].mean(1)


summary = {"configs": {}, "comparisons": [], "n_users": int(n_users)}
boots = {}
for c, RR in runs.items():
    pu = {m: np.stack([per_user(r)[m] for r in RR["ranks"]]).astype(float) for m in METRICS}
    row = {"n_seeds": int(RR["ranks"].shape[0])}
    for m in METRICS:
        bt = hboot(pu[m]); boots[(c, m)] = bt
        row[m] = {"mean": float(pu[m].mean()), "lo": float(np.percentile(bt, 2.5)), "hi": float(np.percentile(bt, 97.5)),
                  "seed_means": [float(x) for x in pu[m].mean(1)]}
    if RR["meta"]:
        row["curves_val_ndcg10"] = [[e["val_NDCG@10"] for e in x["log"]] for x in RR["meta"]]
        row["epoch_sec"] = [[e["sec"] for e in x["log"]] for x in RR["meta"]]
        row["train_min"] = float(np.mean([x["train_sec"] for x in RR["meta"]]) / 60)
        row["best_epoch"] = [x["best_epoch"] for x in RR["meta"]]
        row["n_params"] = RR["meta"][0]["n_params"]
    summary["configs"][c] = row
    runs[c]["pu"] = pu

for m in ("R@10", "NDCG@10", "Incr@10"):
    fam = []
    for a, b in COMPARISONS:
        if a not in runs or b not in runs:
            continue
        A, Bm = runs[a]["pu"][m], runs[b]["pu"][m]
        d_user = A.mean(0) - Bm.mean(0)
        obs = d_user.mean()
        signs = rng.choice([-1.0, 1.0], (FLIPS, n_users))
        null = (signs * d_user).mean(1)
        p_perm = float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (FLIPS + 1))
        bt = boots[(a, m)] - boots[(b, m)]  # same user resample index per replicate -> paired over users
        sa, sb = A.mean(1), Bm.mean(1)
        p_t = float(stats.ttest_ind(sa, sb, equal_var=False).pvalue) if len(sa) > 1 and len(sb) > 1 else float("nan")
        fam.append(dict(metric=m, a=a, b=b, diff=float(obs), lo=float(np.percentile(bt, 2.5)),
                        hi=float(np.percentile(bt, 97.5)), rel=float(obs / Bm.mean()) if Bm.mean() else None,
                        p_signflip=p_perm, p_seed_welch=p_t))
    # Holm over the family (per metric) on the sign-flip p
    order = np.argsort([f["p_signflip"] for f in fam]); k = len(fam); running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * fam[i]["p_signflip"]))
        fam[i]["p_holm"] = running
    summary["comparisons"] += fam

json.dump(summary, open(f"{R}/summary.json", "w"), indent=1)
# per-user test ranks of every run (int16 is enough: catalog < 32k) so figures/stats are reproducible without runs/
np.savez_compressed(f"{R}/test_ranks.npz", **{c: v["ranks"].astype(np.int16) for c, v in runs.items()})

f = lambda x: f"{100 * x['mean']:.2f} [{100 * x['lo']:.2f}, {100 * x['hi']:.2f}]"
lines = [f"# Results: {DS} (leave-one-out test, full-catalog ranking, {n_users:,} users)", "",
         "Values are % with 95% hierarchical-bootstrap CIs (users x seeds). Incr@K = share of test users whose "
         "target the model ranks in its top-K while the union of the top-K of Pop, RecentPop and ItemKNN misses it.", "",
         "| model | seeds | R@2 | R@10 | R@50 | NDCG@10 | Incr@2 | Incr@10 | train min/run | best epochs |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for c, r in summary["configs"].items():
    lines.append(f"| {c} | {r['n_seeds']} | " + " | ".join(f(r[m]) for m in METRICS) +
                 f" | {r.get('train_min', 0):.0f} | {r.get('best_epoch', '-')} |")
lines += ["", f"Ensemble union (3 members, top-K each): R@2 {100 * union['K2'].mean():.2f}, R@10 {100 * union['K10'].mean():.2f}, "
          f"R@50 {100 * union['K50'].mean():.2f}.", "",
          "## Planned comparisons (effect = A - B, percentage points)", "",
          "| metric | A | B | diff (pp) [95% CI] | rel. | p sign-flip | p Holm | p seed Welch |", "|---|---|---|---|---|---|---|---|"]
for x in summary["comparisons"]:
    lines.append(f"| {x['metric']} | {x['a']} | {x['b']} | {100 * x['diff']:+.2f} [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}] | "
                 f"{100 * x['rel']:+.1f}% | {x['p_signflip']:.4f} | {x['p_holm']:.4f} | {x['p_seed_welch']:.3f} |")
open(f"{R}/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
