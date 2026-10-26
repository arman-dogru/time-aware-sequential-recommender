"""Post-hoc audit numbers requested in review round 1 (no training; reads committed results + data):

  results/ml1m/epoch_cap.json      which runs used all 40 epochs (no early stop) and how much validation
                                   NDCG@10 still rose over their last 10 epochs  -> docs/figures/epoch_cap.png
  results/ml1m/posthoc.json        (a) hstu-full vs hstu-trel+trope per seed, user-only and
                                   hierarchical bootstrap CIs; (b) the unplanned neg:S2048-H16 vs full-softmax
                                   comparison (exploratory: not in the Holm family)
  results/<ds>/unseen_targets.json test targets never seen in any training sequence
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasr.stats import bootstrap_weights, hboot, signflip_p

CAP = 40
rng = np.random.default_rng(31)
s = json.load(open("results/ml1m/summary.json"))
R = np.load("results/ml1m/test_ranks.npz")

# --- epoch cap -----------------------------------------------------------------------------------
rows = []
for c, r in s["configs"].items():
    for k, (be, cv) in enumerate(zip(r.get("best_epoch", []), r.get("curves_val_ndcg10", []))):
        rows.append(dict(config=c, seed_index=k, epochs_run=len(cv), best_epoch=be, hit_cap=len(cv) == CAP,
                         val_ndcg10_gain_last10=float(max(cv) - max(cv[:-10])) if len(cv) == CAP else None))
cap = {"cap": CAP, "n_runs": len(rows), "n_hit_cap": sum(r["hit_cap"] for r in rows), "runs": rows}
json.dump(cap, open("results/ml1m/epoch_cap.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(11, 6.5))
y = np.arange(len(rows))
ax.barh(y, [r["epochs_run"] for r in rows], color=["#d62728" if r["hit_cap"] else "#1f77b4" for r in rows], alpha=0.75)
ax.scatter([r["best_epoch"] for r in rows], y, color="k", marker="|", s=120, label="best (selected) epoch")
for yi, r in zip(y, rows):
    if r["hit_cap"]:
        ax.text(r["epochs_run"] + 0.4, yi, f"+{100 * r['val_ndcg10_gain_last10']:.2f}", va="center", fontsize=7)
ax.axvline(CAP, color="k", ls="--", lw=0.8)
ax.set_yticks(y, [f"{r['config']} s{r['seed_index']}" for r in rows], fontsize=7); ax.invert_yaxis()
ax.set_xlabel("epochs run (red = hit the 40-epoch cap, no early stop); label on capped runs = val NDCG@10 gain (pp) over the last 10 epochs")
ax.set_title(f"ML-1M: {cap['n_hit_cap']} of {cap['n_runs']} runs hit the 40-epoch cap", fontsize=10)
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)); ax.set_xlim(0, 45)
fig.tight_layout(); fig.savefig("docs/figures/epoch_cap.png", dpi=110); plt.close(fig)

# --- post-hoc comparisons ------------------------------------------------------------------------
W = bootstrap_weights(R["sasrec"].shape[1], 2000, rng)
pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def compare(a, b):
    A, B = (R[a] <= 10).astype(float), (R[b] <= 10).astype(float)
    d_user = A.mean(0) - B.mean(0)
    out = dict(a=a, b=b, diff=float(d_user.mean()), seeds_a=int(len(A)), seeds_b=int(len(B)),
               per_seed_R10_a=[float(x) for x in A.mean(1)], per_seed_R10_b=[float(x) for x in B.mean(1)],
               ci_hierarchical=pct(hboot(A, W, rng) - hboot(B, W, rng)),
               ci_user_only=pct((d_user[None] @ W.T / W.shape[1])[0]),
               p_signflip_user=signflip_p(d_user, 10000, rng))
    if len(A) == len(B) and len(A) > 1:
        dseed = A.mean(1) - B.mean(1)
        out["per_seed_diff"] = [float(x) for x in dseed]
    return out


post = {"note": "exploratory / audit; not part of the Holm family in summary.json",
        "hstu-full_vs_hstu-trel+trope": compare("hstu-full", "hstu-trel+trope"),
        "neg:S2048-H16_vs_hstu-trel+trope": compare("neg:S2048-H16", "hstu-trel+trope")}
json.dump(post, open("results/ml1m/posthoc.json", "w"), indent=1)

# --- test targets never seen in training ---------------------------------------------------------
for ds in ("ml1m", "amz_Industrial_and_Scientific_u10k"):
    z = np.load(f"data/processed/{ds}.npz"); it, off = z["items"], z["offsets"]
    seen = np.zeros(int(z["n_items"]) + 1, bool)
    for u in range(len(off) - 1):
        seen[it[off[u]:off[u + 1] - 2]] = True
    tg = it[off[1:] - 1]
    n = int((~seen[tg]).sum())
    json.dump({"test_targets": int(len(tg)), "unseen_in_train": n, "share": n / len(tg),
               "recall_ceiling": 1 - n / len(tg)}, open(f"results/{ds}/unseen_targets.json", "w"), indent=1)
print(json.dumps({k: {kk: v for kk, v in x.items() if not kk.startswith("per_seed_R")} for k, x in post.items() if k != "note"}, indent=1))
print(cap["n_hit_cap"], "of", cap["n_runs"], "hit the cap")
