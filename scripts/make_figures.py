"""All result figures (docs/figures/*.png) from committed result files:
results/<ds>/summary.json, results/<ds>/test_ranks.npz, results/<ds>/ranks/ensemble_union_hits.npz,
results/bench.json, plus (for the learned-bias figure) runs/<ds>/hstu-trel/seed0/model.pt, whose
extracted values are saved to results/<ds>/learned_time_bias.json so the figure can be redrawn."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasr.configs import MAIN
from tasr.models import N_TBUCKETS

DS = {"ml1m": "MovieLens-1M", "amz_Industrial_and_Scientific_u10k": "Amazon'23 Industrial & Scientific (10k users)"}
FIG = "docs/figures"; os.makedirs(FIG, exist_ok=True)
BASE = ["Pop", "RecentPop", "ItemKNN", "Ensemble-RRF"]
rng = np.random.default_rng(7)
S = {ds: json.load(open(f"results/{ds}/summary.json")) for ds in DS if os.path.exists(f"results/{ds}/summary.json")}


def color(c):
    return "#999999" if c in BASE else ("#d62728" if c.startswith("sasrec") else "#1f77b4")


# A. overview -------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(S), figsize=(7 * len(S), 6), squeeze=False)
for ax, (ds, s) in zip(axes[0], S.items()):
    names = [c for c in BASE + MAIN if c in s["configs"]]
    m = [100 * s["configs"][c]["R@10"]["mean"] for c in names]
    lo = [100 * s["configs"][c]["R@10"]["lo"] for c in names]; hi = [100 * s["configs"][c]["R@10"]["hi"] for c in names]
    y = np.arange(len(names))
    ax.barh(y, m, xerr=[np.subtract(m, lo), np.subtract(hi, m)], color=[color(c) for c in names], capsize=3)
    ax.set_yticks(y, [f"{c} (n={s['configs'][c]['n_seeds']})" for c in names]); ax.invert_yaxis()
    ax.set_xlabel("test Recall@10 (%), 95% CI over users x seeds"); ax.set_title(DS[ds])
    for yi, v in zip(y, m):
        ax.text(v + 0.3, yi, f"{v:.1f}", va="center", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/recall10_overview.png", dpi=110); plt.close(fig)

# B. forest plot of planned comparisons -------------------------------------------------------------
fig, axes = plt.subplots(1, len(S), figsize=(8.5 * len(S), 6), squeeze=False)
for ax, (ds, s) in zip(axes[0], S.items()):
    comp = [c for c in s["comparisons"] if c["metric"] == "R@10"]
    y = np.arange(len(comp))
    lo_all, hi_all = min(100 * c["lo"] for c in comp), max(100 * c["hi"] for c in comp)
    pad = 0.02 * (hi_all - lo_all)
    for yi, c in zip(y, comp):
        p_sig = c["p_holm"] < 0.05
        ci_sig = c["lo"] > 0 or c["hi"] < 0  # hierarchical CI (users x seeds) excludes zero
        sig = p_sig and ci_sig  # significant only when the test AND the CI agree
        col = "#2ca02c" if sig and c["diff"] > 0 else ("#d62728" if sig else "#555555")
        ax.errorbar(100 * c["diff"], yi, xerr=[[100 * (c["diff"] - c["lo"])], [100 * (c["hi"] - c["diff"])]],
                    fmt="o", color=col, mfc=col if (sig or not p_sig) else "white", capsize=3)
        flag = "  (Holm p<0.05 but CI spans 0)" if p_sig and not ci_sig else ""
        ax.text(100 * c["hi"] + pad, yi, f" p_Holm={c['p_holm']:.3f}{flag}", va="center", fontsize=7)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlim(min(lo_all, 0) - 4 * pad, hi_all + 22 * pad)
    ax.set_yticks(y, [f"{c['a']}  vs  {c['b']}" for c in comp]); ax.invert_yaxis()
    ax.set_xlabel("difference in test Recall@10 (pp), 95% CI over users x seeds\ncolored = Holm p<0.05 AND CI excludes 0; hollow grey = Holm p<0.05 but CI spans 0"); ax.set_title(DS[ds])
fig.tight_layout(); fig.savefig(f"{FIG}/comparisons_forest.png", dpi=110); plt.close(fig)

# C. training curves -------------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(S), figsize=(7 * len(S), 4.5), squeeze=False)
for ax, (ds, s) in zip(axes[0], S.items()):
    for c in MAIN:
        if c not in s["configs"]:
            continue
        cur = s["configs"][c]["curves_val_ndcg10"]
        n = max(len(x) for x in cur)
        arr = np.full((len(cur), n), np.nan)
        for i, x in enumerate(cur):
            arr[i, :len(x)] = x
        ax.plot(np.arange(1, n + 1), 100 * np.nanmean(arr, 0), label=c, color=None, ls="--" if c.startswith("sasrec") else "-")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation NDCG@10 (%), mean over seeds"); ax.set_title(DS[ds]); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/training_curves.png", dpi=110); plt.close(fig)

# D. negative-sampling heatmap (ML-1M) ------------------------------------------------------------
if "ml1m" in S:
    s = S["ml1m"]["configs"]
    Ss, Hs = (128, 512, 2048), (0, 16)
    grid = np.full((len(Ss), len(Hs)), np.nan); gridn = np.full_like(grid, np.nan)
    for i, a in enumerate(Ss):
        for j, b in enumerate(Hs):
            c = f"neg:S{a}-H{b}"
            if c in s:
                grid[i, j] = 100 * s[c]["R@10"]["seed_means"][0]; gridn[i, j] = 100 * s[c]["NDCG@10"]["seed_means"][0]
    if np.isfinite(grid).any():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        ref = s.get("hstu-trel+trope")
        for ax, g, lab, key in ((axes[0], grid, "Recall@10", "R@10"), (axes[1], gridn, "NDCG@10", "NDCG@10")):
            im = ax.imshow(g, cmap="viridis", origin="lower")
            for i in range(len(Ss)):
                for j in range(len(Hs)):
                    if np.isfinite(g[i, j]):
                        ax.text(j, i, f"{g[i, j]:.1f}", ha="center", va="center", color="w", fontsize=10)
            ax.set_xticks(range(len(Hs)), [f"H={h}" for h in Hs]); ax.set_yticks(range(len(Ss)), [f"S={x}" for x in Ss])
            ax.set_xlabel("ItemKNN hard negatives per position"); ax.set_ylabel("shared uniform negatives per batch")
            t = f"test {lab} (%), hstu-trel+trope, sampled softmax, seed 0"
            if ref:
                t += f"\nfull-softmax reference: {100 * ref[key]['mean']:.1f}"
            ax.set_title(t, fontsize=9); plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(f"{FIG}/negatives_heatmap.png", dpi=110); plt.close(fig)

# E. incremental recall ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(S), figsize=(7 * len(S), 4.8), squeeze=False)
for ax, (ds, s) in zip(axes[0], S.items()):
    names = [c for c in ["ItemKNN"] + MAIN if c in s["configs"]]
    x = np.arange(len(names))
    for k, (met, off) in enumerate((("Incr@2", -0.2), ("Incr@10", 0.2))):
        m = [100 * s["configs"][c][met]["mean"] for c in names]
        lo = [100 * s["configs"][c][met]["lo"] for c in names]; hi = [100 * s["configs"][c][met]["hi"] for c in names]
        ax.bar(x + off, m, 0.4, yerr=[np.subtract(m, lo), np.subtract(hi, m)], capsize=2, label=met)
    ax.set_xticks(x, names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("% of test users: model hits in top-K,\nensemble union (Pop+RecentPop+ItemKNN, top-K each) misses")
    ax.set_title(DS[ds]); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/incremental_recall.png", dpi=110); plt.close(fig)

# F. recall by time gap ---------------------------------------------------------------------------
GAPS = [(-1, 60, "<1 min"), (60, 3600, "1 min-1 h"), (3600, 86400, "1 h-1 d"), (86400, 30 * 86400, "1-30 d"), (30 * 86400, 1e12, ">30 d")]
SHOW = ["sasrec", "hstu-notime", "hstu-trel", "hstu-trel+trope", "hstu-full"]
fig, axes = plt.subplots(1, len(S), figsize=(7 * len(S), 4.5), squeeze=False)
for ax, (ds, s) in zip(axes[0], S.items()):
    z = np.load(f"data/processed/{ds}.npz"); off, ts = z["offsets"], z["ts"]
    gap = ts[off[1:] - 1] - ts[off[1:] - 2]
    R = np.load(f"results/{ds}/test_ranks.npz")
    out = {"buckets": [g[2] for g in GAPS], "n_users": [], "models": {}}
    for lo_, hi_, _ in GAPS:
        out["n_users"].append(int(((gap > lo_) & (gap <= hi_)).sum()))
    w = 0.8 / len(SHOW)
    for k, c in enumerate([c for c in SHOW if c in R.files]):
        hit = (R[c] <= 10).mean(0)  # seed-averaged per user
        vals = []
        for lo_, hi_, _ in GAPS:
            h = hit[(gap > lo_) & (gap <= hi_)]
            bt = [h[rng.integers(0, len(h), len(h))].mean() for _ in range(500)] if len(h) else [np.nan]
            vals.append((float(h.mean()) if len(h) else np.nan, float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))))
        out["models"][c] = vals
        v = np.array(vals) * 100
        ax.bar(np.arange(len(GAPS)) - 0.4 + w * (k + 0.5), v[:, 0], w, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], capsize=2, label=c)
    ax.set_xticks(range(len(GAPS)), [f"{g[2]}\n(n={n})" for g, n in zip(GAPS, out["n_users"])], fontsize=8)
    ax.set_xlabel("gap between last history event and the test target"); ax.set_ylabel("test Recall@10 (%)")
    ax.set_title(DS[ds]); ax.legend(fontsize=7)
    json.dump(out, open(f"results/{ds}/recall_by_gap.json", "w"), indent=1)
fig.tight_layout(); fig.savefig(f"{FIG}/recall_by_gap.png", dpi=110); plt.close(fig)

# G. learned relative time bias -------------------------------------------------------------------
fig, axes = plt.subplots(1, len(S), figsize=(7 * len(S), 4), squeeze=False)
for ax, ds in zip(axes[0], S):
    jp = f"results/{ds}/learned_time_bias.json"
    mp = f"runs/{ds}/hstu-trel/seed0/model.pt"
    if os.path.exists(mp):
        import torch
        sd = torch.load(mp, map_location="cpu")
        json.dump({k: sd[k].tolist() for k in sd if k.endswith("time_bias")}, open(jp, "w"))
    if not os.path.exists(jp):
        continue
    tb = json.load(open(jp))
    edges = 2 ** (np.arange(N_TBUCKETS) / 2.0) - 1  # lower edge (s) of each log_sqrt2 bucket
    for k, v in tb.items():
        for h, row in enumerate(v):
            ax.plot(np.maximum(edges, 0.5), row, label=f"{k.split('.')[1] and 'layer ' + k.split('.')[1]} head {h}")
    for sec, lab in ((60, "1 min"), (3600, "1 h"), (86400, "1 d"), (30 * 86400, "30 d"), (365 * 86400, "1 y")):
        ax.axvline(sec, color="grey", lw=0.6, ls=":"); ax.text(sec, ax.get_ylim()[0], lab, fontsize=7, rotation=90, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(0.5, 3e9)
    ax.set_xlabel("query time - key event time (s), bucket lower edge"); ax.set_ylabel("learned additive attention bias")
    ax.set_title(f"{DS[ds]}: learned relative-time bias (hstu-trel, seed 0)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/learned_time_bias.png", dpi=110); plt.close(fig)

# H. latency / throughput -------------------------------------------------------------------------
if os.path.exists("results/bench.json"):
    b = json.load(open("results/bench.json"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    models = sorted({r["model"] for r in b["inference"]}, key=lambda m: list(dict.fromkeys(r["model"] for r in b["inference"])).index(m))
    for ax, bs in zip(axes[:2], (1, 256)):
        for mname in models:
            for th, ls in ((1, "--"), (2, "-")):
                rows = sorted([r for r in b["inference"] if r["model"] == mname and r["batch"] == bs and r["threads"] == th], key=lambda r: r["L"])
                ax.plot([r["L"] for r in rows], [r["median_ms"] for r in rows], ls, marker="o", label=f"{mname} t={th}")
        knn = [r for r in b["itemknn"] if r["batch"] == bs]
        for r in knn:
            ax.axhline(r["median_ms"], color="grey", lw=0.8, ls="--" if r["threads"] == 1 else "-")
        ax.set_xlabel("history length L"); ax.set_ylabel("median latency per request batch (ms)")
        ax.set_title(f"inference, batch={bs} (encode + score 3.7k items + top-10)\ngrey = ItemKNN", fontsize=9)
        ax.set_yscale("log"); ax.legend(fontsize=6)
    tr = b["train"]
    labs = [f"{r['model']}\nt={r['threads']}" for r in tr]
    axes[2].bar(range(len(tr)), [r["seq_per_s"] for r in tr], color=[color(r["model"]) for r in tr])
    axes[2].set_xticks(range(len(tr)), labs, fontsize=6, rotation=45, ha="right")
    axes[2].set_ylabel("training sequences / s (L=100, batch 128)")
    axes[2].set_title(f"training step throughput (load avg {b['loadavg_start'][0]:.0f} on {b['cpus']} CPUs)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/latency_throughput.png", dpi=110); plt.close(fig)
print("figures:", sorted(os.listdir(FIG)))

# I. query-time ablation (eval-only; scripts/eval_query_time.py) --------------------------------------
QT = {ds: json.load(open(f"results/{ds}/query_time_ablation.json")) for ds in DS if os.path.exists(f"results/{ds}/query_time_ablation.json")}
if QT:
    fig, axes = plt.subplots(1, len(QT), figsize=(9 * len(QT), 5.5), squeeze=False)
    for ax, (ds, q) in zip(axes[0], QT.items()):
        names = [c for c in MAIN if c in q["configs"]]
        x = np.arange(len(names))
        a = [100 * q["configs"][c]["R10_target"] for c in names]; b = [100 * q["configs"][c]["R10_lasthist"] for c in names]
        ax.bar(x - 0.2, a, 0.4, label="query time = target timestamp (protocol)")
        ax.bar(x + 0.2, b, 0.4, label="query time = last history timestamp (gap set to 0 s)")
        for xi, c in zip(x, names):
            e = q["configs"][c]
            ax.text(xi, max(a[list(names).index(c)], b[list(names).index(c)]) * 1.02,
                    f"{100 * e['diff']:+.2f}\n[{100 * e['lo']:+.2f},\n{100 * e['hi']:+.2f}]", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x, [f"{c}\n(n={len(q['configs'][c]['seeds'])})" for c in names], rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("test Recall@10 (%)"); ax.set_title(f"{DS[ds]}: eval-only query-time ablation (labels: diff pp [95% CI])", fontsize=10)
        ax.set_ylim(0, max(a + b) * 1.45); ax.legend(fontsize=9, loc="upper left", ncol=1)
    fig.tight_layout(); fig.savefig(f"{FIG}/query_time_ablation.png", dpi=110); plt.close(fig)
    print("query_time_ablation.png")
