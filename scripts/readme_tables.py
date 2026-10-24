"""Render the README result tables from results/<ds>/summary.json into the README between
<!-- TABLES:BEGIN --> and <!-- TABLES:END --> markers (so the README numbers are never hand-copied)."""
import json, re

DS = {"ml1m": "MovieLens-1M (6,040 users, 3,706 items)",
      "amz_Industrial_and_Scientific_u10k": "Amazon'23 Industrial & Scientific, 10k-user subsample (21,949 items)"}
ORDER = ["Pop", "RecentPop", "ItemKNN", "Ensemble-RRF", "sasrec", "sasrec+trel", "hstu-notime", "hstu-trel", "hstu-trope",
         "hstu-trel+trope", "hstu-full", "hstu-timeonly", "hstu-tt:incr1"]
f = lambda x: f"{100 * x['mean']:.2f} [{100 * x['lo']:.2f}, {100 * x['hi']:.2f}]"
out = []
for ds, title in DS.items():
    s = json.load(open(f"results/{ds}/summary.json"))
    out += [f"#### {title}", "", "| model | seeds | Recall@2 | Recall@10 | NDCG@10 | Incr@10 vs ensemble |", "|---|---|---|---|---|---|"]
    for c in ORDER:
        if c in s["configs"]:
            r = s["configs"][c]
            out.append(f"| {c} | {r['n_seeds']} | {f(r['R@2'])} | **{f(r['R@10'])}** | {f(r['NDCG@10'])} | {f(r['Incr@10'])} |")
    out += ["", f"Planned comparisons (Recall@10, percentage points; Holm over the family; full table incl. NDCG@10 and Incr@10 in "
            f"[`results/{ds}/summary.md`](results/{ds}/summary.md)):", "",
            "| A vs B | diff [95% CI] | p sign-flip | p Holm |", "|---|---|---|---|"]
    for x in s["comparisons"]:
        if x["metric"] == "R@10":
            out.append(f"| {x['a']} vs {x['b']} | {100 * x['diff']:+.2f} [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}] | "
                       f"{x['p_signflip']:.4f} | {x['p_holm']:.4f} |")
    out.append("")
txt = open("README.md").read()
new = "<!-- TABLES:BEGIN -->\n" + "\n".join(out) + "\n<!-- TABLES:END -->"
txt = re.sub(r"<!-- TABLES:BEGIN -->.*<!-- TABLES:END -->", lambda m: new, txt, flags=re.S)
open("README.md", "w").write(txt)
print("\n".join(out))
