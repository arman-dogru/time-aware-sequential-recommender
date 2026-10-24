"""Eval-only query-time ablation (no retraining).

At test time every time feature is computed relative to the query time q of the last position. The
protocol (docs/method.md) uses the timestamp of the held-out target event as q, i.e. the request time.
On ML-1M that reveals that 74% of test targets come < 1 min after the last history event (rating bursts).
Here we re-evaluate every trained checkpoint with q := timestamp of the LAST HISTORY EVENT instead
(the model then cannot see how much time passed before the request) and report the paired change.
Writes results/<ds>/query_time_ablation.json."""
import glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasr.data import SeqData
from tasr.metrics import ranks_from_scores, summarize
from tasr.models import SeqRec
from tasr.stats import bootstrap_weights, hboot, signflip_p

torch.set_num_threads(int(os.environ.get("THREADS", "2")))
rng = np.random.default_rng(2024)
DSETS = {"ml1m": 100, "amz_Industrial_and_Scientific_u10k": 50}


@torch.no_grad()
def ranks(model, data, mode, bs=512):
    out = []
    E = model.item_table()
    for s in range(0, data.n_users, bs):
        x, t, q, tg, hist = data.eval_batch(range(s, min(s + bs, data.n_users)), "test")
        if mode == "last_history":
            q = q.clone(); q[:, -1] = t[:, -1]
        h = model.encode(x, t, q)[:, -1]
        out.append(ranks_from_scores(h @ E.T, tg, data.history_mask(hist)))
    return np.concatenate(out)


for ds, L in DSETS.items():
    data = SeqData(f"data/processed/{ds}.npz", L)
    res = {"note": __doc__.split("\n")[0], "configs": {}}
    for mp in sorted(glob.glob(f"runs/{ds}/*/seed*/model.pt")):
        cfg, seed = mp.split("/")[2], mp.split("/")[3]
        a = json.load(open(mp.replace("model.pt", "metrics.json")))["args"]
        m = SeqRec(data.n_items, a["arch"], a["d"], a["layers"], a["heads"], a["max_len"], a["dropout"],
                   a["pos"], bool(a["trel"]), bool(a["trope"]), bool(a["tabs"]))
        m.load_state_dict(torch.load(mp, map_location="cpu")); m.eval()
        r_t, r_h = ranks(m, data, "target"), ranks(m, data, "last_history")
        ref = np.load(mp.replace("model.pt", "test_ranks.npy"))
        assert np.array_equal(r_t, ref), f"{mp}: re-evaluation does not reproduce test_ranks.npy"
        res["configs"].setdefault(cfg, {"seeds": [], "hit10_target": [], "hit10_lasthist": [],
                                        "time_features": bool(a["trel"] or a["trope"] or a["tabs"])})
        e = res["configs"][cfg]
        e["seeds"].append(seed); e["hit10_target"].append((r_t <= 10).tolist()); e["hit10_lasthist"].append((r_h <= 10).tolist())
        e.setdefault("per_seed", []).append({"seed": seed, "target": summarize(r_t), "last_history": summarize(r_h)})
        print(ds, cfg, seed, "R@10 target %.4f  last-history %.4f" % (np.mean(r_t <= 10), np.mean(r_h <= 10)), flush=True)
    for cfg, e in res["configs"].items():
        A, B = np.array(e.pop("hit10_target"), float), np.array(e.pop("hit10_lasthist"), float)
        W = bootstrap_weights(A.shape[1], 2000, rng)
        bt = hboot(B - A, W, rng)  # paired over users and seeds (same checkpoint, two query times)
        d = (B - A).mean(0)
        e.update(R10_target=float(A.mean()), R10_lasthist=float(B.mean()), diff=float(d.mean()),
                 lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5)),
                 p_signflip=signflip_p(d, 10000, rng) if np.any(d) else 1.0)
    os.makedirs(f"results/{ds}", exist_ok=True)
    json.dump(res, open(f"results/{ds}/query_time_ablation.json", "w"), indent=1)
