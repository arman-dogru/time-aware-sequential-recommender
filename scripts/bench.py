"""CPU latency/throughput microbenchmarks. Writes results/bench.json.

Inference = encode the history + score all items + top-10, per request batch. Training = one
optimizer step of the production loss (sampled softmax, 256 shared + 16 hard negatives).
Weights are random (latency does not depend on training). The box is shared and heavily loaded,
so we record load average and report median and p90 over many repetitions."""
import argparse, json, os, platform, time
import numpy as np
import torch

from tasr.data import SeqData
from tasr.models import SeqRec
from tasr.baselines import ItemKNN

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--out", default="results/bench.json")
a = ap.parse_args()

VARIANTS = {"sasrec": dict(arch="sasrec", pos="abs", trel=False),
            "hstu-notime": dict(arch="hstu", pos="rel", trel=False),
            "hstu-trel+trope": dict(arch="hstu", pos="rel", trel=True, trope=True),
            "hstu-full": dict(arch="hstu", pos="rel", trel=True, trope=True, tabs=True)}
data = SeqData("data/processed/ml1m.npz", 200)


def timeit(fn, reps):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.percentile(ts, 90))


res = {"host": platform.processor() or platform.machine(), "cpus": os.cpu_count(), "torch": torch.__version__,
       "loadavg_start": os.getloadavg(), "inference": [], "train": [], "itemknn": []}
users = np.arange(256)
for threads in (1, 4):
    torch.set_num_threads(threads)
    for L in (50, 100, 200):
        data.max_len = L
        for name, kw in VARIANTS.items():
            torch.manual_seed(0)
            m = SeqRec(data.n_items, L=L, **kw).eval()
            for bs in (1, 32, 256):
                x, t, q, tg, hist = data.eval_batch(users[:bs], "test")
                def inf():
                    with torch.no_grad():
                        h = m.encode(x, t, q)[:, -1]
                        torch.topk(h @ m.item_table().T, 10, 1)
                med, p90 = timeit(inf, a.reps if bs < 256 else max(5, a.reps // 3))
                res["inference"].append(dict(model=name, threads=threads, L=L, batch=bs, median_ms=1e3 * med,
                                             p90_ms=1e3 * p90, req_per_s=bs / med))
                print(res["inference"][-1], flush=True)
    # training throughput at L=100, batch 128
    data.max_len = 100
    for name, kw in VARIANTS.items():
        torch.manual_seed(0)
        m = SeqRec(data.n_items, L=100, **kw)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        x, y, t, q = data.train_batch(users[:128])
        def step():
            h = m.encode(x, t, q); v = y > 0; hv = h[v]; E = m.item_table()
            neg = torch.randint(1, data.n_items + 1, (256,))
            lg = torch.cat([(hv * E[y[v]]).sum(-1, keepdim=True), hv @ E[neg].T,
                            torch.einsum("nd,nkd->nk", hv, E[torch.randint(1, data.n_items + 1, (len(hv), 16))])], 1) / 0.05
            loss = -torch.log_softmax(lg, 1)[:, 0].mean(); opt.zero_grad(); loss.backward(); opt.step()
        med, p90 = timeit(step, max(5, a.reps // 3))
        res["train"].append(dict(model=name, threads=threads, L=100, batch=128, median_ms=1e3 * med, p90_ms=1e3 * p90,
                                 seq_per_s=128 / med, n_params=sum(p.numel() for p in m.parameters())))
        print(res["train"][-1], flush=True)
    data.max_len = 100
    knn = ItemKNN(data)
    for bs in (1, 32, 256):
        x, t, q, tg, hist = data.eval_batch(users[:bs], "test")
        med, p90 = timeit(lambda: torch.topk(knn.scores(x), 10, 1), a.reps)
        res["itemknn"].append(dict(threads=threads, batch=bs, median_ms=1e3 * med, p90_ms=1e3 * p90, req_per_s=bs / med))
res["loadavg_end"] = os.getloadavg()
json.dump(res, open(a.out, "w"), indent=1)
