"""Train one neural config with one seed; writes metrics.json + per-user test ranks."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from tasr.data import SeqData
from tasr.models import SeqRec
from tasr.metrics import ranks_from_scores, summarize
from tasr.baselines import ItemKNN


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--arch", default="hstu", choices=["hstu", "sasrec"])
    p.add_argument("--pos", default="rel", choices=["abs", "rel", "none"])
    p.add_argument("--trel", type=int, default=1)
    p.add_argument("--trope", type=int, default=0)
    p.add_argument("--tabs", type=int, default=0)
    p.add_argument("--loss", default="sampled", choices=["sampled", "full"])
    p.add_argument("--n-shared", type=int, default=256)
    p.add_argument("--n-hard", type=int, default=16)
    p.add_argument("--hard-pool", type=int, default=50)
    p.add_argument("--pos-aware", type=int, default=1)
    p.add_argument("--incr-beta", type=float, default=0.0)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--max-len", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default=None, help="default runs/<dataset>")
    p.add_argument("--data", default="data/processed/ml1m.npz")
    a = p.parse_args(argv)
    if a.epochs < 1:
        p.error("--epochs must be >= 1 (the model is selected on validation after at least one epoch)")
    return a


@torch.no_grad()
def evaluate(model, data, split, bs=512):
    model.eval()
    ranks = []
    E = model.item_table()
    for s in range(0, data.n_users, bs):
        users = range(s, min(s + bs, data.n_users))
        x, t, q, tg, hist = data.eval_batch(users, split)
        h = model.encode(x, t, q)[:, -1]
        ranks.append(ranks_from_scores(h @ E.T, tg, data.history_mask(hist)))
    model.train()
    return np.concatenate(ranks)


def main(argv=None):
    a = get_args(argv)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    ds = os.path.basename(a.data).replace(".npz", "")
    out = os.path.join(a.out or os.path.join("runs", ds), a.name, f"seed{a.seed}")
    data = SeqData(a.data, a.max_len)  # load (and fail) before creating the run dir, so a bad path leaves nothing behind
    model = SeqRec(data.n_items, a.arch, a.d, a.layers, a.heads, a.max_len, a.dropout,
                   a.pos, bool(a.trel), bool(a.trope), bool(a.tabs))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    neighbors = None
    if a.loss == "sampled" and a.n_hard > 0:
        neighbors = ItemKNN(data).neighbors[:, : a.hard_pool]
    incr = None
    if a.incr_beta > 0:
        incr = dict(np.load(a.data.replace(".npz", "_incr_flags.npz")))
        assert int(incr["max_len"]) == a.max_len  # from scripts/incremental_flags.py
    os.makedirs(out, exist_ok=True)
    train_hist = [data.train_seq(u)[0] for u in range(data.n_users)]
    rng = np.random.default_rng(a.seed)
    best, best_state, bad, log = -1.0, None, 0, []
    t_start = time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(data.n_users)
        tot, n_pos, t_ep = 0.0, 0, time.time()
        for s in range(0, len(perm), a.batch):
            users = perm[s: s + a.batch]
            x, y, t, q = data.train_batch(users)
            h = model.encode(x, t, q)  # (B,L,d)
            valid = y > 0
            E = model.item_table()
            hv, yv = h[valid], y[valid]
            if a.loss == "full":
                logits = hv @ E.T / a.tau
                logits[:, 0] = float("-inf")
                loss_vec = F.cross_entropy(logits, yv, reduction="none")
            else:
                pos_logit = (hv * E[yv]).sum(-1, keepdim=True)
                shared = torch.randint(1, data.n_items + 1, (a.n_shared,))
                neg_logit = hv @ E[shared].T  # shared across the whole batch: one (N,S) matmul
                neg_ids = shared[None].expand(len(yv), -1)
                if neighbors is not None:
                    xv = x[valid]
                    pick = torch.randint(0, neighbors.shape[1], (len(yv), a.n_hard))
                    hard = neighbors[xv][torch.arange(len(yv))[:, None], pick]
                    neg_logit = torch.cat([neg_logit, torch.einsum("nd,nkd->nk", hv, E[hard])], 1)
                    neg_ids = torch.cat([neg_ids, hard], 1)
                bad_neg = (neg_ids == yv[:, None]) | (neg_ids == 0)  # 0 = padding / missing neighbour
                if a.pos_aware:  # never treat a known positive of this user as a negative
                    hm = data.history_mask([train_hist[u] for u in users])
                    row = torch.arange(len(users))[:, None].expand_as(y)[valid]
                    bad_neg |= hm[row[:, None], neg_ids]
                neg_logit = neg_logit.masked_fill(bad_neg, float("-inf"))
                logits = torch.cat([pos_logit, neg_logit], 1) / a.tau
                loss_vec = -torch.log_softmax(logits, 1)[:, 0]
            w = torch.ones_like(loss_vec)
            if incr is not None:
                miss = data.train_flags(incr, users)  # (B,L) bool: target missed by the ensemble
                w = 1.0 + a.incr_beta * miss[valid].float()
                w = w / w.mean()
            loss = (w * loss_vec).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(yv); n_pos += len(yv)
        vr = evaluate(model, data, "valid")
        vm = summarize(vr)
        rec = dict(epoch=ep + 1, loss=tot / n_pos, sec=time.time() - t_ep, **{f"val_{k}": v for k, v in vm.items()})
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if vm["NDCG@10"] > best:
            best, bad = vm["NDCG@10"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    assert best_state is not None  # unreachable: get_args enforces --epochs >= 1
    model.load_state_dict(best_state)
    tr = evaluate(model, data, "test")
    vr = evaluate(model, data, "valid")
    res = dict(args=vars(a), best_epoch=int(np.argmax([r["val_NDCG@10"] for r in log]) + 1),
               train_sec=time.time() - t_start, n_params=sum(p.numel() for p in model.parameters()),
               valid=summarize(vr), test=summarize(tr), log=log)
    json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    np.save(os.path.join(out, "test_ranks.npy"), tr.astype(np.int32))
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    print("TEST", json.dumps(res["test"]), flush=True)


if __name__ == "__main__":
    main()
