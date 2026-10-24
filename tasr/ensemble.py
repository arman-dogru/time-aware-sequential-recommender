"""Retrieval 'ensemble' = {global Pop, RecentPop, sequential ItemKNN}; used for incremental recall."""
import numpy as np
import torch

from tasr.baselines import ItemKNN, RecentPop, popularity
from tasr.metrics import ranks_from_scores


class Ensemble:
    def __init__(self, data, knn_m=10, knn_decay=0.8, window_days=30):
        self.data = data
        self.pop = torch.from_numpy(popularity(data))
        self.rpop = RecentPop(data, window_days)
        self.knn = ItemKNN(data, m=knn_m, decay=knn_decay)

    def member_scores(self, x, q_last):
        B = x.shape[0]
        return {"Pop": self.pop[None].expand(B, -1).clone(),
                "RecentPop": self.rpop.scores(q_last),
                "ItemKNN": self.knn.scores(x) + 1e-6 * self.pop[None]}  # pop tie-break for cold histories

    def ranks(self, split, bs=512):
        """per member ranks + RRF fusion ranks for the eval split."""
        out = {k: [] for k in ("Pop", "RecentPop", "ItemKNN", "Ensemble-RRF")}
        for s in range(0, self.data.n_users, bs):
            users = range(s, min(s + bs, self.data.n_users))
            x, t, q, tg, hist = self.data.eval_batch(users, split)
            hm = self.data.history_mask(hist)
            sc = self.member_scores(x, q[:, -1].numpy())
            rrf = 0
            for k, v in sc.items():
                r = ranks_from_scores(v, tg, hm)
                out[k].append(r)
                v = v.masked_fill(hm, float("-inf"))
                order = torch.argsort(v, 1, descending=True)
                rk = torch.empty_like(order); rk.scatter_(1, order, torch.arange(1, v.shape[1] + 1).expand_as(order))
                rrf = rrf + 1.0 / (60 + rk.float())
            out["Ensemble-RRF"].append(ranks_from_scores(rrf, tg, hm))
        return {k: np.concatenate(v) for k, v in out.items()}
