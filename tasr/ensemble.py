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

    def ranks(self, split, bs=512, users=None, rrf=True):
        """per member ranks (+ RRF fusion ranks) for the eval split, optionally on a user subset."""
        all_users = np.arange(self.data.n_users) if users is None else np.asarray(users)
        out = {k: [] for k in ("Pop", "RecentPop", "ItemKNN") + (("Ensemble-RRF",) if rrf else ())}
        for s in range(0, len(all_users), bs):
            users = all_users[s: s + bs]
            x, t, q, tg, hist = self.data.eval_batch(users, split)
            hm = self.data.history_mask(hist)
            sc = self.member_scores(x, q[:, -1].numpy())
            rrf_s = 0
            for k, v in sc.items():
                r = ranks_from_scores(v, tg, hm)
                out[k].append(r)
                if not rrf:
                    continue
                v = v.masked_fill(hm, float("-inf"))
                order = torch.argsort(v, 1, descending=True)
                rk = torch.empty_like(order); rk.scatter_(1, order, torch.arange(1, v.shape[1] + 1).expand_as(order))
                rrf_s = rrf_s + 1.0 / (60 + rk.float())
            if rrf:
                out["Ensemble-RRF"].append(ranks_from_scores(rrf_s, tg, hm))
        return {k: np.concatenate(v) for k, v in out.items()}
