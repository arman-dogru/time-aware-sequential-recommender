"""Non-neural baselines: global popularity, recent (time-windowed) popularity, sequential ItemKNN."""
import numpy as np
import torch


def train_matrix(data):
    rows, cols = [], []
    for u in range(data.n_users):
        it, _ = data.train_seq(u)
        rows.append(np.full(len(it), u)); cols.append(it)
    return np.concatenate(rows), np.concatenate(cols)


def popularity(data):
    _, cols = train_matrix(data)
    return np.bincount(cols, minlength=data.n_items + 1).astype(np.float32)


class RecentPop:
    """Counts of each item in the `window_days` before the query time (training events only)."""

    def __init__(self, data, window_days=30):
        allt = np.concatenate([data.train_seq(u)[1] for u in range(data.n_users)])
        alli = np.concatenate([data.train_seq(u)[0] for u in range(data.n_users)])
        self.t0 = allt.min()
        day = (allt - self.t0) // 86400
        self.n_days = int(day.max()) + 2
        cnt = np.zeros((self.n_days + 1, data.n_items + 1), dtype=np.float32)
        np.add.at(cnt, (day + 1, alli), 1)
        self.cum = np.cumsum(cnt, axis=0)  # cum[d] = events on days < d
        self.w = window_days
        self.glob = popularity(data)

    def scores(self, qts):
        d = np.clip((np.asarray(qts) - self.t0) // 86400, 0, self.n_days)  # exclude query day (strictly past days)
        lo = np.clip(d - self.w, 0, None)
        s = self.cum[d] - self.cum[lo]
        return torch.from_numpy(s + 1e-6 * self.glob[None, :])  # global pop as tie-break


class ItemKNN:
    """Item-item cosine on the binary train user-item matrix; score = sum_j decay^age_j * sim(x_j, .)
    over the last `m` history items."""

    def __init__(self, data, topk=100, m=10, decay=0.8):
        r, c = train_matrix(data)
        n = data.n_items + 1
        import scipy.sparse as sp
        X = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(data.n_users, n))
        X.data[:] = 1.0
        X.sum_duplicates(); X.data[:] = 1.0
        co = (X.T @ X).toarray().astype(np.float32)
        norm = np.sqrt(np.diag(co)) + 1e-9
        sim = co / norm[:, None] / norm[None, :]
        np.fill_diagonal(sim, 0); sim[0] = 0; sim[:, 0] = 0
        # keep top-k neighbours per item (sparsify)
        idx = np.argpartition(-sim, topk, axis=1)[:, :topk]
        keep = np.zeros_like(sim)
        np.put_along_axis(keep, idx, np.take_along_axis(sim, idx, 1), 1)
        self.sim = torch.from_numpy(keep)
        order = np.argsort(-np.take_along_axis(sim, idx, 1), axis=1)
        self.neighbors = torch.from_numpy(np.take_along_axis(idx, order, 1)).long()  # (n, topk) sorted
        self.m, self.decay = m, decay

    def scores(self, x):
        """x: (B, L) left-padded history -> (B, n_items+1) scores."""
        last = x[:, -self.m:]
        w = self.decay ** torch.arange(self.m - 1, -1, -1, dtype=torch.float32)
        w = w[None, -last.shape[1]:] * (last > 0).float()
        return torch.einsum("bm,bmn->bn", w, self.sim[last])
