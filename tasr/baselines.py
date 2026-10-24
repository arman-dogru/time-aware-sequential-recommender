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
    """Counts of each item in the `window_days` before the query time (training events only), at a
    resolution of `bucket_days` (1 for ML-1M; 7 for the 24-year Amazon log to bound memory)."""

    def __init__(self, data, window_days=30, bucket_days=None):
        allt = np.concatenate([data.train_seq(u)[1] for u in range(data.n_users)])
        alli = np.concatenate([data.train_seq(u)[0] for u in range(data.n_users)])
        self.t0 = allt.min()
        span_days = (allt.max() - self.t0) / 86400
        self.res = bucket_days or (1 if span_days < 2000 else 7)
        b = (allt - self.t0) // (86400 * self.res)
        self.n_b = int(b.max()) + 2
        cnt = np.zeros((self.n_b + 1, data.n_items + 1), dtype=np.float32)
        np.add.at(cnt, (b + 1, alli), 1)
        self.cum = np.cumsum(cnt, axis=0)  # cum[k] = events in buckets < k
        self.w = max(1, int(round(window_days / self.res)))
        self.glob = popularity(data)

    def scores(self, qts):
        k = np.clip((np.asarray(qts) - self.t0) // (86400 * self.res), 0, self.n_b)  # strictly past buckets
        lo = np.clip(k - self.w, 0, None)
        s = self.cum[k] - self.cum[lo]
        return torch.from_numpy(s + 1e-6 * self.glob[None, :])  # global pop as tie-break


class ItemKNN:
    """Item-item cosine on the binary train user-item matrix (top-`topk` neighbours kept per item);
    score = sum_j decay^age_j * sim(x_j, .) over the last `m` history items."""

    def __init__(self, data, topk=100, m=10, decay=0.8):
        import scipy.sparse as sp
        r, c = train_matrix(data)
        n = data.n_items + 1
        X = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(data.n_users, n))
        X.sum_duplicates(); X.data[:] = 1.0
        co = (X.T @ X).tocsr()
        norm = np.sqrt(co.diagonal()) + 1e-9
        nb_idx = np.zeros((n, topk), dtype=np.int64)
        nb_val = np.zeros((n, topk), dtype=np.float32)
        for i in range(1, n):
            a, b = co.indptr[i], co.indptr[i + 1]
            j, v = co.indices[a:b], co.data[a:b] / norm[i] / norm[co.indices[a:b]]
            keep = (j != i) & (j != 0)
            j, v = j[keep], v[keep]
            if len(j) > topk:
                sel = np.argpartition(-v, topk)[:topk]; j, v = j[sel], v[sel]
            o = np.argsort(-v, kind="stable")
            nb_idx[i, :len(j)], nb_val[i, :len(j)] = j[o], v[o]
        self.nb_idx, self.nb_val = torch.from_numpy(nb_idx), torch.from_numpy(nb_val)
        self.neighbors = self.nb_idx  # (n, topk) sorted by similarity (0-padded)
        self.n, self.m, self.decay = n, m, decay

    def scores_last(self, last):
        """last: (P, m) item ids (0 = none), most recent last -> (P, n) scores."""
        m = last.shape[1]
        w = (self.decay ** torch.arange(m - 1, -1, -1, dtype=torch.float32))[None] * (last > 0).float()
        vals = (w[..., None] * self.nb_val[last]).reshape(len(last), -1)
        out = torch.zeros(len(last), self.n)
        out.scatter_add_(1, self.nb_idx[last].reshape(len(last), -1), vals)
        out[:, 0] = 0
        return out

    def scores(self, x):
        """x: (B, L) left-padded history -> (B, n_items+1) scores."""
        return self.scores_last(x[:, -self.m:])
