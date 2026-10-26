"""Leave-one-out sequence data for ML-1M (see scripts/prepare_ml1m.py)."""
import numpy as np
import torch


class SeqData:
    def __init__(self, path="data/processed/ml1m.npz", max_len=200):
        z = np.load(path)
        self.items, self.ts, self.off = z["items"], z["ts"], z["offsets"]
        self.n_items = int(z["n_items"])
        self.n_users = len(self.off) - 1
        self.max_len = max_len

    def seq(self, u):
        a, b = self.off[u], self.off[u + 1]
        return self.items[a:b], self.ts[a:b]

    def train_seq(self, u):
        it, t = self.seq(u)
        return it[:-2], t[:-2]

    # --- batches -------------------------------------------------------------
    def _pad(self, rows, L):
        out = np.zeros((len(rows), L), dtype=np.int64)
        for r, x in enumerate(rows):
            x = x[-L:]
            out[r, L - len(x):] = x
        return out

    def train_batch(self, users):
        """inputs x_1..x_{n-1}, targets x_2..x_n, event ts t_1..t_{n-1}, query ts t_2..t_n (last L)."""
        L = self.max_len
        xs, ys, ts, qs = [], [], [], []
        for u in users:
            it, t = self.train_seq(u)
            it, t = it[-(L + 1):], t[-(L + 1):]
            xs.append(it[:-1]); ys.append(it[1:]); ts.append(t[:-1]); qs.append(t[1:])
        return tuple(torch.from_numpy(self._pad(v, L)) for v in (xs, ys, ts, qs))

    def eval_batch(self, users, split):
        """split='valid': history = all but last 2, target = 2nd-to-last; 'test': history = all but last."""
        L = self.max_len
        cut = 2 if split == "valid" else 1
        xs, ts, qs, tg, hist = [], [], [], [], []
        for u in users:
            it, t = self.seq(u)
            h_it, h_t = it[:-cut], t[:-cut]
            xs.append(h_it); ts.append(h_t)
            # query time for position i is the time of event i+1; for the last position it is the target's time
            q = np.append(h_t[1:], t[len(it) - cut])
            qs.append(q); tg.append(it[len(it) - cut]); hist.append(h_it)
        x, t_, q_ = (torch.from_numpy(self._pad(v, L)) for v in (xs, ts, qs))
        return x, t_, q_, torch.tensor(tg), hist

    def history_mask(self, hist_list):
        m = torch.zeros(len(hist_list), self.n_items + 1, dtype=torch.bool)
        for r, h in enumerate(hist_list):
            m[r, torch.from_numpy(h.astype(np.int64))] = True
        m[:, 0] = True
        return m


def last_history_query(t, q):
    """Query-time ablation (scripts/eval_query_time.py): set the query time of the LAST position to the last
    history event's timestamp (gap = 0 s). Every other position's query time is unchanged."""
    q = q.clone()
    q[:, -1] = t[:, -1]
    return q


def window_positions(data, u):
    """indices (into the user's full sequence) of the targets used by train_batch for user u."""
    n_train = data.off[u + 1] - data.off[u] - 2
    start = max(0, n_train - (data.max_len + 1))
    return np.arange(start + 1, n_train)


def _train_flags(self, flags, users):
    L = self.max_len
    rows = [flags["flags"][flags["off"][u]:flags["off"][u + 1]]
            for u in users]
    return torch.from_numpy(self._pad(rows, L)).bool()


SeqData.train_flags = _train_flags
