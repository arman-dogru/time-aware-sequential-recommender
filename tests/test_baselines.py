import numpy as np

from tasr.baselines import RecentPop

DAY = 86400


class Fake:
    """train_seq(u) returns the user's training events directly."""
    def __init__(self, seqs, n_items):
        self.seqs, self.n_items, self.n_users = seqs, n_items, len(seqs)

    def train_seq(self, u):
        it, ts = self.seqs[u]
        return np.array(it), np.array(ts, dtype=np.int64)


T0 = 1_000_000 * DAY // DAY * DAY  # day-aligned origin
DATA = Fake({0: ([1, 2], [T0, T0 + 5 * DAY + 12 * 3600]),   # item1 on day 0, item2 on day 5 at 12:00
             1: ([3, 1], [T0 + 5 * DAY + 13 * 3600, T0 + 2 * DAY])}, n_items=3)


def counts(rp, qts):
    return np.floor(rp.scores([qts]).numpy()[0]).astype(int)  # drop the 1e-6 global-pop tie-break


def test_recentpop_excludes_query_day():
    rp = RecentPop(DATA, window_days=30, bucket_days=1)
    # query on day 5 at 18:00: the day-5 events (12:00, 13:00) are in the query's bucket -> excluded
    assert counts(rp, T0 + 5 * DAY + 18 * 3600)[1:].tolist() == [2, 0, 0]


def test_recentpop_includes_previous_day():
    rp = RecentPop(DATA, window_days=30, bucket_days=1)
    assert counts(rp, T0 + 6 * DAY + 1)[1:].tolist() == [2, 1, 1]


def test_recentpop_window_lower_edge():
    rp = RecentPop(DATA, window_days=3, bucket_days=1)
    # query day 5, window = days 2,3,4: item1 on day 2 counts, item1 on day 0 does not
    assert counts(rp, T0 + 5 * DAY)[1:].tolist() == [1, 0, 0]


def test_recentpop_weekly_buckets_strictly_past():
    rp = RecentPop(DATA, window_days=14, bucket_days=7)
    # day 5 is in week 0 with every event -> nothing strictly past yet
    assert counts(rp, T0 + 5 * DAY + 18 * 3600)[1:].tolist() == [0, 0, 0]
    assert counts(rp, T0 + 7 * DAY)[1:].tolist() == [2, 1, 1]
