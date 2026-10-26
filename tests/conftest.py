import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DAY = 86400
# three users; items 1..6; timestamps in seconds
SEQS = {
    0: ([1, 2, 3, 4, 5], [100, 200, 300, 400, 500]),
    1: ([2, 3, 1, 4, 6, 5], [0, DAY, 2 * DAY, 3 * DAY, 4 * DAY, 4 * DAY + 30]),
    2: ([6, 5, 4, 3, 2], [10, 20, 30, 40, 50]),
}


@pytest.fixture
def toy_npz(tmp_path):
    items, ts, off = [], [], [0]
    for u in sorted(SEQS):
        items += SEQS[u][0]; ts += SEQS[u][1]; off.append(len(items))
    p = tmp_path / "toy.npz"
    np.savez(p, items=np.array(items, np.int32), ts=np.array(ts, np.int64), offsets=np.array(off),
             n_items=np.int64(6), raw_movie_ids=np.arange(1, 7))
    return str(p)
