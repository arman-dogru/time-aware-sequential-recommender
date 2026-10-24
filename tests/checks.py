"""Reusable assertions; tests/test_mutation.py runs them against deliberately broken code."""
import math
import numpy as np
import torch

from tests.conftest import SEQS


def check_ranks(ranks_from_scores):
    # items 0..5; 0 = padding, item 1 is in the history; target = item 3, tied with item 2
    s = torch.tensor([[9.0, 0.5, 0.9, 0.9, 0.1, 0.2]])
    hm = torch.tensor([[True, True, False, False, False, False]])
    assert ranks_from_scores(s, torch.tensor([3]), hm).tolist() == [2]   # ties count against the target
    assert ranks_from_scores(s, torch.tensor([5]), hm).tolist() == [3]   # .9, .9 ahead
    assert ranks_from_scores(s, torch.tensor([4]), hm).tolist() == [4]
    s2 = torch.tensor([[0.0, 5.0, 1.0, 4.0, 3.0, 2.0]])                 # masked item 1 would otherwise be top
    assert ranks_from_scores(s2, torch.tensor([3]), hm).tolist() == [1]


def check_summary(summarize):
    m = summarize([1, 2, 10, 11])
    assert m["R@2"] == 0.5 and m["R@10"] == 0.75 and m["R@50"] == 1.0
    assert math.isclose(m["NDCG@10"], (1 + 1 / math.log2(3) + 1 / math.log2(11)) / 4)
    assert math.isclose(m["MRR"], (1 + 1 / 2 + 1 / 10 + 1 / 11) / 4)


def check_loo(data):
    """train = all but the last 2 events; valid predicts the 2nd-to-last; test predicts the last."""
    users = list(SEQS)
    xv, tv, qv, tgv, hv = data.eval_batch(users, "valid")
    xt, tt, qt, tgt, ht = data.eval_batch(users, "test")
    x, y, t, q = data.train_batch(users)
    for r, u in enumerate(users):
        it, ts = SEQS[u]
        assert data.train_seq(u)[0].tolist() == it[:-2]
        assert tgv[r].item() == it[-2] and tgt[r].item() == it[-1]
        assert hv[r].tolist() == it[:-2] and ht[r].tolist() == it[:-1]
        assert [v for v in xv[r].tolist() if v] == it[:-2][-data.max_len:]
        assert [v for v in xt[r].tolist() if v] == it[:-1][-data.max_len:]
        # query time of the last position = timestamp of the predicted (target) event
        assert qv[r, -1].item() == ts[-2] and qt[r, -1].item() == ts[-1]
        # training never sees the valid/test events, as input or as target
        tr = it[:-2]
        assert [v for v in y[r].tolist() if v] == tr[1:][-data.max_len:]
        assert [v for v in x[r].tolist() if v] == tr[:-1][-data.max_len:]
        n = len([v for v in y[r].tolist() if v])
        assert q[r, -n:].tolist() == ts[1:len(tr)][-n:]   # query time = next event's timestamp
