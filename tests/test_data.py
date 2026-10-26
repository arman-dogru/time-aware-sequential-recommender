import torch

from tasr.data import SeqData
from tests.checks import check_loo
from tests.conftest import SEQS


def test_leave_one_out_boundaries(toy_npz):
    check_loo(SeqData(toy_npz, max_len=10))


def test_leave_one_out_truncated_window(toy_npz):
    check_loo(SeqData(toy_npz, max_len=2))


def test_history_mask(toy_npz):
    d = SeqData(toy_npz, max_len=10)
    m = d.history_mask([d.seq(0)[0][:-1]])  # items 1..4
    assert m.shape == (1, 7)
    assert m[0].tolist() == [True, True, True, True, True, False, False]  # padding + history only


def test_left_padding(toy_npz):
    d = SeqData(toy_npz, max_len=6)
    x, t, q, tg, h = d.eval_batch([0], "test")
    assert x[0].tolist() == [0, 0, 1, 2, 3, 4]
    assert t[0].tolist() == [0, 0, 100, 200, 300, 400]
    assert q[0].tolist() == [0, 0, 200, 300, 400, 500]
