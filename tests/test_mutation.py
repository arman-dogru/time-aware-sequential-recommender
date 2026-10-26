"""Mutation-style tests: deliberately off-by-one implementations must be caught by the same checks."""
import numpy as np
import pytest
import torch

from tasr.data import SeqData
from tasr.metrics import summarize
from tests.checks import check_loo, check_ranks, check_summary


def test_catches_off_by_one_recall_cutoff():
    def mutant(ranks):  # r < k instead of r <= k
        out = summarize(ranks)
        r = np.asarray(ranks)
        out.update({f"R@{k}": float(np.mean(r < k)) for k in (2, 10, 50)})
        return out
    with pytest.raises(AssertionError):
        check_summary(mutant)


def test_catches_off_by_one_rank():
    def mutant(scores, target, hist_mask):  # forgets that rank is 1-based
        s = scores.clone(); s[hist_mask] = float("-inf")
        tgt = s.gather(1, target[:, None])
        return ((s > tgt).sum(1) + (s == tgt).sum(1) - 1).numpy()
    with pytest.raises(AssertionError):
        check_ranks(mutant)


def test_catches_off_by_one_split(toy_npz, monkeypatch):
    d = SeqData(toy_npz, max_len=10)
    orig = SeqData.train_seq
    monkeypatch.setattr(SeqData, "train_seq", lambda self, u: tuple(a[:-1] for a in self.seq(u)))  # leaks the valid event
    with pytest.raises(AssertionError):
        check_loo(d)
    monkeypatch.setattr(SeqData, "train_seq", orig)
    check_loo(d)
