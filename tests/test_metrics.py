import math
import numpy as np

from tasr.metrics import ranks_from_scores, summarize
from tasr.stats import per_user
from tests.checks import check_ranks, check_summary


def test_ranks_ties_and_history_mask():
    check_ranks(ranks_from_scores)


def test_recall_ndcg_mrr_hand_computed():
    check_summary(summarize)


def test_per_user_matches_summarize():
    r = np.array([1, 3, 10, 11, 60])
    union = {"K2": np.zeros(5, bool), "K10": np.zeros(5, bool)}
    pu = per_user(r, union)
    s = summarize(r)
    for k in ("R@2", "R@10", "R@50", "NDCG@10"):
        assert math.isclose(pu[k].mean(), s[k])


def test_incremental_recall_hand_computed():
    r = np.array([1, 5, 20, 2])
    union = {"K2": np.array([False, False, False, True]), "K10": np.array([True, False, False, True])}
    pu = per_user(r, union)
    assert pu["Incr@10"].tolist() == [False, True, False, False]   # user 0 found by the ensemble; user 2 missed by both
    assert pu["Incr@2"].tolist() == [True, False, False, False]    # user 3 is in the ensemble's top-2
