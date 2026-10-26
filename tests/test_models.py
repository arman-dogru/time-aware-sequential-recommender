"""Leakage-critical model and helper checks (added in review round 1: each test kills a mutant that the
earlier suite let survive; see scripts/mutation_check.py)."""
import itertools

import numpy as np
import pytest
import torch

from tasr.baselines import ItemKNN
from tasr.data import SeqData, last_history_query, window_positions
from tasr.models import SeqRec
from tasr.stats import signflip_p

L, N_ITEMS = 8, 20
# arch x pos x (trel, trope, tabs)
CONFIGS = [(a, p, *tf) for a, p, tf in itertools.product(
    ["sasrec", "hstu"], ["abs", "rel", "none"], [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)])]


def make(arch, pos, trel, trope, tabs, seed=0, layers=2):
    torch.manual_seed(seed)
    m = SeqRec(N_ITEMS, arch, 16, layers, 2, L, 0.0, pos, bool(trel), bool(trope), bool(tabs)).eval()
    with torch.no_grad():  # zero-initialised biases would hide index bugs: randomise everything
        for p in m.parameters():
            p.normal_(0, 0.5)
    return m


def batch(seed=0, B=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(1, N_ITEMS + 1, (B, L), generator=g)
    x[0, :3] = 0  # left padding on one row
    gaps = torch.randint(0, 5 * 86400, (B, L), generator=g)
    t = 1_000_000_000 + gaps.cumsum(1)
    q = torch.cat([t[:, 1:], t[:, -1:] + 3600], 1)  # query time = next event's time
    return x, t, q


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: "-".join(map(str, c)))
def test_causal_future_perturbation(cfg):
    """Changing items, event times and query times AFTER position i must not change outputs at <= i."""
    m = make(*cfg)
    x, t, q = batch()
    with torch.no_grad():
        h = m.encode(x, t, q)
        for i in range(3, L - 1):
            x2, t2, q2 = x.clone(), t.clone(), q.clone()
            x2[:, i + 1:] = torch.randint(1, N_ITEMS + 1, x2[:, i + 1:].shape)
            t2[:, i + 1:] += 7 * 86400; q2[:, i + 1:] += 30 * 86400
            h2 = m.encode(x2, t2, q2)
            assert torch.equal(h[:, :i + 1], h2[:, :i + 1]), f"position <= {i} sees the future"
            assert not torch.equal(h[:, i + 1:], h2[:, i + 1:])  # the perturbation is not a no-op


@pytest.mark.parametrize("cfg", [c for c in CONFIGS if any(c[2:])], ids=lambda c: "-".join(map(str, c)))
def test_query_time_only_affects_its_own_position(cfg):
    """q_k is the request time of position k: it may change output k only. A transposed time-bias index
    (bucket(q_j - t_i) instead of bucket(q_i - t_j)) would leak q_k into later positions. One layer: with more
    layers q_k legitimately reaches later positions through hidden state k (q_k = t_{k+1} is visible there anyway)."""
    m = make(*cfg, layers=1)
    x, t, q = batch(1)
    with torch.no_grad():
        h = m.encode(x, t, q)
        for k in range(3, L):
            q2 = q.clone(); q2[:, k] += 10 * 86400
            h2 = m.encode(x, t, q2)
            other = [j for j in range(L) if j != k]
            assert torch.equal(h[:, other], h2[:, other]), f"q[{k}] changes other positions"
            assert not torch.equal(h[:, k], h2[:, k]), f"q[{k}] has no effect on its own position"


def test_time_bias_uses_query_minus_key():
    """With trel only and a bias that is huge for exactly one bucket, position i must attend most to the key j
    whose gap q_i - t_j falls into that bucket."""
    from tasr.models import time_bucket
    t = torch.tensor([[0, 10, 100, 1000, 10000, 100000, 1000000, 2000000]])
    q = torch.full_like(t, 2000000)
    tb = time_bucket(q[:, :, None] - t[:, None, :])
    assert tb[0, -1, 0] == time_bucket(torch.tensor(2000000)) and tb[0, -1, -1] == 0
    # SeqRec.encode must build exactly this (B, L_query, L_key) index
    m = make("hstu", "none", 1, 0, 0)
    seen = {}
    orig = m.blocks[0].bias
    m.blocks[0].bias = lambda B, L_, tb_, dev: seen.setdefault("tb", tb_) is not None and orig(B, L_, tb_, dev)
    with torch.no_grad():
        m.encode(torch.arange(1, L + 1)[None], t, q)
    assert torch.equal(seen["tb"], tb)


def test_last_history_query_changes_only_last_position():
    x, t, q = batch(2)
    q2 = last_history_query(t, q)
    assert torch.equal(q2[:, :-1], q[:, :-1])
    assert torch.equal(q2[:, -1], t[:, -1])
    assert not torch.equal(q2, q) and torch.equal(q, batch(2)[2])  # input not modified in place


def test_signflip_two_sided():
    d = np.random.default_rng(3).uniform(0.05, 0.3, 25)
    p_pos = signflip_p(d, 2000, np.random.default_rng(7))
    p_neg = signflip_p(-d, 2000, np.random.default_rng(7))
    assert p_pos == p_neg and p_pos < 0.01


def test_itemknn_excludes_self(toy_npz):
    knn = ItemKNN(SeqData(toy_npz, max_len=10))
    nb = knn.neighbors.numpy()
    for i in range(1, nb.shape[0]):
        assert i not in nb[i][knn.nb_val[i].numpy() > 0], f"item {i} is its own neighbour"
    assert (knn.nb_val[1:] > 0).any()


@pytest.mark.parametrize("max_len", [2, 3, 10])
def test_window_positions_match_train_batch_targets(toy_npz, max_len):
    """incremental flags are aligned to train_batch targets through window_positions."""
    d = SeqData(toy_npz, max_len=max_len)
    for u in range(d.n_users):
        _, y, _, _ = d.train_batch([u])
        y = y[0][y[0] > 0].numpy()
        it, _ = d.seq(u)
        assert np.array_equal(it[window_positions(d, u)], y)


@pytest.mark.parametrize("cfg", [c for c in CONFIGS if c[1] != "abs"], ids=lambda c: "-".join(map(str, c)))
def test_left_padding_is_invisible(cfg):
    """Padding keys are masked: the same history encoded with and without left padding gives the same
    outputs at the real positions (absolute positions excluded: they shift with the padding by design)."""
    m = make(*cfg)
    x, t, q = batch(4)
    x[:, :3] = 0
    with torch.no_grad():
        h_pad = m.encode(x, t, q)[:, 3:]
        h = m.encode(x[:, 3:], t[:, 3:], q[:, 3:])
    assert torch.allclose(h_pad, h, atol=1e-5)


@pytest.mark.parametrize("arch", ["sasrec", "hstu"])
def test_relative_position_bias_sees_order(arch):
    """One layer, no time features: without positions the last output is invariant to permuting the earlier
    items; the relative-position bias must break that (a reversed/clamped index makes it constant)."""
    x, t, q = batch(5, B=1)
    perm = torch.cat([torch.randperm(L - 1, generator=torch.Generator().manual_seed(1)), torch.tensor([L - 1])])
    outs = {}
    for pos in ("none", "rel"):
        m = make(arch, pos, 0, 0, 0, layers=1)
        with torch.no_grad():
            outs[pos] = (m.encode(x, t, q)[:, -1], m.encode(x[:, perm], t, q)[:, -1])
    assert torch.allclose(*outs["none"], atol=1e-5)
    assert not torch.allclose(*outs["rel"], atol=1e-4)


def test_bad_negatives_positive_aware():
    from tasr.train import bad_negatives
    neg = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    yv, row = torch.tensor([2, 4]), torch.tensor([0, 1])
    hm = torch.zeros(2, 5, dtype=torch.bool); hm[0, 3] = True; hm[1, 1] = True
    assert bad_negatives(neg, yv).tolist() == [[True, False, True, False], [False, False, False, True]]
    assert bad_negatives(neg, yv, hm, row).tolist() == [[True, False, True, True], [True, False, False, True]]


@pytest.mark.parametrize("epochs", ["0", "-1"])
def test_train_rejects_nonpositive_epochs(epochs):
    from tasr.train import get_args
    with pytest.raises(SystemExit):
        get_args(["--name", "x", "--epochs", epochs])
    assert get_args(["--name", "x", "--epochs", "1"]).epochs == 1
