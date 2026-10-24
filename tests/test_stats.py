import numpy as np

from tasr.stats import bootstrap_weights, hboot, holm, signflip_p


def test_holm_hand_computed():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.5, 0.6]), [1.0, 1.0])
    assert np.allclose(holm([0.001]), [0.001])


def test_holm_monotone_and_capped():
    p = np.random.default_rng(0).uniform(size=20)
    h = holm(p)
    o = np.argsort(p)
    assert np.all(np.diff(h[o]) >= 0) and np.all(h <= 1) and np.all(h >= p)


def test_signflip_null_and_extreme():
    rng = np.random.default_rng(0)
    assert signflip_p(np.zeros(50), 999, rng) == 1.0
    assert signflip_p(np.ones(30), 999, rng) == 1 / 1000  # min attainable p


def test_signflip_type1_rate():
    rng = np.random.default_rng(1)
    ps = [signflip_p(rng.normal(size=40), 400, rng) for _ in range(300)]
    assert 0.02 <= np.mean(np.array(ps) < 0.05) <= 0.09


def test_bootstrap_weights_rows():
    W = bootstrap_weights(100, 50, np.random.default_rng(0))
    assert W.shape == (50, 100) and np.all(W.sum(1) == 100)


def test_hboot_constant_and_coverage():
    rng = np.random.default_rng(0)
    W = bootstrap_weights(2000, 1000, rng)
    assert np.allclose(hboot(np.full((3, 2000), 0.25), W, rng), 0.25)
    M = (np.random.default_rng(5).uniform(size=(1, 2000)) < 0.3).astype(float)
    bt = hboot(M, W, rng)
    lo, hi = np.percentile(bt, [2.5, 97.5])
    se = np.sqrt(M.mean() * (1 - M.mean()) / 2000)
    assert lo < M.mean() < hi and abs((hi - lo) - 2 * 1.96 * se) < 0.3 * 2 * 1.96 * se


def test_hboot_resamples_seeds():
    rng = np.random.default_rng(0)
    W = bootstrap_weights(10, 400, rng)
    M = np.stack([np.zeros(10), np.ones(10)])
    vals = set(np.round(hboot(M, W, rng), 6))
    assert vals == {0.0, 0.5, 1.0}  # seed draws {0,0}, {0,1}, {1,1}


def test_paired_difference_uses_same_user_resample():
    rng = np.random.default_rng(0)
    W = bootstrap_weights(500, 300, rng)
    a = (np.random.default_rng(2).uniform(size=(1, 500)) < 0.4).astype(float)
    d = hboot(a + 0.1, W, rng) - hboot(a, W, rng)  # same users, constant shift -> zero-width CI
    assert np.allclose(d, 0.1, atol=1e-5)
