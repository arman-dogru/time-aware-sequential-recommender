"""Statistics used by scripts/aggregate.py (kept importable so they can be unit-tested)."""
import numpy as np


def per_user(r, union):
    """Per-user metric values from 1-based ranks. union: dict with boolean K2/K10 ensemble-union hits."""
    r = np.asarray(r, float)
    return {"R@2": r <= 2, "R@10": r <= 10, "R@50": r <= 50,
            "NDCG@10": np.where(r <= 10, 1 / np.log2(r + 1), 0.0),
            "Incr@2": (r <= 2) & ~union["K2"], "Incr@10": (r <= 10) & ~union["K10"]}


def bootstrap_weights(n_users, B, rng):
    """(B, U) multiplicities of a with-replacement user resample, one row per replicate."""
    return np.stack([np.bincount(rng.integers(0, n_users, n_users), minlength=n_users) for _ in range(B)]).astype(np.float32)


def hboot(M, W, rng):
    """Hierarchical bootstrap means. M: (S, U) per-seed per-user values; W: (B, U) user weights.
    Each replicate resamples users (row of W) and seeds (S draws with replacement)."""
    n_users = M.shape[1]
    per_seed = (M.astype(np.float32) @ W.T) / n_users  # (S, B)
    S, B = M.shape[0], W.shape[0]
    sidx = rng.integers(0, S, (B, S))
    return per_seed[sidx, np.arange(B)[:, None]].mean(1)


def signflip_p(d_user, flips, rng):
    """Two-sided Monte-Carlo sign-flip test of mean(d_user) = 0; p = (#{|null| >= |obs|} + 1) / (flips + 1)."""
    d_user = np.asarray(d_user, float)
    obs = d_user.mean()
    signs = rng.choice([-1.0, 1.0], (flips, len(d_user)))
    null = (signs * d_user).mean(1)
    return float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (flips + 1))


def holm(pvals):
    """Holm step-down adjusted p-values (monotone, capped at 1), in the input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p, kind="stable")
    k, running, out = len(p), 0.0, np.empty(len(p))
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * p[i]))
        out[i] = running
    return out
