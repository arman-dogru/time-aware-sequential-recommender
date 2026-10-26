import numpy as np
import torch

KS = (2, 10, 50)


def ranks_from_scores(scores, target, hist_mask):
    """1-based rank of target among all items, history items (and padding) excluded."""
    s = scores.clone()
    s[hist_mask] = float("-inf")
    tgt = s.gather(1, target[:, None])
    # pessimistic tie handling: ties count against the target
    return ((s > tgt).sum(1) + (s == tgt).sum(1)).cpu().numpy().astype(np.int32)


def summarize(ranks):
    r = np.asarray(ranks)
    out = {f"R@{k}": float(np.mean(r <= k)) for k in KS}
    out["NDCG@10"] = float(np.mean(np.where(r <= 10, 1.0 / np.log2(r + 1), 0.0)))
    out["MRR"] = float(np.mean(1.0 / r))
    return out
