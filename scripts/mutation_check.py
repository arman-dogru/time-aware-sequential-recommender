"""Single-line mutation check of the test suite. Each mutant is applied to a throw-away copy of tasr/ and
tests/ (the working tree is never touched) and `pytest -x tests` is run; a mutant is KILLED when pytest fails.
Writes results/mutation_check.json and exits 1 if any mutant survives. Extra arguments are passed to pytest
(e.g. --ignore=tests/test_models.py to score the pre-round-1 suite); then nothing is written."""
import json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = [  # (file, original, mutant, what it breaks)
    ("tasr/models.py", "causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))",
     "causal = torch.ones(L, L, dtype=torch.bool, device=x.device)", "full attention (sees the future)"),
    ("tasr/models.py", "causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))",
     "causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)", "attends one step ahead"),
    ("tasr/models.py", "tb = time_bucket(q[:, :, None] - t[:, None, :])", "tb = time_bucket(q[:, None, :] - t[:, :, None])",
     "transposed time-bias index"),
    ("tasr/models.py", "tb = time_bucket(q[:, :, None] - t[:, None, :])", "tb = time_bucket(t[:, :, None] - t[:, None, :])",
     "time bias ignores the query time"),
    ("tasr/models.py", "h = h + self._cal(self.cal_q, q)", "h = h + self._cal(self.cal_q, q.roll(-1, 1))",
     "calendar of the next position's query time"),
    ("tasr/models.py", "h = h + self._cal(self.cal_q, q)", "h = h + self._cal(self.cal_q, t)", "query calendar fed event times"),
    ("tasr/models.py", "mask = (causal[None] & valid[:, None, :])[:, None].float()",
     "mask = causal[None, None].float().expand(B, 1, L, L)", "padding keys not masked"),
    ("tasr/models.py", "rel = (i[:, None] - i[None, :]).clamp(0, self.L - 1)", "rel = (i[None, :] - i[:, None]).clamp(0, self.L - 1)",
     "reversed relative-position index"),
    ("tasr/train.py", "bad |= hm[row[:, None], neg_ids]", "bad = bad", "positive-aware masking removed"),
    ("tasr/train.py", "bad_neg = bad_negatives(neg_ids, yv, hm, row)", "bad_neg = bad_negatives(neg_ids, yv)",
     "training loop drops positive-aware masking"),
    ("tasr/train.py", "if a.epochs < 1:", "if a.epochs < 0:", "--epochs 0 accepted"),
    ("tasr/stats.py", "np.abs(null) >= abs(obs) - 1e-12", "null >= obs - 1e-12", "one-sided sign-flip"),
    ("tasr/stats.py", "(flips + 1))", "flips)", "sign-flip without the +1 correction"),
    ("tasr/stats.py", "running = max(running, min(1.0, (k - rank) * p[i]))", "running = min(1.0, (k - rank) * p[i])",
     "Holm not monotone"),
    ("tasr/stats.py", "running = max(running, min(1.0, (k - rank) * p[i]))", "running = max(running, min(1.0, (k - rank - 1) * p[i]))",
     "Holm multiplier off by one"),
    ("tasr/stats.py", "return per_seed[sidx, np.arange(B)[:, None]].mean(1)", "return per_seed.mean(0)",
     "bootstrap ignores seed variance"),
    ("tasr/baselines.py", "keep = (j != i) & (j != 0)", "keep = (j != 0)", "ItemKNN keeps self-similarity"),
    ("tasr/baselines.py", "k = np.clip((np.asarray(qts) - self.t0) // (86400 * self.res), 0, self.n_b)",
     "k = np.clip((np.asarray(qts) - self.t0) // (86400 * self.res) + 1, 0, self.n_b)", "RecentPop includes the query bucket"),
    ("tasr/baselines.py", "lo = np.clip(k - self.w, 0, None)", "lo = np.clip(k - self.w - 1, 0, None)", "RecentPop window one bucket too long"),
    ("tasr/data.py", "return np.arange(start + 1, n_train)", "return np.arange(start, n_train - 1)", "incremental-flag window off by one"),
    ("tasr/data.py", "start = max(0, n_train - (data.max_len + 1))", "start = max(0, n_train - data.max_len)", "incremental-flag window start"),
    ("tasr/data.py", "return it[:-2], t[:-2]", "return it[:-1], t[:-1]", "training sees the validation event"),
    ("tasr/data.py", "q = np.append(h_t[1:], t[len(it) - cut])", "q = np.append(h_t[1:], h_t[-1])", "eval query time = last history"),
    ("tasr/data.py", "q[:, -1] = t[:, -1]", "q[:, :] = t[:, :]", "ablation changes every query time"),
    ("tasr/metrics.py", "return ((s > tgt).sum(1) + (s == tgt).sum(1))", "return ((s > tgt).sum(1) + 1)", "optimistic ties"),
    ("tasr/metrics.py", "return ((s > tgt).sum(1) + (s == tgt).sum(1))", "return ((s > tgt).sum(1) + (s == tgt).sum(1) - 1)", "0-based rank"),
    ("tasr/metrics.py", "out = {f\"R@{k}\": float(np.mean(r <= k)) for k in KS}", "out = {f\"R@{k}\": float(np.mean(r < k)) for k in KS}", "Recall cutoff r < k"),
    ("tasr/metrics.py", "s[hist_mask] = float(\"-inf\")", "pass", "history not excluded"),
]


def main():
    py = sys.executable
    res = []
    for f, old, new, what in M:
        tmp = tempfile.mkdtemp(prefix="mut_")
        try:
            for d in ("tasr", "tests"):
                shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d), ignore=shutil.ignore_patterns("__pycache__"))
            p = os.path.join(tmp, f); src = open(p).read()
            assert src.count(old) == 1, f"{f}: mutation site not unique/found: {old}"
            open(p, "w").write(src.replace(old, new))
            r = subprocess.run([py, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *sys.argv[1:], "tests"], cwd=tmp,
                               env={**os.environ, "PYTHONPATH": tmp}, capture_output=True, text=True)
            killed = r.returncode != 0
        finally:
            shutil.rmtree(tmp)
        res.append(dict(file=f, mutant=new, breaks=what, killed=killed))
        print(("KILLED  " if killed else "SURVIVED"), f, "-", what, flush=True)
    n = sum(r["killed"] for r in res)
    print(f"{n}/{len(res)} mutants killed")
    if sys.argv[1:]:
        return
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump({"killed": n, "total": len(res), "mutants": res}, open(os.path.join(ROOT, "results/mutation_check.json"), "w"), indent=1)
    sys.exit(0 if n == len(res) else 1)


if __name__ == "__main__":
    main()
