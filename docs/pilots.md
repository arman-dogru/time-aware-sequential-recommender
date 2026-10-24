# Pilot runs (ML-1M, before the main grid)

These short pilots set the protocol. They are kept for transparency; they are not part of the reported results.
Raw outputs are in the (untracked) `runs_pilot/` directory. The numbers below are copied from its `metrics.json` files.
All use HSTU/SASRec at d=64, 2 layers and L=100, with sampled softmax (256 shared uniform negatives, 16 ItemKNN hard negatives, positive-aware masking).

| run | arch / pos | lr | val Recall@10 by epoch |
|---|---|---|---|
| pilot_hstu_lr1e3 | HSTU / rel | 1e-3 | 0.011, 0.016, 0.048, 0.072, 0.094, 0.117 (stopped) |
| pilot_hstu_lr3e3 | HSTU / rel | 3e-3 | 0.016, 0.045, 0.095, 0.113, 0.128, 0.136, 0.152, 0.171, 0.175, 0.189, 0.200, 0.209, 0.211, 0.221, 0.223 |
| dbg_hstu_abs | HSTU / abs | 3e-3 | 0.039, 0.071, 0.111, 0.145, 0.148 |
| dbg_sas_rel | SASRec / rel | 3e-3 | 0.003, 0.012, 0.024, 0.036, 0.034 |
| dbg_sas_rel_nodrop | SASRec / rel, dropout 0 | 3e-3 | 0.003, 0.008, 0.031, 0.026, 0.033 |

Short 120-step probes (batch 128, about 2.5 epochs, seed 0) gave these validation Recall@10 values:

| arch | loss | val R@10 after 120 steps |
|---|---|---|
| SASRec | full softmax | 0.188 |
| HSTU | full softmax | 0.170 |
| SASRec, no FFN | full softmax | 0.177 |
| HSTU | sampled, S=1024 shared + 16 hard | 0.152 |
| SASRec | sampled, S=1024 shared + 16 hard | 0.025 |
| SASRec | sampled, S=256, no hard, not positive-aware | 0.023 |

Decisions taken from these pilots:

1. **lr 3e-3.** It converges about 2x faster than 1e-3 at equal quality so far.
2. **Full softmax for the architecture × time grid.** The catalog is small enough (3.7k / 25.8k items). Under uniform sampled softmax, SASRec barely trains in this setup, while HSTU does. Comparing the two under that loss would mostly measure that interaction, not the time features. The sampled-softmax levers (shared and hard negatives, positive-aware masking) are studied separately on the time-aware HSTU. A SASRec sampled-softmax point is included so the interaction stays visible.
3. **Small init for auxiliary embeddings.** An early SASRec bug: absolute-position embeddings at the default N(0,1) init swamped the item embeddings (std 0.02 × √d). Fixed before every reported run.
