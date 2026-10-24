# Results: amz_Industrial_and_Scientific_u10k (leave-one-out test, full-catalog ranking, 10,000 users)

Values are % with 95% hierarchical-bootstrap CIs (users x seeds). Incr@K = share of test users whose target the model ranks in its top-K while the union of the top-K of Pop, RecentPop and ItemKNN misses it.

| model | seeds | R@2 | R@10 | R@50 | NDCG@10 | Incr@2 | Incr@10 | train min/run | best epochs |
|---|---|---|---|---|---|---|---|---|---|
| sasrec | 1 | 0.97 [0.78, 1.16] | 2.13 [1.84, 2.43] | 5.00 [4.57, 5.43] | 1.29 [1.10, 1.49] | 0.57 [0.43, 0.72] | 0.88 [0.70, 1.07] | 31 | [6] |
| hstu-notime | 1 | 0.81 [0.64, 0.99] | 1.83 [1.58, 2.10] | 4.04 [3.66, 4.44] | 1.11 [0.95, 1.29] | 0.48 [0.35, 0.62] | 0.73 [0.57, 0.89] | 27 | [6] |
| hstu-trel | 1 | 0.85 [0.67, 1.03] | 1.76 [1.51, 2.02] | 3.98 [3.59, 4.38] | 1.11 [0.94, 1.29] | 0.48 [0.35, 0.62] | 0.67 [0.52, 0.83] | 68 | [6] |
| hstu-trel+trope | 1 | 0.86 [0.68, 1.05] | 1.79 [1.52, 2.06] | 4.09 [3.70, 4.50] | 1.12 [0.94, 1.30] | 0.45 [0.32, 0.59] | 0.69 [0.53, 0.86] | 91 | [6] |
| Pop | 1 | 0.49 [0.36, 0.63] | 1.77 [1.52, 2.05] | 4.50 [4.10, 4.92] | 0.88 [0.73, 1.03] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0 | - |
| RecentPop | 1 | 0.51 [0.37, 0.65] | 1.63 [1.38, 1.89] | 4.31 [3.90, 4.69] | 0.84 [0.70, 0.99] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0 | - |
| ItemKNN | 1 | 0.43 [0.30, 0.55] | 1.09 [0.89, 1.29] | 3.96 [3.57, 4.35] | 0.63 [0.50, 0.75] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0 | - |
| Ensemble-RRF | 1 | 0.59 [0.44, 0.75] | 1.97 [1.70, 2.26] | 5.48 [5.03, 5.94] | 1.03 [0.88, 1.20] | 0.18 [0.10, 0.27] | 0.16 [0.09, 0.24] | 0 | - |

Ensemble union (3 members, top-K each): R@2 1.16, R@10 3.34, R@50 8.10.

## Planned comparisons (effect = A - B, percentage points)

| metric | A | B | diff (pp) [95% CI] | rel. | p sign-flip | p Holm | p seed Welch |
|---|---|---|---|---|---|---|---|
| R@10 | hstu-trel | hstu-notime | -0.07 [-0.17, +0.03] | -3.8% | 0.2602 | 0.5203 | nan |
| R@10 | hstu-trel+trope | hstu-notime | -0.04 [-0.26, +0.17] | -2.2% | 0.7809 | 0.7809 | nan |
| R@10 | hstu-notime | sasrec | -0.30 [-0.56, -0.05] | -14.1% | 0.0318 | 0.0954 | nan |
| R@10 | hstu-trel+trope | sasrec | -0.34 [-0.59, -0.08] | -16.0% | 0.0134 | 0.0536 | nan |
| NDCG@10 | hstu-trel | hstu-notime | +0.00 [-0.05, +0.05] | +0.0% | 0.9929 | 1.0000 | nan |
| NDCG@10 | hstu-trel+trope | hstu-notime | +0.01 [-0.11, +0.13] | +0.7% | 0.9031 | 1.0000 | nan |
| NDCG@10 | hstu-notime | sasrec | -0.18 [-0.34, -0.03] | -14.0% | 0.0239 | 0.0956 | nan |
| NDCG@10 | hstu-trel+trope | sasrec | -0.17 [-0.33, -0.02] | -13.5% | 0.0306 | 0.0956 | nan |
| Incr@10 | hstu-trel | hstu-notime | -0.06 [-0.15, +0.02] | -8.2% | 0.2577 | 0.5168 | nan |
| Incr@10 | hstu-trel+trope | hstu-notime | -0.04 [-0.20, +0.12] | -5.5% | 0.7290 | 0.7290 | nan |
| Incr@10 | hstu-notime | sasrec | -0.15 [-0.35, +0.04] | -17.0% | 0.1723 | 0.5168 | nan |
| Incr@10 | hstu-trel+trope | sasrec | -0.19 [-0.38, +0.01] | -21.6% | 0.0713 | 0.2852 | nan |
