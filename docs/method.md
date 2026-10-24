# Method and evaluation protocol

## Data

| | MovieLens-1M | Amazon Reviews 2023: Industrial & Scientific (5-core), 10k-user subsample |
|---|---|---|
| source | [GroupLens](https://grouplens.org/datasets/movielens/1m/) (sha256-pinned in [`scripts/prepare_ml1m.py`](../scripts/prepare_ml1m.py)) | [McAuley Lab, HF Hub](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (sha256-pinned in [`scripts/prepare_amazon.py`](../scripts/prepare_amazon.py)) |
| users / items / events | 6,040 / 3,706 / 1,000,209 | 10,000 / 21,949 / 81,271 (full: 50,985 / 25,848 / 412,947) |
| median sequence length | 96 | 6 |
| consecutive gaps < 1 min | 88.5% (rating bursts) | 5.7% |
| consecutive gaps > 1 day | 1.5% | 69.6% |
| history window L | 100 | 50 |

Each rating or review is treated as an implicit interaction. Events are sorted by (timestamp, file order).
The split is leave-one-out: the last event is the test target and the second-to-last is the validation target. Ranking is over the
**full catalog**, with the user's history items excluded and ties counted against the target. Details: [`tasr/data.py`](../tasr/data.py) and [`tasr/metrics.py`](../tasr/metrics.py).

The two datasets have very different time structure (see [`docs/figures/data_timeline.png`](figures/data_timeline.png)):
- In ML-1M, time mostly encodes *rating sessions*: users rate many movies within minutes. The test target is usually seconds or minutes after the last history event.
- On Amazon, gaps are days to years, closer to the commerce setting in the Shopify post.

The full Amazon category (51k users) cost about 15 minutes per epoch on the shared, heavily loaded box. A uniform 10k-user subsample (seed 0) keeps it within the CPU budget.

## Query time

Position *i* predicts event *i+1*, and its **query time** is `q_i = t_(i+1)`. At test time that is the timestamp of the
held-out event, i.e. the request time. It is known when a recommendation is requested, but the item is not. This is the
"current session timestamp at inference" from the Shopify post. HSTU's public implementation uses the same convention
(`RelativeBucketedTimeAndPositionBasedBias` uses next-event timestamps). Every time feature is computed relative to `q_i`.
No item information from the target leaks into the input.

## Models

All neural models have d=64, 2 layers, 2 heads, dropout 0.2, tied input/output item embeddings, L2-normalised
outputs and cosine/τ logits (τ=0.05). Training uses AdamW at lr 3e-3 with batch 128, early stopping on validation NDCG@10
(patience 4, max 40 epochs), and the best-validation checkpoint is the one evaluated on test.

| name | backbone | positions | time features |
|---|---|---|---|
| sasrec | softmax attention + FFN (pre-LN) | learned absolute | none |
| sasrec+trel | same | learned absolute | relative time bias |
| hstu-notime | HSTU block: SiLU pointwise attention, U-gating | relative-position bias | none |
| hstu-trel | HSTU | relative-position bias | relative time bias: per-head scalar for each of 64 log_√2 buckets of `q_i − t_j` (seconds) |
| hstu-trope | HSTU | relative-position bias | time RoPE: q rotated by `ω·q_i`, k by `ω·t_j`; 8 frequencies, periods 1 min to 8 years, on half of each head |
| hstu-trel+trope | HSTU | relative-position bias | both |
| hstu-full | HSTU | relative-position bias | both, plus calendar embeddings (hour, day-of-week, month-of-year) of events and of the query time |
| hstu-timeonly | HSTU | **none** | both |
| hstu-tt:incr1 | as hstu-trel+trope | | plus ensemble-incremental training: loss weight `1 + β·[ensemble misses target]`, β=1, renormalised to mean 1 |

**Loss.** The architecture × time grid uses the **full softmax** over the catalog. In pilots ([`docs/pilots.md`](pilots.md)), SASRec barely trains under
uniform sampled softmax while HSTU does, so the sampled loss would confound the architecture comparison. The sampled-softmax
levers are studied separately on hstu-trel+trope:
- `S` uniform negatives shared by the whole batch (one `N×S` matmul);
- `H` hard negatives per position, drawn from the current item's top-50 ItemKNN neighbours (another model's predictions used as negatives, as in the post);
- positive-aware masking, so that no item from the user's training history is used as a negative.

## Baselines and the retrieval ensemble

- **Pop:** global training counts.
- **RecentPop:** item counts in the 30 days before the query time. The window is tuned on validation; weekly buckets are used on Amazon.
- **ItemKNN:** cosine similarity on the binary user×item matrix with the top 100 neighbours kept. Scores are a recency-decayed sum over the last m history items (m=20, decay 0.5, tuned on validation).

The **ensemble** is the union of each member's top-K (3K candidates). This is a deliberately generous reference.
**Incr@K** is the share of test users whose target the model ranks in its top-K while the ensemble union misses it. That is the
"incremental recall" of the Shopify post: positives that the other retrievers do not already find.
For training (`--incr-beta`), the miss flags come from the ensemble at default settings (m=10, decay 0.8), computed from each
training prefix with already-seen items excluded ([`scripts/incremental_flags.py`](../scripts/incremental_flags.py)).

## Statistics

Implemented in [`scripts/aggregate.py`](../scripts/aggregate.py):

- **CIs:** a 95% *hierarchical* bootstrap (2,000 replicates) that resamples test users *and* training seeds, so the interval
  covers both test-set noise and seed-to-seed training variance. Differences use the same user resample for both arms,
  which makes them paired over users.
- **Tests:** a Monte-Carlo sign-flip permutation test (10,000 flips) on per-user, seed-averaged differences, plus a Welch t-test on
  per-seed means. The Welch test has low power with 2-3 seeds and is reported only as a sanity check.
- **Multiplicity:** Holm correction over the 12 planned comparisons in [`tasr/configs.py`](../tasr/configs.py), separately per metric.
  The primary metric is Recall@10; NDCG@10 and Incr@10 are secondary.
- **Seeds:** 3 for the core configs, 2 for the secondary ones, 1 for the negatives heatmap (CPU budget). The sign-flip test treats
  users as the unit and does not account for seed variance. Single-seed comparisons should be read with that in mind; the
  hierarchical CI does include seed variance wherever there are at least 2 seeds.
