"""The experiment grid.

Main grid (architecture x time features): every config shares data, d=64, 2 layers, 2 heads,
dropout 0.2, AdamW lr 3e-3, batch 128, early stopping on validation NDCG@10 (patience 4, max 40
epochs) and the *full* softmax over the catalog (cosine logits, tau=0.05). Full softmax is affordable
at 3.7k / 25.8k items and removes the loss as a confounder: in pilots SASRec barely trains under the
uniform sampled softmax (val R@10 0.03 after 5 epochs) while it trains fine under full softmax.

Negative-sampling study (Shopify's 'shared + positive-aware hard negatives' lever): sampled softmax
on hstu-trel+trope with S shared uniform negatives per batch and H ItemKNN hard negatives per
position (sampled from the current item's top-50 neighbours), positive-aware masking on/off.
"""

COMMON = ["--lr", "3e-3", "--epochs", "40", "--patience", "4"]
HSTU_TT = ["--arch", "hstu", "--pos", "rel", "--trel", "1", "--trope", "1"]
FULL = ["--loss", "full"]

CONFIGS = {
    # --- architecture x time features (full softmax) ----------------------------------------------
    "sasrec":          FULL + ["--arch", "sasrec", "--pos", "abs", "--trel", "0"],
    "sasrec+trel":     FULL + ["--arch", "sasrec", "--pos", "abs", "--trel", "1"],
    "hstu-notime":     FULL + ["--arch", "hstu", "--pos", "rel", "--trel", "0"],
    "hstu-trel":       FULL + ["--arch", "hstu", "--pos", "rel", "--trel", "1"],
    "hstu-trope":      FULL + ["--arch", "hstu", "--pos", "rel", "--trel", "0", "--trope", "1"],
    "hstu-trel+trope": FULL + HSTU_TT,
    "hstu-full":       FULL + HSTU_TT + ["--tabs", "1"],
    "hstu-timeonly":   FULL + ["--arch", "hstu", "--pos", "none", "--trel", "1", "--trope", "1"],
    # --- ensemble-incremental training: upweight targets the Pop/RecentPop/ItemKNN ensemble misses --
    "hstu-tt:incr1":   FULL + HSTU_TT + ["--incr-beta", "1"],
}
# --- negative-sampling sweep (sampled softmax), heatmap S x H -----------------------------------------
for S in (128, 512, 2048):
    for H in (0, 16):
        CONFIGS[f"neg:S{S}-H{H}"] = HSTU_TT + ["--loss", "sampled", "--n-shared", str(S), "--n-hard", str(H)]
CONFIGS["neg:S512-H16-nopa"] = HSTU_TT + ["--loss", "sampled", "--n-shared", "512", "--n-hard", "16", "--pos-aware", "0"]
CONFIGS["neg:sasrec-S512-H16"] = ["--arch", "sasrec", "--pos", "abs", "--trel", "0", "--loss", "sampled",
                                  "--n-shared", "512", "--n-hard", "16"]

MAIN = [c for c in CONFIGS if not c.startswith("neg:")]
# CPU budget (box load ~35-42 on 8 cores, 2-6 h per ML-1M run): 3 seeds for the core configs, 1 for the rest of
# the main grid and for the negatives heatmap (H=64 column dropped).
CORE = ["sasrec", "hstu-notime", "hstu-trel", "hstu-trel+trope", "hstu-full"]
SECONDARY = [c for c in MAIN if c not in CORE]
# Amazon (10k-user subsample, L=50): the architecture x time core only (CPU budget)
AMAZON = ["sasrec", "hstu-notime", "hstu-trel", "hstu-trel+trope", "hstu-full"]

# planned pairwise comparisons (a, b): effect = a - b. Holm-corrected as one family per metric.
COMPARISONS = [
    ("hstu-trel", "hstu-notime"),          # relative time bias
    ("hstu-trope", "hstu-notime"),         # time rotary
    ("hstu-trel+trope", "hstu-notime"),    # both time encodings (headline time ablation)
    ("hstu-full", "hstu-trel+trope"),      # + absolute calendar embeddings
    ("hstu-timeonly", "hstu-trel+trope"),  # drop positional bias, keep time
    ("sasrec+trel", "sasrec"),             # time bias on the SASRec backbone
    ("hstu-notime", "sasrec"),             # architecture, no time
    ("hstu-trel+trope", "sasrec"),         # time-aware HSTU vs SASRec
    ("hstu-tt:incr1", "hstu-trel+trope"),  # incremental-recall training
    ("neg:S512-H16", "neg:S512-H0"),       # hard negatives
    ("neg:S512-H16-nopa", "neg:S512-H16"), # positive-aware masking off
    ("neg:S512-H16", "hstu-trel+trope"),   # sampled vs full softmax
]
