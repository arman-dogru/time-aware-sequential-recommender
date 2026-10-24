"""The experiment grid. Every neural config shares data, loss (sampled softmax with 256 shared + 16
ItemKNN hard negatives, positive-aware, tau=0.05), d=64, 2 layers, 2 heads, L=100, dropout 0.2,
AdamW lr 3e-3, batch 128, early stopping on validation NDCG@10 (patience 4, max 40 epochs)."""

COMMON = ["--lr", "3e-3", "--epochs", "40", "--patience", "4"]

CONFIGS = {
    # --- architecture x time features ------------------------------------------------------------
    "sasrec":            ["--arch", "sasrec", "--pos", "abs", "--trel", "0"],
    "sasrec+trel":       ["--arch", "sasrec", "--pos", "abs", "--trel", "1"],
    "hstu-notime":       ["--arch", "hstu", "--pos", "rel", "--trel", "0"],
    "hstu-trel":         ["--arch", "hstu", "--pos", "rel", "--trel", "1"],
    "hstu-trope":        ["--arch", "hstu", "--pos", "rel", "--trel", "0", "--trope", "1"],
    "hstu-trel+trope":   ["--arch", "hstu", "--pos", "rel", "--trel", "1", "--trope", "1"],
    "hstu-full":         ["--arch", "hstu", "--pos", "rel", "--trel", "1", "--trope", "1", "--tabs", "1"],
    "hstu-timeonly":     ["--arch", "hstu", "--pos", "none", "--trel", "1", "--trope", "1"],
    # --- training-objective ablations on hstu-trel+trope -----------------------------------------
    "hstu-tt:no-hard":   ["--arch", "hstu", "--pos", "rel", "--trel", "1", "--trope", "1", "--n-hard", "0"],
    "hstu-tt:no-posaware": ["--arch", "hstu", "--pos", "rel", "--trel", "1", "--trope", "1", "--pos-aware", "0"],
    "hstu-tt:incr1":     ["--arch", "hstu", "--pos", "rel", "--trel", "1", "--trope", "1", "--incr-beta", "1"],
    "hstu-tt:fullsoftmax": ["--arch", "hstu", "--pos", "rel", "--trel", "1", "--trope", "1", "--loss", "full"],
}

# planned pairwise comparisons (a, b): effect = a - b. Holm-corrected as one family.
COMPARISONS = [
    ("hstu-trel", "hstu-notime"),          # relative time bias
    ("hstu-trope", "hstu-notime"),         # time rotary
    ("hstu-trel+trope", "hstu-notime"),    # both time encodings
    ("hstu-full", "hstu-trel+trope"),      # + absolute calendar embeddings
    ("hstu-timeonly", "hstu-trel+trope"),  # drop positional bias, keep time
    ("sasrec+trel", "sasrec"),             # time bias on the SASRec backbone
    ("hstu-notime", "sasrec"),             # architecture, no time
    ("hstu-trel+trope", "sasrec"),         # headline: time-aware HSTU vs SASRec
    ("hstu-tt:no-hard", "hstu-trel+trope"),
    ("hstu-tt:no-posaware", "hstu-trel+trope"),
    ("hstu-tt:incr1", "hstu-trel+trope"),
    ("hstu-tt:fullsoftmax", "hstu-trel+trope"),
]
