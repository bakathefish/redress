"""v4: the nnPU MLP fitted on all five macroregions (15 networks: three priors x five seeds,
each seed holding out one region for early stopping), scoring every support row. Same
architecture, loss and schedule as v4_mlp.py. Writes mlp_scores{SUF}.parquet with
score_mean / score_std per row and mlp_final{SUF}.json (thresholds at the union burden,
the inner 1% negative fraction, and 0.5%). Usage: python v4_mlp_final.py [suffix]
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "recovery")
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
sys.argv = [sys.argv[0]] + ([SUF] if SUF else [])
# reuse the challenger's definitions without running its outer loop
src = open(os.path.join("recovery", "v4_mlp.py"), encoding="utf-8").read()
head = src[: src.index("oof_mean = np.full(n, np.nan)")]
head = head.replace(
    'd = pd.read_parquet(os.path.join(OUT, "features.parquet"))',
    'd = pd.read_parquet(os.path.join(OUT, f"features{SUF}.parquet"))',
)
head = head.replace(
    'man = json.load(open(os.path.join(OUT, "feature_manifest.json")))',
    'man = json.load(open(os.path.join(OUT, f"feature_manifest{SUF}.json")))',
)
head = head.replace(
    "torch.set_num_threads(4)",
    "torch.set_num_threads(int(os.environ.get('MLP_THREADS', '8')))",
)
head = "SUF = %r\n" % SUF + head
exec(compile(head, "v4_mlp_head", "exec"))  # noqa: S102 (defines Net, train_net, score, X, masks, REGIONS, PRIORS, SEEDS)

T0 = time.time()
allm = np.ones(n, bool)
med = np.median(X, axis=0)
iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
iqr = np.where(iqr > 1e-6, iqr, 1.0).astype(np.float32)
scale = (med.astype(np.float32), iqr)
nets, log, frac_neg1 = [], [], []
for prior in PRIORS:
    for seed in SEEDS:
        val = REGIONS[seed % len(REGIONS)]
        trm = region != val
        valm = region == val
        net, ap, eps = train_net(
            trm, valm, prior, 77000 + int(prior * 1e4) * 10 + seed, scale
        )
        nets.append(net)
        vN = np.where(valm & is_neg)[0]
        vS = np.random.default_rng(seed).choice(np.where(valm)[0], 20000, replace=False)
        sN, _ = score([net], vN, scale)
        sS, _ = score([net], vS, scale)
        frac_neg1.append(float((sS >= np.quantile(sN, 0.99)).mean()))
        log.append(
            dict(
                prior=prior,
                seed=seed,
                val_region=val,
                val_prauc=round(float(ap), 4),
                epochs=eps,
            )
        )
        print(
            f"  final prior {prior} seed {seed} val {val}: PR-AUC {ap:.3f} after {eps} epochs ({time.time() - T0:.0f}s)",
            flush=True,
        )
mean, std = score(nets, np.arange(n), scale)
f_burden = float(picked.mean())
f_neg1 = float(np.mean(frac_neg1))
t_b, t_n, t_5 = (
    float(np.quantile(mean, 1 - f_burden)),
    float(np.quantile(mean, 1 - f_neg1)),
    float(np.quantile(mean, 0.995)),
)
out = d[["source_id"]].copy()
out["score_mean"], out["score_std"] = mean.astype(np.float32), std.astype(np.float32)
out.to_parquet(os.path.join(OUT, f"mlp_scores{SUF}.parquet"), index=False)
os.makedirs(os.path.join(OUT, f"mlp_models{SUF}"), exist_ok=True)
for i, net in enumerate(nets):
    torch.save(
        net.state_dict(), os.path.join(OUT, f"mlp_models{SUF}", f"net_{i:02d}.pt")
    )
np.save(os.path.join(OUT, f"mlp_models{SUF}", "scale.npy"), np.stack(scale))
fin = dict(
    networks=log,
    union_burden_fraction=f_burden,
    frac_for_neg1pct=f_neg1,
    t_burden=t_b,
    t_neg1=t_n,
    t_05=t_5,
    n_above_burden=int((mean >= t_b).sum()),
    n_above_05=int((mean >= t_5).sum()),
    seconds=round(time.time() - T0),
)
json.dump(fin, open(os.path.join(OUT, f"mlp_final{SUF}.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in fin.items() if k != "networks"}), flush=True)
