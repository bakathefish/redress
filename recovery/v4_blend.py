"""v4: deployment scores from a rank-average blend of the final tree ensemble and the final
nnPU MLP (used only if the out-of-fold blend passed the promotion rule). Produces
scores_blend.parquet and final_blend.json in the shape v4_candidates.py expects, so the
candidate chain runs unchanged with the suffix "_blend". Thresholds: union burden fraction,
the mean of the two models' inner 1% negative fractions, and 0.5%.
Usage: python v4_blend.py [tree suffix] [mlp suffix]
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "recovery"
TS = sys.argv[1] if len(sys.argv) > 1 else ""
MS = sys.argv[2] if len(sys.argv) > 2 else ""
tree = pd.read_parquet(os.path.join(OUT, f"scores{TS}.parquet"))
mlp = pd.read_parquet(os.path.join(OUT, f"mlp_scores{MS}.parquet"))
ft = json.load(open(os.path.join(OUT, f"final{TS}.json")))
fm = json.load(open(os.path.join(OUT, f"mlp_final{MS}.json")))
assert (tree.source_id.values == mlp.source_id.values).all()
rt = pd.Series(tree.ranking_score.values).rank(pct=True).values
rm = pd.Series(mlp.score_mean.values).rank(pct=True).values
blend = (0.5 * (rt + rm)).astype(np.float32)
sc = tree.copy()
sc["tree_score"] = tree.ranking_score.values
sc["mlp_score"] = mlp.score_mean.values
sc["mlp_score_std"] = mlp.score_std.values
sc["ranking_score"] = blend
sc["score_std"] = (0.5 * np.abs(rt - rm)).astype(
    np.float32
)  # disagreement between the two rankers
sc["score_percentile"] = (pd.Series(blend).rank(pct=True).values * 100).astype(
    np.float32
)
f_b = ft["union_burden_fraction"]
f_n = 0.5 * (ft["frac_for_neg1pct"] + fm["frac_for_neg1pct"])
t_b, t_n, t_5 = (
    float(np.quantile(blend, 1 - f_b)),
    float(np.quantile(blend, 1 - f_n)),
    float(np.quantile(blend, 0.995)),
)
sc["tier"] = np.where(
    blend >= t_b, "equal_burden", np.where(blend >= t_5, "high_recall", "")
)
above5 = blend >= t_5
dis_cut = float(np.quantile(sc.score_std.values[above5], 0.95))
sc["disagreement_flag"] = sc.score_std.values > dis_cut
sc.to_parquet(os.path.join(OUT, "scores_blend.parquet"), index=False)
fin = dict(ft)
fin.update(
    feature_set="_blend",
    blend="rank average of the final tree ensemble and the final nnPU MLP (15 networks)",
    tree_final=f"final{TS}.json",
    mlp_final=f"mlp_final{MS}.json",
    mlp_models_sha256=hashlib.sha256(
        b"".join(
            open(os.path.join(OUT, f"mlp_models{MS}", f), "rb").read()
            for f in sorted(os.listdir(os.path.join(OUT, f"mlp_models{MS}")))
        )
    ).hexdigest(),
    frac_for_neg1pct=f_n,
    t_burden=t_b,
    t_neg1=t_n,
    t_05=t_5,
    n_above_burden=int((blend >= t_b).sum()),
    n_above_neg1=int((blend >= t_n).sum()),
    n_above_05=int(above5.sum()),
    disagreement_cut=dis_cut,
)
json.dump(fin, open(os.path.join(OUT, "final_blend.json"), "w"), indent=1)
print(
    json.dumps(
        {
            k: fin[k]
            for k in ("t_burden", "t_neg1", "t_05", "n_above_burden", "n_above_05")
        }
    )
)
