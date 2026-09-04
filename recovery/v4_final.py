"""v4 build step 7: the final PU+N LightGBM ensemble on all five macroregions, scoring every
source in the model's support. Settings (negative-anchor weight, tree count, the catalog
fraction that gives a 1% targeted-negative rate) come from an inner selection over the five
regions, never from the deployment scores. Thresholds:
  equal burden  : the union of seven's selection fraction on the support (1,133 of 466,419)
  1% negatives  : the inner-selected catalog fraction
  high recall   : 0.5% of the support
Also flags rows outside the positives' feature range (3 or more features) and rows where the
48 bags disagree unusually (std above the 95th percentile of the rows above the high-recall
threshold). Saves the 48 boosters, the score table and final{SUF}.json with hashes.
Usage: python v4_final.py [suffix]
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "recovery")
import v4lib  # noqa: E402

SUF = sys.argv[1] if len(sys.argv) > 1 else ""
v4lib.N_JOBS = int(os.environ.get("V4_NJOBS", "4"))
OUT = v4lib.OUT
REGIONS = v4lib.REGIONS
D = v4lib.Data(SUF)
ALL = np.arange(len(D.features))
T0 = time.time()

a_sel, k_sel, f_neg1, summ = D.inner_select(
    REGIONS, ALL, tag="final", n_bags=8, seed_base=900000
)
print(
    f"final settings: anchor {a_sel}, trees {k_sel}, catalog fraction at 1% negatives {f_neg1:.5f} ({time.time() - T0:.0f}s)",
    flush=True,
)
seeds = [990000 + s for s in range(48)]
models = D.train_bags(np.ones(D.n, bool), a_sel, k_sel, seeds, ALL)
mean, std = D.predict(models, np.arange(D.n), ALL)
f_burden = float(D.picked.mean())
t_b = float(np.quantile(mean, 1 - f_burden))
t_n = float(np.quantile(mean, 1 - f_neg1))
t_5 = float(np.quantile(mean, 0.995))
print(
    f"thresholds: burden {t_b:.4f} (fraction {f_burden:.5f}), neg1 {t_n:.4f}, 0.5% {t_5:.4f} ({time.time() - T0:.0f}s)",
    flush=True,
)

# support flags
Xp = D.X[D.is_pos]
lo, hi = Xp.min(axis=0), Xp.max(axis=0)
pad = 0.10 * (hi - lo)
n_out = ((D.X < lo - pad) | (D.X > hi + pad)).sum(axis=1)
above5 = mean >= t_5
dis_cut = float(np.quantile(std[above5], 0.95)) if above5.any() else float("inf")

mdir = os.path.join(OUT, f"models{SUF}")
os.makedirs(mdir, exist_ok=True)
hs = hashlib.sha256()
for i, m in enumerate(models):
    p = os.path.join(mdir, f"bag_{i:02d}.txt")
    m.save_model(p)
    hs.update(open(p, "rb").read())
models_sha = hs.hexdigest()

sc = D.d[
    [
        "source_id",
        "field",
        "region",
        "id",
        "ra",
        "dec",
        "sky_group",
        "sky_group_size",
        "y",
        "y_strict",
        "picked",
        "ambiguous",
        "in_any_list",
        "in_hviding25_A1",
        "in_hviding25_B1",
        "in_barro25",
        "in_degraaff26",
        "in_perger25",
        "in_kocevski24",
        "sel_labbe23",
        "sel_kokorev24",
        "sel_kocevski24",
        "sel_perezgonzalez24",
        "sel_barro23",
        "sel_greene24",
        "sel_akins24",
        "mag_f444w",
        "snr_f444w",
        "z_phot",
        "r_h_arcsec",
        "has_spec",
        "in_census",
        "spec_tested",
        "spec_vshaped",
    ]
].copy()
sc["ranking_score"] = mean.astype(np.float32)
sc["score_std"] = std.astype(np.float32)
sc["score_percentile"] = (pd.Series(mean).rank(pct=True).values * 100).astype(
    np.float32
)
sc["tier"] = np.where(
    mean >= t_b, "equal_burden", np.where(mean >= t_5, "high_recall", "")
)
sc["n_features_outside_positive_range"] = n_out.astype(np.int16)
sc["ood_flag"] = n_out >= 3
sc["disagreement_flag"] = std > dis_cut
for c in D.features:
    sc[c] = D.d[c].values
sc.to_parquet(os.path.join(OUT, f"scores{SUF}.parquet"), index=False)
fin = {
    "feature_set": SUF or "_v30",
    "features": D.features,
    "feature_manifest_sha256": hashlib.sha256(
        open(os.path.join(OUT, f"feature_manifest{SUF}.json"), "rb").read()
    ).hexdigest(),
    "features_parquet_sha256": D.manifest.get("features_parquet_sha256"),
    "models_dir": mdir,
    "models_sha256": models_sha,
    "n_bags": len(models),
    "chosen_anchor": a_sel,
    "chosen_trees": k_sel,
    "inner_selection": summ,
    "support_rows": int(D.n),
    "union_burden_fraction": f_burden,
    "frac_for_neg1pct": f_neg1,
    "t_burden": t_b,
    "t_neg1": t_n,
    "t_05": t_5,
    "n_above_burden": int((mean >= t_b).sum()),
    "n_above_neg1": int((mean >= t_n).sum()),
    "n_above_05": int((mean >= t_5).sum()),
    "positives_above_burden_in_sample": int((D.is_pos & (mean >= t_b)).sum()),
    "negatives_above_burden_in_sample": int((D.is_neg & (mean >= t_b)).sum()),
    "ood_rows_above_05": int((above5 & (n_out >= 3)).sum()),
    "disagreement_cut": dis_cut,
    "lightgbm_params": v4lib.PARAMS,
    "seconds": round(time.time() - T0),
}
json.dump(fin, open(os.path.join(OUT, f"final{SUF}.json"), "w"), indent=1)
print(
    json.dumps(
        {
            k: v
            for k, v in fin.items()
            if k not in ("inner_selection", "features", "lightgbm_params")
        },
        indent=1,
    ),
    flush=True,
)
