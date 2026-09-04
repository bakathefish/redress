"""v4 build step 6c: the wildcard challenger, TabICL (Qu et al. 2025; checkpoint v2 of
2026-02-12), an in-context tabular foundation model with open weights, run exactly like the
TabPFN challenger (which is blocked by a licence gate on this machine). Eight contexts per outer fold, each
holding every training positive, about 2,500 unlabelled rows (half uniform, half matched in
region / F444W bin / S/N quartile, labelled 0 as in bagging PU) and up to 2,000 confirmed
negatives (labelled 0); context predictions averaged. Same five macroregion outer folds;
thresholds from a 20,000-row training-region subsample (equal burden = union fraction).
Scores are produced for the labelled rows and a 20,000-row subsample of every held-out
region (enough to evaluate and to place thresholds); the full deployment set is scored only
if the challenger is promoted.
Usage: python v4_tabpfn.py [feature-manifest suffix]
Outputs: recovery/tabpfn_oof_scores{SUF}.parquet, recovery/tabpfn_results{SUF}.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tabicl import TabICLClassifier

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "recovery"
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
T0 = time.time()
man = json.load(open(os.path.join(OUT, f"feature_manifest{SUF}.json")))
FEATURES = man["features"]
REGIONS = ["CEERS", "GOODS-N", "GOODS-S", "COSMOS", "UDS"]
N_CTX, U_CTX, N_NEG_CTX = 8, 2500, 2000

d = pd.read_parquet(os.path.join(OUT, f"features{SUF}.parquet"))
d = d[d.in_support.values].reset_index(drop=True)
n = len(d)
X = d[FEATURES].values.astype(np.float32)
y = d.y.values
region = d.region.values
is_pos, is_neg = y == 1, y == 0
is_unl = np.isnan(y) & ~d.ambiguous.values.astype(bool)
_lab_groups = np.unique(
    d.sky_group.values[is_pos | is_neg | d.ambiguous.values.astype(bool)]
)
is_unl = is_unl & ~np.isin(d.sky_group.values, _lab_groups)
picked = d.picked.values.astype(bool)
strict = d.y_strict.values.astype(bool)
rcode = pd.Categorical(region, categories=REGIONS).codes.astype(np.int64)
magbin = np.floor(d.m_f444w.values / 0.5).astype(np.int64)
q = np.nanquantile(d.asnr_f444w.values[is_unl], [0.25, 0.5, 0.75])
snrq = np.searchsorted(q, d.asnr_f444w.values).astype(np.int64)
cell = rcode * 100000 + (magbin + 200) * 10 + snrq
u_by_cell = (
    pd.Series(np.where(is_unl)[0]).groupby(cell[is_unl]).apply(np.asarray).to_dict()
)


def context(train_mask, rng):
    P = np.where(train_mask & is_pos)[0]
    N = np.where(train_mask & is_neg)[0]
    U = np.where(train_mask & is_unl)[0]
    per = max(U_CTX // 2 // len(P), 1)
    u_mat = []
    for p in P:
        pool = u_by_cell.get(cell[p], U)
        pool = pool[train_mask[pool]]
        if len(pool) == 0:
            pool = U
        u_mat.append(rng.choice(pool, per, replace=len(pool) < per))
    Ub = np.concatenate(
        [rng.choice(U, U_CTX // 2, replace=False), np.concatenate(u_mat)]
    )
    Nb = rng.choice(N, min(N_NEG_CTX, len(N)), replace=False)
    idx = np.concatenate([P, Ub, Nb])
    lab = np.concatenate([np.ones(len(P)), np.zeros(len(Ub) + len(Nb))]).astype(int)
    return idx, lab


def make_clf(seed):
    return TabICLClassifier(n_estimators=4, device="cpu", random_state=seed, verbose=False)


rng_sub = np.random.default_rng(2026)
score = np.full(n, np.nan)
thr = {k: np.full(n, np.nan) for k in ("t_burden", "t_05")}
results = {"folds": {}}
for R in REGIONS:
    T = [x for x in REGIONS if x != R]
    trm, tem = np.isin(region, T), region == R
    f_T = float(picked[trm].mean())
    te_rows = np.unique(
        np.concatenate(
            [
                np.where(tem & (is_pos | is_neg))[0],
                rng_sub.choice(np.where(tem)[0], 20000, replace=False),
            ]
        )
    )
    sub = rng_sub.choice(np.where(trm)[0], 20000, replace=False)
    rows = np.concatenate([te_rows, sub])
    acc = np.zeros(len(rows))
    for c in range(N_CTX):
        rng = np.random.default_rng(1000 * REGIONS.index(R) + c)
        idx, lab = context(trm, rng)
        clf = make_clf(c)
        clf.fit(X[idx], lab)
        acc += clf.predict_proba(X[rows])[:, 1]
        print(
            f"  {R} context {c}: {len(idx)} rows ({time.time() - T0:.0f}s)", flush=True
        )
    acc /= N_CTX
    score[te_rows] = acc[: len(te_rows)]
    s_sub = acc[len(te_rows) :]
    t_b, t_5 = float(np.quantile(s_sub, 1 - f_T)), float(np.quantile(s_sub, 0.995))
    thr["t_burden"][tem], thr["t_05"][tem] = t_b, t_5
    P, N = tem & is_pos, tem & is_neg
    results["folds"][R] = dict(
        union_burden_train=f_T,
        t_burden=t_b,
        t_05=t_5,
        positives=int(P.sum()),
        recall_at_burden=int((score[P] >= t_b).sum()),
        union_recall=int((picked & P).sum()),
        rule_missed=int((P & ~picked).sum()),
        rule_missed_recovered=int((score[P & ~picked] >= t_b).sum()),
        negatives=int(N.sum()),
        neg_selected_at_burden=int((score[N] >= t_b).sum()),
        subsample_selected_fraction=float((score[te_rows] >= t_b).mean()),
        case_control_prauc=round(
            float(
                average_precision_score(
                    np.r_[np.ones(P.sum()), np.zeros(N.sum())],
                    np.r_[score[P], score[N]],
                )
            ),
            4,
        ),
        seconds=round(time.time() - T0),
    )
    fr = results["folds"][R]
    print(
        f"TabPFN {R}: recall {fr['recall_at_burden']}/{fr['positives']} (union {fr['union_recall']}) | rule-missed {fr['rule_missed_recovered']}/{fr['rule_missed']} | neg {fr['neg_selected_at_burden']}/{fr['negatives']} | PR-AUC {fr['case_control_prauc']}",
        flush=True,
    )
    json.dump(
        results, open(os.path.join(OUT, f"tabicl_results{SUF}.json"), "w"), indent=1
    )

oof = d[["source_id", "region", "y", "y_strict", "picked"]].copy()
oof["score_mean"] = score
oof["score_std"] = np.nan
for k, v in thr.items():
    oof[k] = v
oof["t_neg1"] = np.nan
oof["sel_burden"] = oof.score_mean >= oof.t_burden
oof["sel_neg1"] = False
oof["sel_05"] = oof.score_mean >= oof.t_05
oof.to_parquet(os.path.join(OUT, f"tabicl_oof_scores{SUF}.parquet"), index=False)
sel = oof.sel_burden.values
results["pooled"] = {
    "recall_at_burden": int((sel & is_pos).sum()),
    "union_recall": int((picked & is_pos).sum()),
    "rule_missed_recovered": int((sel & is_pos & ~picked).sum()),
    "strict_missed_recovered": int((sel & strict & ~picked).sum()),
    "neg_selected_at_burden": int((sel & is_neg).sum()),
    "case_control_prauc_pooled": round(
        float(
            average_precision_score(
                np.r_[np.ones(is_pos.sum()), np.zeros(is_neg.sum())],
                np.r_[score[is_pos], score[is_neg]],
            )
        ),
        4,
    ),
    "seconds_total": round(time.time() - T0),
}
json.dump(results, open(os.path.join(OUT, f"tabicl_results{SUF}.json"), "w"), indent=1)
print("TabICL pooled:", json.dumps(results["pooled"]), flush=True)
