"""v4: the primary PU+N LightGBM protocol (nested leave-one-macroregion-out, inner one-standard
-error selection, thresholds from the training regions) run on an alternative feature set,
so it can be compared with the 30-feature run of record under the same rule. Reduced checks:
two shuffled-label repeats, the field adversary and the ensemble dispersion.
Usage: python v4_train_variant.py _v34
Outputs: oof_scores{SUF}.parquet, train_results{SUF}.json, baseline_scores{SUF}.parquet
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "recovery")
import v4lib  # noqa: E402

SUF = sys.argv[1]
v4lib.N_JOBS = int(os.environ.get("V4_NJOBS", "2"))
OUT = v4lib.OUT
REGIONS = v4lib.REGIONS
D = v4lib.Data(SUF)
ALL = np.arange(len(D.features))
n, region, is_pos, is_neg, picked, strict = (
    D.n,
    D.region,
    D.is_pos,
    D.is_neg,
    D.picked,
    D.strict,
)
T0 = time.time()
print(
    f"{SUF}: support rows {n} | positives {is_pos.sum()} | negatives {is_neg.sum()} | unlabelled {D.is_unl.sum()} | features {len(D.features)}",
    flush=True,
)

results = {
    "folds": {},
    "baselines": {},
    "sanity": {},
    "feature_set": SUF,
    "features": D.features,
}
oof_mean = np.full(n, np.nan)
oof_std = np.full(n, np.nan)
thr = {k: np.full(n, np.nan) for k in ("t_burden", "t_neg1", "t_05")}
rng_sub = np.random.default_rng(2026)
for R in REGIONS:
    T = [x for x in REGIONS if x != R]
    trm, tem = np.isin(region, T), region == R
    f_T = float(picked[trm].mean())
    a_sel, k_sel, f_neg1, summ = D.inner_select(
        T, ALL, tag=R, rng_sub=rng_sub, seed_base=500000
    )
    seeds = [100000 * (REGIONS.index(R) + 1) + 777 + s for s in range(48)]
    models = D.train_bags(trm, a_sel, k_sel, seeds, ALL)
    te_rows, tr_rows = np.where(tem)[0], np.where(trm)[0]
    m_te, s_te = D.predict(models, te_rows, ALL)
    m_tr, _ = D.predict(models, tr_rows, ALL)
    t_b, t_n, t_5 = (
        float(np.quantile(m_tr, 1 - f_T)),
        float(np.quantile(m_tr, 1 - f_neg1)),
        float(np.quantile(m_tr, 0.995)),
    )
    oof_mean[te_rows], oof_std[te_rows] = m_te, s_te
    thr["t_burden"][te_rows], thr["t_neg1"][te_rows], thr["t_05"][te_rows] = (
        t_b,
        t_n,
        t_5,
    )
    P, N = tem & is_pos, tem & is_neg
    results["folds"][R] = dict(
        summ,
        union_burden_train=f_T,
        t_burden=t_b,
        t_neg1=t_n,
        t_05=t_5,
        n_held_out=int(tem.sum()),
        selected_at_burden=int((m_te >= t_b).sum()),
        union_selected_in_region=int((picked & tem).sum()),
        positives=int(P.sum()),
        recall_at_burden=int((oof_mean[P] >= t_b).sum()),
        union_recall=int((picked & P).sum()),
        rule_missed=int((P & ~picked).sum()),
        rule_missed_recovered=int((oof_mean[P & ~picked] >= t_b).sum()),
        negatives=int(N.sum()),
        neg_selected_at_burden=int((oof_mean[N] >= t_b).sum()),
        neg_selected_at_neg1=int((oof_mean[N] >= t_n).sum()),
        case_control_prauc=round(
            float(
                average_precision_score(
                    np.r_[np.ones(P.sum()), np.zeros(N.sum())],
                    np.r_[oof_mean[P], oof_mean[N]],
                )
            ),
            4,
        ),
        seconds=round(time.time() - T0),
    )
    fr = results["folds"][R]
    print(
        f"  {R}{SUF}: anchor {a_sel} trees {k_sel} | selects {fr['selected_at_burden']} (union {fr['union_selected_in_region']}) | recall {fr['recall_at_burden']}/{fr['positives']} (union {fr['union_recall']}) | rule-missed {fr['rule_missed_recovered']}/{fr['rule_missed']} | neg {fr['neg_selected_at_burden']}/{fr['negatives']} | PR-AUC {fr['case_control_prauc']} | {fr['seconds']}s",
        flush=True,
    )
    json.dump(
        results, open(os.path.join(OUT, f"train_results{SUF}.json"), "w"), indent=1
    )

oof = D.d[
    [
        "source_id",
        "field",
        "region",
        "id",
        "ra",
        "dec",
        "sky_group",
        "y",
        "y_strict",
        "picked",
        "in_any_list",
        "in_hviding25_B1",
    ]
].copy()
oof["score_mean"], oof["score_std"] = oof_mean, oof_std
for k, v in thr.items():
    oof[k] = v
oof["sel_burden"] = oof.score_mean >= oof.t_burden
oof["sel_neg1"] = oof.score_mean >= oof.t_neg1
oof["sel_05"] = oof.score_mean >= oof.t_05
oof.to_parquet(os.path.join(OUT, f"oof_scores{SUF}.parquet"), index=False)
sb = oof.sel_burden.values
results["primary_pooled"] = {
    "positives_in_support": int(is_pos.sum()),
    "recall_at_burden": int((sb & is_pos).sum()),
    "union_recall": int((picked & is_pos).sum()),
    "rule_missed": int((is_pos & ~picked).sum()),
    "rule_missed_recovered": int((sb & is_pos & ~picked).sum()),
    "strict_recovered": int((sb & strict).sum()),
    "strict_missed_recovered": int((sb & strict & ~picked).sum()),
    "negatives": int(is_neg.sum()),
    "neg_selected_at_burden": int((sb & is_neg).sum()),
    "selected_at_burden": int(sb.sum()),
    "union_selected": int(picked.sum()),
    "selected_at_05": int(oof.sel_05.sum()),
    "recall_at_05": int((oof.sel_05.values & is_pos).sum()),
    "rule_missed_recovered_at_05": int((oof.sel_05.values & is_pos & ~picked).sum()),
    "case_control_prauc_pooled": round(
        float(
            average_precision_score(
                np.r_[np.ones(is_pos.sum()), np.zeros(is_neg.sum())],
                np.r_[oof_mean[is_pos], oof_mean[is_neg]],
            )
        ),
        4,
    ),
}
print(f"PRIMARY{SUF} pooled:", json.dumps(results["primary_pooled"]), flush=True)

# shuffled labels x2
labelled = is_pos | is_neg
bs = pd.DataFrame({"source_id": D.d.source_id.values})
for s in (1, 2):
    sc = np.full(n, np.nan)
    sel = np.zeros(n, bool)
    for R in REGIONS:
        T = [x for x in REGIONS if x != R]
        trm, tem = np.isin(region, T), region == R
        fr = results["folds"][R]
        rng = np.random.default_rng(s)
        lab_rows = np.where(trm & labelled)[0]
        fake = np.zeros(n, bool)
        fake[rng.choice(lab_rows, int((trm & is_pos).sum()), replace=False)] = True
        fake_neg = labelled & ~fake
        seeds = [7000000 + 100000 * REGIONS.index(R) + 1000 * s + b for b in range(8)]
        models = D.train_bags(
            trm,
            fr["chosen_anchor"],
            fr["chosen_trees"],
            seeds,
            ALL,
            pos_mask=fake,
            neg_mask=fake_neg,
        )
        te_rows = np.where(tem & labelled)[0]
        sub = rng_sub.choice(np.where(trm)[0], 50000, replace=False)
        m_te, _ = D.predict(models, te_rows, ALL)
        m_sub, _ = D.predict(models, sub, ALL)
        sc[te_rows] = m_te
        sel[te_rows] = m_te >= np.quantile(m_sub, 1 - fr["union_burden_train"])
    results["baselines"][f"shuffled_labels_{s}"] = {
        "recall_at_burden": int((sel & is_pos).sum()),
        "rule_missed_recovered": int((sel & is_pos & ~picked).sum()),
        "strict_recovered": int((sel & strict).sum()),
        "strict_missed_recovered": int((sel & strict & ~picked).sum()),
        "neg_selected_at_burden": int((sel & is_neg).sum()),
        "case_control_prauc_pooled": round(
            float(
                average_precision_score(
                    np.r_[np.ones(is_pos.sum()), np.zeros(is_neg.sum())],
                    np.r_[sc[is_pos], sc[is_neg]],
                )
            ),
            4,
        ),
        "per_region_recall": {
            R: int((sel & is_pos & (region == R)).sum()) for R in REGIONS
        },
    }
    bs[f"score_shuffled_labels_{s}"] = sc
    print(
        "  shuffled",
        s,
        json.dumps(results["baselines"][f"shuffled_labels_{s}"]),
        flush=True,
    )
bs[labelled].to_parquet(os.path.join(OUT, f"baseline_scores{SUF}.parquet"), index=False)

# field adversary and dispersion
rcode = pd.Categorical(region, categories=REGIONS).codes
rows = np.concatenate(
    [rng_sub.choice(np.where(region == R)[0], 12000, replace=False) for R in REGIONS]
)
rng_sub.shuffle(rows)
folds5 = np.arange(len(rows)) % 5
acc, imp = [], np.zeros(len(D.features))
for f_ in range(5):
    tr, te = rows[folds5 != f_], rows[folds5 == f_]
    m = lgb.train(
        dict(
            objective="multiclass",
            num_class=5,
            learning_rate=0.05,
            num_leaves=31,
            verbose=-1,
            num_threads=8,
            seed=f_,
        ),
        lgb.Dataset(D.X[tr], label=rcode[tr]),
        300,
    )
    acc.append(float((m.predict(D.X[te]).argmax(axis=1) == rcode[te]).mean()))
    imp += m.feature_importance("gain")
top = sorted(zip(imp / imp.sum(), D.features), reverse=True)[:6]
results["sanity"]["field_adversary"] = {
    "accuracy_mean": round(float(np.mean(acc)), 4),
    "chance": 0.2,
    "top_features_gain_share": [(f_, round(float(g), 3)) for g, f_ in top],
}
results["sanity"]["primary_score_std"] = {
    "median_all": float(np.nanmedian(oof_std)),
    "median_positives": float(np.nanmedian(oof_std[is_pos])),
    "median_selected_at_burden": float(np.nanmedian(oof_std[sb])),
}
results["seconds_total"] = round(time.time() - T0)
json.dump(results, open(os.path.join(OUT, f"train_results{SUF}.json"), "w"), indent=1)
print("done", results["seconds_total"], "s", flush=True)
