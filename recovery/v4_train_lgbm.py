"""v4 build step 4+5: the primary PU+N LightGBM ensemble under nested leave-one-macroregion-out,
with the inner one-standard-error selection of tree count and negative-anchor weight, the
equal-burden / negative-rate / 0.5 per cent operating points chosen inside the training
regions, and the baselines and sanity checks the design requires (the design round).

Every reported score for a source comes from an ensemble that never saw that source's
macroregion; every threshold applied to a held-out region was computed on the other four.

Outputs (recovery/):
  oof_scores.parquet   every support row: out-of-fold score mean/std, fold thresholds, selections
  train_results.json   inner curves, chosen settings, thresholds, baseline and sanity summaries
  baseline_scores.parquet  out-of-fold scores of the labelled rows for every variant (paired tests)
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "recovery"
T0 = time.time()

man = json.load(open(os.path.join(OUT, "feature_manifest.json")))
FEATURES = man["features"]
REGIONS = ["CEERS", "GOODS-N", "GOODS-S", "COSMOS", "UDS"]
ANCHORS = [0.0, 0.10, 0.25, 0.50]
CHECK = list(range(50, 1250, 50))  # 24 checkpoints, every 50 trees
N_MAX = 1200
U_PER_POS = 20
PARAMS = dict(
    objective="binary",
    learning_rate=0.025,
    num_leaves=7,
    max_depth=3,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_alpha=0.25,
    reg_lambda=5.0,
    min_split_gain=0.01,
    verbose=-1,
    num_threads=4,
)
N_JOBS = 4  # bags in parallel (threads); LightGBM releases the GIL

# ------------------------------------------------------------------ data (support rows only)
d = pd.read_parquet(os.path.join(OUT, "features.parquet"))
d = d[d.in_support.values].reset_index(drop=True)
n = len(d)
X = d[FEATURES].values.astype(np.float32)
y = d.y.values
region = d.region.values
is_pos = y == 1
is_neg = y == 0
is_amb = d.ambiguous.values.astype(bool)
is_unl = np.isnan(y) & ~is_amb
# twin rows of any labelled or ambiguous object (overlapping tiles, same sky group) never enter the unlabelled pool
_lab_groups = np.unique(d.sky_group.values[is_pos | is_neg | d.ambiguous.values.astype(bool)])
is_unl = is_unl & ~np.isin(d.sky_group.values, _lab_groups)
picked = d.picked.values.astype(bool)
strict = d.y_strict.values.astype(bool)
rcode = pd.Categorical(region, categories=REGIONS).codes.astype(np.int64)
magbin = np.floor(d.m_f444w.values / 0.5).astype(np.int64)
q = np.nanquantile(d.asnr_f444w.values[is_unl], [0.25, 0.5, 0.75])
snrq = np.searchsorted(q, d.asnr_f444w.values).astype(np.int64)
cell = rcode * 100000 + (magbin + 200) * 10 + snrq
cell_rm = rcode * 1000 + (magbin + 200)
u_by_cell = (
    pd.Series(np.where(is_unl)[0]).groupby(cell[is_unl]).apply(np.asarray).to_dict()
)
u_by_rm = (
    pd.Series(np.where(is_unl)[0]).groupby(cell_rm[is_unl]).apply(np.asarray).to_dict()
)
print(
    f"support rows {n} | positives {is_pos.sum()} | negatives {is_neg.sum()} | unlabelled {is_unl.sum()} | ambiguous {is_amb.sum()}",
    flush=True,
)


def make_bag(train_mask, rng, anchor, pos_mask=None, neg_mask=None):
    """Row indices, labels and weights of one PU+N bag drawn inside train_mask."""
    pm = is_pos if pos_mask is None else pos_mask
    nm = is_neg if neg_mask is None else neg_mask
    P = np.where(train_mask & pm)[0]
    N = np.where(train_mask & nm)[0]
    U = np.where(train_mask & is_unl)[0]
    half = U_PER_POS // 2
    u_uni = rng.choice(U, half * len(P), replace=False)
    u_mat = []
    for p in P:
        pool = u_by_cell.get(cell[p])
        if pool is None or len(pool) < 3:
            pool = u_by_rm.get(cell_rm[p])
        if pool is None or len(pool) < 3:
            pool = U
        pool = pool[train_mask[pool]]  # matched rows must lie in the training regions
        if len(pool) == 0:
            pool = U
        u_mat.append(rng.choice(pool, half, replace=len(pool) < half))
    Ub = np.concatenate([u_uni, np.concatenate(u_mat)])
    idx = [P, Ub]
    lab = [np.ones(len(P)), np.zeros(len(Ub))]
    # positives weigh 1 each; unlabelled sum to |P|; negatives sum to anchor * |P|
    wts = [np.ones(len(P)), np.full(len(Ub), len(P) / len(Ub))]
    if anchor > 0 and len(N):
        idx.append(N)
        lab.append(np.zeros(len(N)))
        wts.append(np.full(len(N), anchor * len(P) / len(N)))
    return np.concatenate(idx), np.concatenate(lab), np.concatenate(wts)


def fit_bag(idx, lab, w, seed, n_trees, feats):
    ds = lgb.Dataset(X[idx][:, feats], label=lab, weight=w, free_raw_data=True)
    p = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    return lgb.train(p, ds, num_boost_round=n_trees)


def train_bags(train_mask, anchor, n_trees, seeds, feats, pos_mask=None, neg_mask=None):
    def one(seed):
        rng = np.random.default_rng(seed)
        idx, lab, w = make_bag(train_mask, rng, anchor, pos_mask, neg_mask)
        return fit_bag(idx, lab, w, seed, n_trees, feats)

    return Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(one)(s) for s in seeds)


def predict(models, rows, feats, k=None):
    Xs = X[rows][:, feats]
    preds = np.stack([m.predict(Xs, num_iteration=k) for m in models])
    return preds.mean(axis=0), preds.std(axis=0)


def predict_staged(models, rows, feats):
    Xs = X[rows][:, feats]
    out = np.zeros((len(CHECK), len(rows)))
    for m in models:
        for j, k in enumerate(CHECK):
            out[j] += m.predict(Xs, num_iteration=k)
    return out / len(models)


ALL = np.arange(len(FEATURES))
results = {"folds": {}, "baselines": {}, "sanity": {}}
oof_mean = np.full(n, np.nan)
oof_std = np.full(n, np.nan)
thr_cols = {k: np.full(n, np.nan) for k in ("t_burden", "t_neg1", "t_05")}
rng_sub = np.random.default_rng(2026)


def inner_select(T, feats=ALL, n_bags=8, anchors=ANCHORS, pos_mask=None, tag=""):
    """Nested selection of (anchor, n_trees) using the regions in T only."""
    curves = {a: {"prauc": [], "rec": [], "frac_neg1": [], "npos": []} for a in anchors}
    for r in T:
        tr = np.isin(region, [x for x in T if x != r])
        pm = is_pos if pos_mask is None else pos_mask
        teP = np.where((region == r) & pm)[0]
        teN = np.where((region == r) & is_neg)[0]
        teS = rng_sub.choice(np.where(region == r)[0], 20000, replace=False)
        f_tr = picked[tr].mean()
        rows = np.concatenate([teP, teN, teS])
        lab = np.concatenate([np.ones(len(teP)), np.zeros(len(teN))])
        for a in anchors:
            seeds = [
                1000 * REGIONS.index(r) + 10 * int(a * 100) + b for b in range(n_bags)
            ]
            models = train_bags(tr, a, N_MAX, seeds, feats, pos_mask=pos_mask)
            S = predict_staged(models, rows, feats)
            sP, sN, sS = (
                S[:, : len(teP)],
                S[:, len(teP) : len(teP) + len(teN)],
                S[:, len(teP) + len(teN) :],
            )
            pr, rc, fn = [], [], []
            for j in range(len(CHECK)):
                pr.append(average_precision_score(lab, np.concatenate([sP[j], sN[j]])))
                t = np.quantile(sS[j], 1.0 - f_tr)
                rc.append(int((sP[j] >= t).sum()))
                t1 = np.quantile(sN[j], 0.99)
                fn.append(float((sS[j] >= t1).mean()))
            curves[a]["prauc"].append(pr)
            curves[a]["rec"].append(rc)
            curves[a]["frac_neg1"].append(fn)
            curves[a]["npos"].append(len(teP))
        print(
            f"  inner {tag} held-out {r}: done at {time.time() - T0:.0f}s", flush=True
        )
    # one-standard-error rule on mean case-control PR-AUC, recall within two positives of max
    best = (-1, None, None)
    table = {}
    for a in anchors:
        pr = np.array(curves[a]["prauc"])  # folds x checkpoints
        rc = np.array(curves[a]["rec"]).sum(axis=0)
        table[a] = (pr.mean(axis=0), pr.std(axis=0, ddof=1) / np.sqrt(pr.shape[0]), rc)
        j = int(np.argmax(table[a][0]))
        if table[a][0][j] > best[0]:
            best = (table[a][0][j], a, j)
    se = table[best[1]][1][best[2]]
    rec_max = max(int(table[a][2].max()) for a in anchors)
    cands = []
    for a in anchors:
        m, s, rc = table[a]
        for j in range(len(CHECK)):
            if m[j] >= best[0] - se and rc[j] >= rec_max - 2:
                cands.append((CHECK[j], abs(a - 0.25), a, j))
    cands.sort()
    k_sel, _, a_sel, j_sel = cands[0]
    fn = float(np.mean(np.array(curves[a_sel]["frac_neg1"])[:, j_sel]))
    summary = {
        "chosen_anchor": a_sel,
        "chosen_trees": int(k_sel),
        "best_mean_prauc": round(float(best[0]), 4),
        "best_anchor": best[1],
        "best_trees": int(CHECK[best[2]]),
        "one_se": round(float(se), 4),
        "chosen_mean_prauc": round(float(table[a_sel][0][j_sel]), 4),
        "chosen_inner_recall_sum": int(table[a_sel][2][j_sel]),
        "inner_recall_sum_max": int(rec_max),
        "inner_positives_total": int(sum(curves[a_sel]["npos"])),
        "frac_for_neg1pct": fn,
        "curve_mean_prauc": {
            str(a): [round(float(v), 4) for v in table[a][0]] for a in anchors
        },
        "curve_recall_sum": {str(a): [int(v) for v in table[a][2]] for a in anchors},
        "checkpoints": CHECK,
    }
    return a_sel, int(k_sel), fn, summary


# ------------------------------------------------------------------ primary: nested outer loop
for R in REGIONS:
    T = [x for x in REGIONS if x != R]
    trm = np.isin(region, T)
    tem = region == R
    f_T = float(picked[trm].mean())
    print(
        f"outer fold held-out {R}: union burden in training regions {f_T:.5f}",
        flush=True,
    )
    a_sel, k_sel, f_neg1, summ = inner_select(T, tag=R)
    seeds = [100000 * (REGIONS.index(R) + 1) + s for s in range(48)]
    models = train_bags(trm, a_sel, k_sel, seeds, ALL)
    te_rows = np.where(tem)[0]
    tr_rows = np.where(trm)[0]
    m_te, s_te = predict(models, te_rows, ALL)
    m_tr, _ = predict(models, tr_rows, ALL)
    t_b = float(np.quantile(m_tr, 1.0 - f_T))
    t_n = float(np.quantile(m_tr, 1.0 - f_neg1))
    t_5 = float(np.quantile(m_tr, 0.995))
    oof_mean[te_rows], oof_std[te_rows] = m_te, s_te
    thr_cols["t_burden"][te_rows] = t_b
    thr_cols["t_neg1"][te_rows] = t_n
    thr_cols["t_05"][te_rows] = t_5
    P = tem & is_pos
    N = tem & is_neg
    sel = m_te >= t_b
    selN = oof_mean[np.where(N)[0]] >= t_n
    results["folds"][R] = dict(
        summ,
        union_burden_train=f_T,
        t_burden=t_b,
        t_neg1=t_n,
        t_05=t_5,
        n_held_out=int(tem.sum()),
        selected_at_burden=int(sel.sum()),
        union_selected_in_region=int((picked & tem).sum()),
        positives=int(P.sum()),
        recall_at_burden=int((oof_mean[np.where(P)[0]] >= t_b).sum()),
        union_recall=int((picked & P).sum()),
        rule_missed=int((P & ~picked).sum()),
        rule_missed_recovered=int((oof_mean[np.where(P & ~picked)[0]] >= t_b).sum()),
        negatives=int(N.sum()),
        neg_selected_at_burden=int((oof_mean[np.where(N)[0]] >= t_b).sum()),
        neg_selected_at_neg1=int(selN.sum()),
        case_control_prauc=round(
            float(
                average_precision_score(
                    np.r_[np.ones(P.sum()), np.zeros(N.sum())],
                    np.r_[oof_mean[np.where(P)[0]], oof_mean[np.where(N)[0]]],
                )
            ),
            4,
        ),
        seconds=round(time.time() - T0),
    )
    fr = results["folds"][R]
    print(
        f"  {R}: anchor {a_sel} trees {k_sel} | at equal burden selects {fr['selected_at_burden']} (union {fr['union_selected_in_region']}) | "
        f"recall {fr['recall_at_burden']}/{fr['positives']} (union {fr['union_recall']}) | rule-missed {fr['rule_missed_recovered']}/{fr['rule_missed']} | "
        f"neg rate {fr['neg_selected_at_burden']}/{fr['negatives']} | PR-AUC {fr['case_control_prauc']} | {fr['seconds']}s",
        flush=True,
    )
    json.dump(results, open(os.path.join(OUT, "train_results.json"), "w"), indent=1)

oof = d[
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
oof["score_mean"] = oof_mean
oof["score_std"] = oof_std
for k, v in thr_cols.items():
    oof[k] = v
oof["sel_burden"] = oof.score_mean >= oof.t_burden
oof["sel_neg1"] = oof.score_mean >= oof.t_neg1
oof["sel_05"] = oof.score_mean >= oof.t_05
oof.to_parquet(os.path.join(OUT, "oof_scores.parquet"), index=False)

P_all = is_pos
tot = {
    "positives_in_support": int(P_all.sum()),
    "recall_at_burden": int((oof.sel_burden.values & P_all).sum()),
    "union_recall": int((picked & P_all).sum()),
    "rule_missed": int((P_all & ~picked).sum()),
    "rule_missed_recovered": int((oof.sel_burden.values & P_all & ~picked).sum()),
    "strict": int(strict.sum()),
    "strict_recovered": int((oof.sel_burden.values & strict).sum()),
    "strict_missed": int((strict & ~picked).sum()),
    "strict_missed_recovered": int((oof.sel_burden.values & strict & ~picked).sum()),
    "negatives": int(is_neg.sum()),
    "neg_selected_at_burden": int((oof.sel_burden.values & is_neg).sum()),
    "selected_at_burden": int(oof.sel_burden.sum()),
    "union_selected": int(picked.sum()),
    "selected_at_neg1": int(oof.sel_neg1.sum()),
    "recall_at_neg1": int((oof.sel_neg1.values & P_all).sum()),
    "neg_selected_at_neg1": int((oof.sel_neg1.values & is_neg).sum()),
    "selected_at_05": int(oof.sel_05.sum()),
    "recall_at_05": int((oof.sel_05.values & P_all).sum()),
    "rule_missed_recovered_at_05": int((oof.sel_05.values & P_all & ~picked).sum()),
    "neg_selected_at_05": int((oof.sel_05.values & is_neg).sum()),
    "case_control_prauc_pooled": round(
        float(
            average_precision_score(
                np.r_[np.ones(P_all.sum()), np.zeros(is_neg.sum())],
                np.r_[oof_mean[P_all], oof_mean[is_neg]],
            )
        ),
        4,
    ),
}
results["primary_pooled"] = tot
print("PRIMARY (pooled out-of-fold):", json.dumps(tot), flush=True)
json.dump(results, open(os.path.join(OUT, "train_results.json"), "w"), indent=1)


# ------------------------------------------------------------------ variants and sanity checks
def run_variant(
    name,
    feats=ALL,
    n_bags=8,
    mode="pu",
    pos_train_mask=None,
    shuffle_seed=None,
    anchor=None,
    extra=None,
):
    """Out-of-fold scores on the labelled rows (and a 50k training-region subsample per fold
    for the equal-burden threshold) under the same five outer folds; settings copied from
    the primary fold's inner selection (no new inner loop; a variant is a comparison, not a
    contender, unless it is promoted by the paired rule)."""
    sc = np.full(n, np.nan)
    per_fold = {}
    labelled = np.where(is_pos | is_neg)[0]
    for R in REGIONS:
        T = [x for x in REGIONS if x != R]
        trm = np.isin(region, T)
        tem = region == R
        fr = results["folds"][R]
        a = fr["chosen_anchor"] if anchor is None else anchor
        k = fr["chosen_trees"]
        f_T = fr["union_burden_train"]
        seeds = [
            7000000 + 100000 * REGIONS.index(R) + 1000 * (shuffle_seed or 0) + s
            for s in range(n_bags)
        ]
        te_rows = labelled[tem[labelled]]
        sub = rng_sub.choice(np.where(trm)[0], 50000, replace=False)
        if mode == "pu":
            pm = None
            if pos_train_mask is not None:
                pm = is_pos & pos_train_mask
            models = train_bags(trm, a, k, seeds, feats, pos_mask=pm)
        elif mode == "shuffle":
            rng = np.random.default_rng(shuffle_seed)
            lab_rows = np.where(trm & (is_pos | is_neg))[0]
            fake = np.zeros(n, bool)
            fake[rng.choice(lab_rows, int((trm & is_pos).sum()), replace=False)] = True
            fake_neg = (is_pos | is_neg) & ~fake
            models = train_bags(
                trm, a, k, seeds, feats, pos_mask=fake, neg_mask=fake_neg
            )
        elif mode == "pn":

            def one(seed):
                P = np.where(trm & is_pos)[0]
                N = np.where(trm & is_neg)[0]
                idx = np.concatenate([P, N])
                lab = np.concatenate([np.ones(len(P)), np.zeros(len(N))])
                w = np.concatenate([np.ones(len(P)), np.full(len(N), len(P) / len(N))])
                return fit_bag(idx, lab, w, seed, k, feats)

            models = Parallel(n_jobs=N_JOBS, prefer="threads")(
                delayed(one)(s) for s in seeds
            )
        elif mode == "rules":

            def one(seed):
                rng = np.random.default_rng(seed)
                rows = rng.choice(np.where(trm & is_unl)[0], 120000, replace=False)
                ds = lgb.Dataset(X[rows][:, feats], label=picked[rows].astype(float))
                p = dict(
                    PARAMS, num_leaves=31, max_depth=-1, learning_rate=0.05, seed=seed
                )
                return lgb.train(p, ds, num_boost_round=400)

            models = Parallel(n_jobs=N_JOBS, prefer="threads")(
                delayed(one)(s) for s in seeds[:2]
            )
        else:
            raise ValueError(mode)
        m_te, _ = predict(models, te_rows, feats)
        m_sub, _ = predict(models, sub, feats)
        t_b = float(np.quantile(m_sub, 1.0 - f_T))
        sc[te_rows] = m_te
        per_fold[R] = t_b
    sel = np.zeros(n, bool)
    for R, t_b in per_fold.items():
        rows = np.where((region == R) & (is_pos | is_neg))[0]
        sel[rows] = sc[rows] >= t_b
    out = {
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
        "seconds": round(time.time() - T0),
    }
    if extra:
        out.update(extra)
    print(f"  variant {name}: {json.dumps(out)}", flush=True)
    return sc, sel, out


bs = pd.DataFrame({"source_id": d.source_id.values})
labelled_mask = is_pos | is_neg
fi = {f: i for i, f in enumerate(FEATURES)}
mag_only = np.array(
    [
        fi[f"m_{b}"]
        for b in ("f090w", "f115w", "f150w", "f200w", "f277w", "f356w", "f444w")
    ]
)
morph_only = np.array([fi["log_rh"], fi["log_rh_over_rstar"]])
no_c277444 = np.array([i for f, i in fi.items() if f != "c_f277w_f444w"])
no_size = np.array(
    [i for f, i in fi.items() if f not in ("log_rh", "log_rh_over_rstar")]
)

variants = [
    ("primary_8bag_replica", dict()),
    ("magnitude_only", dict(feats=mag_only)),
    ("morphology_only", dict(feats=morph_only)),
    ("ablate_c_f277w_f444w", dict(feats=no_c277444)),
    ("ablate_size", dict(feats=no_size)),
    ("anchor_0", dict(anchor=0.0)),
    ("anchor_0.50", dict(anchor=0.50)),
    ("pn_lightgbm", dict(mode="pn")),
    ("rules_reproduction", dict(mode="rules")),
]
for c in ("hviding25_A1", "barro25", "degraaff26"):
    variants.append(
        (
            f"leave_out_{c}",
            dict(
                pos_train_mask=~d[f"in_{c}"].values.astype(bool),
                extra={"held_out_catalog": c},
            ),
        )
    )
for s in range(5):
    variants.append((f"shuffled_labels_{s}", dict(mode="shuffle", shuffle_seed=s + 1)))

for name, kw in variants:
    sc, sel, out = run_variant(name, **kw)
    if name.startswith("leave_out_"):
        c = kw["extra"]["held_out_catalog"]
        inc = d[f"in_{c}"].values.astype(bool)
        out["recall_on_left_out_catalog"] = (
            f"{int((sel & is_pos & inc).sum())} of {int((is_pos & inc).sum())}"
        )
        out["primary_recall_on_that_catalog"] = (
            f"{int((oof.sel_burden.values & is_pos & inc).sum())} of {int((is_pos & inc).sum())}"
        )
    results["baselines"][name] = out
    bs[f"score_{name}"] = sc
    bs[f"sel_{name}"] = sel
    json.dump(results, open(os.path.join(OUT, "train_results.json"), "w"), indent=1)

# z_phot variant: 31 features (LightGBM handles missing z_phot natively)
Xz = np.column_stack([X, d.z_phot.values.astype(np.float32)])
X_backup = X
X = Xz
sc, sel, out = run_variant("with_zphot", feats=np.arange(31))
results["baselines"]["with_zphot"] = out
bs["score_with_zphot"] = sc
bs["sel_with_zphot"] = sel
X = X_backup

# field-prediction adversary: can the 30 features tell the macroregion apart?
rows = np.concatenate(
    [rng_sub.choice(np.where(region == R)[0], 12000, replace=False) for R in REGIONS]
)
rng_sub.shuffle(rows)
folds5 = np.arange(len(rows)) % 5
acc, imp = [], np.zeros(len(FEATURES))
for f in range(5):
    tr, te = rows[folds5 != f], rows[folds5 == f]
    ds = lgb.Dataset(X[tr], label=rcode[tr])
    m = lgb.train(
        dict(
            objective="multiclass",
            num_class=5,
            learning_rate=0.05,
            num_leaves=31,
            verbose=-1,
            num_threads=16,
            seed=f,
        ),
        ds,
        300,
    )
    acc.append(float((m.predict(X[te]).argmax(axis=1) == rcode[te]).mean()))
    imp += m.feature_importance("gain")
top = sorted(zip(imp / imp.sum(), FEATURES), reverse=True)[:6]
results["sanity"]["field_adversary"] = {
    "accuracy_mean": round(float(np.mean(acc)), 4),
    "chance": 0.2,
    "top_features_gain_share": [(f, round(float(g), 3)) for g, f in top],
}
# bag dispersion of the primary ensemble
results["sanity"]["primary_score_std"] = {
    "median_all": float(np.nanmedian(oof_std)),
    "median_positives": float(np.nanmedian(oof_std[is_pos])),
    "median_selected_at_burden": float(np.nanmedian(oof_std[oof.sel_burden.values])),
}
bs[labelled_mask].to_parquet(os.path.join(OUT, "baseline_scores.parquet"), index=False)
results["seconds_total"] = round(time.time() - T0)
json.dump(results, open(os.path.join(OUT, "train_results.json"), "w"), indent=1)
print("done in", results["seconds_total"], "s", flush=True)
