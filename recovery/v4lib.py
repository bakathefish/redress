"""Shared machinery of the v4 recovery classifier: data view, PU+N bag sampler, LightGBM
fit/predict, nested inner selection. Mirrors recovery/v4_train_lgbm.py exactly (that
script is the run of record for the out-of-fold numbers; this module serves the final fit,
the feature-set variants and the deployment)."""

import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

OUT = "recovery"
REGIONS = ["CEERS", "GOODS-N", "GOODS-S", "COSMOS", "UDS"]
ANCHORS = [0.0, 0.10, 0.25, 0.50]
CHECK = list(range(50, 1250, 50))
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
N_JOBS = 4


class Data:
    def __init__(self, suffix=""):
        man = json.load(open(os.path.join(OUT, f"feature_manifest{suffix}.json")))
        self.manifest = man
        self.features = man["features"]
        d = pd.read_parquet(os.path.join(OUT, f"features{suffix}.parquet"))
        d = d[d.in_support.values].reset_index(drop=True)
        self.d = d
        self.n = len(d)
        self.X = d[self.features].values.astype(np.float32)
        y = d.y.values
        self.region = d.region.values
        self.is_pos = y == 1
        self.is_neg = y == 0
        self.is_amb = d.ambiguous.values.astype(bool)
        is_unl = np.isnan(y) & ~self.is_amb
        lab_groups = np.unique(
            d.sky_group.values[self.is_pos | self.is_neg | self.is_amb]
        )
        self.is_unl = is_unl & ~np.isin(d.sky_group.values, lab_groups)
        self.picked = d.picked.values.astype(bool)
        self.strict = d.y_strict.values.astype(bool)
        rcode = pd.Categorical(self.region, categories=REGIONS).codes.astype(np.int64)
        magbin = np.floor(d.m_f444w.values / 0.5).astype(np.int64)
        q = np.nanquantile(d.asnr_f444w.values[self.is_unl], [0.25, 0.5, 0.75])
        snrq = np.searchsorted(q, d.asnr_f444w.values).astype(np.int64)
        self.cell = rcode * 100000 + (magbin + 200) * 10 + snrq
        self.cell_rm = rcode * 1000 + (magbin + 200)
        u = np.where(self.is_unl)[0]
        self.u_by_cell = pd.Series(u).groupby(self.cell[u]).apply(np.asarray).to_dict()
        self.u_by_rm = pd.Series(u).groupby(self.cell_rm[u]).apply(np.asarray).to_dict()
        self.t0 = time.time()

    def make_bag(self, train_mask, rng, anchor, pos_mask=None, neg_mask=None):
        pm = self.is_pos if pos_mask is None else pos_mask
        nm = self.is_neg if neg_mask is None else neg_mask
        P = np.where(train_mask & pm)[0]
        N = np.where(train_mask & nm)[0]
        U = np.where(train_mask & self.is_unl)[0]
        half = U_PER_POS // 2
        u_uni = rng.choice(U, half * len(P), replace=False)
        u_mat = []
        for p in P:
            pool = self.u_by_cell.get(self.cell[p])
            if pool is None or len(pool) < 3:
                pool = self.u_by_rm.get(self.cell_rm[p])
            if pool is None or len(pool) < 3:
                pool = U
            pool = pool[train_mask[pool]]
            if len(pool) == 0:
                pool = U
            u_mat.append(rng.choice(pool, half, replace=len(pool) < half))
        Ub = np.concatenate([u_uni, np.concatenate(u_mat)])
        idx = [P, Ub]
        lab = [np.ones(len(P)), np.zeros(len(Ub))]
        wts = [np.ones(len(P)), np.full(len(Ub), len(P) / len(Ub))]
        if anchor > 0 and len(N):
            idx.append(N)
            lab.append(np.zeros(len(N)))
            wts.append(np.full(len(N), anchor * len(P) / len(N)))
        return np.concatenate(idx), np.concatenate(lab), np.concatenate(wts)

    def fit_bag(self, idx, lab, w, seed, n_trees, feats):
        ds = lgb.Dataset(self.X[idx][:, feats], label=lab, weight=w, free_raw_data=True)
        p = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        return lgb.train(p, ds, num_boost_round=n_trees)

    def train_bags(
        self, train_mask, anchor, n_trees, seeds, feats, pos_mask=None, neg_mask=None
    ):
        def one(seed):
            rng = np.random.default_rng(seed)
            idx, lab, w = self.make_bag(train_mask, rng, anchor, pos_mask, neg_mask)
            return self.fit_bag(idx, lab, w, seed, n_trees, feats)

        return Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(one)(s) for s in seeds)

    def predict(self, models, rows, feats, k=None):
        Xs = self.X[rows][:, feats]
        preds = np.stack([m.predict(Xs, num_iteration=k) for m in models])
        return preds.mean(axis=0), preds.std(axis=0)

    def predict_staged(self, models, rows, feats):
        Xs = self.X[rows][:, feats]
        out = np.zeros((len(CHECK), len(rows)))
        for m in models:
            for j, k in enumerate(CHECK):
                out[j] += m.predict(Xs, num_iteration=k)
        return out / len(models)

    def inner_select(
        self, T, feats, n_bags=8, anchors=ANCHORS, rng_sub=None, tag="", seed_base=0
    ):
        """Nested selection of (anchor, n_trees) using the regions in T only: mean case-control
        PR-AUC across the inner held-out regions, one-standard-error rule, equal-burden recall
        within two positives of the maximum; also the catalog fraction at a 1% negative rate."""
        rng_sub = rng_sub or np.random.default_rng(2026)
        curves = {
            a: {"prauc": [], "rec": [], "frac_neg1": [], "npos": []} for a in anchors
        }
        for r in T:
            tr = np.isin(self.region, [x for x in T if x != r])
            teP = np.where((self.region == r) & self.is_pos)[0]
            teN = np.where((self.region == r) & self.is_neg)[0]
            teS = rng_sub.choice(np.where(self.region == r)[0], 20000, replace=False)
            f_tr = self.picked[tr].mean()
            rows = np.concatenate([teP, teN, teS])
            lab = np.concatenate([np.ones(len(teP)), np.zeros(len(teN))])
            for a in anchors:
                seeds = [
                    seed_base + 1000 * REGIONS.index(r) + 10 * int(a * 100) + b
                    for b in range(n_bags)
                ]
                models = self.train_bags(tr, a, N_MAX, seeds, feats)
                S = self.predict_staged(models, rows, feats)
                sP, sN, sS = (
                    S[:, : len(teP)],
                    S[:, len(teP) : len(teP) + len(teN)],
                    S[:, len(teP) + len(teN) :],
                )
                pr, rc, fn = [], [], []
                for j in range(len(CHECK)):
                    pr.append(
                        average_precision_score(lab, np.concatenate([sP[j], sN[j]]))
                    )
                    t = np.quantile(sS[j], 1.0 - f_tr)
                    rc.append(int((sP[j] >= t).sum()))
                    fn.append(float((sS[j] >= np.quantile(sN[j], 0.99)).mean()))
                curves[a]["prauc"].append(pr)
                curves[a]["rec"].append(rc)
                curves[a]["frac_neg1"].append(fn)
                curves[a]["npos"].append(len(teP))
            print(
                f"  inner {tag} held-out {r}: done at {time.time() - self.t0:.0f}s",
                flush=True,
            )
        best = (-1, None, None)
        table = {}
        for a in anchors:
            pr = np.array(curves[a]["prauc"])
            rc = np.array(curves[a]["rec"]).sum(axis=0)
            table[a] = (
                pr.mean(axis=0),
                pr.std(axis=0, ddof=1) / np.sqrt(pr.shape[0]),
                rc,
            )
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
            "curve_recall_sum": {
                str(a): [int(v) for v in table[a][2]] for a in anchors
            },
            "checkpoints": CHECK,
        }
        return a_sel, int(k_sel), fn, summary
