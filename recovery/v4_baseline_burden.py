"""Round 2, reviewer E finding 1/2: recover the REALISED CATALOGUE BURDEN of every baseline,
ablation, shuffled control and the imitation-of-the-rules model.

The run of record (recovery/v4_train_lgbm.py) scored each variant on the 5126 labelled rows
only, so no variant has a catalogue-wide burden anyone can check. This script re-runs every
variant with the record's exact code path -- same data, same masks, same seeds, same bag
counts, same per-fold settings taken from the record's inner selection, same threshold rule --
and scores ALL in-support rows of the held-out region instead of only the labelled ones.

Nothing about any variant is changed. Two things are done differently from the record and only
these two:
  (a) the inner nested selection is NOT re-run; its outputs (chosen_anchor, chosen_trees,
      union_burden_train) are read from recovery/train_results.json, and its consumption of
      the shared rng_sub stream is replayed call for call so that every variant's 50000-row
      training subsample -- and therefore every variant's threshold -- is the same draw the
      record used;
  (b) predictions are taken on every in-support row of the held-out region, of which the
      record's labelled rows are a subset. LightGBM prediction is row-independent, so the
      labelled-row scores are the record's scores and the labelled-row counts must reproduce
      exactly. That reproduction is checked for every variant and reported.

Outputs (recovery/):
  baseline_burden.json      realised burden, reproduction check and matched-burden view
  baseline_burden.md        the same as a readable table
  baseline_oof_full.parquet the catalogue-wide out-of-fold score and selection of every variant

Run from the repository root:
  python recovery/v4_baseline_burden.py
"""

import json
import math
import os
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "recovery"
T0 = time.time()

# ---------------------------------------------------------------- record constants (verbatim)
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


ALL = np.arange(len(FEATURES))

# ---------------------------------------------------------------- the record's fold settings
record = json.load(open(os.path.join(OUT, "train_results.json")))
results = {
    "folds": record["folds"]
}  # run_variant reads chosen_anchor / trees / burden here
for R in REGIONS:
    T = [x for x in REGIONS if x != R]
    f_T_here = float(picked[np.isin(region, T)].mean())
    f_T_rec = record["folds"][R]["union_burden_train"]
    assert f_T_here == f_T_rec, (R, f_T_here, f_T_rec)
print("fold settings and training-region burdens match the record exactly", flush=True)

# ---------------------------------------------------------------- replay of the shared stream
# rng_sub is a single stream shared by the inner selection and by every variant's threshold
# subsample. The inner loop drew 20000 rows once per (outer fold, inner held-out region), in
# the order the record's outer loop visits them, before any variant ran. Replaying those 20
# draws puts the stream in the state each variant's 50000-row draw was taken from.
rng_sub = np.random.default_rng(2026)
_replayed = 0
for R in REGIONS:
    for r in [x for x in REGIONS if x != R]:
        rng_sub.choice(np.where(region == r)[0], 20000, replace=False)
        _replayed += 1
assert _replayed == 20
print(
    f"replayed {_replayed} inner-selection draws on the shared rng_sub stream",
    flush=True,
)

# ---------------------------------------------------------------- full-support variant runner
labelled_all = np.where(is_pos | is_neg)[0]
full_rows_by_region = {R: np.where(region == R)[0] for R in REGIONS}
score_full = {}  # variant -> catalogue-wide out-of-fold score over all support rows
sel_full = {}  # variant -> catalogue-wide selection at the fold's equal-burden threshold


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
    """The record's run_variant, with the held-out scoring widened from the labelled rows to
    every in-support row of the held-out region. Training, seeds, subsample and threshold are
    untouched."""
    sc = np.full(n, np.nan)  # catalogue-wide out-of-fold score
    sel = np.zeros(n, bool)  # catalogue-wide selection at the fold threshold
    per_fold = {}
    labelled = labelled_all
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
        te_rows = labelled[
            tem[labelled]
        ]  # kept for stream fidelity; scoring uses full rows
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
        full = full_rows_by_region[R]
        m_full, _ = predict(models, full, feats)
        m_sub, _ = predict(models, sub, feats)
        t_b = float(np.quantile(m_sub, 1.0 - f_T))
        sc[full] = m_full
        sel[full] = m_full >= t_b
        per_fold[R] = t_b
        print(
            f"    {name} fold {R}: t_burden {t_b:.12f} | selects {int(sel[full].sum())} of {len(full)} | {time.time() - T0:.0f}s",
            flush=True,
        )
    # the record's labelled-row selection, recomputed from the same scores and thresholds
    sel_lab = np.zeros(n, bool)
    for R, t_b in per_fold.items():
        rows = np.where((region == R) & (is_pos | is_neg))[0]
        sel_lab[rows] = sc[rows] >= t_b
    repro = {
        "recall_at_burden": int((sel_lab & is_pos).sum()),
        "rule_missed_recovered": int((sel_lab & is_pos & ~picked).sum()),
        "strict_recovered": int((sel_lab & strict).sum()),
        "strict_missed_recovered": int((sel_lab & strict & ~picked).sum()),
        "neg_selected_at_burden": int((sel_lab & is_neg).sum()),
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
            R: int((sel_lab & is_pos & (region == R)).sum()) for R in REGIONS
        },
    }
    out = {
        "thresholds": {R: per_fold[R] for R in REGIONS},
        "labelled_row_reproduction": repro,
        "selected_rows_per_region": {
            R: int(sel[full_rows_by_region[R]].sum()) for R in REGIONS
        },
        "selected_rows_pooled": int(sel.sum()),
        "burden_fraction_pooled": float(sel.sum()) / n,
        "seconds": round(time.time() - T0),
    }
    if extra:
        out.update(extra)
    score_full[name] = sc
    sel_full[name] = sel
    return out


# ---------------------------------------------------------------- the record's variant list
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

# reviewer E's two priorities first, so a partial run is already usable; the shared rng_sub
# stream forbids reordering, so priority is expressed by writing output after every variant
# rather than by moving anything in the list.
BURDEN = {
    "machine": {
        "node": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm_num_threads": PARAMS["num_threads"],
        "joblib_n_jobs": N_JOBS,
    },
    "support_rows": int(n),
    "positives_in_support": int(is_pos.sum()),
    "rule_missed_in_support": int((is_pos & ~picked).sum()),
    "anchors_in_support": int(is_neg.sum()),
    "union_selected_in_support": int(picked.sum()),
    "variants": {},
}


def dump():
    BURDEN["seconds_total"] = round(time.time() - T0)
    json.dump(BURDEN, open(os.path.join(OUT, "baseline_burden.json"), "w"), indent=1)


for name, kw in variants:
    print(f"  variant {name}", flush=True)
    out = run_variant(name, **kw)
    BURDEN["variants"][name] = out
    dump()

# z_phot variant: 31 features (LightGBM handles missing z_phot natively)
print("  variant with_zphot", flush=True)
Xz = np.column_stack([X, d.z_phot.values.astype(np.float32)])
X_backup = X
X = Xz
BURDEN["variants"]["with_zphot"] = run_variant("with_zphot", feats=np.arange(31))
X = X_backup
dump()

# ---------------------------------------------------------------- reproduction against record
print("checking reproduction against the record", flush=True)
KEYS = (
    "recall_at_burden",
    "rule_missed_recovered",
    "strict_recovered",
    "strict_missed_recovered",
    "neg_selected_at_burden",
    "case_control_prauc_pooled",
    "per_region_recall",
)
for name, out in BURDEN["variants"].items():
    rec = record["baselines"][name]
    mine = out["labelled_row_reproduction"]
    diffs = {k: [rec[k], mine[k]] for k in KEYS if rec[k] != mine[k]}
    out["reproduces_record"] = not diffs
    out["reproduction_diffs"] = diffs
    if diffs:
        print(f"  MISMATCH {name}: {json.dumps(diffs)}", flush=True)
dump()

# ---------------------------------------------------------------- matched-burden view
den = json.load(open(os.path.join(OUT, "denominators.json")))
N_UNION = den["union_in_support_per_region"]
oof = pd.read_parquet(os.path.join(OUT, "oof_scores.parquet"))
assert len(oof) == n
assert (oof.source_id.values == d.source_id.values).all()
primary_score = oof.score_mean.values
score_full["__primary_48bag__"] = primary_score
rule_missed_mask = is_pos & ~picked


def matched_selection(sc):
    """Top N_union rows of each held-out region by score. Ties are broken by the row order of
    features.parquet, the same convention build_numbers.py uses; the tie bounds recorded
    alongside say whether any tie could have changed a count."""
    sel = np.zeros(n, bool)
    ties = {}
    for R in REGIONS:
        rows = full_rows_by_region[R]
        k = N_UNION[R]
        s = sc[rows]
        order = np.argsort(-s, kind="stable")
        take = rows[order[:k]]
        sel[take] = True
        cut = s[order[k - 1]]
        n_above = int((s > cut).sum())
        n_tied = int((s == cut).sum())
        pos_tied = int((is_pos[rows] & (s == cut)).sum())
        rm_tied = int((rule_missed_mask[rows] & (s == cut)).sum())
        ties[R] = {
            "cut_score": float(cut),
            "rows_strictly_above": n_above,
            "rows_tied_at_cut": n_tied,
            "slots_left_for_ties": k - n_above,
            "positives_tied_at_cut": pos_tied,
            "rule_missed_tied_at_cut": rm_tied,
            "tie_free": (n_above + n_tied) == k,
        }
    return sel, ties


def matched_stats(sc):
    sel, ties = matched_selection(sc)
    return {
        "matched_recall_support": int((sel & is_pos).sum()),
        "matched_rule_missed": int((sel & rule_missed_mask).sum()),
        "matched_anchors": int((sel & is_neg).sum()),
        "matched_recall_per_region": {
            R: int((sel & is_pos & (region == R)).sum()) for R in REGIONS
        },
        "matched_selected_per_region": {R: N_UNION[R] for R in REGIONS},
        "matched_selected_pooled": int(sel.sum()),
        "ties": ties,
    }, sel


matched_sel = {}
for name in list(BURDEN["variants"].keys()):
    st, sel = matched_stats(score_full[name])
    BURDEN["variants"][name]["matched"] = st
    matched_sel[name] = sel
st, sel = matched_stats(primary_score)
BURDEN["primary_48bag_matched"] = st
matched_sel["__primary_48bag__"] = sel
BURDEN["primary_48bag_deployed"] = {
    "selected_rows_pooled": int(oof.sel_burden.sum()),
    "selected_rows_per_region": {
        R: int(oof.sel_burden.values[full_rows_by_region[R]].sum()) for R in REGIONS
    },
    "recall_support": int((oof.sel_burden.values & is_pos).sum()),
    "rule_missed": int((oof.sel_burden.values & rule_missed_mask).sum()),
    "anchors": int((oof.sel_burden.values & is_neg).sum()),
}
BURDEN["union"] = {
    "selected_rows_pooled": int(picked.sum()),
    "selected_rows_per_region": {R: N_UNION[R] for R in REGIONS},
    "recall_support": int((picked & is_pos).sum()),
    "rule_missed": 0,
    "anchors": int((picked & is_neg).sum()),
}
dump()


# ---------------------------------------------------------------- paired tests at matched burden
def mcnemar_p(b, c):
    """Exact two-sided McNemar: binomial test on the b + c discordant pairs at p = 0.5.
    The same implementation paper_recovery/build_numbers.py uses."""
    m_ = b + c
    if m_ == 0:
        return 1.0
    mm = min(b, c)
    tail = sum(math.comb(m_, k) for k in range(mm + 1)) / (2.0**m_)
    return min(1.0, 2.0 * tail)


def paired(a_sel, b_sel, subset):
    a = a_sel[subset]
    b = b_sel[subset]
    bb = int((a & ~b).sum())
    cc = int((b & ~a).sum())
    return {
        "n_pairs": int(subset.sum()),
        "both": int((a & b).sum()),
        "a_only": bb,
        "b_only": cc,
        "neither": int((~a & ~b).sum()),
        "p": mcnemar_p(bb, cc),
    }


imit = matched_sel["rules_reproduction"]
prim = matched_sel["__primary_48bag__"]
BURDEN["paired_matched"] = {
    "imitation_vs_union_positives": paired(imit, picked, is_pos),
    "imitation_vs_union_rule_missed": paired(imit, picked, rule_missed_mask),
    "primary_vs_imitation_positives": paired(prim, imit, is_pos),
    "primary_vs_imitation_rule_missed": paired(prim, imit, rule_missed_mask),
    "primary_vs_union_positives": paired(prim, picked, is_pos),
}
BURDEN["paired_deployed"] = {
    "imitation_deployed_vs_union_positives": paired(
        sel_full["rules_reproduction"], picked, is_pos
    ),
    "primary_deployed_vs_imitation_deployed_positives": paired(
        oof.sel_burden.values, sel_full["rules_reproduction"], is_pos
    ),
}
dump()

# ---------------------------------------------------------------- catalogue-wide score release
rel = pd.DataFrame({"source_id": d.source_id.values, "region": region})
for name in BURDEN["variants"]:
    rel["score_" + name] = score_full[name].astype(np.float32)
    rel["sel_" + name] = sel_full[name]
    rel["selmatched_" + name] = matched_sel[name]
rel.to_parquet(os.path.join(OUT, "baseline_oof_full.parquet"), index=False)
BURDEN["released_table"] = "recovery/baseline_oof_full.parquet"
dump()

# ---------------------------------------------------------------- readable table
U = BURDEN["union"]
P48 = BURDEN["primary_48bag_deployed"]
P48M = BURDEN["primary_48bag_matched"]
bad = [k for k, v in BURDEN["variants"].items() if not v["reproduces_record"]]
lines = []
lines.append("# Realised catalogue burden of every v4 variant\n")
lines.append(
    "Round 2, reviewer E findings 1 and 2. Every variant in the run of record was scored only "
    "on the 5126 labelled rows, so no variant had a catalogue-wide burden. This table re-runs "
    "each of them with the record's exact code path and scores all "
    f"{n} in-support rows of each held-out region.\n"
)
mach = BURDEN["machine"]
lines.append(
    f"Machine: {mach['node']}, {mach['platform']}, {mach['cpu_count']} logical cores, "
    f"Python {mach['python']}, LightGBM {mach['lightgbm']}, NumPy {mach['numpy']}, "
    f"LightGBM num_threads {mach['lightgbm_num_threads']} and joblib n_jobs "
    f"{mach['joblib_n_jobs']} exactly as in the record. Wall time "
    f"{BURDEN['seconds_total']} s. "
    + (
        "Every variant reproduced the record's labelled-row numbers exactly (recall, "
        "rule-missed, strict, anchors, pooled case-control PR-AUC and the "
        "per-region recalls all identical to recovery/train_results.json)."
        if not bad
        else "Variants that did NOT reproduce the record's labelled-row numbers: "
        + ", ".join(bad)
        + ". Their burden numbers are not used."
    )
    + "\n"
)
lines.append(
    "The inner nested selection was not re-run; its chosen anchor, tree count and "
    "training-region union burden were read from the record, and the 20 draws it took on the "
    "shared rng_sub stream were replayed call for call so every variant's 50000-row threshold "
    "subsample is the same draw the record used. The training-region union burdens recomputed "
    "from the data equal the record's to the last bit.\n"
)
lines.append(
    "The catalogue-wide out-of-fold scores this table is built from are released in "
    "`recovery/baseline_oof_full.parquet`.\n"
)

lines.append("## Realised burden at the record's equal-burden threshold\n")
lines.append(
    "Selected rows is the realised catalogue burden: in-support rows the variant selects at "
    "the threshold the record set inside the training regions. Recall, rule-missed and anchors "
    "are the record's labelled-row numbers, reproduced here.\n"
)
lines.append(
    "| variant | selected rows | burden vs union | recall of 147 | rule-missed of 41 | anchors of 4979 | pooled AP | reproduces |"
)
lines.append("|---|---:|---:|---:|---:|---:|---:|:--:|")
lines.append(
    f"| union of the seven rules | {U['selected_rows_pooled']} | 1.00x | "
    f"{U['recall_support']} | 0 | {U['anchors']} | - | - |"
)
lines.append(
    f"| primary, 48 bags (deployed) | {P48['selected_rows_pooled']} | "
    f"{P48['selected_rows_pooled'] / U['selected_rows_pooled']:.2f}x | {P48['recall_support']} | "
    f"{P48['rule_missed']} | {P48['anchors']} | "
    f"{record['primary_pooled']['case_control_prauc_pooled']:.4f} | - |"
)
for name, v in BURDEN["variants"].items():
    r = v["labelled_row_reproduction"]
    lines.append(
        f"| {name} | {v['selected_rows_pooled']} | "
        f"{v['selected_rows_pooled'] / U['selected_rows_pooled']:.2f}x | "
        f"{r['recall_at_burden']} | {r['rule_missed_recovered']} | "
        f"{r['neg_selected_at_burden']} | {r['case_control_prauc_pooled']:.4f} | "
        f"{'yes' if v['reproduces_record'] else 'NO'} |"
    )
lines.append("")

lines.append("## Realised burden per region\n")
lines.append("| variant | " + " | ".join(REGIONS) + " | pooled |")
lines.append("|---" + "|---:" * (len(REGIONS) + 1) + "|")
lines.append(
    "| union of the seven rules | "
    + " | ".join(str(N_UNION[R]) for R in REGIONS)
    + f" | {U['selected_rows_pooled']} |"
)
lines.append(
    "| primary, 48 bags (deployed) | "
    + " | ".join(str(P48["selected_rows_per_region"][R]) for R in REGIONS)
    + f" | {P48['selected_rows_pooled']} |"
)
for name, v in BURDEN["variants"].items():
    lines.append(
        f"| {name} | "
        + " | ".join(str(v["selected_rows_per_region"][R]) for R in REGIONS)
        + f" | {v['selected_rows_pooled']} |"
    )
lines.append("")

lines.append("## Matched burden: top N_union rows in each held-out region\n")
lines.append(
    "N_union per region is the union's own in-support selection count: "
    + ", ".join(f"{R} {N_UNION[R]}" for R in REGIONS)
    + f", pooled {U['selected_rows_pooled']}. Every variant is cut to exactly those counts, so "
    "the comparison against the union is burden matched by construction.\n"
)
lines.append(
    "| variant | matched recall of 147 | matched rule-missed of 41 | matched anchors | tie-free |"
)
lines.append("|---|---:|---:|---:|:--:|")
lines.append(
    f"| union of the seven rules | {U['recall_support']} | 0 | {U['anchors']} | - |"
)
lines.append(
    f"| primary, 48 bags | {P48M['matched_recall_support']} | {P48M['matched_rule_missed']} | "
    f"{P48M['matched_anchors']} | "
    f"{'yes' if all(t['tie_free'] for t in P48M['ties'].values()) else 'no'} |"
)
for name, v in BURDEN["variants"].items():
    m = v["matched"]
    lines.append(
        f"| {name} | {m['matched_recall_support']} | {m['matched_rule_missed']} | "
        f"{m['matched_anchors']} | "
        f"{'yes' if all(t['tie_free'] for t in m['ties'].values()) else 'no'} |"
    )
lines.append("")

lines.append("## Paired tests at matched burden\n")
lines.append(
    "| comparison | subset | both | A only | B only | neither | exact two-sided p |"
)
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for lab, key in (
    ("imitation (A) vs union (B), matched", "imitation_vs_union_positives"),
    ("imitation (A) vs union (B), matched", "imitation_vs_union_rule_missed"),
    ("primary (A) vs imitation (B), matched", "primary_vs_imitation_positives"),
    ("primary (A) vs imitation (B), matched", "primary_vs_imitation_rule_missed"),
    ("primary (A) vs union (B), matched", "primary_vs_union_positives"),
):
    t = BURDEN["paired_matched"][key]
    sub = "41 rule-missed" if "rule_missed" in key else "147 positives"
    lines.append(
        f"| {lab} | {sub} | {t['both']} | {t['a_only']} | {t['b_only']} | {t['neither']} | "
        f"{t['p']:.6g} |"
    )
for lab, key in (
    (
        "imitation (A) vs union (B), deployed thresholds",
        "imitation_deployed_vs_union_positives",
    ),
    (
        "primary (A) vs imitation (B), deployed thresholds",
        "primary_deployed_vs_imitation_deployed_positives",
    ),
):
    t = BURDEN["paired_deployed"][key]
    lines.append(
        f"| {lab} | 147 positives | {t['both']} | {t['a_only']} | {t['b_only']} | "
        f"{t['neither']} | {t['p']:.6g} |"
    )
lines.append("")
open(os.path.join(OUT, "baseline_burden.md"), "w", encoding="utf-8").write(
    "\n".join(lines) + "\n"
)
dump()
print("done in", BURDEN["seconds_total"], "s", flush=True)
