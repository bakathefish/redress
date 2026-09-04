"""v4 build step 5: the evaluation tables from the out-of-fold scores.

Reads recovery/oof_scores{SUFFIX}.parquet (primary), mlp_oof_scores.parquet (challenger,
if present), baseline_scores{SUFFIX}.parquet + train_results{SUFFIX}.json, features.parquet
and denominators.json. Writes evaluation{SUFFIX}.json and evaluation{SUFFIX}.md.
Usage: python v4_evaluate.py [suffix]   (suffix selects a feature-set run, e.g. "_v34")

Vocabulary (design round, section 4): recall counts before percentages; the
confirmed-negative number is a "selection rate on targeted spectroscopic non-LRDs", never
precision; PR-AUC is "case-control PR-AUC"; Wilson intervals; the strict seven as counts only.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "recovery"
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
REGIONS = ["CEERS", "GOODS-N", "GOODS-S", "COSMOS", "UDS"]

den = json.load(open(os.path.join(OUT, "denominators.json")))
tr = json.load(open(os.path.join(OUT, f"train_results{SUF}.json")))
f = pd.read_parquet(os.path.join(OUT, "features.parquet"))
f = f[f.in_support.values].reset_index(drop=True)
o = pd.read_parquet(os.path.join(OUT, f"oof_scores{SUF}.parquet"))
assert (o.source_id.values == f.source_id.values).all()
pos = f.y.values == 1
neg = f.y.values == 0
picked = f.picked.values.astype(bool)
strict = f.y_strict.values.astype(bool)
region = f.region.values
mag = f.mag_f444w.values
N_POS_ALL = den["positives_151"]
N_POS_SUP = int(pos.sum())
N_ABSTAIN = N_POS_ALL - N_POS_SUP
N_MISSED_ALL = den["rule_missed_44"]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den_ = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den_
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den_
    return [round(c - h, 3), round(c + h, 3)]


def block(sel, name):
    sel = np.asarray(sel, bool)
    r = int((sel & pos).sum())
    out = {
        "name": name,
        "selected_total": int(sel.sum()),
        "selection_fraction": round(float(sel.mean()), 5),
        "recall_147": r,
        "recall_147_pct": round(100 * r / N_POS_SUP, 1),
        "recall_147_wilson": wilson(r, N_POS_SUP),
        "recall_151_end_to_end": r,
        "recall_151_pct": round(100 * r / N_POS_ALL, 1),
        "rule_missed_41": int((sel & pos & ~picked).sum()),
        "rule_missed_44_end_to_end_pct": round(
            100 * (sel & pos & ~picked).sum() / N_MISSED_ALL, 1
        ),
        "strict_in_support": int((sel & strict).sum()),
        "strict_total_in_support": int(strict.sum()),
        "strict_missed": int((sel & strict & ~picked).sum()),
        "strict_missed_total": int((strict & ~picked).sum()),
        "neg_selected": int((sel & neg).sum()),
        "neg_total": int(neg.sum()),
        "neg_selection_rate_pct": round(100 * (sel & neg).sum() / neg.sum(), 2),
        "per_region_recall": {
            R: f"{int((sel & pos & (region == R)).sum())}/{int((pos & (region == R)).sum())}"
            for R in REGIONS
        },
        "per_region_missed": {
            R: f"{int((sel & pos & ~picked & (region == R)).sum())}/{int((pos & ~picked & (region == R)).sum())}"
            for R in REGIONS
        },
        "per_catalog_recall": {
            c: f"{int((sel & pos & f[f'in_{c}'].values).sum())}/{int((pos & f[f'in_{c}'].values).sum())}"
            for c in ("hviding25_A1", "barro25", "degraaff26")
        },
        "recall_by_f444w_mag": {},
    }
    for lo, hi in ((0, 24), (24, 25), (25, 26), (26, 27), (27, 40)):
        m = pos & (mag >= lo) & (mag < hi)
        out["recall_by_f444w_mag"][f"{lo}-{hi}"] = (
            f"{int((sel & m).sum())}/{int(m.sum())} (union {int((picked & m).sum())})"
        )
    return out


ev = {"denominators": den, "abstentions_missing_band": N_ABSTAIN}
union = block(picked, "union of seven rules")
union["recall_151_end_to_end"] = den["union_selects_of_151"]  # one incomplete-band positive is rule-selected
union["recall_151_pct"] = round(100 * den["union_selects_of_151"] / N_POS_ALL, 1)
ev["union"] = union
prim = block(o.sel_burden.values, "primary at equal burden")
prim["case_control_prauc_pooled"] = round(
    float(
        average_precision_score(
            np.r_[np.ones(pos.sum()), np.zeros(neg.sum())],
            np.r_[o.score_mean.values[pos], o.score_mean.values[neg]],
        )
    ),
    4,
)
prim["chosen_settings_per_fold"] = {
    R: (tr["folds"][R]["chosen_anchor"], tr["folds"][R]["chosen_trees"])
    for R in REGIONS
}
prim["per_fold_selected_vs_union"] = {
    R: f"{tr['folds'][R]['selected_at_burden']} vs {tr['folds'][R]['union_selected_in_region']}"
    for R in REGIONS
}
ev["primary_equal_burden"] = prim
ev["primary_neg1pct"] = block(o.sel_neg1.values, "primary at inner 1% negative rate")
ev["primary_0p5pct"] = block(o.sel_05.values, "primary at 0.5% of the catalog")
# paired gain / loss against the union on the positives
sb = o.sel_burden.values
ev["paired_vs_union"] = {
    "positives_both": int((sb & picked & pos).sum()),
    "positives_model_only": int((sb & ~picked & pos).sum()),
    "positives_union_only": int((~sb & picked & pos).sum()),
    "positives_neither": int((~sb & ~picked & pos).sum()),
    "selected_both": int((sb & picked).sum()),
    "selected_model_only": int((sb & ~picked).sum()),
    "selected_union_only": int((~sb & picked).sum()),
    "jaccard_selected": round(
        float((sb & picked).sum() / max((sb | picked).sum(), 1)), 3
    ),
}
# the rule-missed positives the model recovers, listed
rows = np.where(sb & pos & ~picked)[0]
ev["recovered_rule_missed_list"] = (
    f.iloc[rows][
        [
            "field",
            "id",
            "region",
            "mag_f444w",
            "in_hviding25_A1",
            "in_barro25",
            "in_degraaff26",
            "y_strict",
        ]
    ]
    .assign(score=o.score_mean.values[rows])
    .sort_values("score", ascending=False)
    .to_dict("records")
)
rows = np.where(~sb & pos)[0]
ev["still_missed_list"] = (
    f.iloc[rows][["field", "id", "region", "mag_f444w", "picked", "y_strict"]]
    .assign(
        score=o.score_mean.values[rows],
        score_rank_pct=[
            round(100 * float((o.score_mean.values >= s).mean()), 2)
            for s in o.score_mean.values[rows]
        ],
    )
    .sort_values("score", ascending=False)
    .to_dict("records")
)

# baselines and challenger
ev["baselines"] = tr.get("baselines", {})
ev["sanity"] = tr.get("sanity", {})
shuf = [
    tr["baselines"][k]["recall_at_burden"]
    for k in tr["baselines"]
    if k.startswith("shuffled_labels_")
]
ev["sanity"]["shuffled_labels_recall_at_burden"] = shuf
ev["sanity"]["shuffled_labels_expected_if_random"] = round(
    float(np.mean([tr["folds"][R]["union_burden_train"] for R in REGIONS]) * N_POS_SUP),
    2,
)
# challengers: any *_oof_scores{SUF}.parquet beside the primary (nnPU MLP, TabPFN v2, ...)
for chal, fname in (("mlp", f"mlp_oof_scores{SUF}.parquet"), ("tabpfn", f"tabpfn_oof_scores{SUF}.parquet"), ("tabicl", f"tabicl_oof_scores{SUF}.parquet")):
    cpath = os.path.join(OUT, fname)
    if not os.path.exists(cpath):
        continue
    m = pd.read_parquet(cpath)
    assert (m.source_id.values == f.source_id.values).all()
    cb = block(m.sel_burden.values, f"{chal} at equal burden")
    ok = np.isfinite(m.score_mean.values)
    cb["case_control_prauc_pooled"] = round(float(average_precision_score(np.r_[np.ones((pos & ok).sum()), np.zeros((neg & ok).sum())], np.r_[m.score_mean.values[pos & ok], m.score_mean.values[neg & ok]])), 4)
    ev[f"{chal}_equal_burden"] = cb
    gain_pos = cb["recall_147"] - prim["recall_147"]
    gain_mis = cb["rule_missed_41"] - prim["rule_missed_41"]
    wins = sum(int(cb["per_region_recall"][R].split("/")[0]) > int(prim["per_region_recall"][R].split("/")[0]) for R in REGIONS)
    dneg = cb["neg_selection_rate_pct"] - prim["neg_selection_rate_pct"]
    ev[f"{chal}_promotion"] = {"gain_positives": gain_pos, "gain_rule_missed": gain_mis, "regions_won": wins, "neg_rate_change_points": round(dneg, 2), "promoted": bool((gain_pos >= 3 or gain_mis >= 2) and wins >= 4 and dneg <= 0.3)}
mlp_path = os.path.join(OUT, f"mlp_oof_scores{SUF}.parquet")
if os.path.exists(mlp_path):
    m = pd.read_parquet(mlp_path)
    assert (m.source_id.values == f.source_id.values).all()
    # simple rank-average blend, evaluated at equal burden with fold thresholds re-derived by quantile per region
    rk = 0.5 * (
        pd.Series(o.score_mean.values).rank(pct=True).values
        + pd.Series(m.score_mean.values).rank(pct=True).values
    )
    blend_sel = np.zeros(len(f), bool)
    for R in REGIONS:
        te = region == R
        frac = tr["folds"][R]["union_burden_train"]
        t = np.quantile(rk[~te], 1 - frac)
        blend_sel[te] = rk[te] >= t
    ev["blend_rank_average_equal_burden"] = block(
        blend_sel, "rank-average blend (trees + MLP) at equal burden"
    )

json.dump(
    ev, open(os.path.join(OUT, f"evaluation{SUF}.json"), "w"), indent=1, default=str
)


# ---------------------------------------------------------------- markdown
def row(b):
    return f"| {b['name']} | {b['selected_total']} | {b['recall_147']} ({b['recall_147_pct']}%) | {b['recall_151_end_to_end']} of 151 ({b['recall_151_pct']}%) | {b['rule_missed_41']} of 41 | {b['strict_in_support']} of {b['strict_total_in_support']} | {b['strict_missed']} of {b['strict_missed_total']} | {b['neg_selected']} of {b['neg_total']} ({b['neg_selection_rate_pct']}%) |"


L = [f"# v4 recovery classifier: out-of-fold evaluation{SUF}", ""]
L.append(
    f"Positives: {N_POS_ALL} spectroscopic LRDs, {N_POS_SUP} inside the seven-band support ({N_ABSTAIN} abstentions counted as misses end to end). Rule-missed: {N_MISSED_ALL} ({den['rule_missed_in_support_41']} in support). Confirmed non-LRDs in support: {int(neg.sum())}. Support rows: {len(f)}. Union burden: {den['union_selected_in_support']} selections ({100 * den['union_burden_fraction_in_support']:.3f}%)."
)
L.append("")
L.append(
    "| selection | selected | recall of 147 | end-to-end recall | rule-missed | strict 32 | strict missed | targeted-negative selection rate |"
)
L.append("|---|---|---|---|---|---|---|---|")
for k in ("union", "primary_equal_burden", "primary_neg1pct", "primary_0p5pct", "mlp_equal_burden", "tabpfn_equal_burden", "tabicl_equal_burden", "blend_rank_average_equal_burden"):
    if k in ev:
        L.append(row(ev[k]))
L.append("")
L.append(
    f"Case-control PR-AUC (positives vs targeted negatives, pooled OOF): primary {prim['case_control_prauc_pooled']}"
    + (
        f", MLP {ev['mlp_equal_burden']['case_control_prauc_pooled']}"
        if "mlp_equal_burden" in ev
        else ""
    )
)
L.append("")
L.append("## Per region at equal burden (recall / rule-missed recovered)")
L.append(
    "| region | union | primary | primary rule-missed | selected vs union | settings (anchor, trees) |"
)
L.append("|---|---|---|---|---|---|")
for R in REGIONS:
    L.append(
        f"| {R} | {union['per_region_recall'][R]} | {prim['per_region_recall'][R]} | {prim['per_region_missed'][R]} | {prim['per_fold_selected_vs_union'][R]} | {prim['chosen_settings_per_fold'][R]} |"
    )
L.append("")
L.append("## Paired against the union (positives)")
pv = ev["paired_vs_union"]
L.append(
    f"both {pv['positives_both']}, model only {pv['positives_model_only']}, union only {pv['positives_union_only']}, neither {pv['positives_neither']}; selected sets: both {pv['selected_both']}, model only {pv['selected_model_only']}, union only {pv['selected_union_only']}, Jaccard {pv['jaccard_selected']}"
)
L.append("")
L.append("## By catalog and by F444W magnitude (primary at equal burden)")
L.append(
    "catalogs: " + ", ".join(f"{k} {v}" for k, v in prim["per_catalog_recall"].items())
)
L.append(
    "magnitude: "
    + ", ".join(f"{k}: {v}" for k, v in prim["recall_by_f444w_mag"].items())
)
L.append("")
L.append(
    "## Baselines and sanity checks (equal burden, 8 bags, settings copied from the primary fold)"
)
L.append(
    "| variant | recall of 147 | rule-missed | strict missed | negatives selected | PR-AUC | per region |"
)
L.append("|---|---|---|---|---|---|---|")
for k, b in ev["baselines"].items():
    L.append(
        f"| {k} | {b['recall_at_burden']} | {b['rule_missed_recovered']} | {b['strict_missed_recovered']} | {b['neg_selected_at_burden']} | {b['case_control_prauc_pooled']} | {' '.join(str(v) for v in b['per_region_recall'].values())} |"
        + (
            f" left-out catalog: {b.get('recall_on_left_out_catalog')} (primary {b.get('primary_recall_on_that_catalog')})"
            if "held_out_catalog" in b
            else ""
        )
    )
L.append("")
L.append(
    f"Shuffled-label recall at equal burden ({len(shuf)} repeats): {shuf}; chance level if the labels carried nothing: about {ev['sanity']['shuffled_labels_expected_if_random']}. The floor sits far above chance because every labelled object, positive or negative, was a spectroscopic target: a shuffled model still learns what a targeted object looks like (brightness, colour selection of the programs), so the honest comparison for the primary is against this floor and against the union, not against chance."
)
if "field_adversary" in ev["sanity"]:
    fa = ev["sanity"]["field_adversary"]
    L.append(f"Field adversary: accuracy {fa['accuracy_mean']} (chance {fa['chance']}); top features {fa['top_features_gain_share']}.")
if "primary_score_std" in ev["sanity"]:
    ps = ev["sanity"]["primary_score_std"]
    L.append(f"Ensemble dispersion (std over 48 bags): median all {ps['median_all']:.4f}, positives {ps['median_positives']:.4f}, selected {ps['median_selected_at_burden']:.4f}.")
for chal in ("mlp", "tabpfn", "tabicl"):
    if f"{chal}_promotion" in ev:
        L.append("")
        L.append(f"{chal} promotion rule: {json.dumps(ev[f'{chal}_promotion'])}")
L.append("")
L.append("## Rule-missed positives the primary model recovers at equal burden")
L.append("| field | id | region | F444W | A1 | Barro | deGraaff | strict | score |")
L.append("|---|---|---|---|---|---|---|---|---|")
for r in ev["recovered_rule_missed_list"]:
    L.append(
        f"| {r['field']} | {r['id']} | {r['region']} | {r['mag_f444w']:.2f} | {r['in_hviding25_A1']} | {r['in_barro25']} | {r['in_degraaff26']} | {r['y_strict']} | {r['score']:.3f} |"
    )
L.append("")
L.append(
    "## Positives still missed at equal burden (score percentile among all support rows)"
)
L.append("| field | id | region | F444W | union picks it | strict | score | top % |")
L.append("|---|---|---|---|---|---|---|---|")
for r in ev["still_missed_list"]:
    L.append(
        f"| {r['field']} | {r['id']} | {r['region']} | {r['mag_f444w']:.2f} | {r['picked']} | {r['y_strict']} | {r['score']:.3f} | {r['score_rank_pct']} |"
    )
open(os.path.join(OUT, f"evaluation{SUF}.md"), "w", encoding="utf-8").write(
    "\n".join(L)
)
print("\n".join(L[:40]))
