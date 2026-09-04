"""Every number the recovery paper quotes, read from the tables of record in recovery and
written once to paper/numbers.json and paper/numbers.tex (LaTeX macros).
The paper text uses the macros only, so a number can be audited by diffing this file's
output against the tables; nothing is typed by hand.

Run from the repository root: python paper/build_numbers.py
"""

import json
import math
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "recovery")
OUT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
REG = ["CEERS", "GOODS-N", "GOODS-S", "COSMOS", "UDS"]
WORD = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def J(f):
    return json.load(open(os.path.join(V, f)))


def frac(s):
    m = re.match(r"\s*(\d+)\s*(?:/|of)\s*(\d+)", str(s))
    return int(m.group(1)), int(m.group(2))


def ab_colour(f_blue, f_red):
    """Ordinary AB colour, NaN wherever either catalogue flux is non-positive.

    ROUND 3, Reviewer C findings 1 and 2. The AB colour is -2.5 log10(f_blue / f_red), and the
    paper's own definition, in the Figure~\\ref{fig:planes} caption, is that "a source with a
    non-positive catalogue flux in either band has no AB colour". Computing the ratio first and
    filtering on finiteness afterwards silently admits rows whose two fluxes are both negative:
    the ratio is positive, the logarithm is finite, and the row is counted as having a colour it
    does not have, since the ratio of two negative noise fluxes is the colour of their absolute
    values. Masking before the logarithm is the only place the rule can be enforced once. Every
    AB colour in this file goes through here.
    """
    x = np.asarray(f_blue, dtype=float)
    y = np.asarray(f_red, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = (x > 0) & (y > 0)
    out[ok] = -2.5 * np.log10(x[ok] / y[ok])
    return out


N = {}

# ---------------------------------------------------------------- denominators and labels
den = J("denominators.json")
lc = J("label_counts.json")
fm = J("feature_manifest.json")
N["catalogRows"] = fm["rows"]
N["supportRows"] = den["support_rows"]
N["completeRows"] = den["complete_rows"]
N["nPos"] = den["positives_151"]
N["nPosSupport"] = den["positives_in_support_147"]
N["nPosOutside"] = den["positives_incomplete_bands"]
N["nRuleMissed"] = den["rule_missed_44"]
N["nRuleMissedSupport"] = den["rule_missed_in_support_41"]
N["nNeg"] = lc["labels"]["negatives"]
N["nNegSupport"] = den["negatives_in_support"]
N["nAmbiguous"] = den["ambiguous_excluded"]
N["unionSelected"] = den["union_selected_in_support"]
N["unionBurdenPct"] = round(100 * den["union_burden_fraction_in_support"], 3)
N["listBarro"] = lc["barro25"]["in_fields"]
N["listDeGraaff"] = lc["degraaff26"]["in_fields"]
N["listHvidingA"] = lc["hviding25_A1"]["in_fields"]
N["nRegions"] = 5
N["nFields"] = 9
N["nFeatures"] = fm["n_features"]
N["skyGroupArcsec"] = fm["sky_group_radius_arcsec"]
for r in REG:
    k = r.replace("-", "")
    N["pos" + k] = den["positives_per_region"][r]
    N["posSup" + k] = den["positives_in_support_per_region"][r]
    N["unionSel" + k] = den["union_in_support_per_region"][r]
    N["rows" + k] = den["support_rows_per_region"][r]
    N["negSup" + k] = den["negatives_in_support_per_region"][r]
    N["missedSup" + k] = den["rule_missed_in_support_per_region"][r]

lab = pd.read_parquet(os.path.join(V, "labels.parquet"))
pos = lab[lab.y == 1]
L = ["in_hviding25_A1", "in_barro25", "in_degraaff26"]
inlist = pos[L].astype(bool).any(axis=1)
N["nPosInPublishedList"] = int(inlist.sum())
N["nPosOnlySpectralTest"] = int((~inlist).sum())
N["nPosBarroOnly"] = int(
    (
        pos.in_barro25.astype(bool)
        & ~pos.in_hviding25_A1.astype(bool)
        & ~pos.in_degraaff26.astype(bool)
    ).sum()
)
N["nPosDeGraaffOnly"] = int(
    (
        pos.in_degraaff26.astype(bool)
        & ~pos.in_hviding25_A1.astype(bool)
        & ~pos.in_barro25.astype(bool)
    ).sum()
)
N["nPosInTwoOrMore"] = int((pos[L].astype(bool).sum(axis=1) >= 2).sum())
N["nPosInHvidingB"] = int(pos.in_hviding25_B1.astype(bool).sum())
neg = lab[lab.y == 0]
N["nNegWithSpectrum"] = int(neg.has_spec.astype(bool).sum())
# Reviewer C finding 1: the anchors are gated on the v3 `vshaped_spec` verdict, not on the
# row-level `in_census` flag, so the census this paper describes is the set of spectra the
# V-shape test was run on, that is the rows with a non-null verdict. `in_census` gives 5,076
# and cannot contain the 5,397 anchors; this gives 5,572 and the chain closes exactly:
# 5,572 tested = 46 pass + 5,526 fail, and 5,526 fail = 129 in a list + 5,397 anchors.
_vs = lab.vshaped_spec.astype("boolean")
N["nCensusSpectra"] = int(_vs.notna().sum())
N["nVshapePass"] = int(_vs.eq(True).fillna(False).sum())
_notv = _vs.eq(False).fillna(False)
N["nVshapeFail"] = int(_notv.sum())
N["nVshapeFailInList"] = int((_notv & lab.in_any_list.astype(bool)).sum())
assert N["nVshapePass"] + N["nVshapeFail"] == N["nCensusSpectra"]
assert N["nVshapeFail"] - N["nVshapeFailInList"] == N["nNeg"]

# ---------------------------------------------------------------- per-rule table (support rows)
f = pd.read_parquet(os.path.join(V, "features.parquet"))
sup = f[f.in_support.astype(bool)].copy()
posS = sup[sup.y == 1]
negS = sup[sup.y == 0]
posAll = f[f.y == 1]
rules = [
    "labbe23",
    "kokorev24",
    "kocevski24",
    "perezgonzalez24",
    "barro23",
    "greene24",
    "akins24",
]
RTAG = {
    "labbe23": "Labbe",
    "kokorev24": "Kokorev",
    "kocevski24": "Kocevski",
    "perezgonzalez24": "PerezGonzalez",
    "barro23": "Barro",
    "greene24": "Greene",
    "akins24": "Akins",
}
per_rule = {}
for r in rules:
    s = sup["sel_" + r].astype(bool)
    per_rule[r] = dict(
        selected=int(s.sum()),
        recall151=int(posAll["sel_" + r].astype(bool).sum()),
        recall147=int(posS["sel_" + r].astype(bool).sum()),
        anchors=int(negS["sel_" + r].astype(bool).sum()),
    )
    t = RTAG[r]
    N["rule" + t + "Selected"] = per_rule[r]["selected"]
    N["rule" + t + "Recall"] = per_rule[r]["recall151"]
    # Table 1 quotes the 151 denominator, Figure 3 plots the 147; both now have a macro
    N["rule" + t + "RecallSupport"] = per_rule[r]["recall147"]
    N["rule" + t + "Anchors"] = per_rule[r]["anchors"]
    N["rule" + t + "RecallPct"] = round(100 * per_rule[r]["recall151"] / N["nPos"], 1)
    N["rule" + t + "AnchorPct"] = round(
        100 * per_rule[r]["anchors"] / N["nNegSupport"], 2
    )
N["ruleRecallMin"] = min(v["recall151"] for v in per_rule.values())
N["ruleRecallMax"] = max(v["recall151"] for v in per_rule.values())
# one decimal, so the running text of Section 3.3 matches the per-rule table on the facing
# page (round 1, Reviewer C finding 13 and Reviewer D minor 5).
N["ruleRecallMinPct"] = round(100 * N["ruleRecallMin"] / N["nPos"], 1)
N["ruleRecallMaxPct"] = round(100 * N["ruleRecallMax"] / N["nPos"], 1)
nr = posAll[["sel_" + r for r in rules]].astype(bool).sum(axis=1)
hist = nr.value_counts().sort_index().to_dict()
for k in range(8):
    N["nPosSelectedBy" + WORD[k] + "Rules"] = int(hist.get(k, 0))
N["unionMissPct"] = round(100 * N["nRuleMissed"] / N["nPos"], 1)

# colour and magnitude of the misses (in support)
missS = posS[~posS.picked.astype(bool)]
hitS = posS[posS.picked.astype(bool)]
N["missMedianColour"] = round(float(missS.c_f277w_f444w.median()), 2)
N["hitMedianColour"] = round(float(hitS.c_f277w_f444w.median()), 2)
N["missMedianMag"] = round(float(missS.mag_f444w.median()), 2)
N["hitMedianMag"] = round(float(hitS.mag_f444w.median()), 2)
N["nPosColourBelowOne"] = int((posS.c_f277w_f444w <= 1.0).sum())
# one decimal, so this does not print as the same "48" as blueUnionPct, which is a different
# fraction of a different denominator (round 1, Reviewer C finding 12, Reviewer D major 6).
N["nPosColourBelowOnePct"] = round(100 * N["nPosColourBelowOne"] / N["nPosSupport"], 1)

# ---------------------------------------------------------------- out-of-fold evaluation
ev = J("evaluation.json")
ops = {
    "model": ev["primary_equal_burden"],
    "union": ev["union"],
    "modelNegOne": ev["primary_neg1pct"],
    "modelHalf": ev["primary_0p5pct"],
    "mlp": ev["mlp_equal_burden"],
    "blend": ev["blend_rank_average_equal_burden"],
}
for prefix, e in ops.items():
    N[prefix + "Selected"] = e["selected_total"]
    N[prefix + "RecallSupport"] = e["recall_147"]
    N[prefix + "RecallEnd"] = e["recall_151_end_to_end"]
    N[prefix + "RuleMissed"] = e["rule_missed_41"]
    N[prefix + "Anchors"] = e["neg_selected"]
    N[prefix + "AnchorPct"] = e["neg_selection_rate_pct"]
    N[prefix + "RecallEndPct"] = e["recall_151_pct"]
    N[prefix + "RecallSupportPct"] = e["recall_147_pct"]
    N[prefix + "WilsonLo"] = round(100 * e["recall_147_wilson"][0], 1)
    N[prefix + "WilsonHi"] = round(100 * e["recall_147_wilson"][1], 1)
    N[prefix + "SelectionPct"] = round(100 * e["selection_fraction"], 3)
    for r in REG:
        k = r.replace("-", "")
        a, b = frac(e["per_region_recall"][r])
        N[prefix + "Region" + k] = a
        if "per_region_missed" in e:
            a, b = frac(e["per_region_missed"][r])
            N[prefix + "RegionMissed" + k] = a
    if "per_catalog_recall" in e:
        for c, t in (
            ("hviding25_A1", "HvidingA"),
            ("barro25", "Barro"),
            ("degraaff26", "DeGraaff"),
        ):
            a, b = frac(e["per_catalog_recall"][c])
            N[prefix + "Catalog" + t] = a
            N["catalogN" + t] = b
    if "case_control_prauc_pooled" in e:
        N[prefix + "PrAuc"] = round(e["case_control_prauc_pooled"], 3)
for r in REG:
    k = r.replace("-", "")
    m = re.match(
        r"(\d+) vs (\d+)", ev["primary_equal_burden"]["per_fold_selected_vs_union"][r]
    )
    N["modelSel" + k] = int(m.group(1))
    N["unionSelCheck" + k] = int(m.group(2))
    a, t = ev["primary_equal_burden"]["chosen_settings_per_fold"][r]
    N["fold" + k + "Anchor"] = a
    N["fold" + k + "Trees"] = t
N["modelGain"] = N["modelRecallEnd"] - N["unionRecallEnd"]
N["modelGainSupport"] = N["modelRecallSupport"] - N["unionRecallSupport"]
N["modelAnchorExtra"] = N["modelAnchors"] - N["unionAnchors"]
N["modelSelectedFewer"] = N["unionSelected"] - N["modelSelected"]
N["modelRegionsWon"] = sum(
    N["modelRegion" + r.replace("-", "")] > N["unionRegion" + r.replace("-", "")]
    for r in REG
)
N["modelRuleMissedPct"] = round(100 * N["modelRuleMissed"] / N["nRuleMissed"], 0)
pv = ev["paired_vs_union"]
N["pairedBoth"] = pv["positives_both"]
N["pairedModelOnly"] = pv["positives_model_only"]
N["pairedUnionOnly"] = pv["positives_union_only"]
N["pairedNeither"] = pv["positives_neither"]
N["pairedUnionOnlyEnd"] = N["pairedUnionOnly"] + (
    N["unionRecallEnd"] - N["unionRecallSupport"]
)
N["pairedNeitherEnd"] = (
    N["nPos"] - N["pairedBoth"] - N["pairedModelOnly"] - N["pairedUnionOnlyEnd"]
)
N["setBoth"] = pv["selected_both"]
N["setModelOnly"] = pv["selected_model_only"]
N["setUnionOnly"] = pv["selected_union_only"]
N["setJaccard"] = round(pv["jaccard_selected"], 2)
N["setModelOnlyPct"] = round(100 * pv["selected_model_only"] / N["modelSelected"], 0)

# recall by magnitude and colour bins (OOF selections)
oof = pd.read_parquet(os.path.join(V, "oof_scores.parquet")).set_index("source_id")
supi = sup.set_index("source_id")
supi["sel_burden"] = oof.loc[supi.index, "sel_burden"].values
supi["score"] = oof.loc[supi.index, "score_mean"].values
P = supi[supi.y == 1].copy()
mbins = [0, 24, 25, 26, 27, 40]
mlabels = ["Bright", "TwentyFour", "TwentyFive", "TwentySix", "Faint"]
P["mbin"] = pd.cut(P.mag_f444w, mbins, labels=mlabels)
for b, g in P.groupby("mbin", observed=True):
    N[f"mag{b}N"] = len(g)
    N[f"mag{b}Union"] = int(g.picked.astype(bool).sum())
    N[f"mag{b}Model"] = int(g.sel_burden.sum())
cbins = [-5, 0.5, 1.0, 1.5, 2.0, 10]
clabels = ["BelowHalf", "HalfToOne", "OneToOneHalf", "OneHalfToTwo", "AboveTwo"]
# Round 2, Reviewer C finding 5: these bins are the ones Figure 6 panel (a) draws, and that
# figure and every colour-bin sentence in the text moved to the ordinary AB colour the rules
# impose their thresholds on. Binning them on the asinh feature colour left the record
# disagreeing with the figure it underwrites, so the AB colour is used here, computed exactly
# as the AB block further down does: -2.5 log10(f_f277w / f_f444w) from the catalogue fluxes.
_labab = lab[["source_id", "f_f277w", "f_f444w"]].set_index("source_id")
_rab = _labab.loc[P.index]
P["c_ab_f277w_f444w"] = ab_colour(_rab.f_f277w, _rab.f_f444w)
assert np.isfinite(P.c_ab_f277w_f444w).all(), "a positive has no finite AB colour"
P["cbin"] = pd.cut(P.c_ab_f277w_f444w, cbins, labels=clabels)
for b, g in P.groupby("cbin", observed=True):
    N[f"col{b}N"] = len(g)
    N[f"col{b}Union"] = int(g.picked.astype(bool).sum())
    N[f"col{b}Model"] = int(g.sel_burden.sum())
blue = P[P.c_f277w_f444w <= 1.0]
N["blueN"] = len(blue)
N["blueUnion"] = int(blue.picked.astype(bool).sum())
N["blueModel"] = int(blue.sel_burden.sum())
# one decimal, so blueUnionPct does not print as the same "48" as nPosColourBelowOnePct
# (round 1, Reviewer C finding 12, Reviewer D major 6).
N["blueUnionPct"] = round(100 * N["blueUnion"] / N["blueN"], 1)
N["blueModelPct"] = round(100 * N["blueModel"] / N["blueN"], 1)
red = P[P.c_f277w_f444w > 1.0]
N["redN"] = len(red)
N["redUnion"] = int(red.picked.astype(bool).sum())
N["redModel"] = int(red.sel_burden.sum())
faint = P[P.mag_f444w > 25]
N["faintN"] = len(faint)
N["faintUnion"] = int(faint.picked.astype(bool).sum())
N["faintModel"] = int(faint.sel_burden.sum())

# pooled ranking curve points
sc = supi.score.values
ys = (supi.y == 1).values
order = np.argsort(-sc)
cum = np.cumsum(ys[order])
for n_, w in (
    (250, "TwoFifty"),
    (500, "FiveHundred"),
    (1000, "Thousand"),
    (2000, "TwoThousand"),
    (5000, "FiveThousand"),
    (10000, "TenThousand"),
):
    N["curveTop" + w] = int(cum[n_ - 1])

# ---------------------------------------------------------------- baselines and sanity
bl = ev["baselines"]
for name, tag in (
    ("primary_8bag_replica", "Replica"),
    ("magnitude_only", "MagOnly"),
    ("morphology_only", "MorphOnly"),
    ("ablate_c_f277w_f444w", "AblateColour"),
    ("ablate_size", "AblateSize"),
    ("anchor_0", "AnchorZero"),
    ("anchor_0.50", "AnchorHalf"),
    ("pn_lightgbm", "PN"),
    ("rules_reproduction", "Imitation"),
    ("with_zphot", "WithZphot"),
):
    if name in bl:
        b = bl[name]
        N["bl" + tag + "Recall"] = b["recall_at_burden"]
        N["bl" + tag + "Missed"] = b["rule_missed_recovered"]
        N["bl" + tag + "Anchors"] = b["neg_selected_at_burden"]
        N["bl" + tag + "PrAuc"] = round(b["case_control_prauc_pooled"], 3)
for c, tag in (
    ("hviding25_A1", "HvidingA"),
    ("barro25", "Barro"),
    ("degraaff26", "DeGraaff"),
):
    b = bl["leave_out_" + c]
    N["loco" + tag + "Recall"] = b["recall_at_burden"]
    N["loco" + tag + "Missed"] = b["rule_missed_recovered"]
    N["loco" + tag + "Anchors"] = b["neg_selected_at_burden"]
    N["loco" + tag + "LeftOut"], N["loco" + tag + "LeftOutN"] = frac(
        b["recall_on_left_out_catalog"]
    )
    N["loco" + tag + "PrimaryOnCatalog"], _ = frac(b["primary_recall_on_that_catalog"])
shuf = ev["sanity"]["shuffled_labels_recall_at_burden"]
N["shuffledN"] = len(shuf)
N["shuffledMin"] = min(shuf)
N["shuffledMax"] = max(shuf)
N["shuffledChance"] = ev["sanity"]["shuffled_labels_expected_if_random"]
sa = ev["sanity"]
N["fieldAdversaryAcc"] = round(100 * sa["field_adversary"]["accuracy_mean"], 1)
N["fieldAdversaryChance"] = round(100 * sa["field_adversary"]["chance"], 0)
N["ensembleStdAll"] = round(sa["primary_score_std"]["median_all"], 4)
N["ensembleStdSelected"] = round(
    sa["primary_score_std"]["median_selected_at_burden"], 4
)
N["ensembleStdPositives"] = round(sa["primary_score_std"]["median_positives"], 4)
N["imitationGap"] = N["blImitationRecall"] - N["unionRecallSupport"]
N["modelOverImitation"] = N["modelRecallSupport"] - N["blImitationRecall"]
N["modelOverImitationMissed"] = N["modelRuleMissed"] - N["blImitationMissed"]

# ---------------------------------------------------------------- challengers and selection
ms = J("model_selection.json")
for tag, rec in (
    ("TreesThirty", ms["trees_30_features"]),
    ("TreesThirtyFour", ms["trees_34_features"]),
    ("Mlp", ms["mlp_30_features"]),
    ("Blend", ms["blend_trees30_mlp"]),
):
    N["sel" + tag + "Recall"] = rec["recall_147"]
    N["sel" + tag + "Missed"] = rec["rule_missed_41"]
    N["sel" + tag + "Anchors"] = rec["neg_selected"]
N["selMlpRegionsWon"] = ms["mlp_promotion_vs_trees30"]["regions_won"]
N["selBlendRegionsWon"] = ms["blend_promotion_vs_trees30"]["regions_won"]
N["selTreesThirtyFourRegionsWon"] = ms["trees34_promotion_vs_trees30"]["regions_won"]
N["selRuleGainPos"] = 3
N["selRuleGainMissed"] = 2
N["selRuleRegions"] = 4
N["selRuleNegPoints"] = 0.3
e30, e34 = ms["external_check_trees_30"], ms["external_check_trees_34"]
N["extThirtyFourZphotLowPct"] = e34["zphot_lt_3_pct"]
N["extThirtyZphotLowPct"] = e30["zphot_lt_3_pct"]
N["extThirtyFourSecureLow"] = e34["secure_z_le_3"]
N["extThirtySecureLow"] = e30["secure_z_le_3"]
N["extThirtyFourN"] = e34["equal_burden_candidates"]
N["extThirtyN"] = e30["equal_burden_candidates"]
N["extThirtyFourSecureHigh"] = e34["secure_z_gt_3"]
N["extThirtySecureHigh"] = e30["secure_z_gt_3"]
N["mlpNetworks"] = 15

# ---------------------------------------------------------------- model, folds, thresholds
fin = J("final.json")
tr = J("train_results.json")
N["nBags"] = fin["n_bags"]
N["finalAnchor"] = fin["chosen_anchor"]
N["finalTrees"] = fin["chosen_trees"]
N["tBurden"] = round(fin["t_burden"], 3)
N["tNegOne"] = round(fin["t_neg1"], 3)
N["tHalf"] = round(fin["t_05"], 3)
N["finalAboveBurden"] = fin["n_above_burden"]
N["finalAboveHalf"] = fin["n_above_05"]
N["unlabelledPerPositive"] = 20
lp = fin["lightgbm_params"]
N["lgbLeaves"] = lp["num_leaves"]
N["lgbDepth"] = lp["max_depth"]
N["lgbLearningRate"] = lp["learning_rate"]
N["lgbMinChild"] = lp["min_child_samples"]
N["lgbFeatureFraction"] = lp["feature_fraction"]
N["lgbBaggingFraction"] = lp["bagging_fraction"]
N["lgbRegAlpha"] = lp["reg_alpha"]
N["lgbRegLambda"] = lp["reg_lambda"]
N["anchorGridMax"] = 0.5
N["treeGridMax"] = 1200
N["treeGridStep"] = 50
N["lofoSeconds"] = tr["seconds_total"]
N["lofoMinutes"] = int(round(tr["seconds_total"] / 60))
imp = J("feature_importance_gain.json")
top = sorted(imp.items(), key=lambda kv: -kv[1])
for i, (k, v) in enumerate(top[:8]):
    N["impName" + WORD[i + 1]] = k
    N["impPct" + WORD[i + 1]] = round(100 * v, 1)
# one decimal, so the printed total equals the printed parts (32.4 + 31.2 = 63.6)
N["impTopTwoPct"] = round(100 * (top[0][1] + top[1][1]), 1)
N["impSizePct"] = round(100 * (imp["log_rh"] + imp["log_rh_over_rstar"]), 1)
# the V curvature, the feature that encodes the shape the class is named for (Reviewer A
# round 2, minor finding 16): it is the sharpest evidence for the section's own conclusion
# that the model smooths the rules' colour and compactness rather than learning the V shape.
N["impCurvaturePct"] = round(100 * imp["v_curv"], 1)

# ---------------------------------------------------------------- candidates
cc = J("candidate_counts.json")
N["candTotal"] = cc["candidates_total"]
N["candEqualBurden"] = cc["candidates_equal_burden_tier"]
N["candHighRecall"] = cc["candidates_high_recall_tier"]
N["candRowsAboveHalf"] = cc["rows_above_high_recall_threshold"]
N["candExclPublished"] = cc["excluded"]["published-or-labelled"]
N["candExclRule"] = cc["excluded"]["rule-selected"]
N["candExclDup"] = cc["excluded"]["duplicate-sky-group"]
N["candUntested"] = cc["status_counts"]["untested"]
N["candSecureLow"] = cc["status_counts"]["secure low-z"]
N["candSecureHigh"] = cc["status_counts"]["secure z>3 (V untested)"]
N["candEbUntested"] = cc["status_counts_equal_burden"]["untested"]
N["candEbSecureLow"] = cc["status_counts_equal_burden"]["secure low-z"]
N["candEbSecureHigh"] = cc["status_counts_equal_burden"]["secure z>3 (V untested)"]
N["candWithArchive"] = cc["candidates_with_any_archive_record"]
N["candEligibleRefit"] = cc["eligible_records_to_refit"]
N["candEbOod"] = cc["ood_flagged_equal_burden"]
N["candEbMedianMag"] = round(cc["mag_f444w_median_equal_burden"], 1)
for r in REG:
    N["candEb" + r.replace("-", "")] = cc["per_region_equal_burden"][r]
cp = J("compactness_counts.json")
N["compMeasured"] = cp["measured"]
N["compMeasuredEb"] = cp["measured_equal_burden"]
N["compLabbeEb"] = cp["compact_labbe_equal_burden"]
N["compAkinsEb"] = cp["compact_akins_equal_burden"]
N["compLabbeAll"] = cp["compact_labbe_all"]
# one decimal, to match every comparable percentage in the paper (Reviewer C minor 17), and
# the matching percentage for the Akins criterion, which Section 6.3 quoted as a bare count
# although it is the larger fraction (Reviewer D minor 23).
N["compLabbeEbPct"] = round(
    100 * cp["compact_labbe_equal_burden"] / cp["measured_equal_burden"], 1
)
N["compAkinsEbPct"] = round(
    100 * cp["compact_akins_equal_burden"] / cp["measured_equal_burden"], 1
)
N["followupN"] = cp["followup_tier"]
N["compUnmeasurable"] = cp["candidates"] - cp["measured"]
ap = J("archive_purity.json")
a, e, fu, u = (
    ap["model_candidates_all"],
    ap["model_candidates_equal_burden"],
    ap["model_candidates_followup"],
    ap["union_novel_picks"],
)
N["apAllZphotLowPct"] = a["zphot_lt_3_pct"]
N["apEbZphotLowPct"] = e["zphot_lt_3_pct"]
N["apEbZphotLow"] = e["zphot_lt_3"]
N["apEbSecureHigh"] = e["secure_z_gt_3"]
N["apEbSecureLow"] = e["secure_z_le_3"]
N["apEbWithArchive"] = e["with_any_archive_record"]
N["apFuSecureHigh"] = fu["secure_z_gt_3"]
N["apFuSecureLow"] = fu["secure_z_le_3"]
N["apFuN"] = fu["n"]
N["apUnionN"] = u["n"]
N["apUnionSecureHigh"] = u["secure_z_gt_3"]
N["apUnionSecureLow"] = u["secure_z_le_3"]
N["apUnionZphotLowPct"] = u["zphot_lt_3_pct"]
N["knownZphotLow"], N["knownZphotLowN"] = frac(ap["known_positives_zphot_lt_3"])
N["knownZphotLowPct"] = round(100 * N["knownZphotLow"] / N["knownZphotLowN"], 1)
c = pd.read_csv(os.path.join(V, "candidates.csv"))
N["candInsideRange"] = int((c.n_features_outside_positive_range == 0).sum())
N["candOod"] = int(c.ood_flag.astype(bool).sum())
eb = c[c.tier == "equal_burden"]
N["candEbMedianZphot"] = round(float(eb.z_phot.median()), 1)
N["candEbMedianColour"] = round(float(eb.c_f277w_f444w.median()), 2)
N["candEbColourAboveOnePct"] = round(100 * float((eb.c_f277w_f444w > 1.0).mean()), 0)
# one decimal, so that it stays the exact complement of nPosColourBelowOnePct now that that
# macro carries one decimal (round 1, Reviewer C finding 12).
N["posColourAboveOnePct"] = round(100 * float((posS.c_f277w_f444w > 1.0).mean()), 1)
N["posMedianColour"] = round(float(posS.c_f277w_f444w.median()), 2)
N["posMedianMag"] = round(float(posS.mag_f444w.median()), 1)
N["posMedianZphot"] = round(float(posS.z_phot.median()), 1)
N["candEbRedSlopePct"] = round(100 * float((eb.slope_red < 0).mean()), 0)
N["posRedSlopePct"] = round(100 * float((posS.slope_red < 0).mean()), 0)
rand = sup.sample(20000, random_state=0)
N["randRedSlopePct"] = round(100 * float((rand.slope_red < 0).mean()), 0)
N["followupMedianMag"] = round(float(c[c.followup_tier].mag_f444w.median()), 1)

# ================================================================ round 1 review additions
# Everything below was added for the round 1 referee reports of the author's review record
# (not released), rulings R1 to R11. Nothing above this line changed. Each macro
# carries the table of record it is read from and the exact expression that defines it.


# --- exact two-sided McNemar, and Spearman, without adding a dependency -------------
def mcnemar_p(b, c):
    """Exact two-sided McNemar: binomial test on the b + c discordant pairs at p = 0.5.
    Identical to scipy.stats.binomtest(min(b, c), b + c, 0.5).pvalue (verified 2026-09-04)."""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    tail = sum(math.comb(n, k) for k in range(m + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def pfmt(p):
    """A p value with two significant figures, stored as a string so LaTeX never prints
    4.88281e-04. Down to 1e-4 it is a plain decimal; below that it is math-mode scientific
    notation, because ten decimal places in running prose is not readable. Every macro built
    with this is used inside math mode in main.tex. (Round 1 TEXT worker: the scientific form
    was added when the Mann-Whitney colour p of Section 3.4 came out at 6.3e-9; it changes how
    two of the McNemar values print, not what they are.)"""
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 1e-4:
        for d in range(2, 12):
            s = f"{p:.{d}f}"
            if (
                float(s) != 0.0
                and len(s.rstrip("0").replace("0.", "").lstrip("0")) >= 2
            ):
                return s
    e = math.floor(math.log10(p))
    m = p / (10.0**e)
    return "%.1f\\times 10^{%d}" % (m, e)


def spearman(a, b):
    """Spearman rho as the Pearson correlation of average ranks (ties handled by pandas)."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


# --- published photometric catalogues: Kocevski et al. and the Perger et al. compilation
# labels.parquet flags in_kocevski24 / in_perger25, built in recovery/build_labels.py from
# inputs/prior_art_catalogues/external_labels_raw.csv by a 0.5 arcsec sky match (R1, R2).
posK = pos.in_kocevski24.astype(bool)
posP = pos.in_perger25.astype(bool)
N["nPosInKocevskiCat"] = int(posK.sum())  # of the 151 positives
N["nPosInPergerCat"] = int(posP.sum())
N["nPosInEitherCat"] = int((posK | posP).sum())
missPos = pos[~pos.picked.astype(bool)]  # the 44 rule-missed positives
N["nMissedInKocevskiCat"] = int(missPos.in_kocevski24.astype(bool).sum())
N["nMissedInPergerCat"] = int(missPos.in_perger25.astype(bool).sum())
N["nMissedInEitherCat"] = int(
    (missPos.in_kocevski24.astype(bool) | missPos.in_perger25.astype(bool)).sum()
)
# the robust floor: in no re-implemented rule and in neither published catalogue
noSel = missPos[
    ~(missPos.in_kocevski24.astype(bool) | missPos.in_perger25.astype(bool))
]
N["nNoSelection"] = len(noSel)
N["nNoSelectionPct"] = round(100 * len(noSel) / N["nPos"], 1)
# nearest simple fraction 1/d to nNoSelection / nPos, rendered in words
_fr = N["nNoSelection"] / N["nPos"]
_d = min((3, 4, 5, 6, 7), key=lambda d: abs(_fr - 1.0 / d))
N["nNoSelectionFractionWords"] = "one in " + WORD[_d].lower()
noSelSup = posAll[
    (~posAll.picked.astype(bool))
    & (~posAll.in_kocevski24.astype(bool))
    & (~posAll.in_perger25.astype(bool))
    & posAll.in_support.astype(bool)
]
N["nNoSelectionSupport"] = len(noSelSup)
N["modelNoSelectionRecovered"] = int(
    oof.loc[noSelSup.source_id.values, "sel_burden"].sum()
)
# the kocevski24 module against the catalogue it re-implements (A1)
kcSup = sup[sup.in_kocevski24.astype(bool)]
N["kocevskiCatRows"] = len(kcSup)  # published catalogue rows inside the support
N["kocevskiCatRecall"] = N[
    "nPosInKocevskiCat"
]  # its recovery of the 151 (same quantity)
N["kocevskiModuleOnCatalogue"] = int(kcSup.sel_kocevski24.astype(bool).sum())
hvA = f[f.in_hviding25_A1.astype(bool)]
N["kocevskiCatOfHviding"] = int(hvA.in_kocevski24.astype(bool).sum())
N["kocevskiModuleOfHviding"] = int(hvA.sel_kocevski24.astype(bool).sum())
# the two modules that do validate against the same external sample (A1, A14): the only
# external check on the re-implementation is Hviding et al. table A1, where the published
# per-selection completenesses are quoted in the text from that paper.
N["kokorevModuleOfHviding"] = int(hvA.sel_kokorev24.astype(bool).sum())
N["barroModuleOfHviding"] = int(hvA.sel_barro23.astype(bool).sum())

# --- the second equal-burden comparison: seven rules plus both published catalogues (R2)
# in-support rows, deduplicated to one row per sky group exactly as recovery/v4_candidates.py
# does (above.duplicated("sky_group", keep="first") after sorting by descending score).
upc = (
    sup.picked.astype(bool)
    | sup.in_kocevski24.astype(bool)
    | sup.in_perger25.astype(bool)
)
N["unionPlusCatsSelected"] = int(sup[upc].sky_group.nunique())
N["unionPlusCatsRecall"] = int(
    (
        posAll.picked.astype(bool)
        | posAll.in_kocevski24.astype(bool)
        | posAll.in_perger25.astype(bool)
    ).sum()
)
_ord = (
    sup.assign(_s=oof.loc[sup.source_id.values, "score_mean"].to_numpy())
    .sort_values("_s", ascending=False)
    .head(N["unionPlusCatsSelected"])
)
N["modelAtUnionPlusCatsBurden"] = int((_ord.y == 1).sum())
# rows the union selects outside the seven-band support (A13)
N["unionOutsideSupportRows"] = int(f.picked.astype(bool).sum()) - int(
    sup.picked.astype(bool).sum()
)

# --- paired tests (B1, B3, B16, A10). Discordant counts b = the first named selection only,
# c = the second only, over the 147 in-support positives unless stated.
_P = posS.copy()
_P["sel_burden"] = oof.loc[_P.source_id.values, "sel_burden"].to_numpy()
_m = _P.sel_burden.to_numpy(bool)
_u = _P.picked.to_numpy(bool)
N["mcnemarModelUnionB"] = int((_m & ~_u).sum())
N["mcnemarModelUnionC"] = int((_u & ~_m).sum())
N["mcnemarModelUnionP"] = pfmt(
    mcnemar_p(N["mcnemarModelUnionB"], N["mcnemarModelUnionC"])
)
# Round 2 ruling S4 (Reviewer B finding B2-3): the McNemar test on the rule-missed subset is
# deleted. That subset is defined by the union's failure, so the union's discordant count is
# zero by construction rather than by measurement, the null the test evaluates cannot obtain,
# and the p value carried no information. The descriptive count the text now prints is
# modelRuleMissed, computed above. mcnemarRuleMissedB, mcnemarRuleMissedC and
# mcnemarRuleMissedP are removed from the record and from numbers.tex.
_b = _P[_P.c_f277w_f444w <= 1.0]
N["mcnemarBlueB"] = int(
    (_b.sel_burden.to_numpy(bool) & ~_b.picked.to_numpy(bool)).sum()
)
N["mcnemarBlueC"] = int(
    (_b.picked.to_numpy(bool) & ~_b.sel_burden.to_numpy(bool)).sum()
)
N["mcnemarBlueP"] = pfmt(mcnemar_p(N["mcnemarBlueB"], N["mcnemarBlueC"]))
_r = _P[_P.c_f277w_f444w > 1.0]
N["redDiscordantModel"] = int(
    (_r.sel_burden.to_numpy(bool) & ~_r.picked.to_numpy(bool)).sum()
)
N["redDiscordantUnion"] = int(
    (_r.picked.to_numpy(bool) & ~_r.sel_burden.to_numpy(bool)).sum()
)
# baseline_scores.parquet holds the per-object selections of every variant on the labelled rows
bsc = pd.read_parquet(os.path.join(V, "baseline_scores.parquet")).set_index("source_id")
_i = bsc.loc[_P.source_id.values, "sel_rules_reproduction"].to_numpy(bool)
_n = bsc.loc[_P.source_id.values, "sel_pn_lightgbm"].to_numpy(bool)
N["mcnemarImitationUnionB"] = int((_i & ~_u).sum())
N["mcnemarImitationUnionC"] = int((_u & ~_i).sum())
N["mcnemarImitationUnionP"] = pfmt(
    mcnemar_p(N["mcnemarImitationUnionB"], N["mcnemarImitationUnionC"])
)
N["mcnemarPrimaryImitationB"] = int((_m & ~_i).sum())
N["mcnemarPrimaryImitationC"] = int((_i & ~_m).sum())
N["mcnemarPrimaryImitationP"] = pfmt(
    mcnemar_p(N["mcnemarPrimaryImitationB"], N["mcnemarPrimaryImitationC"])
)
N["mcnemarPrimaryPNB"] = int((_m & ~_n).sum())
N["mcnemarPrimaryPNC"] = int((_n & ~_m).sum())
N["mcnemarPrimaryPNP"] = pfmt(mcnemar_p(N["mcnemarPrimaryPNB"], N["mcnemarPrimaryPNC"]))

# --- matched per-region burden (B4, R5): within each region rank the in-support rows by the
# pooled out-of-fold score and keep exactly the union's row count in that region.
_sup = sup.assign(_s=oof.loc[sup.source_id.values, "score_mean"].to_numpy())
_tot, _won = 0, 0
for r in REG:
    k = r.replace("-", "")
    g = _sup[_sup.region == r].sort_values("_s", ascending=False)
    v = int((g.head(N["unionSel" + k]).y == 1).sum())
    N["matchedRegion" + k] = v
    _tot += v
    _won += v > N["unionRegion" + k]
N["matchedRegionTotal"] = _tot
N["matchedRegionsWon"] = _won

# --- shuffled-label floors on the other two axes (B2)
# train_results.json baselines.shuffled_labels_0..4, the same five controls whose recall
# already gives shuffledMin and shuffledMax above.
shufb = [tr["baselines"]["shuffled_labels_%d" % i] for i in range(N["shuffledN"])]
N["shuffledMissedMin"] = min(x["rule_missed_recovered"] for x in shufb)
N["shuffledMissedMax"] = max(x["rule_missed_recovered"] for x in shufb)
N["shuffledAnchorsMin"] = min(x["neg_selected_at_burden"] for x in shufb)
N["shuffledAnchorsMax"] = max(x["neg_selected_at_burden"] for x in shufb)

# --- the ambiguous sources, counted as anchors (A7, R7)
ambS = sup[sup.ambiguous.astype(bool)]
N["nAmbiguousSupport"] = len(ambS)
N["ambModelSelected"] = int(oof.loc[ambS.source_id.values, "sel_burden"].sum())
N["ambUnionSelected"] = int(ambS.picked.astype(bool).sum())
N["modelAnchorExtraWithAmb"] = (N["modelAnchors"] + N["ambModelSelected"]) - (
    N["unionAnchors"] + N["ambUnionSelected"]
)

# --- what the negative anchors actually are (A4, R6) ------------------------------------
# The anchor rule in recovery/build_labels.py is
#     notv = d.vshaped_spec.astype("boolean").eq(False).fillna(False); neg = notv & ~d.in_any_list
# so an anchor is a source whose v3 `vshaped_spec` is exactly False. That column is built by
# the census code of the author's earlier work (not released) as an OR of the per-record
# `v_shaped` verdict over the source's frozen eligible records, NA where the source holds
# none; every frozen eligible record is an unlensed secure (grade >= 3) PRISM spectrum at
# z > 3. That code sets `v_shaped` False whenever `testable` is False, so vshaped_spec False
# mixes "tested and failed" with "eligible but the fit was refused". The v3 `testable` column
# splits the two. (The features/labels columns spec_tested, in_census and has_spec come from
# a different, row-level cross-match in the same earlier work and do not gate the anchors.)
#
# ROUND 3, Reviewer C finding 4: the split is now read from recovery/anchor_test_status.csv,
# a released table built by recovery/v4_round3_analysis.py, so a reader can reproduce these
# six macros from the release alone. The v3 parquet is still read here and the two are asserted
# equal, so the export can never drift from the table it was cut from.
s3 = pd.read_csv(
    os.path.join(V, "anchor_test_status.csv"),
    usecols=["source_id", "testable"],
).set_index("source_id")
_s3v3 = pd.read_parquet(
    os.path.join(ROOT, "inputs", "sources_v3.parquet"),
    columns=["source_id", "testable"],
).set_index("source_id")
assert (
    s3.testable.astype("boolean")
    .eq(_s3v3.testable.reindex(s3.index).astype("boolean"))
    .fillna(False)
    | (s3.testable.isna() & _s3v3.testable.reindex(s3.index).isna())
).all(), "anchor_test_status.csv disagrees with sources_v3.parquet on testable"
_negAll = lab[lab.y == 0]
_t = s3.loc[_negAll.source_id.values, "testable"].astype("boolean")
N["anchorsTested"] = int(_t.eq(True).fillna(False).sum())
N["anchorsUntested"] = int(_t.eq(False).fillna(False).sum())
N["anchorsOutsideCensus"] = int((~_negAll.in_census.fillna(False)).sum())
_ts = s3.loc[negS.source_id.values, "testable"].astype("boolean").eq(True).fillna(False)
_tsv = _ts.to_numpy()
N["anchorsTestedSupport"] = int(_tsv.sum())
N["anchorsUntestedSupport"] = int((~_tsv).sum())
N["modelAnchorsTested"] = int(
    (oof.loc[negS.source_id.values, "sel_burden"].to_numpy(bool) & _tsv).sum()
)
N["unionAnchorsTested"] = int((negS.picked.to_numpy(bool) & _tsv).sum())
N["modelAnchorTestedPct"] = round(
    100 * N["modelAnchorsTested"] / N["anchorsTestedSupport"], 2
)
N["unionAnchorTestedPct"] = round(
    100 * N["unionAnchorsTested"] / N["anchorsTestedSupport"], 2
)

# --- the positives are pre-selected on compactness (A12, R8) ----------------------------
# criterion r_h < 1.5 r_star, i.e. the paper's log radius ratio log_rh_over_rstar < log10(1.5);
# r_h is the catalogue half-light radius and r_star the frozen per-field stellar radius
# (recovery/v4_features.py line 128). Not the Labbe aperture measurement, which exists only
# for the candidates.
_cmp = np.log10(1.5)
N["posCompactPct"] = round(100 * float((posS.log_rh_over_rstar < _cmp).mean()), 1)
N["catCompactPct"] = round(100 * float((sup.log_rh_over_rstar < _cmp).mean()), 1)

# --- the deployed ranking is not the validated ranking (B20) ----------------------------
scf = pd.read_parquet(
    os.path.join(V, "scores.parquet"), columns=["source_id", "ranking_score"]
).set_index("source_id")
N["insampleOofSpearman"] = round(
    spearman(
        scf.loc[sup.source_id.values, "ranking_score"].to_numpy(float),
        oof.loc[sup.source_id.values, "score_mean"].to_numpy(float),
    ),
    3,
)
# a candidate is "above the out-of-fold threshold" if its out-of-fold score clears the 0.5 per
# cent threshold of its own held-out fold (oof_scores.parquet sel_05).
_cb = ~oof.loc[c.source_id.values, "sel_05"].to_numpy(bool)
N["candBelowOofThreshold"] = int(_cb.sum())
N["candBelowOofThresholdPct"] = round(100 * float(_cb.mean()), 1)

# --- the 0.5 per cent operating point and the imitation learner (B13, B7) ---------------
N["modelHalfRealisedPct"] = round(100 * ev["primary_0p5pct"]["selection_fraction"], 3)
# recovery/v4_train_lgbm.py, mode "rules": dict(PARAMS, num_leaves=31, max_depth=-1,
# learning_rate=0.05, seed=seed), num_boost_round=400, seeds[:2], on a 120000-row draw from
# the training-region unlabelled rows.
N["imitationLeaves"] = 31
N["imitationRounds"] = 400
N["imitationLearningRate"] = 0.05
N["imitationBags"] = 2
N["imitationUnlabelled"] = 120000

# --- archival rates against the observed denominator, and the high-recall tier (A15, R9) -
N["apEbObserved"] = ap["model_candidates_equal_burden"]["with_any_archive_record"]
N["apEbLowOfObserved"] = round(100 * N["apEbSecureLow"] / N["apEbObserved"], 1)
hrc = c[c.tier == "high_recall"]
N["apHrN"] = len(hrc)
N["apHrObserved"] = int((hrc.n_archive_records > 0).sum())
N["apHrSecureLow"] = int((hrc.status == "secure low-z").sum())
N["apHrSecureHigh"] = int((hrc.status == "secure z>3 (V untested)").sum())

# --- realised burdens of the challengers (B14) ------------------------------------------
N["selMlpSelected"] = ms["mlp_30_features"]["selected_total"]
N["selBlendSelected"] = ms["blend_trees30_mlp"]["selected_total"]
N["selTreesThirtyFourSelected"] = ms["trees_34_features"]["selected_total"]

# --- Reviewer C additions ----------------------------------------------------------------
# C5: the catalogue rows are not distinct sources. Sky groups at skyGroupArcsec, over all
# 630,869 rows of labels.parquet; sky_groups_total in denominators.json reproduces exactly.
N["nSkyGroups"] = den["sky_groups_total"]
_gs = f.sky_group.value_counts()
N["nRowsInMultiRowGroups"] = int(_gs[_gs > 1].sum())
_nf = f.groupby("sky_group").field.nunique()
N["nGroupsSpanningFields"] = int((_nf > 1).sum())
N["nRowsSpanningFields"] = int(_gs[_nf[_nf > 1].index].sum())

# C4: the support is band-complete AND every feature finite AND a positive stellar radius.
# recovery/v4_features.py line 134:
#     support = full & np.isfinite(X.values).all(axis=1) & (d.r_star_v1_arcsec.values > 0)
# where `full` is bands_complete, all seven fluxes and all seven errors measured.
N["nBandCompleteRows"] = N[
    "completeRows"
]  # same quantity as completeRows, named for the text
N["nSupportExcludedFeatures"] = N["completeRows"] - N["supportRows"]

# C: the positive-list decomposition, so 58 + 57 + 29 + 5 = 149 closes against
# nPosInPublishedList; nPosBarroOnly, nPosDeGraaffOnly and nPosInTwoOrMore already exist.
N["nPosHvidingOnly"] = int(
    (
        pos.in_hviding25_A1.astype(bool)
        & ~pos.in_barro25.astype(bool)
        & ~pos.in_degraaff26.astype(bool)
    ).sum()
)
assert (
    N["nPosHvidingOnly"]
    + N["nPosBarroOnly"]
    + N["nPosDeGraaffOnly"]
    + N["nPosInTwoOrMore"]
    == N["nPosInPublishedList"]
), "positive-list decomposition does not close"

# C14: the blue and red split on the ordinary AB colour the rules cut on, from the catalogue
# fluxes in labels.parquet, beside the asinh feature colour the existing macros use.
_ab = lab[["source_id", "f_f277w", "f_f444w"]].set_index("source_id")
_pab = _ab.loc[posS.source_id.values]
_cab = ab_colour(_pab.f_f277w, _pab.f_f444w)
assert np.isfinite(_cab).all(), "an in-support positive has no AB colour"
_pm = oof.loc[posS.source_id.values, "sel_burden"].to_numpy(bool)
_pu = posS.picked.to_numpy(bool)
_bab = _cab <= 1.0
N["blueNAB"] = int(_bab.sum())
N["blueUnionAB"] = int((_bab & _pu).sum())
N["blueModelAB"] = int((_bab & _pm).sum())
N["redNAB"] = int((~_bab).sum())
N["redUnionAB"] = int((~_bab & _pu).sum())
N["redModelAB"] = int((~_bab & _pm).sum())
# 99th percentile of |asinh colour minus AB colour| over the support rows where both are finite
# ROUND 3, Reviewer C finding 2 (ruling C3-1): the same mask as _ab_bd. Support rows whose
# two long-wavelength fluxes are both negative have no AB colour and must not enter the
# percentile; 1,061 such rows inflated the quoted divergence from 1.81 to 1.89 mag.
_sab = _ab.loc[sup.source_id.values]
_sabx = _sab.f_f277w.to_numpy(float)
_saby = _sab.f_f444w.to_numpy(float)
_sabok = (_sabx > 0) & (_saby > 0)
_scab = np.full(len(_sab), np.nan)
with np.errstate(divide="ignore", invalid="ignore"):
    _scab[_sabok] = -2.5 * np.log10(_sabx[_sabok] / _saby[_sabok])
_gap = np.abs(sup.c_f277w_f444w.to_numpy(float) - _scab)
N["colourAsinhAbGap"] = round(float(np.nanpercentile(_gap[np.isfinite(_gap)], 99)), 2)

# ================================================================ round 1 TEXT worker additions
# Added by the round 1 TEXT worker for findings the director accepted (DIRECTOR_round1.md R11)
# that need a number the ANALYSIS worker's list does not carry. Every one is read from the
# tables of record; nothing above this line changed except the six rounding fixes marked in
# place. Provenance is given per macro.

# --- which rules actually reach the blue positives (A6, Reviewer D major 5) --------------
# labbe23 and kokorev24 are satisfied by a branch (red1) that contains no F277W-F444W
# threshold at all; the other four impose one in every branch. Split the union's blue
# recoveries between the two groups so the causal sentence can be checked.
# Round 2, Reviewer C finding 6: three rules impose no F277W-F444W threshold in every
# branch, not two. labbe23 and kokorev24 are satisfied by a branch (red1) that carries no
# such threshold, and kocevski24 carries none at all, so kocevski24 belongs in this count;
# with it the macro equals blueUnionAB and matches the sentence it exists to support.
_blue = posS[posS.c_f277w_f444w <= 1.0]
N["blueUnionBranch"] = int(
    (
        _blue.sel_labbe23.astype(bool)
        | _blue.sel_kokorev24.astype(bool)
        | _blue.sel_kocevski24.astype(bool)
    ).sum()
)
N["blueUnionColourOnly"] = int(
    (
        _blue.sel_perezgonzalez24.astype(bool)
        | _blue.sel_barro23.astype(bool)
        | _blue.sel_greene24.astype(bool)
        | _blue.sel_akins24.astype(bool)
    ).sum()
)
# confirmed LRDs that are actually blue in F277W-F444W (Reviewer D minor 24)
N["nPosColourNegative"] = int((posS.c_f277w_f444w < 0).sum())

# --- the same colour bins on the ordinary AB colour the rules cut on (Reviewer C 14) -----
# The seven rules impose their F277W-F444W thresholds on ordinary AB colours, while the
# feature c_f277w_f444w is a difference of inverse-hyperbolic-sine magnitudes. Every
# colour-bin statement in the text is quoted on the AB colour, so each asinh macro above
# gets an AB counterpart here, computed exactly as the blueNAB block is: minus 2.5 times
# log10(f_f277w / f_f444w) from the catalogue fluxes in labels.parquet. Recomputed locally
# rather than reusing that block's temporaries so this block stands on its own.
_abflux2 = lab[["source_id", "f_f277w", "f_f444w"]].set_index("source_id")


def _ab_colour(source_ids):
    # ROUND 3, Reviewer C findings 1 and 2, applied here too. This was the last AB colour in
    # the file still computed by dividing first and filtering on finiteness afterwards, which
    # admits rows whose two fluxes are both negative. It feeds the positives, where nothing
    # moves because all of them have both fluxes positive, and the equal-burden candidates,
    # where candEbColourABDen is likewise unaffected. Routed through the shared mask anyway,
    # so the rule holds everywhere and not only at the two sites a reviewer happened to name.
    r = _abflux2.loc[list(source_ids)]
    return ab_colour(r.f_f277w, r.f_f444w)


_cabpos = _ab_colour(posS.source_id.values)
assert np.isfinite(_cabpos).all(), "a positive has no finite AB colour"
_babpos = _cabpos <= 1.0
# the AB counterparts of nPosColourBelowOne and nPosColourBelowOnePct. The count is the same
# quantity as blueNAB by construction, so the two cannot drift apart; both names exist
# because the text reads the bin one way in Section 3 and the other way in Section 5.
N["nPosColourBelowOneAB"] = int(_babpos.sum())
assert N["nPosColourBelowOneAB"] == N["blueNAB"], "AB blue bin disagrees with blueNAB"
N["nPosColourBelowOnePctAB"] = round(
    100 * N["nPosColourBelowOneAB"] / N["nPosSupport"], 1
)
N["posColourAboveOnePctAB"] = round(100 * float((~_babpos).mean()), 1)
N["posMedianColourAB"] = round(float(np.median(_cabpos)), 2)
N["blueUnionPctAB"] = round(100 * N["blueUnionAB"] / N["blueNAB"], 1)
N["blueModelPctAB"] = round(100 * N["blueModelAB"] / N["blueNAB"], 1)
# which rules reach the AB-blue positives: the AB counterparts of blueUnionBranch and
# blueUnionColourOnly (A6, Reviewer D major 5).
_blueAB = posS[_babpos]
N["blueUnionBranchAB"] = int(
    (
        _blueAB.sel_labbe23.astype(bool)
        | _blueAB.sel_kokorev24.astype(bool)
        | _blueAB.sel_kocevski24.astype(bool)
    ).sum()
)
assert N["blueUnionBranchAB"] == N["blueUnionAB"], (
    "the union's AB-blue recoveries do not all come from the three rules without a "
    "F277W-F444W threshold in every branch"
)
N["blueUnionColourOnlyAB"] = int(
    (
        _blueAB.sel_perezgonzalez24.astype(bool)
        | _blueAB.sel_barro23.astype(bool)
        | _blueAB.sel_greene24.astype(bool)
        | _blueAB.sel_akins24.astype(bool)
    ).sum()
)
N["nPosColourNegativeAB"] = int((_cabpos < 0).sum())
# the equal-burden candidates on the same colour, for the Section 6.1 comparison sentence
_cabeb = _ab_colour(eb.source_id.values)
_cabebf = _cabeb[np.isfinite(_cabeb)]
N["candEbMedianColourAB"] = round(float(np.median(_cabebf)), 2)
N["candEbColourAboveOnePctAB"] = round(100 * float((_cabebf > 1.0).mean()), 0)

# --- prose forms and simple derived counts ----------------------------------------------
# Reviewer D minor 21: MNRAS spells numbers below ten in running prose; the numeral stays in
# tables and figure labels, so the same quantity needs both forms.
N["nRegionsWord"] = WORD[N["nRegions"]].lower()
# Reviewer D minor 7: the Jaccard overlap needs its denominator stated, which is the size of
# the union of the two selected row sets.
N["setUnionRows"] = N["setBoth"] + N["setModelOnly"] + N["setUnionOnly"]

# --- the parent set of the negative anchors (A4, C1, ruling R6) --------------------------
# No macro is added here. The chain the text prints is the ANALYSIS worker's canonical set,
# written above: nCensusSpectra (the sources the V-shape test was run on) = nVshapePass +
# nVshapeFail, and nVshapeFail = nVshapeFailInList + nNeg, both asserted at their definition.
# An earlier draft of this block defined a second, identical set of names; they were removed
# so the paper cannot quote two macros for one quantity.

# --- the ambiguous sources counted as anchors, as totals (A7, ruling R7) -----------------
N["unionAnchorsWithAmb"] = N["unionAnchors"] + N["ambUnionSelected"]
N["modelAnchorsWithAmb"] = N["modelAnchors"] + N["ambModelSelected"]

# --- colour and magnitude of the misses: the tests (A9) ----------------------------------
# Two-sided Mann-Whitney U on the 41 in-support rule-missed positives against the 106 the
# rules recover. scipy is a pinned dependency (requirements.txt).
from scipy.stats import mannwhitneyu  # noqa: E402

N["colourMWUp"] = pfmt(
    float(
        mannwhitneyu(
            missS.c_f277w_f444w.to_numpy(float),
            hitS.c_f277w_f444w.to_numpy(float),
            alternative="two-sided",
        ).pvalue
    )
)
N["magMWUp"] = pfmt(
    float(
        mannwhitneyu(
            missS.mag_f444w.to_numpy(float),
            hitS.mag_f444w.to_numpy(float),
            alternative="two-sided",
        ).pvalue
    )
)

# --- the leave-one-list-out size confound (B17) ------------------------------------------
# Removing a list removes that many training positives; the three results are monotone in
# what is left, which is a complete alternative explanation for their ordering.
N["locoRemainHvidingA"] = N["nPosSupport"] - N["catalogNHvidingA"]
N["locoRemainDeGraaff"] = N["nPosSupport"] - N["catalogNDeGraaff"]
N["locoRemainBarro"] = N["nPosSupport"] - N["catalogNBarro"]

# --- how bad the learned selection's failures are (Reviewer D major 16) ------------------
# evaluation.json still_missed_list carries score_rank_pct, already 100 times the fraction of
# the support scoring at least as high (recovery/v4_evaluate.py).
_sm = pd.DataFrame(ev["still_missed_list"])
_smn = _sm[~_sm.picked.astype(bool)]  # the pairedNeither in-support positives
N["pairedNeitherTopTwo"] = int((_smn.score_rank_pct <= 2.0).sum())
N["worstMissRankPct"] = round(float(_smn.score_rank_pct.max()), 1)

# --- candidates below the strictest detection gate of the rules (A22) --------------------
# Four of the seven rules require F444W brighter than 27.7 or a F444W signal-to-noise above
# 12, so for a candidate outside that gate "no rule selects it" is partly a detection limit.
_cf = f.set_index("source_id").loc[c.source_id.values]
_gate = (_cf.mag_f444w.to_numpy(float) > 27.7) | (_cf.snr_f444w.to_numpy(float) <= 12)
N["candBelowDetection"] = int(_gate.sum())
N["candEbBelowDetection"] = int((_gate & (c.tier.to_numpy() == "equal_burden")).sum())

# --- dropping the two positives the V-shape test admits (A25) ----------------------------
# Those two are separated from the anchors by construction, because the same fixed test
# defines both classes. Nothing in the headline changes when they are dropped.
_vonly = set(pos.loc[~pos[L].astype(bool).any(axis=1), "source_id"])
_kept = posAll[~posAll.source_id.isin(_vonly)]
_keptS = _kept[_kept.in_support.astype(bool)]
N["nPosExclVtest"] = len(_kept)
N["nPosSupportExclVtest"] = len(_keptS)
N["unionRecallExclVtest"] = int(_kept.picked.astype(bool).sum())
N["modelRecallExclVtest"] = int(oof.loc[_keptS.source_id.values, "sel_burden"].sum())

# --- why a single pooled threshold cannot match the union region by region (B4) ----------
_ubf = [
    100 * N["unionSel" + r.replace("-", "")] / N["rows" + r.replace("-", "")]
    for r in REG
]
N["unionBurdenRegionMin"] = round(min(_ubf), 3)
N["unionBurdenRegionMax"] = round(max(_ubf), 3)

# --- the inner selection band and the argmax cell (B6, B30) ------------------------------
# train_results.json folds[region]: best_anchor and best_trees are the argmax cell, one_se the
# width of the band the rule admits, in case-control PR-AUC over the four inner folds.
_fj = [tr["folds"][r] for r in REG]
N["innerBestAnchor"] = sorted({x["best_anchor"] for x in _fj})[0]
assert len({x["best_anchor"] for x in _fj}) == 1, "argmax anchor differs between folds"
N["innerBestTreesMin"] = min(x["best_trees"] for x in _fj)
N["innerBestTreesMax"] = max(x["best_trees"] for x in _fj)
N["innerSeMin"] = round(min(x["one_se"] for x in _fj), 3)
N["innerSeMax"] = round(max(x["one_se"] for x in _fj), 3)

# --- the photometric-redshift ablation against its own comparator (B12, D3) --------------
# Every ablation is an eight-bag run, so the comparator is the eight-bag replica, not the
# deployed 48-bag primary; this is the "one more" the text used to assert by hand.
N["blWithZphotOverReplica"] = N["blWithZphotRecall"] - N["blReplicaRecall"]

# --- realised catalogue burden of every variant (round 2, reviewer E findings 1 and 2) ----
# The run of record scored every baseline, ablation, shuffled control and the imitation of the
# rules on the 5126 labelled rows only, so none of them had a catalogue-wide burden a reader
# could check, and the imitation comparison could not be burden matched. recovery/
# v4_baseline_burden.py re-runs every variant with the record's exact code path -- same data,
# masks, seeds, bag counts, per-fold settings and threshold rule, with the shared rng_sub
# stream replayed so the threshold subsamples are the record's own draws -- and scores every
# in-support row of the held-out region instead of only the labelled ones. It reproduces the
# record's labelled-row numbers for every variant; the assert below is that check.
bu = J("baseline_burden.json")
assert all(v["reproduces_record"] for v in bu["variants"].values()), [
    k for k, v in bu["variants"].items() if not v["reproduces_record"]
]
for _name, _tag in (
    ("primary_8bag_replica", "Replica"),
    ("magnitude_only", "MagOnly"),
    ("morphology_only", "MorphOnly"),
    ("ablate_c_f277w_f444w", "AblateColour"),
    ("ablate_size", "AblateSize"),
    ("anchor_0", "AnchorZero"),
    ("anchor_0.50", "AnchorHalf"),
    ("pn_lightgbm", "PN"),
    ("rules_reproduction", "Imitation"),
    ("with_zphot", "WithZphot"),
):
    _v = bu["variants"][_name]
    N["bl" + _tag + "Selected"] = _v["selected_rows_pooled"]
    N["bl" + _tag + "MatchedRecall"] = _v["matched"]["matched_recall_support"]
    N["bl" + _tag + "MatchedMissed"] = _v["matched"]["matched_rule_missed"]
    N["bl" + _tag + "MatchedAnchors"] = _v["matched"]["matched_anchors"]
for _c, _tag in (
    ("hviding25_A1", "HvidingA"),
    ("barro25", "Barro"),
    ("degraaff26", "DeGraaff"),
):
    _v = bu["variants"]["leave_out_" + _c]
    N["loco" + _tag + "Selected"] = _v["selected_rows_pooled"]
    N["loco" + _tag + "MatchedRecall"] = _v["matched"]["matched_recall_support"]
    N["loco" + _tag + "MatchedMissed"] = _v["matched"]["matched_rule_missed"]
    N["loco" + _tag + "MatchedAnchors"] = _v["matched"]["matched_anchors"]
# the imitation of the rules, named directly because Section 5.2 argues from it
_im = bu["variants"]["rules_reproduction"]
N["imitationSelected"] = _im["selected_rows_pooled"]
N["imitationRecallSupport"] = _im["labelled_row_reproduction"]["recall_at_burden"]
N["imitationRuleMissed"] = _im["labelled_row_reproduction"]["rule_missed_recovered"]
N["imitationAnchors"] = _im["labelled_row_reproduction"]["neg_selected_at_burden"]
N["imitationMatchedRecall"] = _im["matched"]["matched_recall_support"]
N["imitationMatchedRuleMissed"] = _im["matched"]["matched_rule_missed"]
N["imitationBurdenRatio"] = round(_im["selected_rows_pooled"] / N["unionSelected"], 2)
_iu = bu["paired_matched"]["imitation_vs_union_positives"]
N["mcnemarImitationUnionMatchedB"] = _iu["a_only"]
N["mcnemarImitationUnionMatchedC"] = _iu["b_only"]
N["mcnemarImitationUnionMatchedP"] = pfmt(_iu["p"])
_pi = bu["paired_matched"]["primary_vs_imitation_positives"]
N["mcnemarPrimaryImitationMatchedB"] = _pi["a_only"]
N["mcnemarPrimaryImitationMatchedC"] = _pi["b_only"]
N["mcnemarPrimaryImitationMatchedP"] = pfmt(_pi["p"])
_shf = [bu["variants"]["shuffled_labels_%d" % i] for i in range(N["shuffledN"])]
N["shuffledSelectedMin"] = min(x["selected_rows_pooled"] for x in _shf)
N["shuffledSelectedMax"] = max(x["selected_rows_pooled"] for x in _shf)
N["shuffledMatchedMin"] = min(x["matched"]["matched_recall_support"] for x in _shf)
N["shuffledMatchedMax"] = max(x["matched"]["matched_recall_support"] for x in _shf)
N["shuffledMatchedMissedMin"] = min(x["matched"]["matched_rule_missed"] for x in _shf)
N["shuffledMatchedMissedMax"] = max(x["matched"]["matched_rule_missed"] for x in _shf)

# =========================================================================================
# ROUND 2, ANALYSIS WORKER B: the macros the round-2 director file lists under
# "ANALYSIS worker B". Every value below is read from the tables of record; nothing above
# this header is changed. Provenance is given per block.
# =========================================================================================

from scipy.stats import fisher_exact  # noqa: E402

# --- the Barro et al. list and its own published colour cut (S6, A2-1) -------------------
# labels/features flag in_barro25 for the members of the spectroscopic list of
# Barro et al. 2025 that fall in these nine fields; sel_barro23 is this paper's
# re-implementation of that group's own published photometric cut (Table 1). The two
# denominators differ: barroListN is the whole positive set (the number Reviewer A quotes),
# catalogNBarro is the in-support subset already defined above.
_pbarro = posAll[posAll.in_barro25.astype(bool)]
N["barroListN"] = len(_pbarro)
N["barroModuleOnBarroList"] = int(_pbarro.sel_barro23.astype(bool).sum())
_pbarroS = posS[posS.in_barro25.astype(bool)]
N["barroModuleOnBarroListSupport"] = int(_pbarroS.sel_barro23.astype(bool).sum())
assert len(_pbarroS) == N["catalogNBarro"], "in-support Barro denominator moved"
# how much of the positive set and of the rule-missed set that one list carries
_bonly = (
    posAll.in_barro25.astype(bool)
    & ~posAll.in_hviding25_A1.astype(bool)
    & ~posAll.in_degraaff26.astype(bool)
)
N["nPosBarroOnlyAll"] = int(_bonly.sum())
N["unionBarroOnly"] = int(posAll[_bonly].picked.astype(bool).sum())
_missAll = posAll[~posAll.picked.astype(bool)]  # the nRuleMissed positives
N["nRuleMissedBarroOnly"] = int(
    (
        _missAll.in_barro25.astype(bool)
        & ~_missAll.in_hviding25_A1.astype(bool)
        & ~_missAll.in_degraaff26.astype(bool)
    ).sum()
)

# --- what each published catalogue alone leaves in the floor (S6, A2-2) ------------------
# floorKocevskiOnly: rule-missed positives not in the Kocevski et al. catalogue, i.e. the
# floor the paper would report if the Perger et al. compilation were not used at all.
N["floorKocevskiOnly"] = int((~_missAll.in_kocevski24.astype(bool)).sum())
N["floorPergerOnly"] = int((~_missAll.in_perger25.astype(bool)).sum())
# the floor with GOODS-North dropped, the field no published catalogue covers
_floor = _missAll[
    ~(_missAll.in_kocevski24.astype(bool) | _missAll.in_perger25.astype(bool))
]
assert len(_floor) == N["nNoSelection"], "floor set moved"
N["nFloorInGOODSN"] = int((_floor.region == "GOODS-N").sum())
N["floorNoGoodsN"] = int((_floor.region != "GOODS-N").sum())
N["floorNoGoodsNDen"] = int((posAll.region != "GOODS-N").sum())
N["floorNoGoodsNPct"] = round(100 * N["floorNoGoodsN"] / N["floorNoGoodsNDen"], 1)

# --- per-field coverage of the two published catalogues (S6, A2-2) ----------------------
# Counted over every catalogue row, which is the denominator that describes the catalogues'
# sky coverage; the in-support counts differ because ngdeep contributes no support rows at
# all and because incomplete-band rows are dropped. Both catalogue flags are one row per
# object already (316 and 477 flagged rows, 316 and 477 distinct sky groups), so the sky
# group deduplication the candidate ledger applies changes nothing here.
_FIELDMAC = {
    "ceers-full": "CeersFull",
    "gdn": "Gdn",
    "gds": "Gds",
    "gds-sw": "GdsSw",
    "ngdeep": "Ngdeep",
    "primer-cosmos-east": "PrimerCosmosEast",
    "primer-cosmos-west": "PrimerCosmosWest",
    "primer-uds-north": "PrimerUdsNorth",
    "primer-uds-south": "PrimerUdsSouth",
}
assert set(_FIELDMAC) == set(f.field.unique()), "field list moved"
for _fl, _mac in _FIELDMAC.items():
    _g = f[f.field == _fl]
    N["kocevskiCatRows" + _mac] = int(_g.in_kocevski24.astype(bool).sum())
    N["pergerCatRows" + _mac] = int(_g.in_perger25.astype(bool).sum())
    N["catRows" + _mac] = len(_g)
N["kocevskiCatRowsAll"] = int(f.in_kocevski24.astype(bool).sum())
N["pergerCatRowsAll"] = int(f.in_perger25.astype(bool).sum())
assert N["kocevskiCatRowsAll"] == sum(
    N["kocevskiCatRows" + m] for m in _FIELDMAC.values()
)
assert N["pergerCatRowsAll"] == sum(N["pergerCatRows" + m] for m in _FIELDMAC.values())
# region aliases: the GOODS-North region is the single field gdn, so the region-level and
# field-level counts are the same number, and it is the one the text names.
N["kocevskiCatRowsGOODSN"] = N["kocevskiCatRowsGdn"]
N["pergerCatRowsGOODSN"] = N["pergerCatRowsGdn"]
# the Perger compilation inside the support, one row per sky group, the counterpart of the
# existing kocevskiCatRows (which is the in-support Kocevski count).
_pergSup = sup[sup.in_perger25.astype(bool)]
N["pergerCatRows"] = int(_pergSup.sky_group.nunique())
N["kocevskiCatRowsSupport"] = int(
    sup[sup.in_kocevski24.astype(bool)].sky_group.nunique()
)
# the existing kocevskiCatRows counts in-support rows rather than sky groups; the two agree
# because no catalogue object is flagged on two rows, and the assert keeps them from drifting.
assert N["kocevskiCatRowsSupport"] == N["kocevskiCatRows"], (
    "a Kocevski catalogue object is flagged on more than one support row"
)

# --- the frozen V-shape test read on the positives (S7, A2-3) ---------------------------
# features.parquet carries the v4 accounting of the same frozen test Appendix C describes:
# has_spec (an eligible spectrum exists), spec_tested (the test was run), spec_vshaped (the
# test establishes the V shape). The v3 table inputs/sources_v3.parquet carries an
# earlier accounting on these positives (134 eligible, 130 testable, 42 established) and is
# not the set of record for this paper; the v4 columns are what Section 2.3 and the anchor
# definition use, and they are what Reviewer A recomputed.
_T = posAll.spec_tested.astype(bool)
_V = posAll.spec_vshaped.astype(bool)
_hit = posAll.picked.astype(bool)
N["posEligible"] = int(posAll.has_spec.astype(bool).sum())
N["posTested"] = int(_T.sum())
N["posVshaped"] = int(_V.sum())
N["posHitTested"] = int((_T & _hit).sum())
N["posHitVshaped"] = int((_V & _hit).sum())
N["posMissTested"] = int((_T & ~_hit).sum())
N["posMissVshaped"] = int((_V & ~_hit).sum())
N["posHitVshapedPct"] = round(100 * N["posHitVshaped"] / N["posHitTested"], 1)
N["posMissVshapedPct"] = round(100 * N["posMissVshaped"] / N["posMissTested"], 1)
# exactly one positive in the whole catalogue carries an established V shape without the
# tested flag, so posVshaped is not a strict subset of posTested; the count is emitted so the
# text can say so rather than print a ratio that does not nest.
N["posVshapedNotTested"] = int((_V & ~_T).sum())
N["fisherVshapeP"] = pfmt(
    float(
        fisher_exact(
            [
                [N["posHitVshaped"], N["posHitTested"] - N["posHitVshaped"]],
                [N["posMissVshaped"], N["posMissTested"] - N["posMissVshaped"]],
            ],
            alternative="two-sided",
        )[1]
    )
)
# The same three counts on the denominator that nests: a positive "carries a verdict" if the
# test was run on it or the test establishes its V shape. That differs from posTested by the
# single row above, so posVshaped is a subset of posVerdict while it is not a subset of
# posTested, and a ratio printed on these denominators cannot be read as an arithmetic slip.
# The conclusion is the same on either denominator, p = 0.53 against 0.54.
_Vd = _T | _V
N["posVerdict"] = int(_Vd.sum())
N["posHitVerdict"] = int((_Vd & _hit).sum())
N["posMissVerdict"] = int((_Vd & ~_hit).sum())
N["fisherVshapeVerdictP"] = pfmt(
    float(
        fisher_exact(
            [
                [N["posHitVshaped"], N["posHitVerdict"] - N["posHitVshaped"]],
                [N["posMissVshaped"], N["posMissVerdict"] - N["posMissVshaped"]],
            ],
            alternative="two-sided",
        )[1]
    )
)
assert N["posVshaped"] <= N["posVerdict"], "V-shaped passes outnumber the verdicts"
# the same three counts on the two subsets the paper leads with: the rule-missed positives
# the learned selection recovers at equal burden, and the floor of Section 3.3.
_selb = oof["sel_burden"].reindex(posAll.source_id.values).eq(True).to_numpy(bool)
_recmiss = posAll[(~_hit).to_numpy() & _selb]
assert len(_recmiss) == N["modelRuleMissed"], "recovered rule-missed set moved"
N["recoveredMissedTested"] = int(_recmiss.spec_tested.astype(bool).sum())
N["recoveredMissedVshaped"] = int(_recmiss.spec_vshaped.astype(bool).sum())
N["floorTested"] = int(_floor.spec_tested.astype(bool).sum())
N["floorVshaped"] = int(_floor.spec_vshaped.astype(bool).sum())

# --- the brown-dwarf colour screen the rules already carry (S8, A2-7) -------------------
# kokorev24 requires F115W-F200W above -0.5 and greene24 folds the same floor into its
# window. Reported on the asinh colour c_f115w_f200w, the feature the model sees and the
# same convention every other tier colour in the paper uses, because F115W is a dropout band
# here: the ordinary AB colour is undefined for 34 of the 362 equal-burden candidates, 10 of
# the 126 follow-up sources and 11 of the 151 positives, whose F115W flux is not positive.
# The AB counterparts are emitted beside them, as the F277W-F444W block above does.
_BD = -0.5
N["bdEb"] = int((eb.c_f115w_f200w < _BD).sum())
N["bdEbPct"] = round(100 * N["bdEb"] / len(eb), 1)
_fu = c[c.followup_tier.astype(bool)]
assert len(_fu) == N["followupN"], "follow-up tier size moved"
N["bdFu"] = int((_fu.c_f115w_f200w < _BD).sum())
N["bdFuPct"] = round(100 * N["bdFu"] / len(_fu), 1)
N["bdPos"] = int((posAll.c_f115w_f200w < _BD).sum())
N["bdPosPct"] = round(100 * N["bdPos"] / len(posAll), 1)
_abflux3 = lab[["source_id", "f_f115w", "f_f200w"]].set_index("source_id")


def _ab_bd(source_ids):
    # ROUND 3, Reviewer C finding 1 (ruling C3-1): the AB colour exists only where both
    # catalogue fluxes are positive, which is the definition the colour-plane figure caption
    # states. Filtering on a finite logarithm instead admitted rows whose two fluxes are both
    # negative, where the ratio is positive and the logarithm is the colour of the absolute
    # values. Masking before the logarithm moves bdEbAB from 44 to 37, bdEbNoColourAB from 34
    # to 43, bdFuAB from 17 to 16 and bdFuNoColourAB from 10 to 11; the positives row was
    # correct either way, because all of them have both fluxes positive. The mask itself lives
    # in ab_colour, so the rule has one implementation in this file and not two.
    r = _abflux3.loc[list(source_ids)]
    return ab_colour(r.f_f115w.to_numpy(float), r.f_f200w.to_numpy(float))


for _tag, _ids, _n in (
    ("Eb", eb.source_id.values, len(eb)),
    ("Fu", _fu.source_id.values, len(_fu)),
    ("Pos", posAll.source_id.values, len(posAll)),
):
    _v = _ab_bd(_ids)
    _fin = np.isfinite(_v)
    N["bd" + _tag + "AB"] = int((_v[_fin] < _BD).sum())
    N["bd" + _tag + "NoColourAB"] = int((~_fin).sum())

# --- the follow-up tier's colours and its catastrophic photometric redshifts (S8) --------
# Same asinh F277W-F444W colour as candEbMedianColour and posMedianColour, so the three are
# comparable; z_phot is the catalogue photometric redshift, which the ranking does not use.
N["fuMedianColour"] = round(float(_fu.c_f277w_f444w.median()), 2)
N["fuAboveOne"] = int((_fu.c_f277w_f444w > 1.0).sum())
N["fuAboveOneHalf"] = int((_fu.c_f277w_f444w > 1.5).sum())
N["fuZphotAboveTen"] = int((_fu.z_phot > 10).sum())

# --- one compactness criterion applied to the positives and to both candidate tiers (S8) -
# The criterion is r_h < 1.5 r_star on the catalogue half-light radius against the frozen
# per-field stellar radius, i.e. log_rh_over_rstar < log10(1.5), the same criterion the
# existing posCompactPct uses and the only one every row in all three sets carries. The
# Labbe aperture measurement exists only for the candidates, so it cannot be the shared
# criterion; the positives have no aperture photometry measured on the mosaics.
_crit = np.log10(1.5)
N["posCompactCritPct"] = round(100 * float((posS.log_rh_over_rstar < _crit).mean()), 1)
assert N["posCompactCritPct"] == N["posCompactPct"], "compactness criterion drifted"
N["ebCompactCritPct"] = round(100 * float((eb.log_rh_over_rstar < _crit).mean()), 1)
N["fuCompactCritPct"] = round(100 * float((_fu.log_rh_over_rstar < _crit).mean()), 1)

# --- the deployed ranking against the validated ranking in the tail that matters (B2-11) -
# The whole-support Spearman above is dominated by rows at the noise floor. These two read
# the same two score columns inside the candidate threshold: the overlap of the two rankings'
# top candRowsAboveHalf sets, and the Spearman inside the in-sample top one per cent.
_ids = sup.source_id.to_numpy()
_ins = scf.loc[_ids, "ranking_score"].to_numpy(float)
_oo = oof.loc[_ids, "score_mean"].to_numpy(float)
N["topsetN"] = N["candRowsAboveHalf"]
_ti = set(_ids[np.argsort(-_ins, kind="stable")[: N["topsetN"]]])
_to = set(_ids[np.argsort(-_oo, kind="stable")[: N["topsetN"]]])
N["topsetOverlap"] = len(_ti & _to)
N["topsetOverlapPct"] = round(100 * N["topsetOverlap"] / N["topsetN"], 1)
N["topsetOnePctN"] = int(len(sup) * 0.01)
_sel1 = np.argsort(-_ins, kind="stable")[: N["topsetOnePctN"]]
N["spearmanTopOnePct"] = round(spearman(_ins[_sel1], _oo[_sel1]), 3)

# --- the size of the matched per-region wins, and the multiplicity of the paired tests ---
# (S5, B2-13, B2-14). matchedRegion* and unionRegion* are computed above; these are their
# differences, so "leads in all five regions" can be printed with its margins.
_marg = []
for _r in REG:
    _k = _r.replace("-", "")
    N["margin" + _k] = N["matchedRegion" + _k] - N["unionRegion" + _k]
    _marg.append(N["margin" + _k])
N["marginMin"] = min(_marg)
N["marginMax"] = max(_marg)
# how many of the five margins are a single object, which is the scale Section 4.4's
# promotion rule treats as a tie (B2-13). Spelled out, because the prose spells numbers
# below ten.
# spelled forms, because MNRAS spells numbers below ten in running prose and the abstract
# and Section 5.1 print the margin range there.
N["marginMinWord"] = WORD[N["marginMin"]].lower()
N["marginMaxWord"] = WORD[N["marginMax"]].lower()
N["nMarginOne"] = sum(1 for m in _marg if m == 1)
N["nMarginOneWord"] = WORD[N["nMarginOne"]].lower()
# five wins out of five regions, read as a sign test: two-sided binomial at p = 0.5. The
# value is the exact rational 2 / 2**5; it is written out in full rather than at the two
# significant figures pfmt gives, because rounding an exact 0.0625 down to 0.062 would
# misstate where it sits relative to 0.05.
N["signTestP"] = "%.4f" % (2.0 * 0.5 ** N["nRegions"])
# ROUND 3, Reviewer B finding B3-3 (ruling B3-3): the family is every exact test the paper
# prints, which is eight, not five. Six exact McNemar tests: model against union (the
# primary, mcnemarModelUnionP), the blue bin (mcnemarBlueP), imitation against union at
# matched burden (mcnemarImitationUnionMatchedP), primary against imitation at matched burden
# (mcnemarPrimaryImitationMatchedP), the same on the rule-missed subset
# (mcnemarPrimaryImitationMatchedRmP, printed in Section 5.3 and omitted from the round 2
# enumeration), and primary against the positive-negative baseline (mcnemarPrimaryPNP). Plus
# the Fisher exact test of Section 2.2 (fisherVshapeTestedP) and the exact binomial sign test
# of Section 5.1 (signTestP). Round 4, Reviewer A finding 1 adds one more exact test the paper
# prints, the two-sided Fisher exact test of the compactness split in Section 3.4
# (fisherCompactP), so the family is nine. Bonferroni on the primary test only; it survives
# either count.
N["bonferroniFamily"] = 9
N["bonferroniHeadlineP"] = pfmt(
    min(
        1.0,
        N["bonferroniFamily"]
        * mcnemar_p(N["mcnemarModelUnionB"], N["mcnemarModelUnionC"]),
    )
)
N["bonferroniFamilyWord"] = WORD[N["bonferroniFamily"]].lower()

# --- the learned selection's own failure set, not the "neither" set (B2-9) ---------------
# evaluation.json still_missed_list is the model's in-support misses, all of them: the
# pairedNeither positives plus the pairedUnionOnly ones the rules recover and it does not.
# The existing pairedNeitherTopTwo is computed on the smaller "neither" subset and is left
# untouched; these two describe the set the sentence claims to describe.
_smAll = pd.DataFrame(ev["still_missed_list"])
N["nMissedSupport"] = len(_smAll)
assert N["nMissedSupport"] == N["nPosSupport"] - N["modelRecallSupport"], (
    "still_missed_list is not the model's in-support failure set"
)
N["nMissedTopTwoPct"] = int((_smAll.score_rank_pct <= 2.0).sum())
N["nMissedUnionOnlyTopTwoPct"] = int(
    (_smAll[_smAll.picked.astype(bool)].score_rank_pct <= 2.0).sum()
)

# =========================================================================================
# ROUND 2, TEXT WORKER: the macros the round-2 rulings need that neither analysis worker
# emitted. Every value is read from the tables of record; nothing above this header is
# changed except the four in-place corrections marked in place (the deleted rule-missed
# McNemar, the Bonferroni family size, the AB colour bins and the blue-branch rule set).
# =========================================================================================

# --- the two positives the V-shape test admits, and the three it does not (C2-1, C2-2) ----
# recovery/build_labels.py sets pos = (in_spec_list | y_strict), where y_strict is the
# `positive` column of inputs/sources_v3.parquet, the author's earlier compact
# V-shaped prism set. Five sources in no published list pass the frozen V-shape test; the
# criterion that separates the two admitted from the three not admitted is recoverable from
# the record and is the compactness criterion r_h < 1.5 r_star, that is log_ratio_v1 below
# log10(1.5): the two admitted carry 0.100 and 0.098, the three not admitted 0.235, 0.309
# and 0.641. All five are eligible, testable and modelable in the v3 table, so compactness
# is the only criterion that separates them.
_pass = lab.vshaped_spec.astype("boolean").eq(True).fillna(False).to_numpy()
_unl = _pass & ~lab.in_any_list.astype(bool).to_numpy()
N["nVshapePassUnlisted"] = int(_unl.sum())
_unlrows = lab[_unl]
_compact = _unlrows.log_ratio_v1.to_numpy(float) < np.log10(1.5)
N["nVshapePassUnlistedCompact"] = int(_compact.sum())
N["nVshapePassUnlistedNotAdded"] = (
    N["nVshapePassUnlisted"] - N["nVshapePassUnlistedCompact"]
)
N["nVshapePassUnlistedAmbiguous"] = int(
    _unlrows.in_hviding25_B1.astype(bool).to_numpy().sum()
)
assert N["nVshapePassUnlistedCompact"] == N["nPosOnlySpectralTest"], (
    "the compactness criterion does not reproduce the two admitted positives"
)
assert (_unlrows.y.fillna(0).astype(int).to_numpy() == _compact.astype(int)).all(), (
    "an unlisted V-shape passer is a positive without being compact, or the reverse"
)

# --- the realised burden spread of the four challengers (C2-7, B2-16) --------------------
# The challenger table's caption quoted "up to six per cent" as typed prose; the four
# realised burdens are modelSelected and the three challenger burdens, spread max over min.
_chb = [
    N["modelSelected"],
    N["selTreesThirtyFourSelected"],
    N["selMlpSelected"],
    N["selBlendSelected"],
]
N["challengerBurdenSpreadPct"] = round(100 * (max(_chb) / min(_chb) - 1), 1)

# --- the denominator of the equal-burden tier's AB colour fraction (C2-11) ---------------
# candEbColourAboveOnePctAB is already computed over the rows with a finite AB colour, which
# is the convention Figure 2 uses ("a source with a non-positive catalogue flux in either
# band has no AB colour and is not plotted"). The denominator is emitted so the sentence can
# name it instead of leaving the reader to assume all candEqualBurden rows.
N["candEbColourABDen"] = int(np.isfinite(_cabeb).sum())
N["candEbNoColourAB"] = N["candEqualBurden"] - N["candEbColourABDen"]

# --- the matched-burden cost and rule-missed axes of the imitation comparison (S1) --------
# baseline_burden.json carries the anchors let through at matched burden for the primary and
# for every variant, and the paired matched tests on the rule-missed subset. Section 5.2's
# anchor-rate sentence is rewritten from these: at matched burden the imitation lets through
# fewer anchors than the primary, so the labels do not show on that axis either.
N["primaryMatchedRecall"] = bu["primary_48bag_matched"]["matched_recall_support"]
N["primaryMatchedRuleMissed"] = bu["primary_48bag_matched"]["matched_rule_missed"]
N["primaryMatchedAnchors"] = bu["primary_48bag_matched"]["matched_anchors"]
N["imitationMatchedAnchors"] = bu["variants"]["rules_reproduction"]["matched"][
    "matched_anchors"
]
assert N["primaryMatchedRecall"] == N["matchedRegionTotal"], (
    "the burden re-run's matched recall disagrees with the paper's matched per-region total"
)
_iurm = bu["paired_matched"]["imitation_vs_union_rule_missed"]
N["mcnemarImitationUnionMatchedRmB"] = _iurm["a_only"]
N["mcnemarImitationUnionMatchedRmC"] = _iurm["b_only"]
N["mcnemarImitationUnionMatchedRmP"] = pfmt(_iurm["p"])
_pirm = bu["paired_matched"]["primary_vs_imitation_rule_missed"]
N["mcnemarPrimaryImitationMatchedRmB"] = _pirm["a_only"]
N["mcnemarPrimaryImitationMatchedRmC"] = _pirm["b_only"]
N["mcnemarPrimaryImitationMatchedRmP"] = pfmt(_pirm["p"])
_pum = bu["paired_matched"]["primary_vs_union_positives"]
N["mcnemarPrimaryUnionMatchedB"] = _pum["a_only"]
N["mcnemarPrimaryUnionMatchedC"] = _pum["b_only"]
N["mcnemarPrimaryUnionMatchedP"] = pfmt(_pum["p"])
# the imitation's per-region realised burden, which misses the union's in both directions:
# it under-selects in CEERS and over-selects in the largest region.
_imreg = bu["variants"]["rules_reproduction"]["selected_rows_per_region"]
_unreg = bu["union"]["selected_rows_per_region"]
for _r in REG:
    N["imitationSel" + _r.replace("-", "")] = int(_imreg[_r])
N["imitationOverUDSPct"] = int(round(100 * (_imreg["UDS"] / _unreg["UDS"] - 1)))

# --- the four positives that are blue in F277W-F444W, and who recovers them (D M6) --------
# Section 3.4 says no rule with a F277W-F444W threshold can reach them; the learned selection
# does not reach them either, and Table D2 shows all four, so the count is stated.
_negcol = _cabpos < 0
N["nPosColourNegativeNeitherAB"] = int(
    (_negcol & ~_P.sel_burden.to_numpy(bool) & ~_P.picked.to_numpy(bool)).sum()
)
assert N["nPosColourNegativeNeitherAB"] == N["nPosColourNegativeAB"], (
    "a F277W-F444W-blue positive is recovered by one of the two selections"
)

# --- how much a working kocevski24 could have recovered (A2-13) --------------------------
# The module fails closed below z = 4.75 because the redshift-keyed band sets need HST
# F606W and F814W there. Split the rule-missed positives on the catalogue photometric
# redshift at that boundary: below it the band requirement bites, at or above it the rule
# was evaluated on its own terms and still did not select.
_KOCZ = 4.75
N["nRuleMissedZphotBelowHst"] = int((_missAll.z_phot < _KOCZ).sum())
N["nRuleMissedZphotAboveHst"] = int((_missAll.z_phot >= _KOCZ).sum())
assert (
    N["nRuleMissedZphotBelowHst"] + N["nRuleMissedZphotAboveHst"] == N["nRuleMissed"]
), "a rule-missed positive has no catalogue photometric redshift"

# =========================================================================================
# ROUND 3, ANALYSIS WORKER (director's DIRECTOR_round3.md, rulings T1-T4, T6, T9)
# Every number below is read from recovery/published_catalogues_extra.parquet (built by
# recovery/v4_round3_analysis.py, a new record that changes nothing existing) and from
# recovery/baseline_oof_full.parquet. Existing macros are untouched.
# =========================================================================================
_x = pd.read_parquet(os.path.join(V, "published_catalogues_extra.parquet")).set_index(
    "source_id"
)
_R = posAll.set_index("source_id").join(_x, how="left")
assert len(_R) == N["nPos"], "the round-3 join changed the positive count"
_z = _R.z_spec
assert _z.notna().all(), "a positive has no spectroscopic redshift"
_Rmiss = ~_R.picked.astype(bool)
_Rfloor = _Rmiss & ~(_R.in_kocevski24.astype(bool) | _R.in_perger25.astype(bool))
assert int(_Rfloor.sum()) == N["nNoSelection"], "the round-3 floor set moved"
assert int(_Rmiss.sum()) == N["nRuleMissed"], "the round-3 rule-missed set moved"

# --- T1. the spectroscopic redshifts of the positives ------------------------------------
# Source per object: the redshift column of the published list the object is in, else the
# DAWN JWST Archive v4.4 secure (grade >= 3) redshift within skyGroupArcsec, else the
# catalogue photometric redshift. In the run of record no positive falls through to the
# photometric redshift, so the distribution below is spectroscopic throughout.
_src = _R.z_spec_source.value_counts()
N["posZspecSourceList"] = int(_src.get("list", 0))
N["posZspecSourceArchive"] = int(_src.get("archive", 0))
N["posZspecSourcePhotoz"] = int(_src.get("zphot", 0))
assert (
    N["posZspecSourceList"] + N["posZspecSourceArchive"] + N["posZspecSourcePhotoz"]
    == N["nPos"]
), "the redshift-source counts do not add to the positives"
assert N["posZspecSourceList"] == N["nPosInPublishedList"], (
    "a positive in a published list has no list redshift"
)
# the two independent redshifts agree wherever both exist, which is the check that licenses
# mixing them
_bz = _R.z_list.notna() & _R.z_archive.notna()
N["posZspecCrossCheckN"] = int(_bz.sum())
N["posZspecCrossCheckMaxDiff"] = round(
    float((_R.z_list - _R.z_archive)[_bz].abs().max()), 3
)
N["posZspecMedian"] = round(float(_z.median()), 2)
N["posZspecMin"] = round(float(_z.min()), 2)
N["posZspecMax"] = round(float(_z.max()), 2)
N["posZspecBelowThree"] = int((_z < 3).sum())
N["posZspecBelowFour"] = int((_z < 4).sum())
for _lo, _hi, _tag in (
    (-np.inf, 3.0, "BelowThree"),
    (3.0, 4.0, "ThreeFour"),
    (4.0, 5.0, "FourFive"),
    (5.0, 7.0, "FiveSeven"),
    (7.0, 9.0, "SevenNine"),
    (9.0, np.inf, "AboveNine"),
):
    N["posZspecBin" + _tag] = int(((_z >= _lo) & (_z < _hi)).sum())
assert (
    sum(
        N["posZspecBin" + t]
        for t in (
            "BelowThree",
            "ThreeFour",
            "FourFive",
            "FiveSeven",
            "SevenNine",
            "AboveNine",
        )
    )
    == N["nPos"]
), "the redshift bins do not partition the positives"


def _zcut(mask, tag):
    """Union recovery, rule-missed and floor on a redshift-restricted positive set."""
    n = int(mask.sum())
    N["nPos" + tag] = n
    N["unionRecall" + tag] = int((mask & _R.picked.astype(bool)).sum())
    N["nRuleMissed" + tag] = int((mask & _Rmiss).sum())
    N["floor" + tag] = int((mask & _Rfloor).sum())
    N["floor" + tag + "Pct"] = round(100 * N["floor" + tag] / n, 1)
    N["nRuleMissed" + tag + "Pct"] = round(100 * N["nRuleMissed" + tag] / n, 1)


# z >= 4: every rule that states a redshift range is inside its own domain there
_zcut(_z >= 4.0, "ZgeFour")
# the union of the ranges the seven papers state for themselves: Labbe et al. 3 < z < 7,
# Kokorev et al. 4 < z < 9, Kocevski et al. z > 4, Barro et al. z = 5-9, Greene et al.
# z > 5, Akins et al. z ~ 5-9, and Perez-Gonzalez et al. no stated range. The union of the
# six stated intervals is z > 3, two of them being unbounded above.
_zcut(_z > 3.0, "ZinRange")

# --- T2. the bound on the HST-band mechanism ---------------------------------------------
# kocevski24 keys its band set on the redshift it is given, which in this pipeline is the
# catalogue photometric redshift, so nMissedInKocevskiBelowZphot is the count that describes
# what the code did; nMissedInKocevskiBelowZ is the same split on the spectroscopic redshift,
# which is where the objects actually are. Both are upper bounds on how many of the eleven
# catalogue-listed misses that mechanism can explain.
_kcm = _R[_Rmiss & _R.in_kocevski24.astype(bool)]
assert len(_kcm) == N["nMissedInKocevskiCat"], "the Kocevski-catalogue miss set moved"
N["nMissedInKocevskiBelowZ"] = int((_kcm.z_spec < _KOCZ).sum())
N["nMissedInKocevskiBelowZphot"] = int((_kcm.z_phot < _KOCZ).sum())

# --- T4. how the headline moves under a stricter definition of a confirmed LRD -----------
_L3 = ["in_hviding25_A1", "in_barro25", "in_degraaff26"]
_tp = _R[_L3].astype(bool).sum(axis=1) >= 2
N["nPosTwoPlus"] = int(_tp.sum())
assert N["nPosTwoPlus"] == N["nPosInTwoOrMore"], (
    "nPosTwoPlus is the round-3 name for nPosInTwoOrMore and must equal it"
)
N["unionRecallTwoPlus"] = int((_tp & _R.picked.astype(bool)).sum())
N["unionMissTwoPlus"] = int((_tp & _Rmiss).sum())
N["unionMissTwoPlusPct"] = round(100 * N["unionMissTwoPlus"] / N["nPosTwoPlus"], 1)
N["floorTwoPlus"] = int((_tp & _Rfloor).sum())
N["floorTwoPlusPct"] = round(100 * N["floorTwoPlus"] / N["nPosTwoPlus"], 1)
# which list the floor sits on, and how much of it the de Graaff et al. V-shape decision
# declined rather than never saw
_fl = _R[_Rfloor]
N["floorInBarro"] = int(_fl.in_barro25.astype(bool).sum())
N["floorBarroOnly"] = int(
    (
        _fl.in_barro25.astype(bool)
        & ~_fl.in_hviding25_A1.astype(bool)
        & ~_fl.in_degraaff26.astype(bool)
    ).sum()
)
_flt = _fl[_fl.spec_tested.astype(bool)]
assert len(_flt) == N["floorTested"], "the tested-floor denominator moved"
N["floorNotInDeGraaffVtest"] = int((~_flt.in_degraaff26.astype(bool)).sum())

# --- T3. the published-catalogue census ---------------------------------------------------
# Perger et al. 2025 Sect. 2 names the seventeen catalogues its 919 objects were collected
# from, and its machine-readable table carries, per object, the ADS bibcode of the paper it
# was first published in.
# Kokorev et al. 2024 and Akins et al. 2024 are both among them, so the compilation already
# carries the two catalogues Reviewer A asked for. pergerSources is the list, in the order
# the paper prints it.
N["pergerNSources"] = 17
N["pergerSources"] = (
    "Ubler et al. 2023; Kokorev et al. 2023; Harikane et al. 2023; "
    "Labbe et al. 2023a; Maiolino et al. 2023; Killi et al. 2023; Labbe et al. 2023b; "
    "Barro et al. 2024; Matthee et al. 2024; Greene et al. 2024; "
    "Perez-Gonzalez et al. 2024; Williams et al. 2024; Kokorev et al. 2024; "
    "Wang et al. 2024; Kocevski et al. 2024; Akins et al. 2024; Furtak et al. 2024"
)
_pcount = J("round3_analysis_counts.json")["perger_by_paper"]
# rows of the 919 the compilation attributes to each catalogue, and how many of those fall
# inside the nine fields. The compilation lists each object once under one originating paper,
# so these are lower bounds on what each catalogue contributes, which is why the floor is
# reported as an upper bound on the count in no published selection.
N["pergerRowsKokorev"] = int(_pcount["kokorev2024"]["rows"])
N["pergerRowsAkins"] = int(_pcount["akins2024"]["rows"])
N["pergerRowsKocevski"] = int(_pcount["kocevski2024"]["rows"])
N["kokorevCatRows"] = int(_pcount["kokorev2024"]["in_our_fields"])
N["akinsCatRows"] = int(_pcount["akins2024"]["in_our_fields"])
assert N["kokorevCatRows"] == int(_x.perger_kokorev2024.sum()), (
    "Kokorev field count moved"
)
assert N["akinsCatRows"] == int(_x.perger_akins2024.sum()), "Akins field count moved"
# the sample sizes those two catalogues publish for themselves, quoted from their abstracts
# (Kokorev et al. 2024, "260 reddened AGN candidates at 4 < z_phot < 9"; Akins et al. 2024,
# "a sample of 434 little red dots ... selected from the 0.54 deg^2 COSMOS-Web survey")
N["kokorevPublished"] = 260
N["akinsPublished"] = 434
_kok = _R.perger_kokorev2024.fillna(False).astype(bool) | _R.eprint_kokorev2024.fillna(
    False
).astype(bool)
_ak = _R.perger_akins2024.fillna(False).astype(bool) | _R.eprint_akins2024.fillna(
    False
).astype(bool)
N["nPosInKokorevCat"] = int(_kok.sum())
N["nPosInAkinsCat"] = int(_ak.sum())
N["nMissedInKokorevCat"] = int((_Rmiss & _kok).sum())
N["nMissedInAkinsCat"] = int((_Rmiss & _ak).sum())
# the floor recomputed against all four published catalogues
_floorAll = _Rmiss & ~(
    _R.in_kocevski24.astype(bool) | _R.in_perger25.astype(bool) | _kok | _ak
)
N["floorAllCats"] = int(_floorAll.sum())
N["floorAllCatsPct"] = round(100 * N["floorAllCats"] / N["nPos"], 1)
N["floorMovedByExtraCats"] = N["nNoSelection"] - N["floorAllCats"]

# --- T9. the left-out models on their own list, at the union's burden ---------------------
# The Table tab:loco column is each left-out model at its own threshold, which realises fewer
# rows than the union keeps. selmatched_* in baseline_oof_full.parquet is the same run cut
# back to the union's row count region by region, the comparison Section 5.4 mandates.
_bof = pd.read_parquet(os.path.join(V, "baseline_oof_full.parquet")).set_index(
    "source_id"
)
for _c, _tag in (
    ("hviding25_A1", "HvidingA"),
    ("barro25", "Barro"),
    ("degraaff26", "DeGraaff"),
):
    _on = posS[posS["in_" + _c].astype(bool)].source_id.values
    assert len(_on) == N["catalogN" + _tag], f"the {_c} in-support denominator moved"
    N["loco" + _tag + "MatchedOnList"] = int(
        _bof.loc[_on, "selmatched_leave_out_" + _c].astype(bool).sum()
    )
    assert N["loco" + _tag + "LeftOutN"] == N["catalogN" + _tag], (
        f"the {_c} left-out denominator disagrees with the catalogue denominator"
    )

# --- T6. the matched-burden gain, with and without the two size features ------------------
N["matchedGainPrimary"] = N["blReplicaMatchedRecall"] - N["unionRecallSupport"]
N["matchedGainNoSize"] = N["blAblateSizeMatchedRecall"] - N["unionRecallSupport"]
N["matchedGainSizeShare"] = round(
    100 * (N["matchedGainPrimary"] - N["matchedGainNoSize"]) / N["matchedGainPrimary"],
    0,
)

# =========================================================================================
# ROUND 3, TEXT WORKER (director's DIRECTOR_round3.md, rulings B3-2, B3-3, B3-5, B3-12,
# E3-3 and A3-12). Every macro below is read from a table of record or from a record file
# already read above; nothing existing is changed.
# =========================================================================================

# --- B3-2. the fourth challenger, the blend of the 34-feature trees and the perceptron ----
# recovery/model_selection.json holds both blends; only blend_trees30_mlp reached Table 4,
# and the omitted one is the strongest model in the record on every column the table prints.
# regionsWon is recomputed here from the per-region recalls rather than read, because
# model_selection.json carries no blend_trees34 promotion record against trees_30.
_b34 = ms["blend_trees34_mlp"]
N["selBlendTwoSelected"] = _b34["selected_total"]
N["selBlendTwoRecall"] = _b34["recall_147"]
N["selBlendTwoMissed"] = _b34["rule_missed_41"]
N["selBlendTwoAnchors"] = _b34["neg_selected"]
_p30 = ms["trees_30_features"]["per_region_recall"]
N["selBlendTwoRegionsWon"] = int(
    sum(
        int(_b34["per_region_recall"][_r].split("/")[0]) > int(_p30[_r].split("/")[0])
        for _r in _p30
    )
)

# --- B3-2/B3-7. the realised-burden spread must now run over four challengers -------------
# The existing macro was computed over three. Recomputed here over the primary and all four
# challengers, which is the set Table 4 prints after the fourth row is added.
_cb = [
    N["modelSelected"],
    N["selTreesThirtyFourSelected"],
    N["selMlpSelected"],
    N["selBlendSelected"],
    N["selBlendTwoSelected"],
]
N["challengerBurdenMin"] = int(min(_cb))
N["challengerBurdenMax"] = int(max(_cb))
N["challengerBurdenSpreadPct"] = round(100 * (max(_cb) / min(_cb) - 1), 1)

# --- E3-3. the V-shape reading on the released spec_tested flag alone ---------------------
# The paper's counts nested only under "tested OR established"; a referee recomputing from
# the released tables uses spec_tested by itself and gets a different pair. Both are emitted:
# the strict flag reading is the one the text now prints, and posVshapedNotTested (1) is the
# single object that separates the two.
N["posHitTestedVshaped"] = int((_V & _T & _hit).sum())
N["posMissTestedVshaped"] = int((_V & _T & ~_hit).sum())
N["fisherVshapeTestedP"] = pfmt(
    float(
        fisher_exact(
            [
                [
                    N["posHitTestedVshaped"],
                    N["posHitTested"] - N["posHitTestedVshaped"],
                ],
                [
                    N["posMissTestedVshaped"],
                    N["posMissTested"] - N["posMissTestedVshaped"],
                ],
            ],
            alternative="two-sided",
        )[1]
    )
)
assert (
    N["posHitTestedVshaped"] + N["posMissTestedVshaped"] + N["posVshapedNotTested"]
    == N["posVshaped"]
), "the tested V-shape split does not close against posVshaped"

# --- B3-5. the matched-burden gaps the abstract and the Summary must print ----------------
# The apportionment "most of the gain" is a ratio of a point estimate to an unresolved
# increment. These are its two terms, so the text can print the counts instead.
N["matchedGapPrimaryUnion"] = N["primaryMatchedRecall"] - N["unionRecallSupport"]
N["matchedGapImitationUnion"] = N["imitationMatchedRecall"] - N["unionRecallSupport"]

# --- B3-12. the base rate of the case-control PR-AUC --------------------------------------
# modelPrAuc is computed between the in-support positives and the in-support anchors, so its
# base rate is nPosSupport of nPosSupport + nNegSupport.
N["prAucBaseRatePct"] = round(
    100 * N["nPosSupport"] / (N["nPosSupport"] + N["nNegSupport"]), 1
)

# --- D3 minor 4. one number style for the promotion rule's regional clause ----------------
# selRuleRegions is 4; the text prints it as a word beside nRegionsWord so that a single
# comparison does not mix a digit and a word.
N["selRuleRegionsWord"] = WORD[N["selRuleRegions"]].lower()

# --- A3-12. how the rule-missed recoveries split between floor and catalogue --------------
# modelNoSelectionRecovered (15) of the modelRuleMissed (20) are in no published photometric
# selection at all; the rest are listed in a published catalogue and missed by the code only.
N["modelRuleMissedInCat"] = N["modelRuleMissed"] - N["modelNoSelectionRecovered"]

# --- F11 (round 4, Reviewer B finding 5). The union's burden inside the training regions ---
# The per-fold threshold is set to the burden fraction the union realises inside that fold's
# four training regions, which is not the pooled unionBurdenPct of Section 3.2. These two
# macros give the range over the five folds, so the text and the architecture figure can stop
# quoting the pooled figure for a per-fold quantity.
_ubt = [tr["folds"][r]["union_burden_train"] for r in REG]
N["unionBurdenTrainMin"] = round(100 * min(_ubt), 3)
N["unionBurdenTrainMax"] = round(100 * max(_ubt), 3)
assert N["unionBurdenTrainMin"] == 0.194, N["unionBurdenTrainMin"]
assert N["unionBurdenTrainMax"] == 0.257, N["unionBurdenTrainMax"]
# Cross-check against the per-region macros of record: leaving region R out, the union keeps
# (unionSelected - unionSel_R) of (supportRows - rows_R) rows. Same min and max to three
# decimals, computed from the denominators file rather than from train_results.json.
_ubt_rec = [
    100
    * (N["unionSelected"] - N["unionSel" + r.replace("-", "")])
    / (N["supportRows"] - N["rows" + r.replace("-", "")])
    for r in REG
]
assert round(min(_ubt_rec), 3) == N["unionBurdenTrainMin"], (
    round(min(_ubt_rec), 3),
    N["unionBurdenTrainMin"],
)
assert round(max(_ubt_rec), 3) == N["unionBurdenTrainMax"], (
    round(max(_ubt_rec), 3),
    N["unionBurdenTrainMax"],
)

# --- F24 (round 4, Reviewer A finding 1). The size axis of the incompleteness --------------
# Four of the seven rules impose a compactness criterion, so what the rules lose they lose on
# size as well as on colour. Compact means r_h < 1.5 r_star, the same criterion the text uses
# for the positives in Section 2.2, on the in-support positives only. Every value is asserted
# against the director's recount.
_cmp_pos = posS.r_h_arcsec < 1.5 * posS.r_star_v1_arcsec
_cmp_miss = missS.r_h_arcsec < 1.5 * missS.r_star_v1_arcsec
_cmp_hit = hitS.r_h_arcsec < 1.5 * hitS.r_star_v1_arcsec
N["nPosCompactSupport"] = int(_cmp_pos.sum())
N["nPosNonCompactSupport"] = int((~_cmp_pos).sum())
N["missCompact"] = int(_cmp_miss.sum())
N["hitCompact"] = int(_cmp_hit.sum())
N["missCompactPct"] = round(100 * N["missCompact"] / N["nRuleMissedSupport"], 1)
N["hitCompactPct"] = round(100 * N["hitCompact"] / N["unionRecallSupport"], 1)
assert N["nPosCompactSupport"] == 130, N["nPosCompactSupport"]
assert N["nPosNonCompactSupport"] == 17, N["nPosNonCompactSupport"]
assert N["missCompact"] == 30, N["missCompact"]
assert N["hitCompact"] == 100, N["hitCompact"]
assert N["missCompactPct"] == 73.2, N["missCompactPct"]
assert N["hitCompactPct"] == 94.3, N["hitCompactPct"]
N["fisherCompactP"] = pfmt(
    float(
        fisher_exact(
            [
                [N["missCompact"], N["nRuleMissedSupport"] - N["missCompact"]],
                [N["hitCompact"], N["unionRecallSupport"] - N["hitCompact"]],
            ],
            alternative="two-sided",
        )[1]
    )
)
assert N["fisherCompactP"] == "0.00084", N["fisherCompactP"]
_ratio_miss = (missS.r_h_arcsec / missS.r_star_v1_arcsec).to_numpy(float)
_ratio_hit = (hitS.r_h_arcsec / hitS.r_star_v1_arcsec).to_numpy(float)
N["missMedianRatio"] = round(float(np.median(_ratio_miss)), 2)
N["hitMedianRatio"] = round(float(np.median(_ratio_hit)), 2)
assert N["missMedianRatio"] == 1.28, N["missMedianRatio"]
assert N["hitMedianRatio"] == 1.15, N["hitMedianRatio"]
# Recovery split on compactness, model against union, on the same in-support positives.
_Pc = P[P.r_h_arcsec < 1.5 * P.r_star_v1_arcsec]
_Pn = P[~(P.r_h_arcsec < 1.5 * P.r_star_v1_arcsec)]
N["modelRecallCompact"] = int(_Pc.sel_burden.sum())
N["modelRecallNonCompact"] = int(_Pn.sel_burden.sum())
N["unionRecallCompact"] = int(_Pc.picked.astype(bool).sum())
N["unionRecallNonCompact"] = int(_Pn.picked.astype(bool).sum())
assert N["modelRecallCompact"] == 119, N["modelRecallCompact"]
assert N["modelRecallNonCompact"] == 4, N["modelRecallNonCompact"]
assert N["unionRecallCompact"] == 100, N["unionRecallCompact"]
assert N["unionRecallNonCompact"] == 6, N["unionRecallNonCompact"]
# The colour result does not depend on the size one: the same colour comparison restricted to
# the compact positives, formatted exactly as colourMWUp is.
_mc = missS[_cmp_miss]
_hc = hitS[_cmp_hit]
N["missMedianColourCompact"] = round(float(_mc.c_f277w_f444w.median()), 2)
N["hitMedianColourCompact"] = round(float(_hc.c_f277w_f444w.median()), 2)
N["colourMWUpCompact"] = pfmt(
    float(
        mannwhitneyu(
            _mc.c_f277w_f444w.to_numpy(float),
            _hc.c_f277w_f444w.to_numpy(float),
            alternative="two-sided",
        ).pvalue
    )
)
assert N["missMedianColourCompact"] == 0.57, N["missMedianColourCompact"]
assert N["hitMedianColourCompact"] == 1.19, N["hitMedianColourCompact"]
assert N["colourMWUpCompact"] == "1.3\\times 10^{-6}", N["colourMWUpCompact"]

# --- F35 (round 4, Reviewer E finding 1). Where kocevski24 fails closed -------------------
# The module's slope criteria need HST F606W and F814W below z = 4.75, which the DJA
# catalogues carry only where ACS overlaps NIRCam. Split the published Kocevski et al.
# catalogue rows in support by the photometric redshift the module keys on, over the range
# its redshift table covers, so the two selection rates reconcile with the 306 and the 110
# already printed.
_kc = sup[sup.in_kocevski24.astype(bool)]
_kb = _kc[(_kc.z_phot > 2) & (_kc.z_phot < 4.75)]
_ka = _kc[_kc.z_phot >= 4.75]
N["kocevskiCatBelowN"] = int(len(_kb))
N["kocevskiCatBelowSel"] = int(_kb.sel_kocevski24.astype(bool).sum())
N["kocevskiCatAboveN"] = int(len(_ka))
N["kocevskiCatAboveSel"] = int(_ka.sel_kocevski24.astype(bool).sum())
assert N["kocevskiCatBelowN"] == 56, N["kocevskiCatBelowN"]
assert N["kocevskiCatBelowSel"] == 11, N["kocevskiCatBelowSel"]
assert N["kocevskiCatAboveN"] == 207, N["kocevskiCatAboveN"]
assert N["kocevskiCatAboveSel"] == 98, N["kocevskiCatAboveSel"]

# ---------------------------------------------------------------- write
json.dump(N, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=float)


def latex_num(v):
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}".replace(",", "{,}")
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer():
            return f"{int(v):,}".replace(",", "{,}")
        return f"{v:g}"
    return str(v).replace("_", "\\_")


lines = [
    "% generated by paper/build_numbers.py from the tables of record; do not edit"
]
for k, v in N.items():
    if v is None:
        continue
    name = k
    for dch, word in zip("0123456789", WORD):
        name = name.replace(dch, word)
    name = re.sub(r"[^A-Za-z]", "", name)
    lines.append(f"\\newcommand{{\\{name}}}{{{latex_num(v)}}}")
open(os.path.join(OUT, "numbers.tex"), "w", encoding="utf-8").write(
    "\n".join(lines) + "\n"
)
print(len(N), "numbers written")
for k in (
    "nPos",
    "unionRecallEnd",
    "modelRecallEnd",
    "modelRuleMissed",
    "modelAnchors",
    "unionAnchors",
    "pairedUnionOnlyEnd",
    "pairedNeitherEnd",
    "blImitationRecall",
    "blueUnion",
    "blueModel",
    "candTotal",
    "followupN",
    "modelPrAuc",
    "modelWilsonLo",
    "modelWilsonHi",
    "modelRegionsWon",
    "impTopTwoPct",
    "nPosOnlySpectralTest",
    "lofoMinutes",
):
    print(k, N.get(k))
