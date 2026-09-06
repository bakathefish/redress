"""Round 9 (2026-09-06 evening): every new number of the astra-r5 revision, from tables on disk.

No retraining. Reads recovery (labels, features, oof_scores, baseline_oof_full,
baseline_scores, published_catalogues_extra, anchor_test_status, evaluation, candidates), the
public clone's recovery/ tables (barro24b flags, positive_apertures.csv) and the module package
selections/ of the public clone. Writes:

  recovery/round9_numbers.json           every number, by macro name
  paper/numbers.tex                         one delimited block appended (idempotent)
  <clone>/recovery/label_provenance.csv         one row per labelled or ambiguous source
  <clone>/recovery/test_status_reconciliation.csv  every row where the two test exports differ
  <clone>/recovery/positive_match_alternatives.csv all catalog rows of multi-row positive groups
  <clone>/recovery/rule_audit_positives.csv     per rule and positive: outcome and first failing criterion
  <clone>/recovery/candidates.csv               five new columns, every existing column unchanged
  recovery/regional_budget_curve.json     the curve Figure 9 draws
  recovery/still_missed_region_rank.json  within-region percentile ranks for Table 11
  recovery/data_table.json                per-field depths and constants for the data table

Run from the build repository root:  python paper/round9_compute_2026_09_06.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from math import comb

import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", FutureWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "recovery")
R9 = os.path.join(ROOT, "recovery")
CLONE = ROOT  # the release tables live beside the script
REC = os.path.join(CLONE, "recovery")
NUMBERS_TEX = os.path.join(ROOT, "paper", "numbers.tex")
NUMBERS_TEX_REC = NUMBERS_TEX  # one numbers.tex in the clone
os.makedirs(R9, exist_ok=True)
sys.path.insert(0, os.path.join(CLONE, "selections"))

SEL = [
    "sel_labbe23",
    "sel_kokorev24",
    "sel_kocevski24",
    "sel_perezgonzalez24",
    "sel_barro23",
    "sel_greene24",
    "sel_akins24",
]
RULES = [s[4:] for s in SEL]
RTAG = {
    "labbe23": "Labbe",
    "kokorev24": "Kokorev",
    "kocevski24": "Kocevski",
    "perezgonzalez24": "PerezGonzalez",
    "barro23": "Barro",
    "greene24": "Greene",
    "akins24": "Akins",
}
REGIONS = ["CEERS", "GOODS-N", "GOODS-S", "COSMOS", "UDS"]
RK = {r: r.replace("-", "") for r in REGIONS}
BANDS = ["f090w", "f115w", "f150w", "f200w", "f277w", "f356w", "f444w"]
N: dict = {}
rng = np.random.default_rng(20260906)


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0**n)


def pfmt(p):
    if p >= 0.01:
        return "%.2f" % p
    if p >= 0.001:
        return "%.3f" % p
    if p >= 0.0001:
        return "%.4f" % p
    return "%.1e" % p


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# ------------------------------------------------------------------ tables of record
lab = pd.read_parquet(os.path.join(V, "labels.parquet"))
fea = pd.read_parquet(
    os.path.join(V, "features.parquet"),
    columns=[
        "source_id",
        "region",
        "sky_group",
        "sky_group_size",
        "in_support",
        "r_h_arcsec",
        "r_star_v1_arcsec",
        "c_f277w_f444w",
    ],
)
oof = pd.read_parquet(os.path.join(V, "oof_scores.parquet"))
bof = pd.read_parquet(os.path.join(V, "baseline_oof_full.parquet"))
bsc = pd.read_parquet(os.path.join(V, "baseline_scores.parquet"))
ext = pd.read_parquet(
    os.path.join(V, "published_catalogues_extra.parquet"),
    columns=[
        "source_id",
        "z_spec",
        "z_spec_source",
        "z_list",
        "z_archive",
        "archive_grade",
    ],
)
tst = pd.read_csv(os.path.join(V, "anchor_test_status.csv"))
ev = json.load(open(os.path.join(V, "evaluation.json")))
den = json.load(open(os.path.join(V, "denominators.json")))
NU = den["union_in_support_per_region"]
b24 = pd.read_parquet(os.path.join(REC, "barro24b_flags.parquet"))
pap = pd.read_csv(os.path.join(REC, "positive_apertures.csv"))
v34 = pd.read_parquet(
    os.path.join(V, "oof_scores_v34.parquet"),
    columns=["source_id", "score_mean", "sel_burden"],
)
mlp = pd.read_parquet(
    os.path.join(V, "mlp_oof_scores.parquet"),
    columns=["source_id", "score_mean", "sel_burden"],
)

# the support frame: one row per in-support catalog row, with everything joined
d = oof.merge(fea.drop(columns=["region"]), on="source_id", how="left")
d = d.merge(
    lab[
        ["source_id"]
        + SEL
        + [
            "ambiguous",
            "in_hviding25_A1",
            "in_degraaff26",
            "in_barro25",
            "in_perger25",
            "in_kocevski24",
            "vshaped_spec",
            "spec_tested",
            "spec_vshaped",
            "z_phot",
            "mag_f444w",
            "f_f277w",
            "f_f444w",
        ]
    ],
    on="source_id",
    how="left",
)
d = d.merge(
    bof[
        [
            "source_id",
            "score_rules_reproduction",
            "selmatched_rules_reproduction",
            "selmatched_primary_8bag_replica",
            "selmatched_pn_lightgbm",
            "score_primary_8bag_replica",
            "score_pn_lightgbm",
        ]
    ],
    on="source_id",
    how="left",
)
d = d.merge(b24, on="source_id", how="left")
d = d.merge(
    v34.rename(columns={"score_mean": "score_v34", "sel_burden": "sel_v34"}),
    on="source_id",
    how="left",
)
d = d.merge(
    mlp.rename(columns={"score_mean": "score_mlp", "sel_burden": "sel_mlp"}),
    on="source_id",
    how="left",
)
assert d.in_support.all() and len(d) == den["support_rows"]
d["picked"] = d.picked.astype(bool)
d["ispos"] = d.y == 1
d["isneg"] = d.y == 0
d["u8"] = d.picked | d.sel_barro24b.fillna(False).astype(bool)


def matched(score_col, K=None):
    """Top-K rows of each region by a score, K the union's regional row count by default."""
    K = K or NU
    sel = np.zeros(len(d), bool)
    for R, g in d.groupby("region"):
        idx = g.sort_values(score_col, ascending=False, kind="stable").index[: K[R]]
        sel[idx] = True
    return sel


# ------------------------------------------------------------------ 1. region-matched primary
d["selm"] = matched("score_mean")
pos = d[d.ispos]
m, u = pos.selm.to_numpy(bool), pos.picked.to_numpy(bool)
N["matchedRecallSupport"] = int(m.sum())
N["matchedRecallEnd"] = int(
    m.sum()
)  # the four positives outside the support are misses
N["matchedRecallEndPct"] = round(100 * N["matchedRecallEnd"] / 151, 1)
N["matchedRecallSupportPct"] = round(100 * N["matchedRecallSupport"] / 147, 1)
lo, hi = wilson(N["matchedRecallSupport"], 147)
N["matchedWilsonLo"], N["matchedWilsonHi"] = round(lo, 1), round(hi, 1)
N["matchedPairedBoth"] = int((m & u).sum())
N["matchedPairedModelOnly"] = int((m & ~u).sum())
N["matchedPairedUnionOnly"] = int((u & ~m).sum())
N["matchedPairedNeither"] = int((~m & ~u).sum())
N["matchedPairedUnionOnlyEnd"] = (
    N["matchedPairedUnionOnly"] + 1
)  # the one positive outside the support the rules select
N["matchedPairedNeitherEnd"] = (
    151
    - N["matchedPairedBoth"]
    - N["matchedPairedModelOnly"]
    - N["matchedPairedUnionOnlyEnd"]
)
N["mcnemarMatchedB"] = N["matchedPairedModelOnly"]
N["mcnemarMatchedC"] = N["matchedPairedUnionOnly"]
N["mcnemarMatchedP"] = pfmt(mcnemar_p(N["mcnemarMatchedB"], N["mcnemarMatchedC"]))
N["matchedAnchors"] = int((d.selm & d.isneg).sum())
N["matchedAnchorPct"] = round(100 * N["matchedAnchors"] / 4979, 2)
N["matchedAnchorExtra"] = N["matchedAnchors"] - int((d.picked & d.isneg).sum())
N["matchedRuleMissed"] = int((m & ~u).sum())
N["matchedRuleMissedPct"] = round(100 * N["matchedRuleMissed"] / 44, 0)
N["matchedEffectSupportPts"] = round(100 * (N["matchedRecallSupport"] - 106) / 147, 1)
N["matchedEffectEndPts"] = round(100 * (N["matchedRecallEnd"] - 107) / 151, 1)
N["matchedGapPrimaryUnion"] = N["matchedRecallSupport"] - 106
for R in REGIONS:
    g = pos[pos.region == R]
    N["matchedRegion" + RK[R]] = int(g.selm.sum())
    N["matchedRegionRuleMissed" + RK[R]] = int((g.selm & ~g.picked).sum())
# the spectroscopic-criterion subset at matched burden
sc = pos[pos.in_hviding25_A1.astype(bool) | pos.in_degraaff26.astype(bool)]
N["matchedRecallSpecCritSupport"] = int(sc.selm.sum())
N["matchedRecallSpecCrit"] = int(
    sc.selm.sum()
)  # every spectroscopic-criterion positive outside the support is a miss
N["matchedPairedModelOnlySpecCrit"] = int((sc.selm & ~sc.picked).sum())
N["matchedPairedUnionOnlySpecCrit"] = int((sc.picked & ~sc.selm).sum())
N["matchedRecallSpecCritPct"] = round(100 * N["matchedRecallSpecCrit"] / 92, 1)
# the deployed points, restated for the table
N["deployedAnchorExtra"] = int((d.sel_burden & d.isneg).sum()) - int(
    (d.picked & d.isneg).sum()
)

# ------------------------------------------------------------------ 2. the no-PU pairing at matched burden
rep = pos.selmatched_primary_8bag_replica.to_numpy(bool)
pn = pos.selmatched_pn_lightgbm.to_numpy(bool)
N["blPNMatchedRecallCheck"] = int(pn.sum())
N["mcnemarReplicaPNMatchedB"] = int((rep & ~pn).sum())
N["mcnemarReplicaPNMatchedC"] = int((pn & ~rep).sum())
N["mcnemarReplicaPNMatchedP"] = pfmt(
    mcnemar_p(N["mcnemarReplicaPNMatchedB"], N["mcnemarReplicaPNMatchedC"])
)

# ------------------------------------------------------------------ 3. the test family
TESTS = [
    "mcnemar, learned against union, region-matched (primary)",
    "mcnemar, learned against union, deployed thresholds",
    "mcnemar, imitation against union, matched",
    "mcnemar, primary against imitation, matched",
    "mcnemar, primary against imitation on the rule-missed, matched",
    "mcnemar, replica against positives-against-comparison, matched",
    "mcnemar, replica against no-size ablation, matched",
    "mcnemar, learned against union on the compact subset, matched",
    "mcnemar, learned against union of eight, matched",
    "fisher, compactness of missed against kept",
    "fisher, V-shape pass of missed against kept",
    "sign test over five regions",
    "mann-whitney, colour of missed against kept",
    "mann-whitney, magnitude of missed against kept",
    "mann-whitney, colour on the compact positives",
]
N["testFamily"] = len(TESTS)
N["testFamilyWord"] = {14: "fourteen", 15: "fifteen", 16: "sixteen"}[len(TESTS)]
N["bonferroniMatchedP"] = pfmt(
    min(1.0, len(TESTS) * mcnemar_p(N["mcnemarMatchedB"], N["mcnemarMatchedC"]))
)
N["bonferroniDeployedP"] = pfmt(min(1.0, len(TESTS) * mcnemar_p(20, 3)))

# ------------------------------------------------------------------ 4. effect sizes and redshift strata
pz = pos.merge(ext, on="source_id", how="left")
ok = (pz.f_f277w > 0) & (pz.f_f444w > 0)
pz["cab"] = np.where(ok, -2.5 * np.log10(pz.f_f277w / pz.f_f444w), np.nan)
pz["compact"] = pz.r_h_arcsec < 1.5 * pz.r_star_v1_arcsec
pz["ratio"] = pz.r_h_arcsec / pz.r_star_v1_arcsec
miss, hit = pz[~pz.picked], pz[pz.picked]


def boot(a, b, stat, n=20000):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    v = np.array(
        [stat(rng.choice(a, len(a))) - stat(rng.choice(b, len(b))) for _ in range(n)]
    )
    return stat(a) - stat(b), np.percentile(v, 2.5), np.percentile(v, 97.5)


for key, col, stat in (
    ("Colour", "cab", np.nanmedian),
    ("Mag", "mag_f444w", np.median),
    ("Ratio", "ratio", np.median),
):
    e, l, h = boot(miss[col], hit[col], stat)
    N["effect" + key + "Median"] = round(float(e), 2)
    N["effect" + key + "Lo"] = round(float(l), 2)
    N["effect" + key + "Hi"] = round(float(h), 2)
e, l, h = boot(miss.compact, hit.compact, np.mean)
N["effectCompactPts"] = round(100 * float(e), 0)
N["effectCompactLo"] = round(100 * float(l), 0)
N["effectCompactHi"] = round(100 * float(h), 0)
for tag, lo_, hi_ in (("Low", 0, 4), ("Mid", 4, 6), ("High", 6, 99)):
    mm = miss[(miss.z_spec >= lo_) & (miss.z_spec < hi_)]
    hh = hit[(hit.z_spec >= lo_) & (hit.z_spec < hi_)]
    N["zStrat" + tag + "MissN"] = len(mm)
    N["zStrat" + tag + "HitN"] = len(hh)
    N["zStrat" + tag + "MissColour"] = round(float(np.nanmedian(mm.cab)), 2)
    N["zStrat" + tag + "HitColour"] = round(float(np.nanmedian(hh.cab)), 2)
    N["zStrat" + tag + "MissMag"] = round(float(mm.mag_f444w.median()), 2)
    N["zStrat" + tag + "HitMag"] = round(float(hh.mag_f444w.median()), 2)

# ------------------------------------------------------------------ 5. the ambiguous categories
amb = d[d.ambiguous.astype(bool)]
photamb = amb[amb.y.isna()]
b1amb = amb[amb.y == 0]
N["ambPhotSupport"] = len(photamb)
N["ambBOneSupport"] = len(b1amb)
N["ambPhotUnion"] = int(photamb.picked.sum())
N["ambPhotModel"] = int(photamb.sel_burden.sum())
N["ambPhotMatched"] = int(photamb.selm.sum())
N["ambBOneUnion"] = int(b1amb.picked.sum())
N["ambBOneModel"] = int(b1amb.sel_burden.sum())
N["ambBOneMatched"] = int(b1amb.selm.sum())
neg_u = int((d.picked & d.isneg).sum())
neg_m = int((d.selm & d.isneg).sum())
N["sensRateUnionWithAmb"] = round(
    100 * (neg_u + N["ambPhotUnion"]) / (4979 + len(photamb)), 2
)
N["sensRateMatchedWithAmb"] = round(
    100 * (neg_m + N["ambPhotMatched"]) / (4979 + len(photamb)), 2
)
N["sensDenWithAmb"] = 4979 + len(photamb)

# ------------------------------------------------------------------ 6. source-to-row resolution
la = pd.read_csv(os.path.join(V, "label_audit.csv"))
posall = lab[lab.y == 1].merge(
    fea[["source_id", "sky_group", "sky_group_size", "in_support", "region"]],
    on="source_id",
)
posall = posall.merge(
    la[["source_id", "n_sources_within_0p5"]], on="source_id", how="left"
)
rf = pd.read_parquet(os.path.join(V, "rule_flags.parquet"))
allrows = pd.read_parquet(
    os.path.join(V, "features.parquet"),
    columns=["source_id", "field", "id", "ra", "dec", "sky_group", "in_support", "y"],
)
grp = allrows[allrows.sky_group.isin(posall.sky_group)].merge(
    rf[["source_id", "picked"]].rename(columns={"picked": "rule_selected"}),
    on="source_id",
)
grp = grp.merge(
    oof[["source_id", "score_mean", "sel_burden"]], on="source_id", how="left"
)
grp = grp.merge(
    pd.Series(d.selm.values, index=d.source_id.values, name="sel_matched"),
    left_on="source_id",
    right_index=True,
    how="left",
)
any_u = grp.groupby("sky_group").rule_selected.any()
any_m = grp.groupby("sky_group").sel_burden.apply(
    lambda s: bool(s.fillna(False).astype(bool).any())
)
any_mm = grp.groupby("sky_group").sel_matched.apply(
    lambda s: bool(s.fillna(False).astype(bool).any())
)
N["anyCopyUnionRecall"] = int(posall.sky_group.map(any_u).sum())
N["anyCopyModelRecall"] = int(posall.sky_group.map(any_m).sum())
N["anyCopyMatchedRecall"] = int(posall.sky_group.map(any_mm).sum())
sing = pos[pos.sky_group_size == 1]
N["singletonPositives"] = len(sing)
N["singletonMatchedRecall"] = int(sing.selm.sum())
N["singletonUnionRecall"] = int(sing.picked.sum())
N["singletonMcnemarB"] = int((sing.selm & ~sing.picked).sum())
N["singletonMcnemarC"] = int((sing.picked & ~sing.selm).sum())
N["singletonMcnemarP"] = pfmt(mcnemar_p(N["singletonMcnemarB"], N["singletonMcnemarC"]))
multi = grp[grp.sky_group.isin(posall[posall.sky_group_size >= 2].sky_group)].copy()
ref = posall.set_index("sky_group")[["ra", "dec"]]
multi["is_positive_row"] = multi.y == 1
multi["offset_arcsec_from_positive_row"] = 3600 * np.hypot(
    (multi.ra - multi.sky_group.map(ref.ra)) * np.cos(np.deg2rad(multi.dec)),
    multi.dec - multi.sky_group.map(ref.dec),
)
multi = multi.sort_values(["sky_group", "is_positive_row"], ascending=[True, False])
multi[
    [
        "sky_group",
        "field",
        "id",
        "source_id",
        "ra",
        "dec",
        "is_positive_row",
        "offset_arcsec_from_positive_row",
        "in_support",
        "rule_selected",
        "score_mean",
        "sel_burden",
        "sel_matched",
    ]
].to_csv(os.path.join(REC, "positive_match_alternatives.csv"), index=False)
N["multiRowPositiveGroups"] = int(posall.sky_group_size.ge(2).sum())
N["multiRowAlternativeRows"] = int(len(multi) - N["multiRowPositiveGroups"])
N["multiRowAlternativesRuleSelected"] = int(
    multi[~multi.is_positive_row].rule_selected.sum()
)
N["multiRowAlternativesModelSelected"] = int(
    multi[~multi.is_positive_row].sel_burden.fillna(False).astype(bool).sum()
)

# ------------------------------------------------------------------ 7. the label provenance table and the reconciliation
prov = lab[lab.y.notna() | lab.ambiguous.astype(bool)].copy()
prov = prov.merge(
    fea[["source_id", "region", "sky_group", "sky_group_size", "in_support"]],
    on="source_id",
    how="left",
)
prov = prov.merge(
    tst[["source_id", "elig", "testable", "vshaped_spec"]].rename(
        columns={
            "elig": "spectrum_eligible",
            "testable": "test_evaluable_source",
            "vshaped_spec": "verdict_source",
        }
    ),
    on="source_id",
    how="left",
)
prov = prov.merge(ext, on="source_id", how="left")


def final_class(r):
    if r.y == 1:
        return "spectroscopic LRD"
    if r.y == 0:
        return "comparison object"
    return "ambiguous, used in neither class"


def amb_reason(r):
    if r.y == 1:
        return ""
    lists = [
        n
        for n, c in (
            ("Kocevski et al. 2024 catalog", "in_kocevski24"),
            ("Perger et al. 2025 compilation", "in_perger25"),
            ("Barro et al. 2026 list", "in_barro25"),
        )
        if bool(r[c])
    ]
    if r.y == 0 and bool(r.in_hviding25_B1):
        return "broad-line galaxy of Hviding et al. (2025) table B1; not an LRD list; comparison label kept"
    if pd.isna(r.y):
        return "not established V-shaped but listed in: " + "; ".join(lists)
    return ""


prov["final_class"] = prov.apply(final_class, axis=1)
prov["ambiguity_reason"] = prov.apply(amb_reason, axis=1)
prov["verdict_rule"] = (
    "spectroscopic LRD list membership"
    if False
    else np.where(
        prov.y == 1,
        np.where(
            prov[["in_hviding25_A1", "in_barro25", "in_degraaff26"]]
            .astype(bool)
            .any(axis=1),
            "published spectroscopic LRD list",
            "fixed V-shape test and r_h < 1.5 r_star",
        ),
        np.where(
            prov.y == 0,
            "not established V-shaped and in no LRD list",
            "not established V-shaped, in a photometric LRD list",
        ),
    )
)
cols = [
    "source_id",
    "field",
    "id",
    "ra",
    "dec",
    "region",
    "sky_group",
    "sky_group_size",
    "in_support",
    "bands_complete",
    "in_hviding25_A1",
    "in_hviding25_B1",
    "in_barro25",
    "in_degraaff26",
    "in_perger25",
    "in_kocevski24",
    "spectrum_eligible",
    "test_evaluable_source",
    "verdict_source",
    "spec_tested",
    "spec_vshaped",
    "y",
    "y_strict",
    "ambiguous",
    "final_class",
    "verdict_rule",
    "ambiguity_reason",
    "z_spec",
    "z_spec_source",
    "z_list",
    "z_archive",
    "archive_grade",
]
prov = prov[cols].rename(
    columns={
        "spec_tested": "test_evaluable_row",
        "spec_vshaped": "verdict_row",
        "y": "label",
        "y_strict": "label_strict",
    }
)
prov.to_csv(os.path.join(REC, "label_provenance.csv"), index=False)
N["provenanceRows"] = len(prov)
rec_ = prov[
    (
        prov.test_evaluable_source.fillna(False).astype(bool)
        != prov.test_evaluable_row.fillna(False).astype(bool)
    )
    | (
        prov.verdict_source.fillna(False).astype(bool)
        != prov.verdict_row.fillna(False).astype(bool)
    )
].copy()
rec_["difference"] = np.where(
    rec_.verdict_source.fillna(False).astype(bool)
    != rec_.verdict_row.fillna(False).astype(bool),
    "verdict written from a source-level spectrum match after the row-level flag was fixed",
    "evaluable at the source level; row-level census flag not set",
)
rec_["label_changes"] = False
rec_["printed_count_changes"] = False
rec_[
    [
        "source_id",
        "field",
        "id",
        "label",
        "final_class",
        "test_evaluable_source",
        "test_evaluable_row",
        "verdict_source",
        "verdict_row",
        "difference",
        "label_changes",
        "printed_count_changes",
    ]
].to_csv(os.path.join(REC, "test_status_reconciliation.csv"), index=False)
N["reconRows"] = len(rec_)
N["reconAnchorRows"] = int((rec_.label == 0).sum())
N["reconPositiveRows"] = int((rec_.label == 1).sum())
N["reconVerdictDiffers"] = int(
    (
        rec_.verdict_source.fillna(False).astype(bool)
        != rec_.verdict_row.fillna(False).astype(bool)
    ).sum()
)
N["reconVerdictDiffersAnchors"] = int(
    (
        (rec_.label == 0)
        & (
            rec_.verdict_source.fillna(False).astype(bool)
            != rec_.verdict_row.fillna(False).astype(bool)
        )
    ).sum()
)
N["reconEvaluableDiffersAnchors"] = int(
    (
        (rec_.label == 0)
        & (
            rec_.test_evaluable_source.fillna(False).astype(bool)
            != rec_.test_evaluable_row.fillna(False).astype(bool)
        )
    ).sum()
)
N["reconEvaluableDiffersPositives"] = int(
    (
        (rec_.label == 1)
        & (
            rec_.test_evaluable_source.fillna(False).astype(bool)
            != rec_.test_evaluable_row.fillna(False).astype(bool)
        )
    ).sum()
)

# ------------------------------------------------------------------ 8. the rule audit on the positives
from redress import contracts  # noqa: E402
from redress.cuts import CUTS  # noqa: E402

PIV = {
    "f090w": 9022.92,
    "f115w": 11543.01,
    "f150w": 15007.45,
    "f200w": 19886.48,
    "f277w": 27623.47,
    "f356w": 35682.28,
    "f410m": 40820.73,
    "f444w": 44037.14,
    "f606w": 5920.82,
    "f814w": 8056.88,
}
P = (
    lab[lab.y == 1]
    .merge(
        fea[["source_id", "in_support"]],
        on="source_id",
    )
    .merge(
        pap[["source_id", "labbe_compactness", "compactness_f444w", "measurable"]],
        on="source_id",
        how="left",
    )
)
P = P.merge(oof[["source_id", "sel_burden"]], on="source_id", how="left").merge(
    pd.Series(d.selm.values, index=d.source_id.values, name="sel_matched"),
    left_on="source_id",
    right_index=True,
    how="left",
)
n = len(P)
phot = pd.DataFrame(
    {
        "object_id": P.source_id.astype(str).values,
        "field": P.field.values,
        "ra_deg": P.ra.values,
        "dec_deg": P.dec.values,
        "source_catalog": ["dja-v7"] * n,
        "compactness_f444w": P.compactness_f444w.values,
        "flux_radius_f444w_arcsec": P.r_h_arcsec.values,
        "nearest_neighbor_arcsec": [np.nan] * n,
    }
)
for b in BANDS + ["f410m", "f606w", "f814w"]:
    if b in BANDS:
        f = P["f_" + b].to_numpy(float)
        e = P["e_" + b].to_numpy(float)
        cov = np.isfinite(f) & np.isfinite(e) & (e > 0)
        phot["f_%s_ujy" % b] = np.where(cov, f, np.nan)
        phot["e_%s_ujy" % b] = np.where(cov, e, np.nan)
        phot["cov_" + b] = cov
    else:
        phot["f_%s_ujy" % b] = np.nan
        phot["e_%s_ujy" % b] = np.nan
        phot["cov_" + b] = False
contracts.validate_photometry_table(phot, allowed_fields=set(phot.field))
out = {}
out["labbe23"] = CUTS["labbe23"](phot, P.labbe_compactness.values)
out["kokorev24"] = CUTS["kokorev24"](phot, P.labbe_compactness.values)
out["kocevski24"] = CUTS["kocevski24"](
    phot, P.z_phot.values, P.r_star_v1_arcsec.values, PIV
)
out["perezgonzalez24"] = CUTS["perezgonzalez24"](phot)
out["barro23"] = CUTS["barro23"](phot)
out["greene24"] = CUTS["greene24"](phot)
out["akins24"] = CUTS["akins24"](phot)


def first_failure(rule, o, i, r):
    """Name the category of the first failing criterion for row i under rule `rule`."""

    def g(k):
        return bool(o[k][i]) if k in o else True

    if rule in ("labbe23", "kokorev24"):
        if not g("detection"):
            return "detection gate"
        if not (g("red1") or g("red2")):
            return "color criterion"
        if rule == "kokorev24" and not g("bd_removal"):
            return "brown-dwarf color"
        if not g("compact_valid"):
            return "aperture unmeasurable"
        if not g("compact"):
            return "compactness criterion"
        return "none"
    if rule == "akins24":
        if not g("detection"):
            return "detection gate"
        if not g("red_color"):
            return "color criterion"
        if not g("compact_valid"):
            return "aperture unmeasurable"
        if not g("compact"):
            return "compactness criterion"
        return "none"
    if rule == "kocevski24":
        z = P.z_phot.values[i]
        if not g("detection"):
            return "detection gate"
        if np.isfinite(z) and z < 4.75:
            return "HST band set required"
        if not (g("beta_opt_red") and g("beta_uv_window")):
            return "slope criterion"
        if not g("size"):
            return "size criterion"
        if not (g("lineboost_356") and g("lineboost_410")):
            return "line-boost criterion"
        return "none"
    if rule == "perezgonzalez24":
        if not g("mag_gate"):
            return "detection gate"
        if not (g("red_color") and g("blue_color")):
            return "color criterion"
        if not g("bd_retention"):
            return "brown-dwarf color"
        return "none"
    if rule == "barro23":
        if not g("mag_gate"):
            return "detection gate"
        if not g("red_color"):
            return "color criterion"
        return "none"
    if rule == "greene24":
        if not g("detection"):
            return "detection gate"
        if not (g("vshape_blue") and g("vshape_red")):
            return "color criterion"
        return "none"
    return "none"


rows = []
audit = {}
for rule in RULES:
    o = out[rule]
    rec_sel = P["sel_" + rule].astype(bool).values
    rer_sel = np.asarray(o["selected"], bool)
    cats = []
    for i in range(n):
        c = "selected" if rec_sel[i] else first_failure(rule, o, i, P.iloc[i])
        if rec_sel[i] and not rer_sel[i]:
            note = "record selects; seven-band rerun with fresh apertures does not"
        elif (not rec_sel[i]) and rer_sel[i]:
            note = "seven-band rerun with fresh apertures selects; record does not"
        else:
            note = ""
        cats.append(c)
        rows.append(
            dict(
                rule=rule,
                source_id=int(P.source_id.values[i]),
                field=P.field.values[i],
                id=int(P.id.values[i]),
                selected_record=bool(rec_sel[i]),
                selected_rerun=bool(rer_sel[i]),
                outcome=c,
                note=note,
                z_phot=float(P.z_phot.values[i]),
                in_support=bool(P.in_support.values[i]),
            )
        )
    cats = pd.Series(cats)
    audit[rule] = cats.value_counts().to_dict()
    audit[rule]["agree"] = int((rec_sel == rer_sel).sum())
    t = RTAG[rule]
    N["audit" + t + "Selected"] = int(rec_sel.sum())
    N["audit" + t + "Detection"] = int((cats == "detection gate").sum())
    N["audit" + t + "Color"] = int((cats == "color criterion").sum()) + int(
        (cats == "brown-dwarf color").sum()
    )
    N["audit" + t + "Compact"] = int((cats == "compactness criterion").sum()) + int(
        (cats == "size criterion").sum()
    )
    N["audit" + t + "Unavailable"] = int((cats == "aperture unmeasurable").sum()) + int(
        (cats == "HST band set required").sum()
    )
    N["audit" + t + "Slope"] = int((cats == "slope criterion").sum()) + int(
        (cats == "line-boost criterion").sum()
    )
    N["audit" + t + "Agree"] = audit[rule]["agree"]
pd.DataFrame(rows).to_csv(os.path.join(REC, "rule_audit_positives.csv"), index=False)
N["auditRowsTotal"] = len(rows)
N["auditAgreeTotal"] = sum(audit[r]["agree"] for r in RULES)
N["auditDisagreeTotal"] = N["auditRowsTotal"] - N["auditAgreeTotal"]
# the size-threshold sensitivity for kocevski24: positives within 20 per cent of the size cut
ratio = P.r_h_arcsec / P.r_star_v1_arcsec
N["auditKocevskiNearSizeCut"] = int(((ratio > 1.5 / 1.2) & (ratio < 1.5 * 1.2)).sum())
N["auditKocevskiSizeFailAll"] = int((ratio >= 1.5).sum())
N["auditKocevskiZphotBelowHst"] = int((P.z_phot < 4.75).sum())
N["auditKocevskiZphotBelowHstSelected"] = int(
    ((P.z_phot < 4.75) & P.sel_kocevski24.astype(bool)).sum()
)
# the union under the rerun
rerun_union = np.zeros(n, bool)
for rule in RULES:
    rerun_union |= np.asarray(out[rule]["selected"], bool)
N["auditUnionRerunRecall"] = int(rerun_union.sum())
N["auditUnionRecordRecall"] = int(P[SEL].astype(bool).any(axis=1).sum())

# ------------------------------------------------------------------ 9. aperture compactness of the positives
N["posApertureMeasured"] = int(pap.measurable.sum())
N["posLabbeCompact"] = int((pap.labbe_compactness < 1.7).sum())
N["posLabbeCompactPct"] = round(100 * N["posLabbeCompact"] / 151, 1)
N["posAkinsCompact"] = int(
    ((pap.compactness_f444w > 0.5) & (pap.compactness_f444w <= 0.7)).sum()
)
N["posAkinsCompactPct"] = round(100 * N["posAkinsCompact"] / 151, 1)
N["posApertureMosaicVersions"] = int(pap.grizliv.nunique())
PL = P[P.in_support & (P.labbe_compactness < 1.7)]
N["posLabbeCompactSupport"] = len(PL)
N["labbeUnionRecall"] = int(PL[SEL].astype(bool).any(axis=1).sum())
N["labbeMatchedRecall"] = int(PL.sel_matched.fillna(False).astype(bool).sum())
N["labbeDeployedRecall"] = int(PL.sel_burden.fillna(False).astype(bool).sum())
PN_ = P[P.in_support & ~(P.labbe_compactness < 1.7)]
N["posLabbeExtendedSupport"] = len(PN_)
N["labbeExtUnionRecall"] = int(PN_[SEL].astype(bool).any(axis=1).sum())
N["labbeExtMatchedRecall"] = int(PN_.sel_matched.fillna(False).astype(bool).sum())
# catalog-radius against aperture criterion on the same positives
cat_c = P.r_h_arcsec < 1.5 * P.r_star_v1_arcsec
N["posCatalogCompactAndLabbe"] = int((cat_c & (P.labbe_compactness < 1.7)).sum())
N["posCatalogCompactNotLabbe"] = int((cat_c & ~(P.labbe_compactness < 1.7)).sum())
N["posLabbeNotCatalogCompact"] = int((~cat_c & (P.labbe_compactness < 1.7)).sum())

# ------------------------------------------------------------------ 10. challengers at matched budget
d["rank_tree"] = d.groupby("region").score_mean.rank(pct=True)
d["rank_v34"] = d.groupby("region").score_v34.rank(pct=True)
d["rank_mlp"] = d.groupby("region").score_mlp.rank(pct=True)
d["blend30"] = 0.5 * (d.rank_tree + d.rank_mlp)
d["blend34"] = 0.5 * (d.rank_v34 + d.rank_mlp)
for tag, col in (
    ("TreesThirtyFour", "score_v34"),
    ("Mlp", "score_mlp"),
    ("Blend", "blend30"),
    ("BlendTwo", "blend34"),
):
    s = matched(col)
    N["chal" + tag + "MatchedRecall"] = int((s & d.ispos).sum())
    N["chal" + tag + "MatchedRuleMissed"] = int((s & d.ispos & ~d.picked).sum())
    N["chal" + tag + "MatchedAnchors"] = int((s & d.isneg).sum())
N["chalPrimaryMatchedRecall"] = N["matchedRecallSupport"]
N["chalPrimaryMatchedRuleMissed"] = N["matchedRuleMissed"]
N["chalPrimaryMatchedAnchors"] = N["matchedAnchors"]

# ------------------------------------------------------------------ 11. leave-one-list-out, unique against shared members
bs = bsc.merge(
    lab[["source_id", "in_hviding25_A1", "in_barro25", "in_degraaff26", "picked"]],
    on="source_id",
).merge(fea[["source_id", "in_support"]], on="source_id")
bs = bs[bs.in_support]
pp = bs.merge(lab[["source_id", "y"]], on="source_id")
pp = pp[pp.y == 1]
for key, col, tag in (
    ("hviding25_A1", "in_hviding25_A1", "HvidingA"),
    ("barro25", "in_barro25", "Barro"),
    ("degraaff26", "in_degraaff26", "DeGraaff"),
):
    others = [c for c in ("in_hviding25_A1", "in_barro25", "in_degraaff26") if c != col]
    mem = pp[pp[col].astype(bool)]
    uniq = mem[~mem[others].astype(bool).any(axis=1)]
    shared = mem[mem[others].astype(bool).any(axis=1)]
    s = "sel_leave_out_" + key
    N["loco" + tag + "UniqueN"] = len(uniq)
    N["loco" + tag + "SharedN"] = len(shared)
    N["loco" + tag + "UniqueLeftOut"] = int(uniq[s].astype(bool).sum())
    N["loco" + tag + "SharedLeftOut"] = int(shared[s].astype(bool).sum())
    N["loco" + tag + "UniqueUnion"] = int(uniq.picked.astype(bool).sum())
    N["loco" + tag + "SharedUnion"] = int(shared.picked.astype(bool).sum())
    N["loco" + tag + "UniqueReplica"] = int(
        uniq.sel_primary_8bag_replica.astype(bool).sum()
    )
    N["loco" + tag + "SharedReplica"] = int(
        shared.sel_primary_8bag_replica.astype(bool).sum()
    )

# ------------------------------------------------------------------ 12. candidates: status framework, eighth rule, redshifts
cand = pd.read_csv(os.path.join(REC, "candidates.csv"))
assert len(cand) == 1207
cand = cand.drop(
    columns=[
        c
        for c in (
            "sel_barro24b",
            "spectrum_status",
            "z_spec_secure",
            "listed_after_freeze",
            "excluded_by_eighth_rule",
        )
        if c in cand.columns
    ]
)
cand = cand.merge(b24, on="source_id", how="left")
cand["sel_barro24b"] = cand.sel_barro24b.fillna(False).astype(bool)


def spectrum_status(r):
    if int(r.n_archive_records) == 0:
        return "no spectrum in the frozen archive index"
    if int(r.n_secure_lowz_records) > 0:
        return "secure redshift at z <= 3"
    if int(r.n_secure_highz_records) > 0:
        return "secure z > 3 from a non-prism spectrum; V-shape test not applicable"
    return "spectrum present, no secure redshift"


cand["spectrum_status"] = cand.apply(spectrum_status, axis=1)
cand["z_spec_secure"] = cand.archive_z_best_grade3
cand["listed_after_freeze"] = cand.in_degraaff26_v2.fillna(False).astype(bool)
cand.to_csv(os.path.join(REC, "candidates.csv"), index=False)
N["candBEightSelected"] = int(cand.sel_barro24b.sum())
N["candEbBEightSelected"] = int(cand[cand.tier == "equal_burden"].sel_barro24b.sum())
N["candFuBEightSelected"] = int(
    cand[cand.followup_tier.astype(bool)].sel_barro24b.sum()
)
N["candTotalNotEight"] = 1207 - N["candBEightSelected"]
N["candEbNotEight"] = 362 - N["candEbBEightSelected"]
st = cand.spectrum_status.value_counts()
N["candStatusNoSpectrum"] = int(st.get("no spectrum in the frozen archive index", 0))
N["candStatusSpectrumNoSecure"] = int(st.get("spectrum present, no secure redshift", 0))
N["candStatusSecureHigh"] = int(
    st.get("secure z > 3 from a non-prism spectrum; V-shape test not applicable", 0)
)
N["candStatusSecureLow"] = int(st.get("secure redshift at z <= 3", 0))
ste = cand[cand.tier == "equal_burden"].spectrum_status.value_counts()
N["candEbStatusNoSpectrum"] = int(ste.get("no spectrum in the frozen archive index", 0))
N["candEbStatusSpectrumNoSecure"] = int(
    ste.get("spectrum present, no secure redshift", 0)
)
N["candListedAfterFreeze"] = int(cand.listed_after_freeze.sum())
N["candEbLabbeOfMeasuredPct"] = round(
    100
    * cand[(cand.tier == "equal_burden") & cand.compactness_measurable.astype(bool)]
    .compact_labbe.astype(bool)
    .mean(),
    1,
)

# ------------------------------------------------------------------ 13. the data table
full = pd.read_parquet(
    os.path.join(V, "labels.parquet"),
    columns=["field", "bands_complete"] + ["e_" + b for b in BANDS],
)
full = full[full.bands_complete]
rstar = (
    fea.merge(lab[["source_id", "field"]], on="source_id")[
        ["field", "r_star_v1_arcsec"]
    ]
    .drop_duplicates()
    .set_index("field")
    .r_star_v1_arcsec
)
dt = {}
for f, g in full.groupby("field"):
    dt[f] = {
        "band_complete_rows": int(len(g)),
        "r_star_arcsec": float(rstar.get(f, np.nan)),
    }
    for b in BANDS:
        dt[f]["depth5sig_" + b] = round(
            float(23.9 - 2.5 * np.log10(5 * np.nanmedian(g["e_" + b]))), 1
        )
json.dump(dt, open(os.path.join(R9, "data_table.json"), "w"), indent=1)
N["depthTableFields"] = len(dt)

# ------------------------------------------------------------------ 14. the regional-budget curve and within-region ranks
K8 = {R: int(g.u8.sum()) for R, g in d.groupby("region")}
fr = np.concatenate(
    [
        np.arange(0.1, 1.0, 0.1),
        [1.0],
        np.arange(1.25, 3.01, 0.25),
        [3.5, 4, 5, 6, 8, 10, 15, 20, 30],
    ]
)
curve = {
    "fractions": fr.tolist(),
    "primary": [],
    "imitation": [],
    "K_union": NU,
    "K_union8": K8,
}
sorted_ = {}
for R, g in d.groupby("region"):
    for name, col in (
        ("primary", "score_mean"),
        ("imitation", "score_rules_reproduction"),
    ):
        s = g.sort_values(col, ascending=False, kind="stable")
        sorted_[(R, name)] = np.cumsum((s.y == 1).to_numpy())
for name in ("primary", "imitation"):
    tot = np.zeros(len(fr))
    for R in REGIONS:
        cum = sorted_[(R, name)]
        for i, f in enumerate(fr):
            k = max(int(round(f * NU[R])), 1)
            tot[i] += cum[min(k, len(cum)) - 1]
    curve[name] = tot.tolist()
curve["union"] = {
    "rows": int(d.picked.sum()),
    "recall": int((d.picked & d.ispos).sum()),
}
s8 = matched("score_mean", K8)
i8 = matched("score_rules_reproduction", K8)
curve["union8"] = {
    "rows": int(d.u8.sum()),
    "recall": int((d.u8 & d.ispos).sum()),
    "primary": int((s8 & d.ispos).sum()),
    "imitation": int((i8 & d.ispos).sum()),
}
curve["deployed"] = {
    "rows": int(d.sel_burden.sum()),
    "recall": int((d.sel_burden & d.ispos).sum()),
}
curve["half"] = {"rows": int(d.sel_05.sum()), "recall": int((d.sel_05 & d.ispos).sum())}
json.dump(curve, open(os.path.join(R9, "regional_budget_curve.json"), "w"), indent=1)
N["curveTwoTimes"] = int(curve["primary"][list(np.round(fr, 2)).index(2.0)])
N["curveHalf"] = int(curve["primary"][list(np.round(fr, 2)).index(0.5)])
N["curveTenTimes"] = int(curve["primary"][list(np.round(fr, 2)).index(10.0)])
N["curveImitationTwoTimes"] = int(curve["imitation"][list(np.round(fr, 2)).index(2.0)])
sm = ev["still_missed_list"]
rows = []
for r in sm:
    g = d[d.region == r["region"]]
    rows.append(
        {
            "field": r["field"],
            "id": r["id"],
            "region": r["region"],
            "score": r["score"],
            "rank_pct_pooled": round(r["score_rank_pct"], 2),
            "rank_pct_region": round(
                100 * float((g.score_mean >= r["score"]).mean()), 2
            ),
        }
    )
json.dump(rows, open(os.path.join(R9, "still_missed_region_rank.json"), "w"), indent=1)
worst = max(rows, key=lambda x: x["rank_pct_region"])
N["worstMissRankRegionPct"] = worst["rank_pct_region"]
N["stillMissedTopTwoRegion"] = int(sum(1 for x in rows if x["rank_pct_region"] <= 2.0))


# per-object outcomes for the figures, and the matched miss count for the pipeline figure
N["matchedMissedEnd"] = 151 - N["matchedRecallEnd"]
pos[["source_id", "region", "picked", "sel_burden", "selm"]].rename(columns={"selm": "sel_matched"}).to_csv(os.path.join(R9, "positive_outcomes.csv"), index=False)
# ------------------------------------------------------------------ write
json.dump(N, open(os.path.join(R9, "round9_numbers.json"), "w"), indent=1)
WORD = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
existing = set(
    re.findall(
        r"\\newcommand\{\\([A-Za-z]+)\}",
        open(NUMBERS_TEX, encoding="utf-8")
        .read()
        .split("% ---- revision 2026-09-06 (astra r5, round 9): begin ----")[0],
    )
)


def latex_num(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return "{:,}".format(int(v)).replace(",", "{,}")
    if isinstance(v, float):
        s = ("%g" % v) if abs(v) >= 1e-3 else ("%.1e" % v)
        if "e" in s:
            return s
        return s
    return str(v)


BEGIN = "% ---- revision 2026-09-06 (astra r5, round 9): begin ----"
END = "% ---- revision 2026-09-06 (astra r5, round 9): end ----"
lines = [
    BEGIN,
    "% source: paper/round9_compute_2026_09_06.py on recovery and the public clone's recovery tables",
]
for k, v in N.items():
    name = k
    for dch, word in zip("0123456789", WORD):
        name = name.replace(dch, word)
    name = re.sub(r"[^A-Za-z]", "", name)
    cmd = "renewcommand" if name in existing else "newcommand"
    lines.append("\\%s{\\%s}{%s}" % (cmd, name, latex_num(v)))
lines.append(END)
for _path in (NUMBERS_TEX, NUMBERS_TEX_REC):
    _head = open(_path, encoding="utf-8").read().split(BEGIN)[0]
    _existing = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", _head))
    _lines = [lines[0], lines[1]]
    for k, v in N.items():
        name = k
        for dch, word in zip("0123456789", WORD):
            name = name.replace(dch, word)
        name = re.sub(r"[^A-Za-z]", "", name)
        cmd = "renewcommand" if name in _existing else "newcommand"
        _lines.append("\\%s{\\%s}{%s}" % (cmd, name, latex_num(v)))
    _lines.append(lines[-1])
    txt = open(_path, encoding="utf-8").read()
    txt = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END) + "\n?", "", txt, flags=re.S)
    open(_path, "w", encoding="utf-8").write(txt.rstrip("\n") + "\n" + "\n".join(_lines) + "\n")
print(len(N), "numbers;", "audit:", json.dumps(audit, indent=0)[:300])
