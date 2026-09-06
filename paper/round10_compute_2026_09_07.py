"""Round 10 compute (2026-09-07): the numbers behind the r7 referee fixes.

Reads the tables of record (recovery), the public clone's release tables, the round 5
Barro membership record and the Skyfire Table 3 catalog; writes scratch/round10/
round10_numbers.json and appends a delimited macro block to both numbers.tex files.
Nothing is retrained. Every number here is a recount of released or recorded tables.

Sections
  1. comparator fidelity: barro24b as coded against the Barro et al. (2024a) selection's
     own output (the Barro et al. (2026) list less its hand-added objects), kocevski24
     against the published Kocevski et al. (2025) catalog
  2. Table 7 from the region-matched columns (selmatched_*), own-threshold kept beside
  3. duplicate and neighbour audit of the positives, the comparison rows and the candidates
  4. paired-difference intervals (Wald and Newcombe method 10) for every paired comparison
  5. compactness cross-tabulation on the 147 in support
  6. shuffled-label controls at both operating points
  7. the sixteen record-versus-rerun disagreements by rule; the record-only column
  8. the 48 sources used in neither class, split by reason; sensitivity rates recomputed
  9. the V-shape ledger at row and at source level
 10. the Kocevski catalog rows: selected, rejected, no redshift bin
 11. per-field stellar radius spread
 12. Skyfire (Kocevski et al. 2026, arXiv:2609.00112) Table 3 against positives and candidates
 13. freeze chronology: the dates the inputs were obtained
"""

from __future__ import annotations

import json
import os
import re
import sys
from math import comb, sqrt

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "recovery")
R5 = os.path.join(ROOT, "recovery", "round5")
R10 = os.path.join(ROOT, "recovery")
CLONE = ROOT  # the release tables live beside the script
REC = os.path.join(CLONE, "recovery")
SKYFIRE = os.path.join(
    os.path.expanduser("~"),
    "review_r7_external",
    "work",
    "skyfire",
    "Table3.Kocevski26.cat",
)
NUMBERS_TEX = os.path.join(ROOT, "paper", "numbers.tex")
NUMBERS_TEX_REC = NUMBERS_TEX  # one numbers.tex in the clone
os.makedirs(R10, exist_ok=True)

SEL = [
    "sel_labbe23",
    "sel_kokorev24",
    "sel_kocevski24",
    "sel_perezgonzalez24",
    "sel_barro23",
    "sel_greene24",
    "sel_akins24",
]
N = {}
audit = {}


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * p)


def pfmt(p):
    if p >= 0.01:
        return "%.2f" % p
    if p >= 0.001:
        return "%.3f" % p
    return "%.1e" % p


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def newcombe10(a, b, c, d, z=1.96):
    """Newcombe (1998) method 10: CI for p1 - p2 from a paired 2x2 (a both, b first
    only, c second only, d neither). p1 = (a+b)/n is the first selection's rate."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    m1, m2, m3, m4 = a + b, c + d, a + c, b + d
    if min(m1, m2, m3, m4) == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        if num > 0:
            num = max(num - n / 2.0, 0.0)
        phi = num / sqrt(m1 * m2 * m3 * m4)
    delta = p1 - p2
    lo = delta - sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    hi = delta + sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    return lo, hi


def wald_paired(b, c, n, z=1.96):
    delta = (b - c) / n
    se = sqrt((b + c) - (b - c) ** 2 / n) / n
    return delta - z * se, delta + z * se


def angsep(ra1, dec1, ra2, dec2):
    """Great-circle separation in arcsec; inputs in degrees; broadcasts."""
    r1, d1, r2, d2 = map(np.deg2rad, (ra1, dec1, ra2, dec2))
    x = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.rad2deg(np.arccos(np.clip(x, -1, 1))) * 3600.0


def self_pairs(df, radius):
    ra = df.ra.to_numpy(float)
    dec = df.dec.to_numpy(float)
    out = []
    for i in range(len(df)):
        s = angsep(ra[i], dec[i], ra, dec)
        for j in np.where((s < radius) & (np.arange(len(df)) > i))[0]:
            out.append((i, int(j), float(s[j])))
    return out


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
        "r_star_v1_arcsec",
        "field",
    ],
)
oof = pd.read_parquet(os.path.join(V, "oof_scores.parquet"))
bof = pd.read_parquet(os.path.join(V, "baseline_oof_full.parquet"))
den = json.load(open(os.path.join(V, "denominators.json")))
NU = den["union_in_support_per_region"]
b24 = pd.read_parquet(os.path.join(REC, "barro24b_flags.parquet"))
b24c = pd.read_parquet(os.path.join(REC, "barro24b_colors.parquet"))
pap = pd.read_csv(os.path.join(REC, "positive_apertures.csv"))
aud = pd.read_csv(os.path.join(REC, "rule_audit_positives.csv"))
prov = pd.read_csv(os.path.join(REC, "label_provenance.csv"), low_memory=False)
cand = pd.read_csv(os.path.join(REC, "candidates.csv"), low_memory=False)
bm = json.load(open(os.path.join(R5, "barro_membership.json")))
hand = sorted(int(x["source_id"]) for x in bm["manualAdditions"])
assert len(hand) == 7

d = oof.merge(fea.drop(columns=["region"]), on="source_id", how="left")
d = d.merge(
    lab[
        ["source_id"]
        + SEL
        + [
            "ambiguous",
            "in_hviding25_A1",
            "in_hviding25_B1",
            "in_degraaff26",
            "in_barro25",
            "in_perger25",
            "in_kocevski24",
            "spec_tested",
            "spec_vshaped",
            "z_phot",
            "r_h_arcsec",
        ]
    ],
    on="source_id",
    how="left",
)
d = d.merge(b24, on="source_id", how="left").merge(
    b24c[["source_id", "barro24b_colors"]], on="source_id", how="left"
)
d = d.merge(bof.drop(columns=["region"]), on="source_id", how="left")
assert d.in_support.all() and len(d) == den["support_rows"]
d["picked"] = d.picked.astype(bool)
d["ispos"] = d.y == 1
d["isneg"] = d.y == 0
d["sel_barro24b"] = d.sel_barro24b.fillna(False).astype(bool)
d["barro24b_colors"] = d.barro24b_colors.fillna(False).astype(bool)
d["hand_added"] = d.source_id.isin(hand)
d["barro_member"] = (
    d.in_barro25.astype(bool) & ~d.hand_added
)  # the selection's own output

# region-matched primary selection: top NU[region] of the primary score in each region
d["selm"] = False
for reg, k in NU.items():
    idx = d[d.region == reg].sort_values("score_mean", ascending=False).index[: int(k)]
    d.loc[idx, "selm"] = True
assert int((d.selm & d.ispos).sum()) == 124, int((d.selm & d.ispos).sum())

# the 151 positives, in support or not
P = lab[lab.y == 1].merge(
    fea[
        [
            "source_id",
            "in_support",
            "region",
            "sky_group",
            "sky_group_size",
            "field",
            "r_star_v1_arcsec",
        ]
    ],
    on="source_id",
    how="left",
    suffixes=("", "_fea"),
)
P = P.merge(b24, on="source_id", how="left").merge(
    b24c[["source_id", "barro24b_colors"]], on="source_id", how="left"
)
P = P.merge(d[["source_id", "selm", "sel_burden"]], on="source_id", how="left")
P["picked"] = P.picked.astype(bool)
P["sel_barro24b"] = P.sel_barro24b.fillna(False).astype(bool)
P["barro24b_colors"] = P.barro24b_colors.fillna(False).astype(bool)
P["selm"] = P.selm.fillna(False).astype(bool)
P["hand_added"] = P.source_id.isin(hand)
P["barro_member"] = P.in_barro25.astype(bool) & ~P.hand_added
assert len(P) == 151
missed = P[~P.picked]
assert len(missed) == 44

# ------------------------------------------------------------------ 1. comparator fidelity
N["bListMembersPos"] = int(P.in_barro25.astype(bool).sum())  # 106
N["bPhotMembersPos"] = int(
    P.barro_member.sum()
)  # 99: the published selection's own output
bp = P[P.barro_member]
N["bModuleOnPhotMembers"] = int(bp.sel_barro24b.sum())
N["bColorsOnPhotMembers"] = int(bp.barro24b_colors.sum())
N["bModuleOnPhotMembersPct"] = round(100 * N["bModuleOnPhotMembers"] / len(bp), 1)
N["bColorsOnPhotMembersPct"] = round(100 * N["bColorsOnPhotMembers"] / len(bp), 1)
N["bPhotMembersFailColor"] = int((~bp.barro24b_colors).sum())
N["bPhotMembersFailCompact"] = int((bp.barro24b_colors & ~bp.sel_barro24b).sum())
bl = P[P.in_barro25.astype(bool)]
N["bModuleOnListMembers"] = int(bl.sel_barro24b.sum())
N["bColorsOnListMembers"] = int(bl.barro24b_colors.sum())
N["bModuleRuleMissed"] = int(missed.sel_barro24b.sum())  # the coded module on the 44
N["bMembershipRuleMissed"] = int(missed.barro_member.sum())  # list membership on the 44
N["bModuleRuleMissedInSupport"] = int(
    (missed.sel_barro24b & missed.in_support.fillna(False).astype(bool)).sum()
)
N["bModuleAnchorsSupport"] = int((d.sel_barro24b & d.isneg).sum())
N["bModulePosSupport"] = int((d.sel_barro24b & d.ispos).sum())
N["sevenPlusMembershipRecall"] = int((P.picked | P.barro_member).sum())
N["sevenPlusMembershipRecallPct"] = round(100 * N["sevenPlusMembershipRecall"] / 151, 1)
N["sevenPlusListRecall"] = int((P.picked | P.in_barro25.astype(bool)).sum())
N["bPhotMembersNotSelectedByModule"] = len(bp) - N["bModuleOnPhotMembers"]
# kocevski24 against the published Kocevski et al. (2025) catalog on the positives
kc = P[P.in_kocevski24.astype(bool)]
N["kocCatPos"] = len(kc)
N["kocModuleOnCatPos"] = int(kc.sel_kocevski24.astype(bool).sum())
N["kocModuleOnCatPosPct"] = round(100 * N["kocModuleOnCatPos"] / max(len(kc), 1), 1)
ka = aud[(aud.rule == "kocevski24") & aud.source_id.isin(kc.source_id)]
N["kocCatPosUnavailable"] = int((ka.outcome == "input unavailable").sum())
N["kocCatPosEvaluable"] = len(kc) - N["kocCatPosUnavailable"]
N["kocModuleOnCatPosEvaluablePct"] = round(
    100 * N["kocModuleOnCatPos"] / max(N["kocCatPosEvaluable"], 1), 1
)
audit["fidelity"] = {
    "barro_phot_members": len(bp),
    "module": N["bModuleOnPhotMembers"],
    "colors": N["bColorsOnPhotMembers"],
    "kocevski_cat_pos": len(kc),
    "kocevski_module": N["kocModuleOnCatPos"],
    "kocevski_unavailable": N["kocCatPosUnavailable"],
}

# ------------------------------------------------------------------ 2. Table 7 from the matched columns
pp = d[d.ispos]
for key, col, tag in (
    ("hviding25_A1", "in_hviding25_A1", "HvidingA"),
    ("barro25", "in_barro25", "Barro"),
    ("degraaff26", "in_degraaff26", "DeGraaff"),
):
    others = [c for c in ("in_hviding25_A1", "in_barro25", "in_degraaff26") if c != col]
    mem = pp[pp[col].astype(bool)]
    uniq = mem[~mem[others].astype(bool).any(axis=1)]
    shared = mem[mem[others].astype(bool).any(axis=1)]
    sm = "selmatched_leave_out_" + key
    so = "sel_leave_out_" + key
    N["loco" + tag + "UniqueLeftOutMatched"] = int(uniq[sm].astype(bool).sum())
    N["loco" + tag + "SharedLeftOutMatched"] = int(shared[sm].astype(bool).sum())
    N["loco" + tag + "AllLeftOutMatched"] = int(mem[sm].astype(bool).sum())
    N["loco" + tag + "UniqueLeftOutOwn"] = int(uniq[so].astype(bool).sum())
    N["loco" + tag + "SharedLeftOutOwn"] = int(shared[so].astype(bool).sum())
    N["loco" + tag + "AllLeftOutOwn"] = int(mem[so].astype(bool).sum())
    N["loco" + tag + "UniqueReplicaMatched"] = int(
        uniq.selmatched_primary_8bag_replica.astype(bool).sum()
    )
    N["loco" + tag + "SharedReplicaMatched"] = int(
        shared.selmatched_primary_8bag_replica.astype(bool).sum()
    )
    N["loco" + tag + "AllReplicaMatched"] = int(
        mem.selmatched_primary_8bag_replica.astype(bool).sum()
    )
    N["loco" + tag + "AllUnion"] = int(mem.picked.sum())
    N["loco" + tag + "AllPrimaryMatched"] = int(mem.selm.sum())
    N["loco" + tag + "AllN"] = len(mem)
    assert (
        N["loco" + tag + "UniqueLeftOutMatched"]
        + N["loco" + tag + "SharedLeftOutMatched"]
        == N["loco" + tag + "AllLeftOutMatched"]
    )

# ------------------------------------------------------------------ 3. duplicates and neighbours
pos_all = lab[lab.y == 1][["source_id", "field", "id", "ra", "dec"]].reset_index(
    drop=True
)
pos_all = pos_all.merge(fea[["source_id", "sky_group"]], on="source_id", how="left")
pairs1 = self_pairs(pos_all, 1.0)
pairs15 = self_pairs(pos_all, 1.5)
N["posPairsOneArcsec"] = len(pairs1)
N["posPairsOneHalfArcsec"] = len(pairs15)
dup = []
for i, j, s in pairs1:
    dup.append(
        {
            "a": "%s %d" % (pos_all.field[i], pos_all.id[i]),
            "b": "%s %d" % (pos_all.field[j], pos_all.id[j]),
            "sep_arcsec": round(s, 2),
            "same_sky_group": bool(pos_all.sky_group[i] == pos_all.sky_group[j]),
        }
    )
audit["positive_pairs_1arcsec"] = dup
if dup:
    N["dupPairSep"] = dup[0]["sep_arcsec"]
# the merged (source-level) recount: pairs within 1 arcsec are one source
merged_drop = set()
for i, j, s in pairs1:
    # keep the row the ranking scores higher; the other is the shred
    si = float(
        P.set_index("source_id").loc[pos_all.source_id[i], "sel_burden"] if False else 0
    )
    merged_drop.add(int(pos_all.source_id[i]))  # drop the first listed of the pair
PM = P[~P.source_id.isin(merged_drop)]
N["nPosMerged"] = len(PM)
N["nPosSupportMerged"] = int(PM.in_support.fillna(False).astype(bool).sum())
N["unionRecallMerged"] = int(PM.picked.sum())
N["nRuleMissedMerged"] = int((~PM.picked).sum())
N["matchedRecallMerged"] = int(PM.selm.sum())
N["matchedRuleMissedMerged"] = int((PM.selm & ~PM.picked).sum())
N["unionRecallMergedPct"] = round(100 * N["unionRecallMerged"] / len(PM), 1)
N["matchedRecallMergedPct"] = round(100 * N["matchedRecallMerged"] / len(PM), 1)
# comparison rows near positives (all labelled rows, any support)
neg_all = lab[lab.y == 0][["source_id", "field", "id", "ra", "dec"]].reset_index(
    drop=True
)
neg_all = neg_all.merge(
    fea[["source_id", "sky_group", "in_support"]], on="source_id", how="left"
)
near = []
for i in range(len(pos_all)):
    s = angsep(
        pos_all.ra[i],
        pos_all.dec[i],
        neg_all.ra.to_numpy(float),
        neg_all.dec.to_numpy(float),
    )
    for j in np.where(s < 1.0)[0]:
        near.append(
            {
                "positive": "%s %d" % (pos_all.field[i], pos_all.id[i]),
                "comparison": "%s %d" % (neg_all.field[j], neg_all.id[j]),
                "sep_arcsec": round(float(s[j]), 2),
                "same_sky_group": bool(pos_all.sky_group[i] == neg_all.sky_group[j]),
                "comparison_in_support": bool(neg_all.in_support[j]),
            }
        )
audit["comparison_rows_within_1arcsec_of_a_positive"] = near
N["negNearPosOneArcsec"] = len(near)
N["negNearPosOneArcsecSupport"] = int(
    sum(1 for x in near if x["comparison_in_support"])
)
N["negNearPosSameGroup"] = int(sum(1 for x in near if x["same_sky_group"]))
_same = [x["sep_arcsec"] for x in near if x["same_sky_group"]]
N["negNearPosSepMin"] = "%.1f" % min(_same)
N["negNearPosSepMax"] = "%.1f" % max(_same)
# candidates
cpairs = self_pairs(cand[["ra", "dec"]].reset_index(drop=True), 1.0)
N["candPairsOneArcsec"] = len(cpairs)
labelled = lab[lab.y.isin([0, 1])][["ra", "dec"]].reset_index(drop=True)
cn = 0
for i in range(len(cand)):
    s = angsep(
        float(cand.ra[i]),
        float(cand.dec[i]),
        labelled.ra.to_numpy(float),
        labelled.dec.to_numpy(float),
    )
    if (s < 1.0).any():
        cn += 1
N["candNearLabelledOneArcsec"] = cn


# ------------------------------------------------------------------ 4. paired-difference intervals
def paired_block(tag, a, b, c, dd):
    n = a + b + c + dd
    lo, hi = newcombe10(a, b, c, dd)
    wlo, whi = wald_paired(b, c, n)
    N[tag + "DiffPts"] = round(100 * (b - c) / n, 1)
    N[tag + "NewcombeLo"] = round(100 * lo, 1)
    N[tag + "NewcombeHi"] = round(100 * hi, 1)
    N[tag + "WaldLo"] = round(100 * wlo, 1)
    N[tag + "WaldHi"] = round(100 * whi, 1)
    N[tag + "N"] = n


ps = d[d.ispos]
a = int((ps.selm & ps.picked).sum())
b = int((ps.selm & ~ps.picked).sum())
c = int((~ps.selm & ps.picked).sum())
dd = int((~ps.selm & ~ps.picked).sum())
assert (a, b, c) == (103, 21, 3), (a, b, c)
paired_block("pairedPrimary", a, b, c, dd)
sb = ps.sel_burden.astype(bool)
paired_block(
    "pairedDeployed",
    int((sb & ps.picked).sum()),
    int((sb & ~ps.picked).sum()),
    int((~sb & ps.picked).sum()),
    int((~sb & ~ps.picked).sum()),
)
u8 = ps.picked | ps.sel_barro24b
# region-matched to the union of eight: top K8[region]
K8 = {reg: int(((d.region == reg) & (d.picked | d.sel_barro24b)).sum()) for reg in NU}
d["selm8"] = False
for reg, k in K8.items():
    idx = d[d.region == reg].sort_values("score_mean", ascending=False).index[:k]
    d.loc[idx, "selm8"] = True
ps = d[d.ispos]
u8 = ps.picked | ps.sel_barro24b
paired_block(
    "pairedEight",
    int((ps.selm8 & u8).sum()),
    int((ps.selm8 & ~u8).sum()),
    int((~ps.selm8 & u8).sum()),
    int((~ps.selm8 & ~u8).sum()),
)
sc = ps[ps.in_hviding25_A1.astype(bool) | ps.in_degraaff26.astype(bool)]
paired_block(
    "pairedSpecCrit",
    int((sc.selm & sc.picked).sum()),
    int((sc.selm & ~sc.picked).sum()),
    int((~sc.selm & sc.picked).sum()),
    int((~sc.selm & ~sc.picked).sum()),
)
N["nPosSpecCritSupport"] = len(sc)
# the comparison-object axis, paired
ng = d[d.isneg]
nb = int((ng.selm & ~ng.picked).sum())
nc = int((~ng.selm & ng.picked).sum())
N["mcnemarAnchorsB"] = nb
N["mcnemarAnchorsC"] = nc
N["mcnemarAnchorsP"] = pfmt(mcnemar_p(nb, nc))
N["anchorsBoth"] = int((ng.selm & ng.picked).sum())

# ------------------------------------------------------------------ 5. compactness cross-tab on the 147
PS = P.merge(
    pap[["source_id", "labbe_compactness", "measurable"]], on="source_id", how="left"
)
PS = PS[PS.in_support.fillna(False).astype(bool)]
cat_c = PS.r_h_arcsec < 1.5 * PS.r_star_v1_arcsec
lab_c = PS.labbe_compactness < 1.7
N["posCatalogCompactAndLabbeSupport"] = int((cat_c & lab_c).sum())
N["posCatalogCompactNotLabbeSupport"] = int((cat_c & ~lab_c).sum())
N["posLabbeNotCatalogCompactSupport"] = int((~cat_c & lab_c).sum())
N["posNeitherCompactSupport"] = int((~cat_c & ~lab_c).sum())
N["posCatalogCompactSupport"] = int(cat_c.sum())
N["posLabbeCompactSupportCheck"] = int(lab_c.sum())

# ------------------------------------------------------------------ 6. shuffled controls at both operating points
rm_own, rm_mat, an_own, an_mat = [], [], [], []
rmiss = d.ispos & ~d.picked
for i in range(5):
    so, sm = "sel_shuffled_labels_%d" % i, "selmatched_shuffled_labels_%d" % i
    rm_own.append(int((d[so].astype(bool) & rmiss).sum()))
    rm_mat.append(int((d[sm].astype(bool) & rmiss).sum()))
    an_own.append(int((d[so].astype(bool) & d.isneg).sum()))
    an_mat.append(int((d[sm].astype(bool) & d.isneg).sum()))
N["shuffledMissedOwnMin"], N["shuffledMissedOwnMax"] = min(rm_own), max(rm_own)
N["shuffledMissedMatchedMin"], N["shuffledMissedMatchedMax"] = min(rm_mat), max(rm_mat)
N["shuffledAnchorsOwnMin"], N["shuffledAnchorsOwnMax"] = min(an_own), max(an_own)
N["shuffledAnchorsMatchedMin"], N["shuffledAnchorsMatchedMax"] = (
    min(an_mat),
    max(an_mat),
)

# ------------------------------------------------------------------ 7. the sixteen disagreements, the record-only column
dis = aud[aud.selected_record != aud.selected_rerun]
N["auditDisagreeTotalCheck"] = len(dis)
by_rule = dis.rule.value_counts().to_dict()
RTAG = {
    "labbe23": "Labbe",
    "kokorev24": "Kokorev",
    "kocevski24": "Kocevski",
    "perezgonzalez24": "PerezGonzalez",
    "barro23": "Barro",
    "greene24": "Greene",
    "akins24": "Akins",
}
for r, t in RTAG.items():
    N["auditDisagree" + t] = int(by_rule.get(r, 0))
    sub = aud[aud.rule == r]
    N["auditRecordOnly" + t] = int(sub.outcome.astype(str).str.startswith("none").sum())
    N["auditRowsSum" + t] = len(sub)
N["auditDisagreeRerunGains"] = int((dis.selected_rerun & ~dis.selected_record).sum())
N["auditDisagreeRerunLoses"] = int((~dis.selected_rerun & dis.selected_record).sum())
audit["disagreements"] = dis[
    ["rule", "field", "id", "selected_record", "selected_rerun", "outcome", "note"]
].to_dict("records")
gain = dis[dis.selected_rerun & ~dis.selected_record]
audit["outcome_values"] = sorted(aud.outcome.astype(str).unique().tolist())
# which object the rerun union gains and through which rule
rec_union = aud.groupby("source_id").selected_record.any()
rer_union = aud.groupby("source_id").selected_rerun.any()
gained = rer_union[rer_union & ~rec_union.reindex(rer_union.index).fillna(False)]
audit["rerun_union_gain"] = [
    {
        "source_id": int(s),
        "rules": aud[
            (aud.source_id == s) & aud.selected_rerun & ~aud.selected_record
        ].rule.tolist(),
    }
    for s in gained.index
]
N["auditRerunUnionGainN"] = len(gained)
if len(gained):
    g0 = aud[aud.source_id == gained.index[0]].iloc[0]
    N["auditRerunGainObject"] = "%s %d" % (g0.field, g0.id)
    N["auditRerunGainRule"] = ", ".join(audit["rerun_union_gain"][0]["rules"])

# ------------------------------------------------------------------ 8. the 48 used in neither class
amb48 = prov[prov.final_class.astype(str).str.startswith("ambiguous")]
N["ambNeitherTotal"] = len(amb48)
in_phot = amb48.in_kocevski24.astype(bool) | amb48.in_perger25.astype(bool)
N["ambNeitherPhotList"] = int(in_phot.sum())
N["ambNeitherBOneNoSpectrum"] = int(
    (~in_phot & amb48.in_hviding25_B1.astype(bool)).sum()
)
N["ambNeitherOther"] = int((~in_phot & ~amb48.in_hviding25_B1.astype(bool)).sum())
amb_ids_phot = set(amb48[in_phot].source_id)
amb_ids_b1 = set(amb48[~in_phot & amb48.in_hviding25_B1.astype(bool)].source_id)
ap = d[d.source_id.isin(amb_ids_phot)]
ab = d[d.source_id.isin(amb_ids_b1)]
N["ambPhotListSupport"] = len(ap)
N["ambBOneNoSpecSupport"] = len(ab)
N["ambPhotListUnion"] = int(ap.picked.sum())
N["ambPhotListMatched"] = int(ap.selm.sum())
N["ambBOneNoSpecUnion"] = int(ab.picked.sum())
N["ambBOneNoSpecMatched"] = int(ab.selm.sum())
neg_u = int((d.picked & d.isneg).sum())
neg_m = int((d.selm & d.isneg).sum())
N["sensDenPhotList"] = 4979 + len(ap)
N["sensRateUnionPhotList"] = round(
    100 * (neg_u + N["ambPhotListUnion"]) / N["sensDenPhotList"], 2
)
N["sensRateMatchedPhotList"] = round(
    100 * (neg_m + N["ambPhotListMatched"]) / N["sensDenPhotList"], 2
)
# the B1 broad-line comparison objects that carry the ambiguous flag (already y = 0)
b1c = d[d.ambiguous.astype(bool) & d.isneg]
N["ambBOneComparisonSupport"] = len(b1c)

# ------------------------------------------------------------------ 9. the V-shape ledger
pv = prov[prov.final_class == "spectroscopic LRD"]
N["ledgerPosRowTested"] = int(pv.test_evaluable_row.astype(bool).sum())
N["ledgerPosRowVshaped"] = int((pv.verdict_row.astype(str) == "True").sum())
N["ledgerPosSourceTested"] = int(pv.test_evaluable_source.astype(bool).sum())
N["ledgerPosSourceVshaped"] = int((pv.verdict_source.astype(str) == "True").sum())
audit["verdict_row_values"] = sorted(pv.verdict_row.astype(str).unique().tolist())
audit["verdict_source_values"] = sorted(pv.verdict_source.astype(str).unique().tolist())
pv2 = pv.merge(P[["source_id", "picked"]], on="source_id", how="left")
for lvl, tcol, vcol in (
    ("Row", "test_evaluable_row", "verdict_row"),
    ("Source", "test_evaluable_source", "verdict_source"),
):
    t = pv2[pv2[tcol].astype(bool)]
    vs = t[vcol].astype(str) == "True"
    N["ledgerHitTested" + lvl] = int(t.picked.sum())
    N["ledgerHitVshaped" + lvl] = int((vs & t.picked).sum())
    N["ledgerMissTested" + lvl] = int((~t.picked).sum())
    N["ledgerMissVshaped" + lvl] = int((vs & ~t.picked).sum())

# ------------------------------------------------------------------ 10. the Kocevski catalog rows
kr = lab[lab.in_kocevski24.astype(bool)]
N["kocCatRows"] = len(kr)
N["kocCatRowsSelected"] = int(kr.sel_kocevski24.astype(bool).sum())
N["kocCatRowsNoZphot"] = int(kr.z_phot.isna().sum())
N["kocCatRowsRejected"] = len(kr) - N["kocCatRowsSelected"] - N["kocCatRowsNoZphot"]
krs = kr.merge(fea[["source_id", "in_support"]], on="source_id")
N["kocCatRowsSupport"] = int(krs.in_support.sum())
N["kocCatRowsSupportSelected"] = int(
    (krs.in_support & krs.sel_kocevski24.astype(bool)).sum()
)

# ------------------------------------------------------------------ 11. stellar radius spread
rs = fea[fea.in_support].groupby("field").r_star_v1_arcsec.first()
N["rStarFieldMin"] = round(float(rs.min()), 4)
N["rStarFieldMax"] = round(float(rs.max()), 4)
N["rStarSpreadDex"] = round(float(np.log10(rs.max() / rs.min())), 3)
audit["r_star_by_field"] = {k: round(float(v), 4) for k, v in rs.items()}

# ------------------------------------------------------------------ 12. Skyfire Table 3
rows = []
for line in open(SKYFIRE, encoding="utf-8"):
    if line.startswith("#") or not line.strip():
        continue
    parts = line.split()
    if len(parts) < 9:
        continue
    try:
        rows.append(
            {
                "skyfire_id": parts[0],
                "ra": float(parts[1]),
                "dec": float(parts[2]),
                "z": float(parts[3]),
                "line": parts[-2],
                "lrd": int(parts[-1]),
            }
        )
    except ValueError:
        continue
sk = pd.DataFrame(rows)
N["skyfireBroadLine"] = len(sk)
N["skyfireLrd"] = int((sk.lrd == 1).sum())
pc = pos_all.merge(
    fea[["source_id", "field"]], on="source_id", how="left", suffixes=("", "_f")
)
sk_pos, sk_cand, sk_lrd_pos, sk_lrd_cand, sk_new = [], [], [], [], []
for i in range(len(sk)):
    s = angsep(
        sk.ra[i], sk.dec[i], pos_all.ra.to_numpy(float), pos_all.dec.to_numpy(float)
    )
    hit = np.where(s < 1.0)[0]
    sc_ = angsep(sk.ra[i], sk.dec[i], cand.ra.to_numpy(float), cand.dec.to_numpy(float))
    chit = np.where(sc_ < 1.0)[0]
    entry = {"skyfire_id": sk.skyfire_id[i], "z": sk.z[i], "lrd": int(sk.lrd[i])}
    if len(hit):
        entry["positive"] = "%s %d" % (pos_all.field[hit[0]], pos_all.id[hit[0]])
        sk_pos.append(entry)
        if sk.lrd[i] == 1:
            sk_lrd_pos.append(entry)
    if len(chit):
        entry["candidate_rank"] = int(cand["rank"][chit[0]])
        entry["candidate"] = "%s %d" % (cand.field[chit[0]], cand.id[chit[0]])
        sk_cand.append(entry)
        if sk.lrd[i] == 1:
            sk_lrd_cand.append(entry)
    if sk.lrd[i] == 1 and not len(hit):
        sk_new.append(entry)
N["skyfireInPositives"] = len(sk_pos)
N["skyfireLrdInPositives"] = len(sk_lrd_pos)
N["skyfireLrdNotInPositives"] = len(sk_new)
N["skyfireInCandidates"] = len(sk_cand)
N["skyfireCandRanks"] = " and ".join(str(c) for c in sorted(x["candidate_rank"] for x in sk_cand))
N["skyfireLrdInCandidates"] = len(sk_lrd_cand)
N["skyfireBroadLineNotLrdInCandidates"] = int(sum(1 for e in sk_cand if e["lrd"] == 0))
# are the new Skyfire LRDs in our catalog at all, and what do the selections say about them
newrows = []
cat_all = lab[["source_id", "field", "id", "ra", "dec", "picked"]].merge(
    fea[["source_id", "in_support", "region"]], on="source_id", how="left"
)
ceers = (
    cat_all[cat_all.region == "CEERS"]
    if "CEERS" in set(cat_all.region.dropna())
    else cat_all[cat_all.field.astype(str).str.contains("ceers")]
)
for e in sk_new:
    r = sk[sk.skyfire_id == e["skyfire_id"]].iloc[0]
    s = angsep(r.ra, r.dec, ceers.ra.to_numpy(float), ceers.dec.to_numpy(float))
    j = int(np.argmin(s))
    m = dict(e)
    m["nearest_catalog_row"] = "%s %d" % (ceers.field.iloc[j], ceers.id.iloc[j])
    m["sep_arcsec"] = round(float(s[j]), 2)
    if s[j] < 1.0:
        m["in_support"] = bool(ceers.in_support.iloc[j])
        m["union_selects"] = bool(ceers.picked.iloc[j])
        oo = d[d.source_id == ceers.source_id.iloc[j]]
        m["region_matched_selects"] = bool(oo.selm.iloc[0]) if len(oo) else None
        m["deployed_selects"] = bool(oo.sel_burden.iloc[0]) if len(oo) else None
    newrows.append(m)
audit["skyfire_lrds_not_in_positives"] = newrows
audit["skyfire_in_positives"] = sk_pos
audit["skyfire_in_candidates"] = sk_cand
inrow = [m for m in newrows if m["sep_arcsec"] < 1.0]
N["skyfireNewLrdInCatalog"] = len(inrow)
N["skyfireNewLrdInSupport"] = int(sum(1 for m in inrow if m.get("in_support")))
N["skyfireNewLrdUnionSelects"] = int(sum(1 for m in inrow if m.get("union_selects")))
N["skyfireNewLrdMatchedSelects"] = int(
    sum(1 for m in inrow if m.get("region_matched_selects"))
)
N["skyfireNewLrdDeployedSelects"] = int(
    sum(1 for m in inrow if m.get("deployed_selects"))
)

# ------------------------------------------------------------------ 14. deployed against out-of-fold, within region
sc_ = pd.read_parquet(os.path.join(V, "scores.parquet"), columns=["source_id", "region", "ranking_score"])
dv = d[["source_id", "region", "score_mean"]].merge(sc_[["source_id", "ranking_score"]], on="source_id", how="left")
fin = json.load(open(os.path.join(V, "final.json")))
t05 = fin.get("t_05", fin.get("thresholds", {}).get("t_05"))
ov_num, ov_den, rhos = 0, 0, {}
from scipy.stats import spearmanr
for reg, g in dv.groupby("region"):
    k = int((g.ranking_score >= t05).sum()) if t05 is not None else int(round(0.005 * len(g)))
    top_fin = set(g.sort_values("ranking_score", ascending=False).source_id.head(k))
    top_oof = set(g.sort_values("score_mean", ascending=False).source_id.head(k))
    ov_num += len(top_fin & top_oof)
    ov_den += k
    g1 = g[g.source_id.isin(top_fin)]
    rhos[reg] = round(float(spearmanr(g1.ranking_score, g1.score_mean).correlation), 3)
N["topsetOverlapRegional"] = ov_num
N["topsetOverlapRegionalDen"] = ov_den
N["topsetOverlapRegionalPct"] = round(100 * ov_num / max(ov_den, 1), 1)
N["spearmanTopRegionMin"] = min(rhos.values())
N["spearmanTopRegionMax"] = max(rhos.values())
audit["spearman_top_by_region"] = rhos
audit["t05_final"] = t05

# ------------------------------------------------------------------ 13. freeze chronology (dates the inputs were obtained)
N["freezeListsDate"] = "30 August 2026"
N["freezeDeGraaffDownloadDate"] = "2 September 2026"
N["freezeArchiveIndexDate"] = "1 September 2026"
N["freezeLabelTableDate"] = "3 September 2026"
N["barroListArxivDate"] = "17 December 2025"
N["dgListArxivDate"] = "26 November 2025"

# ------------------------------------------------------------------ write
json.dump(
    {"numbers": N, "audit": audit},
    open(os.path.join(R10, "round10_numbers.json"), "w"),
    indent=1,
    default=str,
)
WORD = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def latex_num(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return "{:,}".format(int(v)).replace(",", "{,}")
    if isinstance(v, float):
        return "%g" % v
    return str(v)


BEGIN = "% ---- revision 2026-09-07 (r7 referees, round 10): begin ----"
END = "% ---- revision 2026-09-07 (r7 referees, round 10): end ----"
for _path in (NUMBERS_TEX, NUMBERS_TEX_REC):
    txt = open(_path, encoding="utf-8").read()
    txt = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END) + "\n?", "", txt, flags=re.S)
    _existing = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", txt))
    _lines = [BEGIN, "% source: paper/round10_compute_2026_09_07.py"]
    for k, v in N.items():
        name = k
        for dch, word in zip("0123456789", WORD):
            name = name.replace(dch, word)
        name = re.sub(r"[^A-Za-z]", "", name)
        cmd = "renewcommand" if name in _existing else "newcommand"
        _lines.append("\\%s{\\%s}{%s}" % (cmd, name, latex_num(v)))
    _lines.append(END)
    open(_path, "w", encoding="utf-8").write(
        txt.rstrip("\n") + "\n" + "\n".join(_lines) + "\n"
    )
print(len(N), "numbers written")
for k in (
    "bPhotMembersPos",
    "bModuleOnPhotMembers",
    "bColorsOnPhotMembers",
    "bModuleRuleMissed",
    "bMembershipRuleMissed",
    "bModuleAnchorsSupport",
    "sevenPlusMembershipRecall",
    "kocCatPos",
    "kocModuleOnCatPos",
    "kocCatPosUnavailable",
    "posPairsOneArcsec",
    "negNearPosOneArcsec",
    "candPairsOneArcsec",
    "candNearLabelledOneArcsec",
    "nPosMerged",
    "unionRecallMerged",
    "matchedRecallMerged",
    "nRuleMissedMerged",
    "matchedRuleMissedMerged",
    "pairedPrimaryNewcombeLo",
    "pairedPrimaryNewcombeHi",
    "pairedPrimaryWaldLo",
    "pairedPrimaryWaldHi",
    "mcnemarAnchorsB",
    "mcnemarAnchorsC",
    "mcnemarAnchorsP",
    "posCatalogCompactAndLabbeSupport",
    "posCatalogCompactNotLabbeSupport",
    "posLabbeNotCatalogCompactSupport",
    "shuffledMissedOwnMin",
    "shuffledMissedOwnMax",
    "shuffledMissedMatchedMin",
    "shuffledMissedMatchedMax",
    "auditDisagreeLabbe",
    "auditDisagreeKokorev",
    "auditDisagreeKocevski",
    "auditDisagreePerezGonzalez",
    "auditRerunGainObject",
    "auditRerunGainRule",
    "ambNeitherPhotList",
    "ambNeitherBOneNoSpectrum",
    "ambNeitherOther",
    "sensRateUnionPhotList",
    "sensRateMatchedPhotList",
    "ledgerPosRowTested",
    "ledgerPosRowVshaped",
    "ledgerPosSourceTested",
    "ledgerPosSourceVshaped",
    "kocCatRows",
    "kocCatRowsSelected",
    "kocCatRowsNoZphot",
    "kocCatRowsRejected",
    "rStarFieldMin",
    "rStarFieldMax",
    "rStarSpreadDex",
    "skyfireBroadLine",
    "skyfireLrd",
    "skyfireInPositives",
    "skyfireLrdInPositives",
    "skyfireLrdNotInPositives",
    "skyfireInCandidates",
    "skyfireNewLrdInCatalog",
    "skyfireNewLrdUnionSelects",
    "skyfireNewLrdMatchedSelects",
):
    print("%-36s %s" % (k, N.get(k)))
print(json.dumps(audit["disagreements"], indent=0)[:1500])
print(json.dumps(audit["skyfire_lrds_not_in_positives"], indent=0)[:1500])
print(
    json.dumps(audit["comparison_rows_within_1arcsec_of_a_positive"], indent=0)[:1200]
)
