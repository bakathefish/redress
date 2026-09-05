"""Round 5, numbers B: subset counts and two tests, read from the tables of record.

Every quantity here is a variant of one paper/build_numbers.py already computes, so
every construction below is that file's own code path, copied verbatim and annotated with the
line it is copied from. Nothing in build_numbers.py, numbers.json, numbers.tex or recovery
is touched; this script only reads them and writes recovery/round5/subsets.json.

Run from the repository root:
    python paper/round5_subsets.py
"""

import json
import math
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths (build_numbers.py 23-26)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "recovery")
OUTDIR = os.path.join(ROOT, "recovery", "round5")
os.chdir(ROOT)
os.makedirs(OUTDIR, exist_ok=True)
REG = ["CEERS", "GOODS-N", "GOODS-S", "COSMOS", "UDS"]  # build_numbers.py 27


def J(f):  # build_numbers.py 31-32
    return json.load(open(os.path.join(V, f)))


def mcnemar_p(b, c):
    """Exact two-sided McNemar, verbatim from build_numbers.py 541-550."""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    tail = sum(math.comb(n, k) for k in range(m + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def pfmt(p):
    """p value formatter, verbatim from build_numbers.py 552-572."""
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


N = {}
P = {}  # provenance: key -> the build_numbers.py construction it reuses


def put(key, value, prov):
    N[key] = value
    P[key] = prov
    return value


# ---------------------------------------------------------------- tables of record
den = J("denominators.json")  # build_numbers.py 65
lab = pd.read_parquet(os.path.join(V, "labels.parquet"))  # build_numbers.py 95
f = pd.read_parquet(os.path.join(V, "features.parquet"))  # build_numbers.py 134
sup = f[f.in_support.astype(bool)].copy()  # build_numbers.py 135
posS = sup[sup.y == 1]  # build_numbers.py 136
negS = sup[sup.y == 0]  # build_numbers.py 137
posAll = f[f.y == 1]  # build_numbers.py 138
rules = [  # build_numbers.py 139-147
    "labbe23",
    "kokorev24",
    "kocevski24",
    "perezgonzalez24",
    "barro23",
    "greene24",
    "akins24",
]
oof = pd.read_parquet(os.path.join(V, "oof_scores.parquet")).set_index(
    "source_id"
)  # build_numbers.py 277
bu = J("baseline_burden.json")  # build_numbers.py 1116
_bof = pd.read_parquet(os.path.join(V, "baseline_oof_full.parquet")).set_index(
    "source_id"
)  # build_numbers.py 1768-1770
_x = pd.read_parquet(os.path.join(V, "published_catalogues_extra.parquet")).set_index(
    "source_id"
)  # build_numbers.py 1591-1593

# the record's own values this script reproduces or extends, read not typed
nPos = den["positives_151"]
nPosSupport = den["positives_in_support_147"]
nRuleMissed = den["rule_missed_44"]
unionSelected = den["union_selected_in_support"]  # build_numbers.py 77

# the union column is exactly the OR of the seven rule flags; asserted, not assumed
_u7 = f[["sel_" + r for r in rules]].astype(bool).any(axis=1)
assert (_u7 == f.picked.astype(bool)).all(), "picked is not the OR of the seven rules"
assert int(sup.picked.astype(bool).sum()) == unionSelected, (
    "the in-support union row count disagrees with denominators.json"
)

# =========================================================================================
# (a) Hviding et al. (2025) table A1 in our fields, whole and magnitude-limited
# Construction: hvA = f[f.in_hviding25_A1.astype(bool)] (build_numbers.py 622), and the three
# module counts kocevskiModuleOfHviding / kokorevModuleOfHviding / barroModuleOfHviding
# (build_numbers.py 624, 628, 629). "Brighter than 26.5" is mag_f444w < 26.5, the catalogue
# F444W magnitude column the paper's magnitude bins and medians use (build_numbers.py 284).
# =========================================================================================
hvA = f[f.in_hviding25_A1.astype(bool)]
put("hvA1Rows", int(len(hvA)), "build_numbers.py 622 (hvA), = listHvidingA")
put(
    "hvA1SkyGroups",
    int(hvA.sky_group.nunique()),
    "hvA.sky_group.nunique(); the sky-group dedup of build_numbers.py 639",
)
put("hvA1InSupport", int(hvA.in_support.astype(bool).sum()), "hvA.in_support")
put("hvA1Positives", int((hvA.y == 1).sum()), "hvA.y == 1")
_hvbright = hvA[hvA.mag_f444w < 26.5]
put(
    "hvA1BrightN",
    int(len(_hvbright)),
    "hvA[hvA.mag_f444w < 26.5]; mag_f444w as in build_numbers.py 284",
)
put(
    "hvA1BrightNInclusive",
    int((hvA.mag_f444w <= 26.5).sum()),
    "hvA[hvA.mag_f444w <= 26.5]; emitted so the boundary convention is visible",
)
put(
    "hvA1MagFinite",
    int(np.isfinite(hvA.mag_f444w).sum()),
    "np.isfinite(hvA.mag_f444w)",
)
for _tag, _col in (
    ("Kocevski", "sel_kocevski24"),
    ("Kokorev", "sel_kokorev24"),
    ("Barro", "sel_barro23"),
):
    put(
        "hvA1Bright" + _tag,
        int(_hvbright[_col].astype(bool).sum()),
        f"build_numbers.py 624/628/629 construction on the mag < 26.5 subset ({_col})",
    )
    put(
        "hvA1All" + _tag,
        int(hvA[_col].astype(bool).sum()),
        f"build_numbers.py 624/628/629 verbatim ({_col})",
    )
put(
    "hvA1BrightUnion",
    int(_hvbright.picked.astype(bool).sum()),
    "hvA[mag<26.5].picked; picked is the union of the seven (asserted above)",
)
put("hvA1AllUnion", int(hvA.picked.astype(bool).sum()), "hvA.picked")

# =========================================================================================
# (b) the bright positives, and the bright positives on the Pan et al. (2026) footing
# Construction: posAll (build_numbers.py 138) for the end-to-end denominator; picked for the
# union; the equal-burden out-of-fold selection through the reindex of build_numbers.py 1327,
# which is the end-to-end reading (a positive outside the support carries no out-of-fold
# score and cannot be selected). z_spec comes from the round-3 join of build_numbers.py
# 1591-1596.
# =========================================================================================
_R = posAll.set_index("source_id").join(_x, how="left")  # build_numbers.py 1594
assert len(_R) == nPos, "the round-3 join changed the positive count"
_z = _R.z_spec  # build_numbers.py 1596
assert _z.notna().all(), "a positive has no spectroscopic redshift"
_selb = (
    oof["sel_burden"].reindex(posAll.source_id.values).eq(True).to_numpy(bool)
)  # build_numbers.py 1327
_pick = posAll.picked.to_numpy(bool)
_mag = posAll.mag_f444w.to_numpy(float)
_zv = _z.reindex(posAll.source_id.values).to_numpy(float)
put(
    "posMagFinite",
    int(np.isfinite(_mag).sum()),
    "np.isfinite(posAll.mag_f444w); the 151 positives",
)


def _bright_block(mask, tag, prov):
    put("nPos" + tag, int(mask.sum()), prov)
    put(
        "unionRecall" + tag,
        int((mask & _pick).sum()),
        "posAll.picked on that subset (end to end)",
    )
    put(
        "nRuleMissed" + tag,
        int((mask & ~_pick).sum()),
        "~posAll.picked on that subset (end to end)",
    )
    put(
        "modelRecall" + tag,
        int((mask & _selb).sum()),
        "build_numbers.py 1327 sel_burden reindex on that subset (end to end)",
    )
    put(
        "nPosSupport" + tag,
        int((mask & posAll.in_support.to_numpy(bool)).sum()),
        "posAll.in_support on that subset",
    )


_b26 = _mag < 26.0
_bright_block(_b26, "BrightTwentySix", "posAll[mag_f444w < 26]")
# the Pan et al. (2026) footing: the same magnitude limit inside 2.3 < z_spec < 7.4, both
# bounds strict, on the spectroscopic redshift of build_numbers.py 1596.
_pan = _b26 & (_zv > 2.3) & (_zv < 7.4)
_bright_block(_pan, "BrightPan", "posAll[mag_f444w < 26 and 2.3 < z_spec < 7.4]")
put(
    "nPosPanZonly",
    int(((_zv > 2.3) & (_zv < 7.4)).sum()),
    "the redshift window alone, no magnitude limit",
)

# =========================================================================================
# (c) the union without kocevski24, and the union with the published catalogue in its place
# Constructions: the in-support row count as build_numbers.py 77 counts it (rows, asserted
# above); the outside-support rows as build_numbers.py 654-656 counts them; the pooled
# out-of-fold ranking at that burden exactly as build_numbers.py 647-652 builds
# modelAtUnionPlusCatsBurden; the sky-group count exactly as build_numbers.py 639 counts
# unionPlusCatsSelected; in_kocevski24 is the catalogue flag of build_numbers.py 586 and 616.
# =========================================================================================
_six = [r for r in rules if r != "kocevski24"]
_u6_f = f[["sel_" + r for r in _six]].astype(bool).any(axis=1).to_numpy()
_u6_sup = sup[["sel_" + r for r in _six]].astype(bool).any(axis=1).to_numpy()
_u6_posAll = posAll[["sel_" + r for r in _six]].astype(bool).any(axis=1).to_numpy()
_u6_posS = posS[["sel_" + r for r in _six]].astype(bool).any(axis=1).to_numpy()
_u6_negS = negS[["sel_" + r for r in _six]].astype(bool).any(axis=1).to_numpy()

put(
    "unionSixSelected",
    int(_u6_sup.sum()),
    "in-support rows of the six-rule union, counted as build_numbers.py 77 counts unionSelected",
)
put(
    "unionSixOutsideSupportRows",
    int(_u6_f.sum()) - int(_u6_sup.sum()),
    "build_numbers.py 654-656 construction on the six-rule union",
)
put(
    "unionSixRecallEnd",
    int(_u6_posAll.sum()),
    "the six-rule union over posAll (build_numbers.py 138)",
)
put(
    "unionSixRecallSupport",
    int(_u6_posS.sum()),
    "the six-rule union over posS (build_numbers.py 136)",
)
put(
    "unionSixAnchors",
    int(_u6_negS.sum()),
    "the six-rule union over negS (build_numbers.py 137)",
)
put(
    "unionSixSelectedFewer",
    unionSelected - int(_u6_sup.sum()),
    "unionSelected (build_numbers.py 77) minus unionSixSelected",
)
# the pooled out-of-fold ranking at the six-rule union's in-support row count, built exactly
# as modelAtUnionPlusCatsBurden is built (build_numbers.py 647-652)
_ord6 = (
    sup.assign(_s=oof.loc[sup.source_id.values, "score_mean"].to_numpy())
    .sort_values("_s", ascending=False)
    .head(N["unionSixSelected"])
)
put(
    "modelAtUnionSixBurden",
    int((_ord6.y == 1).sum()),
    "build_numbers.py 647-652 verbatim at the six-rule union's row count",
)

# the union in which kocevski24 is replaced by published Kocevski et al. (2024) membership
_k_sup = sup.in_kocevski24.astype(bool).to_numpy()
_k_posAll = posAll.in_kocevski24.astype(bool).to_numpy()
_swap_sup = _u6_sup | _k_sup
put(
    "unionKocCatSelected",
    int(sup[_swap_sup].sky_group.nunique()),
    "build_numbers.py 639 verbatim (sky groups) with in_kocevski24 in place of sel_kocevski24",
)
put(
    "unionKocCatSelectedRows",
    int(_swap_sup.sum()),
    "the same set counted as rows, beside the sky-group count",
)
put(
    "unionKocCatRecallEnd",
    int((_u6_posAll | _k_posAll).sum()),
    "the swapped union over posAll (build_numbers.py 138)",
)
put(
    "unionKocCatRecallSupport",
    int((_u6_posS | posS.in_kocevski24.astype(bool).to_numpy()).sum()),
    "the swapped union over posS",
)
put(
    "unionSelectedSkyGroups",
    int(sup[sup.picked.astype(bool)].sky_group.nunique()),
    "the seven-rule union in support, counted one per sky group, for comparison",
)

# =========================================================================================
# (d) exact two-sided McNemar, eight-bag replica against the no-size ablation, both at
# matched burden, on the 147 in-support positives.
# The matched selections are the selmatched_* columns of baseline_oof_full.parquet, the same
# columns build_numbers.py 1779 reads for the leave-one-list-out models; blReplicaMatchedRecall
# and blAblateSizeMatchedRecall are read from baseline_burden.json at build_numbers.py 1134 and
# are asserted here against those columns. b and c follow the file's own convention
# (build_numbers.py 658-666): b = the first named selection only, c = the second only.
# =========================================================================================
_ps = posS.source_id.values
assert len(_ps) == nPosSupport, "the in-support positive set moved"
_rep = _bof.loc[_ps, "selmatched_primary_8bag_replica"].to_numpy(bool)
_nos = _bof.loc[_ps, "selmatched_ablate_size"].to_numpy(bool)
assert (
    int(_rep.sum())
    == bu["variants"]["primary_8bag_replica"]["matched"]["matched_recall_support"]
), "selmatched_primary_8bag_replica does not reproduce blReplicaMatchedRecall"
assert (
    int(_nos.sum())
    == bu["variants"]["ablate_size"]["matched"]["matched_recall_support"]
), "selmatched_ablate_size does not reproduce blAblateSizeMatchedRecall"
put(
    "blReplicaMatchedRecallCheck",
    int(_rep.sum()),
    "baseline_oof_full.parquet selmatched_primary_8bag_replica on posS; = blReplicaMatchedRecall",
)
put(
    "blAblateSizeMatchedRecallCheck",
    int(_nos.sum()),
    "baseline_oof_full.parquet selmatched_ablate_size on posS; = blAblateSizeMatchedRecall",
)
put(
    "mcnemarReplicaNoSizeB",
    int((_rep & ~_nos).sum()),
    "build_numbers.py 664 convention: replica only",
)
put(
    "mcnemarReplicaNoSizeC",
    int((_nos & ~_rep).sum()),
    "build_numbers.py 665 convention: no-size only",
)
put(
    "mcnemarReplicaNoSizeBoth",
    int((_rep & _nos).sum()),
    "concordant pairs, both selections",
)
put(
    "mcnemarReplicaNoSizeNeither",
    int((~_rep & ~_nos).sum()),
    "concordant pairs, neither selection",
)
_pRNS = mcnemar_p(N["mcnemarReplicaNoSizeB"], N["mcnemarReplicaNoSizeC"])
put(
    "mcnemarReplicaNoSizeP",
    pfmt(_pRNS),
    "build_numbers.py 541-572 (mcnemar_p then pfmt), as every McNemar macro is formatted",
)
put("mcnemarReplicaNoSizePRaw", float(_pRNS), "the same p value unrounded")
assert (
    N["mcnemarReplicaNoSizeB"]
    + N["mcnemarReplicaNoSizeC"]
    + N["mcnemarReplicaNoSizeBoth"]
    + N["mcnemarReplicaNoSizeNeither"]
    == nPosSupport
), "the McNemar table does not close on the 147 in-support positives"
assert (
    N["blReplicaMatchedRecallCheck"] - N["blAblateSizeMatchedRecallCheck"]
    == N["mcnemarReplicaNoSizeB"] - N["mcnemarReplicaNoSizeC"]
), "the discordant counts do not reproduce the difference in matched recall"

# =========================================================================================
# (e) the Bonferroni family with the three Mann-Whitney tests of Section 3.5 added
# The headline test is the primary McNemar of build_numbers.py 658-668; the corrected value is
# min(1, family * p) rounded through pfmt exactly as build_numbers.py 1450-1456 rounds
# bonferroniHeadlineP. The three tests added are colourMWUp (build_numbers.py 1030), magMWUp
# (1039) and colourMWUpCompact (1966), which are the colour, the magnitude and the colour on
# the compact subset.
# =========================================================================================
_Pp = posS.copy()  # build_numbers.py 658-663
_Pp["sel_burden"] = oof.loc[_Pp.source_id.values, "sel_burden"].to_numpy()
_m = _Pp.sel_burden.to_numpy(bool)
_u = _Pp.picked.to_numpy(bool)
put("mcnemarModelUnionBCheck", int((_m & ~_u).sum()), "build_numbers.py 664 verbatim")
put("mcnemarModelUnionCCheck", int((_u & ~_m).sum()), "build_numbers.py 665 verbatim")
_headline = mcnemar_p(N["mcnemarModelUnionBCheck"], N["mcnemarModelUnionCCheck"])
put(
    "bonferroniHeadlineRawP",
    pfmt(_headline),
    "build_numbers.py 541-572 on the headline test",
)
put("bonferroniHeadlineRawPExact", float(_headline), "the same p value unrounded")
put("bonferroniFamilyNine", 9, "build_numbers.py 1449, the family the paper prints")
put(
    "bonferroniHeadlinePNine",
    pfmt(min(1.0, 9 * _headline)),
    "build_numbers.py 1450-1456 verbatim, family 9",
)
put(
    "bonferroniFamilyTwelve",
    12,
    "family 9 plus the three Mann-Whitney tests of Section 3.5",
)
put(
    "bonferroniHeadlinePTwelve",
    pfmt(min(1.0, 12 * _headline)),
    "build_numbers.py 1450-1456 construction with the family set to 12",
)
put(
    "bonferroniHeadlinePTwelveExact",
    float(min(1.0, 12 * _headline)),
    "the same corrected p value unrounded",
)
# the word form, as build_numbers.py 1457 emits bonferroniFamilyWord. The WORD table of
# build_numbers.py 28 stops at nine, so twelve is spelled here.
put(
    "bonferroniFamilyTwelveWord",
    "twelve",
    "the word form of 12, matching the lower case of build_numbers.py 1457",
)

# =========================================================================================
# (h) exact two-sided McNemar, the anchor-weight-0.5 run against the deployed primary, both
# at matched burden, on the 147 in-support positives.
# The 0.5 run's matched selection is selmatched_anchor_0.50 in baseline_oof_full.parquet, the
# column behind blAnchorHalfMatchedRecall (build_numbers.py 1134). The deployed primary has no
# selmatched_ column, because its matched selection is the per-region top-k of the pooled
# out-of-fold score that build_numbers.py 681-694 already builds for matchedRegion*; that
# construction is copied verbatim below and asserted against both the per-region matched
# recalls in baseline_burden.json and primaryMatchedRecall. b and c follow the file's
# convention at lines 664-665: b = the first named selection only, c = the second only.
# =========================================================================================
_supm = sup.assign(
    _s=oof.loc[sup.source_id.values, "score_mean"].to_numpy()
)  # build_numbers.py 681
_pm_ids = set()
_pm_region = {}
for _r in REG:  # build_numbers.py 683-690
    _g = _supm[_supm.region == _r].sort_values("_s", ascending=False)
    _head = _g.head(den["union_in_support_per_region"][_r])
    _pm_ids.update(_head.source_id.tolist())
    _pm_region[_r] = int((_head.y == 1).sum())
assert _pm_region == bu["primary_48bag_matched"]["matched_recall_per_region"], (
    "the per-region top-k does not reproduce the primary's matched recall per region"
)
_prim = posS.source_id.isin(_pm_ids).to_numpy(bool)
assert int(_prim.sum()) == bu["primary_48bag_matched"]["matched_recall_support"], (
    "the per-region top-k does not reproduce primaryMatchedRecall"
)
_ah = _bof.loc[_ps, "selmatched_anchor_0.50"].to_numpy(bool)
assert (
    int(_ah.sum()) == bu["variants"]["anchor_0.50"]["matched"]["matched_recall_support"]
), "selmatched_anchor_0.50 does not reproduce blAnchorHalfMatchedRecall"
put(
    "blAnchorHalfMatchedRecallCheck",
    int(_ah.sum()),
    "baseline_oof_full.parquet selmatched_anchor_0.50 on posS; = blAnchorHalfMatchedRecall",
)
put(
    "primaryMatchedRecallCheck",
    int(_prim.sum()),
    "build_numbers.py 681-694 per-region top-k on posS; = primaryMatchedRecall",
)
put(
    "mcnemarAnchorHalfPrimaryB",
    int((_ah & ~_prim).sum()),
    "build_numbers.py 664 convention: the anchor-0.5 run only",
)
put(
    "mcnemarAnchorHalfPrimaryC",
    int((_prim & ~_ah).sum()),
    "build_numbers.py 665 convention: the deployed primary only",
)
put(
    "mcnemarAnchorHalfPrimaryBoth",
    int((_ah & _prim).sum()),
    "concordant pairs, both selections",
)
put(
    "mcnemarAnchorHalfPrimaryNeither",
    int((~_ah & ~_prim).sum()),
    "concordant pairs, neither selection",
)
_pAHP = mcnemar_p(N["mcnemarAnchorHalfPrimaryB"], N["mcnemarAnchorHalfPrimaryC"])
put(
    "mcnemarAnchorHalfPrimaryP",
    pfmt(_pAHP),
    "build_numbers.py 541-572 (mcnemar_p then pfmt), as every McNemar macro is formatted",
)
put("mcnemarAnchorHalfPrimaryPRaw", float(_pAHP), "the same p value unrounded")
assert (
    N["mcnemarAnchorHalfPrimaryB"]
    + N["mcnemarAnchorHalfPrimaryC"]
    + N["mcnemarAnchorHalfPrimaryBoth"]
    + N["mcnemarAnchorHalfPrimaryNeither"]
    == nPosSupport
), "the McNemar table does not close on the 147 in-support positives"
assert (
    N["blAnchorHalfMatchedRecallCheck"] - N["primaryMatchedRecallCheck"]
    == N["mcnemarAnchorHalfPrimaryB"] - N["mcnemarAnchorHalfPrimaryC"]
), "the discordant counts do not reproduce the difference in matched recall"

# =========================================================================================
# (i) exact two-sided McNemar on the in-support positives that are not compact, learned
# selection at the deployed equal-burden thresholds against the union.
# The split is the r_h < 1.5 r_star criterion of build_numbers.py's F24 block, on the
# catalogue half-light radius against the frozen per-field stellar radius, the same split
# behind modelRecallNonCompact and unionRecallNonCompact. P is build_numbers.py 278-283.
# =========================================================================================
_Pi = sup.set_index("source_id")  # build_numbers.py 278
_Pi["sel_burden"] = oof.loc[_Pi.index, "sel_burden"].values  # build_numbers.py 279
_Pp2 = _Pi[_Pi.y == 1].copy()  # build_numbers.py 281
_Pn = _Pp2[
    ~(_Pp2.r_h_arcsec < 1.5 * _Pp2.r_star_v1_arcsec)
]  # the F24 non-compact split
put(
    "nPosNonCompactSupportCheck",
    int(len(_Pn)),
    "the F24 r_h < 1.5 r_star split; = nPosNonCompactSupport",
)
_nm = _Pn.sel_burden.to_numpy(bool)
_nu = _Pn.picked.to_numpy(bool)
put(
    "modelRecallNonCompactCheck",
    int(_nm.sum()),
    "sel_burden on the non-compact in-support positives; = modelRecallNonCompact",
)
put(
    "unionRecallNonCompactCheck",
    int(_nu.sum()),
    "picked on the non-compact in-support positives; = unionRecallNonCompact",
)
put(
    "mcnemarNonCompactB",
    int((_nm & ~_nu).sum()),
    "build_numbers.py 664 convention: the learned selection only",
)
put(
    "mcnemarNonCompactC",
    int((_nu & ~_nm).sum()),
    "build_numbers.py 665 convention: the union only",
)
put(
    "mcnemarNonCompactBoth",
    int((_nm & _nu).sum()),
    "concordant pairs, both selections",
)
put(
    "mcnemarNonCompactNeither",
    int((~_nm & ~_nu).sum()),
    "concordant pairs, neither selection",
)
_pNC = mcnemar_p(N["mcnemarNonCompactB"], N["mcnemarNonCompactC"])
put(
    "mcnemarNonCompactP",
    pfmt(_pNC),
    "build_numbers.py 541-572 (mcnemar_p then pfmt), as every McNemar macro is formatted",
)
put("mcnemarNonCompactPRaw", float(_pNC), "the same p value unrounded")
assert (
    N["mcnemarNonCompactB"]
    + N["mcnemarNonCompactC"]
    + N["mcnemarNonCompactBoth"]
    + N["mcnemarNonCompactNeither"]
    == N["nPosNonCompactSupportCheck"]
), "the McNemar table does not close on the non-compact in-support positives"

# =========================================================================================
# (j) the pooled out-of-fold ranking at exactly the union's own row count, 1133 rows.
# Built as modelAtUnionPlusCatsBurden is built (build_numbers.py 647-652), at unionSelected
# instead of unionPlusCatsSelected. One pooled threshold, so this is neither
# modelRecallSupport (per-fold thresholds) nor primaryMatchedRecall (per-region matching).
# =========================================================================================
_ord1133 = _supm.sort_values("_s", ascending=False).head(
    unionSelected
)  # build_numbers.py 647-651
put(
    "modelAtUnionBurdenPooled",
    int((_ord1133.y == 1).sum()),
    "build_numbers.py 647-652 verbatim at unionSelected (1133) rows",
)
put(
    "modelAtUnionBurdenPooledRuleMissed",
    int(((_ord1133.y == 1) & ~_ord1133.picked.astype(bool)).sum()),
    "rule-missed positives inside the same 1133 rows",
)
put(
    "modelAtUnionBurdenPooledAnchors",
    int((_ord1133.y == 0).sum()),
    "anchors inside the same 1133 rows",
)
# tie audit at the cut, so the head() is known to be a unique set and not an arbitrary one
_scut = float(_ord1133._s.min())
put("modelAtUnionBurdenPooledCutScore", _scut, "the pooled score at rank 1133")
put(
    "modelAtUnionBurdenPooledTiesAtCut",
    int((_supm._s == _scut).sum()),
    "in-support rows sharing the cut score; 1 means the top-1133 set is unique",
)
# ---------------------------------------------------------------- write
out = {"numbers": N, "provenance": P}
with open(os.path.join(OUTDIR, "subsets.json"), "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print(len(N), "numbers written to", os.path.join(OUTDIR, "subsets.json"))
for k in N:
    print(f"  {k} = {N[k]}")
