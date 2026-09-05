"""Revision compute of 2026-09-06 (referee rounds 7 and 8), from the tables of record; no retraining.

(a) compact-subset matched-burden evaluation (primary, imitation, union) on support rows with
    r_h < 1.5 r_star, top-K per held-out region with K = union rows in that compact subset;
(b) list-to-catalog match audit for the 151 positives: separation to the nearest list position,
    catalog rows within 0.5 arcsec of the list position, sky-group size and extent;
(c) out-of-fold score and fold threshold for every candidate (columns added to candidates.csv);
(d) release tables: rule_flags.parquet (every catalog row) and labelled_rows.csv (every labelled
    or ambiguous row with its flags and out-of-fold outcome);
(e) the Barro et al. (2024b) eighth comparator: colors from barro24b_colors.parquet, compactness
    from barro24b_apertures.csv (step 22), applied with redress.cuts.barro24b semantics; its
    recall, burden, anchors, overlap with the seven, the union of eight, and the learned
    selection against the union of eight at region-matched burden.

Reads the unreleased inputs/labels.parquet, inputs/features.parquet and inputs/oof_scores.parquet
(steps 1 to 3) and the three published list tables under inputs/prior_art_catalogues/; every
other input is a released file in recovery/. Writes recovery/referee_compute_2026_09_06.json
and the release tables named above.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "selections"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
V4 = "inputs"
REC = "recovery"
LISTS = os.path.join("inputs", "prior_art_catalogues")
OUT = "recovery"
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

lab = pd.read_parquet(os.path.join(V4, "labels.parquet"))
fea = pd.read_parquet(
    os.path.join(V4, "features.parquet"),
    columns=[
        "source_id",
        "region",
        "sky_group",
        "sky_group_size",
        "in_support",
        "r_h_arcsec",
        "r_star_v1_arcsec",
    ],
)
oof = pd.read_parquet(os.path.join(V4, "oof_scores.parquet"))
base = pd.read_parquet(
    os.path.join(REC, "baseline_oof_full.parquet"),
    columns=["source_id", "region", "score_rules_reproduction"],
)
cand = pd.read_csv(os.path.join(REC, "candidates.csv"))
print(
    "labels",
    len(lab),
    "features",
    len(fea),
    "oof",
    len(oof),
    "base",
    len(base),
    "cand",
    len(cand),
)
assert int((lab.y == 1).sum()) == 151, int((lab.y == 1).sum())

# ---------------------------------------------------------------- (a) compact-subset evaluation
d = oof.merge(
    fea[["source_id", "in_support", "r_h_arcsec", "r_star_v1_arcsec"]],
    on="source_id",
    how="left",
)
d = d.merge(base[["source_id", "score_rules_reproduction"]], on="source_id", how="left")
d = d.merge(lab[["source_id"] + SEL], on="source_id", how="left")
assert d.in_support.all()
d["union"] = d[SEL].any(axis=1)
compact = d.r_h_arcsec < 1.5 * d.r_star_v1_arcsec
dc = d[compact].copy()
N["compactSupportRows"] = int(len(dc))
N["compactSupportPct"] = round(100.0 * len(dc) / len(d), 1)
N["compactPositives"] = int((dc.y == 1).sum())
N["compactUnionRows"] = int(dc.union.sum())
N["compactUnionRecall"] = int((dc.union & (dc.y == 1)).sum())
N["compactUnionAnchors"] = int((dc.union & (dc.y == 0)).sum())
rec = {"primary": 0, "imitation": 0}
anc = {"primary": 0, "imitation": 0}
rows = 0
for reg, g in dc.groupby("region"):
    K = int(g.union.sum())
    rows += K
    for name, col in (
        ("primary", "score_mean"),
        ("imitation", "score_rules_reproduction"),
    ):
        top = g.nlargest(K, col)
        rec[name] += int((top.y == 1).sum())
        anc[name] += int((top.y == 0).sum())
assert rows == N["compactUnionRows"]
N["compactPrimaryMatchedRecall"] = rec["primary"]
N["compactPrimaryMatchedAnchors"] = anc["primary"]
N["compactImitationMatchedRecall"] = rec["imitation"]
N["compactImitationMatchedAnchors"] = anc["imitation"]
# paired discordant counts, primary vs union, on compact positives
sel_p = pd.Series(False, index=dc.index)
for reg, g in dc.groupby("region"):
    K = int(g.union.sum())
    sel_p.loc[g.nlargest(K, "score_mean").index] = True
pos = dc.y == 1
N["compactMcnemarB"] = int((pos & sel_p & ~dc.union).sum())
N["compactMcnemarC"] = int((pos & ~sel_p & dc.union).sum())
from scipy.stats import binom

b, c = N["compactMcnemarB"], N["compactMcnemarC"]
N["compactMcnemarP"] = (
    float(min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5))) if b + c else 1.0
)
# the same on the non-compact remainder, for the record
dn = d[~compact]
N["nonCompactSupportRows"] = int(len(dn))
N["nonCompactPositives"] = int((dn.y == 1).sum())
N["nonCompactUnionRows"] = int(dn.union.sum())
N["nonCompactUnionRecall"] = int((dn.union & (dn.y == 1)).sum())
r = 0
for reg, g in dn.groupby("region"):
    K = int(g.union.sum())
    r += int((g.nlargest(K, "score_mean").y == 1).sum())
N["nonCompactPrimaryMatchedRecall"] = r
print("(a)", {k: v for k, v in N.items() if "ompact" in k})


# ---------------------------------------------------------------- (b) match audit
def radec_cols(cols):
    lc = [c.lower() for c in cols]
    ra = next(
        c
        for c, l in zip(cols, lc)
        if l in ("ra", "ra_deg", "radeg", "raj2000", "_ra", "ra_1", "alpha", "ra_d")
    )
    dec = next(
        c
        for c, l in zip(cols, lc)
        if l
        in ("dec", "dec_deg", "decdeg", "dej2000", "_dec", "dec_1", "delta", "dec_d")
    )
    return ra, dec


lists = {
    "hviding25_A1": "hviding2025_zenodo_RUBIES-Hviding25-A1-lrds.fits",
    "barro25": "barro2025_github_Barro25_LRD_NIRSpec_bestfit_properties.fits",
    "degraaff26": "degraaff2026_zenodo_17665942_v0.1_lrds_withdups_blackbody_eline_fits.fits",
}
lpos = []
for name, fn in lists.items():
    t = fits.open(os.path.join(LISTS, fn))[1].data
    cols = list(t.columns.names)
    ra, dec = radec_cols(cols)
    print("   list", name, "columns", ra, dec, "rows", len(t))
    a = np.asarray(t[ra], float)
    b_ = np.asarray(t[dec], float)
    ok = np.isfinite(a) & np.isfinite(b_)
    lpos.append(pd.DataFrame({"list": name, "ra": a[ok], "dec": b_[ok]}))
lpos = pd.concat(lpos, ignore_index=True)
catc = SkyCoord(lab.ra.values * u.deg, lab.dec.values * u.deg)
lc = SkyCoord(lpos.ra.values * u.deg, lpos.dec.values * u.deg)
pos_lab = lab[lab.y == 1].copy()
pc = SkyCoord(pos_lab.ra.values * u.deg, pos_lab.dec.values * u.deg)
idx, sep, _ = pc.match_to_catalog_sky(lc)
pos_lab["sep_arcsec"] = sep.arcsec
# the two spectral-test positives have no list position
listed = pos_lab.sep_arcsec < 0.5
N["matchListedPositives"] = int(listed.sum())
s = pos_lab.sep_arcsec[listed]
N["matchSepMedianArcsec"] = round(float(np.median(s)), 3)
N["matchSepPNinetyArcsec"] = round(float(np.percentile(s, 90)), 3)
N["matchSepMaxArcsec"] = round(float(s.max()), 3)
N["matchSepBelowPointOne"] = int((s < 0.1).sum())
N["matchSepBelowPointTwo"] = int((s < 0.2).sum())
# multiplicity: catalog rows within 0.5 arcsec of each matched list position, counted per field
mult = []
for i, row in pos_lab[listed].iterrows():
    lp = lc[idx[pos_lab.index.get_loc(i)]]
    same_field = lab.field.values == row.field
    seps = lp.separation(catc[same_field]).arcsec
    mult.append(int((seps < 0.5).sum()))
mult = np.array(mult)
N["matchMultiplicityOne"] = int((mult == 1).sum())
N["matchMultiplicityTwoPlus"] = int((mult >= 2).sum())
N["matchMultiplicityMax"] = int(mult.max())
# sky-group size and extent for the positives
fpos = fea[fea.source_id.isin(pos_lab.source_id)]
N["posSkyGroupSizeOne"] = int((fpos.sky_group_size == 1).sum())
N["posSkyGroupSizeTwoPlus"] = int((fpos.sky_group_size >= 2).sum())
N["posSkyGroupSizeMax"] = int(fpos.sky_group_size.max())
# extent of every multi-row group (max pairwise separation), from the label table positions
g = lab.merge(fea[["source_id", "sky_group", "sky_group_size"]], on="source_id")
multi = g[g.sky_group_size >= 2]
ext = []
for sg, gg in multi.groupby("sky_group"):
    c = SkyCoord(gg.ra.values * u.deg, gg.dec.values * u.deg)
    if len(c) > 12:
        c = c[:12]
    m = np.max([c[i].separation(c).arcsec.max() for i in range(len(c))])
    ext.append(m)
ext = np.array(ext)
N["skyGroupsMultiRow"] = int(len(ext))
N["skyGroupExtentMedianArcsec"] = round(float(np.median(ext)), 3)
N["skyGroupExtentMaxArcsec"] = round(float(ext.max()), 3)
N["skyGroupsExtentAbovePointFive"] = int((ext > 0.5).sum())
posg = multi[multi.y == 1].sky_group.unique()
N["posSkyGroupExtentMaxArcsec"] = (
    round(
        float(
            max(
                [
                    ext[list(multi.groupby("sky_group").groups.keys()).index(sg)]
                    for sg in posg
                ]
            )
        ),
        3,
    )
    if len(posg)
    else 0.0
)
print(
    "(b)", {k: v for k, v in N.items() if k.startswith(("match", "posSky", "skyGroup"))}
)

# ---------------------------------------------------------------- (c) out-of-fold scores for candidates
# idempotent: a rerun on a candidates.csv that already carries the columns replaces them
cand = cand.drop(columns=[c for c in cand.columns if c.startswith("oof_")], errors="ignore")
cj = cand.merge(
    oof[["source_id", "score_mean", "t_05", "t_burden", "sel_05", "sel_burden"]].rename(
        columns={
            "score_mean": "oof_score",
            "t_05": "oof_t_05",
            "t_burden": "oof_t_burden",
            "sel_05": "oof_above_05",
            "sel_burden": "oof_above_burden",
        }
    ),
    on="source_id",
    how="left",
)
assert cj.oof_score.notna().all()
N["candOofBelowHalf"] = int((~cj.oof_above_05.astype(bool)).sum())
N["candOofAboveBurden"] = int(cj.oof_above_burden.astype(bool).sum())
N["candEbOofAboveBurden"] = int(
    (cj.oof_above_burden.astype(bool) & (cj.tier == "equal_burden")).sum()
)
N["candFuOofBelowHalf"] = (
    int((~cj.oof_above_05.astype(bool) & (cj.followup_tier.astype(bool))).sum())
    if "followup_tier" in cj
    else None
)
cj.to_csv(os.path.join(REC, "candidates.csv"), index=False)
print("(c)", {k: v for k, v in N.items() if k.startswith("cand")})

# ---------------------------------------------------------------- (d) release tables
flags = lab[["source_id", "field", "id"] + SEL + ["picked"]].copy()
flags.to_parquet(os.path.join(OUT, "rule_flags.parquet"), index=False)
N["ruleFlagsRows"] = int(len(flags))
N["ruleFlagsUnionRows"] = int(flags.picked.sum())
labelled = lab[(lab.y.notna()) | (lab.ambiguous.astype(bool))][
    [
        "source_id",
        "field",
        "id",
        "ra",
        "dec",
        "y",
        "y_strict",
        "ambiguous",
        "bands_complete",
        "in_hviding25_A1",
        "in_hviding25_B1",
        "in_barro25",
        "in_degraaff26",
        "in_perger25",
        "in_kocevski24",
        "spec_tested",
        "spec_vshaped",
        "mag_f444w",
        "snr_f444w",
        "r_h_arcsec",
        "r_star_v1_arcsec",
        "z_phot",
    ]
    + SEL
    + ["picked"]
].copy()
labelled = labelled.merge(
    fea[["source_id", "region", "sky_group", "sky_group_size", "in_support"]],
    on="source_id",
    how="left",
)
labelled = labelled.merge(
    oof[["source_id", "score_mean", "sel_burden", "sel_neg1", "sel_05"]].rename(
        columns={
            "score_mean": "oof_score",
            "sel_burden": "oof_sel_burden",
            "sel_neg1": "oof_sel_neg1",
            "sel_05": "oof_sel_05",
        }
    ),
    on="source_id",
    how="left",
)
labelled.to_csv(os.path.join(OUT, "labelled_rows.csv"), index=False)
N["labelledRows"] = int(len(labelled))
N["labelledPositives"] = int((labelled.y == 1).sum())
N["labelledNegatives"] = int((labelled.y == 0).sum())
N["labelledAmbiguous"] = int(
    (labelled.ambiguous.astype(bool) & labelled.y.isna()).sum()
)
N["labelledBOneNegatives"] = int(
    ((labelled.y == 0) & labelled.in_hviding25_B1.astype(bool)).sum()
)
print("(d)", {k: v for k, v in N.items() if k.startswith(("rule", "labelled"))})

from scipy.stats import binom

# ---------------------------------------------------------------- (e) Barro et al. (2024b) comparator

col = pd.read_parquet(os.path.join(REC, "barro24b_colors.parquet"))
ap = pd.read_csv(os.path.join(REC, "barro24b_apertures.csv"))
lab8 = pd.read_parquet(
    os.path.join(V4, "labels.parquet"),
    columns=["source_id", "y", "y_strict", "ambiguous", "picked", "in_hviding25_B1"]
    + SEL,
)
fea8 = pd.read_parquet(
    os.path.join(V4, "features.parquet"), columns=["source_id", "region", "in_support"]
)
oof8 = pd.read_parquet(
    os.path.join(V4, "oof_scores.parquet"), columns=["source_id", "score_mean"]
)
base8 = pd.read_parquet(
    os.path.join(REC, "baseline_oof_full.parquet"),
    columns=["source_id", "score_rules_reproduction"],
)

assert len(col) == len(lab8) == 630869, (len(col), len(lab8))
surv = col[col.barro24b_colors]
assert len(ap) == len(surv), (len(ap), len(surv))
assert set(ap.source_id) == set(surv.source_id)

d = lab8.merge(col[["source_id", "barro24b_colors"]], on="source_id", how="left")
d = d.merge(
    ap[["source_id", "measurable", "ratio_r05_r02", "grizliv", "reason"]],
    on="source_id",
    how="left",
)
d = d.merge(fea8, on="source_id", how="left")
d = d.merge(oof8, on="source_id", how="left").merge(base8, on="source_id", how="left")

ratio = d.ratio_r05_r02.to_numpy(float)
compact = np.isfinite(ratio) & (ratio > 0) & (ratio < 1.5)
d["sel_barro24b"] = d.barro24b_colors.fillna(False).to_numpy(bool) & compact
d["union7"] = d[SEL].any(axis=1)
d["union8"] = d.union7 | d.sel_barro24b
pos = d.y == 1
neg = d.y == 0
strict = d.y_strict == 1

# ---------------------------------------------------------------- survivors and apertures
N["b24ColorSurvivors"] = int(d.barro24b_colors.sum())
N["b24ColorSurvivorPositives"] = int((d.barro24b_colors & pos).sum())
N["b24Unmeasurable"] = int(((d.barro24b_colors) & (d.measurable != True)).sum())
N["b24UnmeasurablePositives"] = int(
    ((d.barro24b_colors) & (d.measurable != True) & pos).sum()
)
reasons = ap[ap.measurable != True].reason.value_counts().to_dict()
N["b24UnmeasurableReasons"] = reasons
N["b24GrizliVersions"] = sorted(set(str(g) for g in ap.grizliv.dropna()))
N["b24Selected"] = int(d.sel_barro24b.sum())
N["b24Recall"] = int((d.sel_barro24b & pos).sum())
N["b24RecallStrict"] = int((d.sel_barro24b & strict).sum())
N["b24Anchors"] = int((d.sel_barro24b & neg).sum())
N["b24AnchorsB1"] = int((d.sel_barro24b & neg & (d.in_hviding25_B1 == True)).sum())
N["b24SelectedInSupport"] = int((d.sel_barro24b & (d.in_support == True)).sum())
N["b24AmbiguousSelected"] = int((d.sel_barro24b & (d.ambiguous == True)).sum())
# overlap with the seven
N["b24OnlyPositives"] = int((d.sel_barro24b & ~d.union7 & pos).sum())
N["b24AndUnionPositives"] = int((d.sel_barro24b & d.union7 & pos).sum())
N["b24Missed"] = int((~d.sel_barro24b & pos).sum())
N["union7Recall"] = int((d.union7 & pos).sum())
N["union8Recall"] = int((d.union8 & pos).sum())
N["union8Selected"] = int(d.union8.sum())
N["union8SelectedInSupport"] = int((d.union8 & (d.in_support == True)).sum())
N["union8Anchors"] = int((d.union8 & neg).sum())
N["union7Missed"] = int((~d.union7 & pos).sum())
N["union8Missed"] = int((~d.union8 & pos).sum())
N["b24RecoversOfUnion7Missed"] = int((d.sel_barro24b & ~d.union7 & pos).sum())
# "misses in no matched selection" now measured
N["b24RecallStrictMissedByUnion7"] = int((d.sel_barro24b & ~d.union7 & strict).sum())

# ---------------------------------------------------------------- learned vs union-of-eight, region-matched
s = d[d.in_support == True].copy()
assert s.score_mean.notna().all()
rec = {"primary": 0, "imitation": 0}
anc = {"primary": 0, "imitation": 0}
rows = 0
sel_p = pd.Series(False, index=s.index)
for reg, g in s.groupby("region"):
    K = int(g.union8.sum())
    rows += K
    for name, c in (
        ("primary", "score_mean"),
        ("imitation", "score_rules_reproduction"),
    ):
        top = g.nlargest(K, c)
        rec[name] += int((top.y == 1).sum())
        anc[name] += int((top.y == 0).sum())
    sel_p.loc[g.nlargest(K, "score_mean").index] = True
assert rows == N["union8SelectedInSupport"]
N["union8SupportRecall"] = int((s.union8 & (s.y == 1)).sum())
N["union8SupportAnchors"] = int((s.union8 & (s.y == 0)).sum())
N["primaryMatchedUnion8Recall"] = rec["primary"]
N["primaryMatchedUnion8Anchors"] = anc["primary"]
N["imitationMatchedUnion8Recall"] = rec["imitation"]
N["imitationMatchedUnion8Anchors"] = anc["imitation"]
p = s.y == 1
N["union8McnemarB"] = int((p & sel_p & ~s.union8).sum())
N["union8McnemarC"] = int((p & ~sel_p & s.union8).sum())

b, c = N["union8McnemarB"], N["union8McnemarC"]
N["union8McnemarP"] = (
    float(min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5))) if b + c else 1.0
)

json.dump(N, open(os.path.join(REC, "referee_compute_2026_09_06.json"), "w"), indent=1)
for k, v in N.items():
    print(k, v)
d[["source_id", "sel_barro24b"]].to_parquet(
    os.path.join(REC, "barro24b_flags.parquet"), index=False
)
