"""Round 10 release (2026-09-07): make the public tables carry what the paper says they carry.

Writes into the public clone (recovery/):
  labelled_rows.csv        + the 30 model features, the seven catalog total fluxes and
                             uncertainties (microjansky), five AB colours, the region-matched
                             selections, the eighth-rule flag and the hand-added flag
  label_provenance.csv     + hand_added_barro26
  oof_scores_primary.parquet  the primary 48-bag model's out-of-fold score for every one of
                             the 466,419 support rows, with the fold thresholds, the three
                             threshold selections and the two region-matched selections
  candidates.csv           + the seven fluxes and uncertainties, seven AB magnitudes and the
                             brown-dwarf screen flag
  columns.md               a column dictionary for the three tables above
and repairs recovery/v4_features.py (import path) and requirements.txt (sedpy, scipy).

Run from the build repository root: python paper/round10_release_tables.py
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "recovery")
R5 = os.path.join(ROOT, "recovery", "round5")
CLONE = ROOT  # the release tables live beside the script
REC = os.path.join(CLONE, "recovery")
BANDS = ["f090w", "f115w", "f150w", "f200w", "f277w", "f356w", "f444w"]
ZP = 23.9  # microjansky to AB
BASE_LR = set("""source_id field id ra dec y y_strict ambiguous bands_complete in_hviding25_A1 in_hviding25_B1 in_barro25 in_degraaff26 in_perger25 in_kocevski24 spec_tested spec_vshaped mag_f444w snr_f444w r_h_arcsec r_star_v1_arcsec z_phot sel_labbe23 sel_kokorev24 sel_kocevski24 sel_perezgonzalez24 sel_barro23 sel_greene24 sel_akins24 picked region sky_group sky_group_size in_support oof_score oof_sel_burden oof_sel_neg1 oof_sel_05""".split())

man = json.load(open(os.path.join(V, "feature_manifest.json")))
FEATURES = man["features"]
assert len(FEATURES) == 30
den = json.load(open(os.path.join(V, "denominators.json")))
NU = den["union_in_support_per_region"]
hand = sorted(
    int(x["source_id"])
    for x in json.load(open(os.path.join(R5, "barro_membership.json")))[
        "manualAdditions"
    ]
)

lab = pd.read_parquet(os.path.join(V, "labels.parquet"))
fea = pd.read_parquet(
    os.path.join(V, "features.parquet"), columns=["source_id", "region"] + FEATURES
)
oof = pd.read_parquet(os.path.join(V, "oof_scores.parquet"))
b24 = pd.read_parquet(os.path.join(REC, "barro24b_flags.parquet"))


def ab_colour(f1, f2):
    f1 = np.asarray(f1, float)
    f2 = np.asarray(f2, float)
    ok = np.isfinite(f1) & np.isfinite(f2) & (f1 > 0) & (f2 > 0)
    out = np.full(len(f1), np.nan)
    out[ok] = -2.5 * np.log10(f1[ok] / f2[ok])
    return out


def ab_mag(f):
    f = np.asarray(f, float)
    ok = np.isfinite(f) & (f > 0)
    out = np.full(len(f), np.nan)
    out[ok] = ZP - 2.5 * np.log10(f[ok])
    return out


# ------------------------------------------------------------------ region-matched selections on the support
o = oof.merge(b24, on="source_id", how="left")
o["sel_barro24b"] = o.sel_barro24b.fillna(False).astype(bool)
o["picked"] = o.picked.astype(bool)
o["sel_region_matched"] = False
o["sel_region_matched_eight"] = False
for reg, k in NU.items():
    g = o[o.region == reg].sort_values("score_mean", ascending=False)
    o.loc[g.index[: int(k)], "sel_region_matched"] = True
    k8 = int(
        (o.region == reg).sum()
        and ((o.region == reg) & (o.picked | o.sel_barro24b)).sum()
    )
    o.loc[g.index[:k8], "sel_region_matched_eight"] = True
assert int((o.sel_region_matched & (o.y == 1)).sum()) == 124
pos_e = int((o.sel_region_matched_eight & (o.y == 1)).sum())
print("region-matched positives 124; to the union of eight:", pos_e)

# ------------------------------------------------------------------ oof_scores_primary.parquet
keep = [
    "source_id",
    "field",
    "region",
    "id",
    "ra",
    "dec",
    "sky_group",
    "y",
    "picked",
    "in_any_list",
    "score_mean",
    "score_std",
    "t_burden",
    "t_neg1",
    "t_05",
    "sel_burden",
    "sel_neg1",
    "sel_05",
    "sel_region_matched",
    "sel_region_matched_eight",
    "sel_barro24b",
]
op = o[keep].copy()
op["y"] = op.y.astype("Int8")
op.to_parquet(os.path.join(REC, "oof_scores_primary.parquet"), index=False)
print("oof_scores_primary.parquet", op.shape)

# ------------------------------------------------------------------ labelled_rows.csv
lr = pd.read_csv(os.path.join(REC, "labelled_rows.csv"), low_memory=False)
lr = lr[[c for c in lr.columns if c in BASE_LR]]  # idempotent: drop what this script adds
base_cols = list(lr.columns)
lr = lr.merge(fea[["source_id"] + FEATURES], on="source_id", how="left")
fl = lab[["source_id"] + ["f_" + b for b in BANDS] + ["e_" + b for b in BANDS]]
lr = lr.merge(fl, on="source_id", how="left")
for b in BANDS:
    lr = lr.rename(columns={"f_" + b: "f_%s_ujy" % b, "e_" + b: "e_%s_ujy" % b})
for b1, b2 in (
    ("f115w", "f200w"),
    ("f150w", "f200w"),
    ("f200w", "f356w"),
    ("f200w", "f444w"),
    ("f277w", "f444w"),
):
    lr["c_%s_%s_ab" % (b1, b2)] = ab_colour(lr["f_%s_ujy" % b1], lr["f_%s_ujy" % b2])
lr = lr.merge(
    o[["source_id", "sel_region_matched", "sel_region_matched_eight", "sel_barro24b"]],
    on="source_id",
    how="left",
)
for c in ("sel_region_matched", "sel_region_matched_eight", "sel_barro24b"):
    lr[c] = lr[c].fillna(False).astype(bool)
lr["hand_added_barro26"] = lr.source_id.isin(hand)
assert int(lr.hand_added_barro26.sum()) == 7
assert len(lr) == 5596
lr.to_csv(os.path.join(REC, "labelled_rows.csv"), index=False)
print("labelled_rows.csv", lr.shape, "added", len(lr.columns) - len(base_cols))

# ------------------------------------------------------------------ label_provenance.csv
lp = pd.read_csv(os.path.join(REC, "label_provenance.csv"), low_memory=False)
lp["hand_added_barro26"] = lp.source_id.isin(hand)
assert int(lp.hand_added_barro26.sum()) == 7
lp.to_csv(os.path.join(REC, "label_provenance.csv"), index=False)
print("label_provenance.csv", lp.shape)

# ------------------------------------------------------------------ candidates.csv
cd = pd.read_csv(os.path.join(REC, "candidates.csv"), low_memory=False)
cd = cd[[c for c in cd.columns if not re.match(r"^(f|e)_f\d{3}[wm]_ujy$|^mag_f\d{3}[wm]_ab$|^bd_screen$", c)]]
n0 = cd.shape[1]
cd = cd.merge(fl, on="source_id", how="left")
for b in BANDS:
    cd = cd.rename(columns={"f_" + b: "f_%s_ujy" % b, "e_" + b: "e_%s_ujy" % b})
    cd["mag_%s_ab" % b] = ab_mag(cd["f_%s_ujy" % b])
cd["bd_screen"] = cd.c_f115w_f200w < -0.5
assert int((cd.bd_screen & (cd.tier.astype(str).str.contains("equal"))).sum()) == 49, int((cd.bd_screen & cd.tier.astype(str).str.contains("equal")).sum())
cd.to_csv(os.path.join(REC, "candidates.csv"), index=False)
print("candidates.csv", cd.shape, "was", n0, "columns")

# ------------------------------------------------------------------ columns.md
DESC = {
    "source_id": "integer key of the catalog row across the nine fields (the join key of every released table)",
    "field": "DJA grizli v7 catalog name (ceers-full, gdn, gds, gds-sw, ngdeep, primer-cosmos-east, primer-cosmos-west, primer-uds-north, primer-uds-south)",
    "id": "catalog id within the field",
    "ra": "catalog right ascension, J2000, degrees",
    "dec": "catalog declination, J2000, degrees",
    "region": "one of the five held-out sky regions (CEERS, GOODS-N, GOODS-S, COSMOS, UDS)",
    "sky_group": "friends-of-friends group id at 0.5 arcsec; a group is one object where objects are counted",
    "sky_group_size": "number of catalog rows in the sky group",
    "in_support": "row lies in the seven-band support (all seven wide NIRCam bands covered)",
    "bands_complete": "all seven bands carry a finite flux and uncertainty",
    "y": "training label: 1 spectroscopic LRD, 0 comparison object, empty for ambiguous or unlabeled",
    "y_strict": "1 for the 32 sources the spectral test with the compactness criterion admits, else 0 or empty",
    "ambiguous": "flagged ambiguous: removed from the unlabeled pool (and, for the 48 of final class ambiguous, from both classes)",
    "picked": "selected by the union of the seven rules (flags of record, full band set)",
    "in_any_list": "member of any input list or catalog",
    "in_hviding25_A1": "member of Hviding et al. (2025) table A1",
    "in_hviding25_B1": "member of Hviding et al. (2025) table B1 (broad-line galaxies)",
    "in_barro25": "member of the Barro et al. (2026) spectroscopic list",
    "in_degraaff26": "member of the de Graaff et al. (2026) v0.1 list",
    "in_perger25": "member of the Perger et al. (2025) compilation",
    "in_kocevski24": "member of the published Kocevski et al. (2025) catalog",
    "hand_added_barro26": "one of the seven objects Barro et al. (2026) added by hand; in_barro25 and not hand_added_barro26 is the Barro et al. (2024a) selection's own output",
    "spec_tested": "row-level flag: the V-shape test was evaluated on a spectrum attached to this row",
    "spec_vshaped": "row-level flag: that spectrum passed the V-shape test",
    "mag_f444w": "catalog total AB magnitude in F444W",
    "snr_f444w": "catalog signal-to-noise in F444W",
    "r_h_arcsec": "catalog half-light radius (flux_radius) in arcsec",
    "r_star_v1_arcsec": "per-field stellar half-light radius used to normalise r_h",
    "z_phot": "catalog photometric redshift (EAZY, DJA)",
    "sel_labbe23": "selected by the labbe23 module (Labbe et al. 2025), flag of record",
    "sel_kokorev24": "selected by the kokorev24 module (Kokorev et al. 2024), flag of record",
    "sel_kocevski24": "selected by the kocevski24 module (Kocevski et al. 2025), flag of record",
    "sel_perezgonzalez24": "selected by the perezgonzalez24 module (Perez-Gonzalez et al. 2024), flag of record",
    "sel_barro23": "selected by the barro23 module (Barro et al. 2024b), flag of record",
    "sel_greene24": "selected by the greene24 module (Greene et al. 2024), flag of record",
    "sel_akins24": "selected by the akins24 module (Akins et al. 2025), flag of record",
    "sel_barro24b": "selected by the eighth module, barro24b (Barro et al. 2024a), coded after the freeze",
    "oof_score": "out-of-fold ranking score of the primary 48-bag model (fold model that never saw the row's region)",
    "oof_sel_burden": "above the fold's equal-burden threshold",
    "oof_sel_neg1": "above the fold's 1 percent comparison-set threshold",
    "oof_sel_05": "above the fold's 0.5 percent training-row threshold",
    "sel_region_matched": "in the top K of the out-of-fold ranking within its region, K the union of seven's in-support row count there (the primary comparison)",
    "sel_region_matched_eight": "the same with K the union of eight's row count in the region",
    "score_mean": "out-of-fold ranking score, mean over the 48 bags of the fold model",
    "score_std": "standard deviation of the 48 bag outputs",
    "t_burden": "the fold's equal-burden threshold",
    "t_neg1": "the fold's 1 percent comparison-set threshold",
    "t_05": "the fold's 0.5 percent training-row threshold",
    "sel_burden": "score_mean at or above t_burden",
    "sel_neg1": "score_mean at or above t_neg1",
    "sel_05": "score_mean at or above t_05",
    "rank": "rank in the candidate catalog by the deployed model's score",
    "tier": "equal-burden or high-recall tier of the deployed thresholds",
    "status": "frozen archival status (see the paper, Section 6.2)",
    "ranking_score": "in-sample score of the deployed model fitted on all five regions",
    "score_percentile": "percentile of ranking_score over the support",
    "ood_flag": "one or more features outside the range spanned by the positives",
    "n_features_outside_positive_range": "count of such features",
    "disagreement_flag": "bag disagreement above the release cut",
    "nearest_list": "nearest input-list object",
    "nearest_list_sep_arcsec": "its separation in arcsec",
    "n_archive_records": "archival spectra within 0.5 arcsec in the frozen DJA index",
    "n_eligible_records": "of those, prism spectra eligible for the V-shape test",
    "n_secure_lowz_records": "records with a secure (grade 3) redshift at z <= 3",
    "n_secure_highz_records": "records with a secure redshift above 3",
    "archive_best_grade": "best redshift grade among the records",
    "archive_z_best_grade3": "the secure redshift where one exists",
    "archive_gratings": "gratings of the records",
    "archive_files": "spectrum files of the records",
    "in_census": "row in the spectroscopic census of the earlier work",
    "archive_verdict": "verdict of the frozen archival refit (none produced)",
    "compactness_f444w": "Akins et al. (2025) ratio: flux in 0.2 arcsec over 0.5 arcsec diameter apertures on the F444W mosaic",
    "labbe_compactness": "Labbe et al. (2025) ratio: flux in 0.4 arcsec over 0.2 arcsec diameter apertures",
    "flux_d020": "F444W aperture flux, 0.2 arcsec diameter, microjansky",
    "flux_d040": "F444W aperture flux, 0.4 arcsec diameter, microjansky",
    "flux_d050": "F444W aperture flux, 0.5 arcsec diameter, microjansky",
    "compactness_measurable": "the apertures lie inside the valid footprint",
    "compactness_reason": "why not, when not measurable",
    "compactness_mosaic": "grizli mosaic version the apertures were measured on",
    "compactness_pixscale": "mosaic pixel scale, arcsec",
    "compact_akins": "0.5 < compactness_f444w <= 0.7",
    "compact_labbe": "labbe_compactness < 1.7",
    "followup_tier": "member of the follow-up tier defined in Section 6.3",
    "oof_t_burden": "the row's own fold equal-burden threshold",
    "oof_t_05": "the row's own fold 0.5 percent threshold",
    "oof_above_burden": "oof_score at or above oof_t_burden",
    "oof_above_05": "oof_score at or above oof_t_05",
    "bd_screen": "brown-dwarf screen: F115W-F200W feature colour below -0.5 (a heuristic photometric screen, not a stellar classification)",
    "sel_barro24b_flag": "",
    "in_degraaff26_v2": "member of the de Graaff et al. (2026) v2 list (post-freeze status field)",
    "zspec_degraaff26_v2": "its spectroscopic redshift",
    "spectrum_status": "five-way spectrum status of Section 6.2",
    "z_spec_secure": "secure archival redshift where one exists",
    "listed_after_freeze": "listed as a spectroscopic LRD by a later list version (post-freeze status field)",
    "spectrum_eligible": "the source has a spectrum eligible for the V-shape test",
    "test_evaluable_source": "the test could be evaluated on at least one eligible spectrum of the source",
    "verdict_source": "source-level verdict: True if any eligible spectrum passes",
    "test_evaluable_row": "row-level: evaluated on the spectrum attached to this row",
    "verdict_row": "row-level verdict",
    "label": "training label as in labelled_rows.csv",
    "label_strict": "strict label",
    "final_class": "spectroscopic LRD, comparison object, or ambiguous (used in neither class)",
    "verdict_rule": "which rule fixed the class",
    "ambiguity_reason": "why the source is ambiguous (empty for the seven B1 broad-line galaxies without an eligible spectrum)",
    "z_spec": "spectroscopic redshift used in the paper",
    "z_spec_source": "list or archive",
    "z_list": "redshift quoted by the input list",
    "z_archive": "archival redshift",
    "archive_grade": "its grade",
}
PATTERN = [
    (
        r"^m_(f\d{3}[wm])$",
        "inverse-hyperbolic-sine magnitude feature of {0} (no zero point; add 23.9 to compare with AB at high S/N)",
    ),
    (r"^asnr_(f\d{3}[wm])$", "asinh signal-to-noise feature of {0}"),
    (
        r"^c_(f\d{3}[wm])_(f\d{3}[wm])$",
        "feature colour {0} minus {1} on the asinh scale",
    ),
    (
        r"^c_(f\d{3}[wm])_(f\d{3}[wm])_ab$",
        "AB colour {0} minus {1} from the catalog total fluxes (empty where either flux is not positive)",
    ),
    (
        r"^f_(f\d{3}[wm])_ujy$",
        "catalog total flux in {0}, microjansky (DJA <band>_tot_1)",
    ),
    (r"^e_(f\d{3}[wm])_ujy$", "its uncertainty, microjansky (DJA <band>_etot_1)"),
    (
        r"^mag_(f\d{3}[wm])_ab$",
        "AB magnitude in {0} from the total flux, 23.9 - 2.5 log10(flux)",
    ),
    (r"^slope_blue$", "weighted observed-frame slope over the blue bands (feature)"),
    (r"^slope_red$", "weighted observed-frame slope over the red bands (feature)"),
    (r"^v_curv$", "V-curvature feature"),
    (r"^log_rh$", "log10 of the catalog half-light radius in arcsec"),
    (r"^log_rh_over_rstar$", "log10 of r_h over the per-field stellar radius"),
    (r"^n_snr_ge1$", "number of bands with S/N >= 1"),
    (r"^n_snr_ge3$", "number of bands with S/N >= 3"),
]


def describe(col):
    if col in DESC and DESC[col]:
        return DESC[col]
    for pat, txt in PATTERN:
        m = re.match(pat, col)
        if m:
            return txt.format(*[g.upper() for g in m.groups()])
    return "(see the paper)"


lines = [
    "# Column dictionary",
    "",
    "Units: fluxes in microjansky, magnitudes AB unless the column name says asinh feature,",
    "coordinates J2000 degrees, sizes arcsec. Boolean columns are True/False. The join key of",
    "every table is `source_id`. The feature columns are the 30 model inputs listed in",
    "`feature_manifest.json`, with the asinh softening per band recorded there.",
    "",
]
for name, df in (
    ("candidates.csv", cd),
    ("labelled_rows.csv", lr),
    ("label_provenance.csv", lp),
    ("oof_scores_primary.parquet", op),
):
    lines += [
        "## %s (%d rows, %d columns)" % (name, len(df), df.shape[1]),
        "",
        "| column | meaning |",
        "|---|---|",
    ]
    for c in df.columns:
        lines.append("| `%s` | %s |" % (c, describe(c)))
    lines.append("")
open(os.path.join(REC, "columns.md"), "w", encoding="utf-8", newline="\n").write(
    "\n".join(lines)
)
missing = sorted(
    {
        c
        for df in (cd, lr, lp, op)
        for c in df.columns
        if describe(c) == "(see the paper)"
    }
)
print("columns.md written; undescribed:", missing)

# ------------------------------------------------------------------ repairs
p = os.path.join(REC, "v4_features.py")
t = open(p, encoding="utf-8").read()
if 'sys.path.insert(0, "src")' in t:
    t = t.replace('sys.path.insert(0, "src")', 'sys.path.insert(0, "selections")')
    open(p, "w", encoding="utf-8", newline="\n").write(t)
    print("v4_features.py: import path repaired")
p = os.path.join(CLONE, "requirements.txt")
t = open(p, encoding="utf-8").read()
adds = []
try:
    import sedpy

    sv = getattr(sedpy, "__version__", None)
except Exception:
    sv = None
try:
    import scipy

    sc = scipy.__version__
except Exception:
    sc = None
if "sedpy" not in t and sv:
    adds.append("sedpy==%s" % sv)
if "scipy" not in t and sc:
    adds.append("scipy==%s" % sc)
if adds:
    open(p, "w", encoding="utf-8", newline="\n").write(
        t.rstrip("\n") + "\n" + "\n".join(adds) + "\n"
    )
print("requirements added:", adds, "(sedpy", sv, "scipy", sc, ")")
