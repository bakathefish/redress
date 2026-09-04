"""v4 build step 8: the candidate catalog from the final scores.

A candidate is a source that (1) lies in the model's support, (2) scores at or above the
frozen high-recall threshold (0.5% of the support; the equal-burden threshold defines the
primary tier inside it), (3) is selected by none of the seven published rules in any of its
catalog copies, (4) has no 0.5-arcsec sky-group member in any published LRD list (Hviding
2025 A1 and B1, Barro 2025, de Graaff 2026, Perger 2025, Kocevski 2024) or in the labelled
spectroscopic census, and (5) is the highest-scoring row of its sky group. Every candidate is
matched to the DJA v4.4 spectrum index (all records within 0.5 arcsec) so the catalog says
what the archive can test; the refit itself is v4_refit_archive.py.

Also writes the full ledger of every row above the high-recall threshold with the reason it
is not a candidate (rule-selected, published, labelled, duplicate), so nothing is hidden.
Usage: python v4_candidates.py [suffix]
Outputs: candidates{SUF}.csv, candidates_ledger{SUF}.csv, candidate_counts{SUF}.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "recovery"
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
fin = json.load(open(os.path.join(OUT, f"final{SUF}.json")))
sc = pd.read_parquet(os.path.join(OUT, f"scores{SUF}.parquet"))
feats = fin["features"]
allrows = pd.read_parquet(
    os.path.join(OUT, "features.parquet"),
    columns=[
        "source_id",
        "sky_group",
        "picked",
        "in_any_list",
        "in_hviding25_B1",
        "y",
        "ambiguous",
        "in_census",
        "spec_tested",
        "spec_vshaped",
    ],
)

# group-level flags over ALL catalog rows (a twin copy outside the support still counts)
listed = (
    allrows.in_any_list.values
    | allrows.in_hviding25_B1.values
    | allrows.y.notna().values
    | allrows.ambiguous.values
    | allrows.spec_vshaped.fillna(False).values.astype(bool)  # the census V-shaped 46 are published by the paper
)
listed_groups = set(allrows.sky_group.values[listed])
picked_groups = set(allrows.sky_group.values[allrows.picked.values.astype(bool)])
census_groups = set(allrows.sky_group.values[allrows.in_census.values.astype(bool)])

above = sc[sc.ranking_score >= fin["t_05"]].copy()
above["reason_excluded"] = ""
above.loc[above.sky_group.isin(picked_groups), "reason_excluded"] = "rule-selected"
above.loc[above.sky_group.isin(listed_groups), "reason_excluded"] = (
    "published-or-labelled"
)
above = above.sort_values("ranking_score", ascending=False)
dup = above.duplicated("sky_group", keep="first")
above.loc[dup & (above.reason_excluded == ""), "reason_excluded"] = (
    "duplicate-sky-group"
)
cand = above[above.reason_excluded == ""].copy()
cand["rank"] = np.arange(1, len(cand) + 1)

# nearest published-list entry (any list, within 2 arcsec) for the record
D = os.path.join("inputs", "prior_art_catalogues")
lists = {
    "hviding25_A1": "hviding2025_zenodo_RUBIES-Hviding25-A1-lrds.fits",
    "hviding25_B1": "hviding2025_zenodo_RUBIES-Hviding25-B1-bl.fits",
    "barro25": "barro2025_github_Barro25_LRD_NIRSpec_bestfit_properties.fits",
    "degraaff26": "degraaff2026_zenodo_17665942_v0.1_lrds_withdups_blackbody_eline_fits.fits",
}
c = SkyCoord(cand.ra.values * u.deg, cand.dec.values * u.deg)
cand["nearest_list"] = ""
cand["nearest_list_sep_arcsec"] = np.nan
for name, fn in lists.items():
    tab = Table.read(os.path.join(D, fn))
    rc = [x for x in tab.colnames if x.lower() in ("ra", "ra_deg", "right_ascension")][
        0
    ]
    dc = [x for x in tab.colnames if x.lower() in ("dec", "dec_deg", "declination")][0]
    ra, dec = np.asarray(tab[rc], float), np.asarray(tab[dc], float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    b = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg)
    _, sep, _ = c.match_to_catalog_sky(b)
    better = (sep.arcsec < 2.0) & (
        np.isnan(cand.nearest_list_sep_arcsec.values)
        | (sep.arcsec < cand.nearest_list_sep_arcsec.values)
    )
    cand.loc[better, "nearest_list"] = name
    cand.loc[better, "nearest_list_sep_arcsec"] = sep.arcsec[better]
e = pd.read_csv(os.path.join(D, "external_labels_raw.csv"))
for name in ("perger25", "kocevski24"):
    ee = e[e.src == name]
    b = SkyCoord(ee.ra.values * u.deg, ee.dec.values * u.deg)
    _, sep, _ = c.match_to_catalog_sky(b)
    better = (sep.arcsec < 2.0) & (
        np.isnan(cand.nearest_list_sep_arcsec.values)
        | (sep.arcsec < cand.nearest_list_sep_arcsec.values)
    )
    cand.loc[better, "nearest_list"] = name
    cand.loc[better, "nearest_list_sep_arcsec"] = sep.arcsec[better]
assert not (cand.nearest_list_sep_arcsec < 0.5).any(), "novelty screen failed"

# archive join: every DJA v4.4 record within 0.5 arcsec
idx = pd.read_csv(
    os.path.join(D, "dja_v4.4_zenodo_dja_msaexp_emission_lines_v4.4.csv.gz"),
    usecols=["file", "root", "ra", "dec", "grating", "grade", "z_best", "version"],
    low_memory=False,
)
idx["grade"] = pd.to_numeric(idx.grade, errors="coerce")
idx["z_best"] = pd.to_numeric(idx.z_best, errors="coerce")
b = SkyCoord(idx.ra.values * u.deg, idx.dec.values * u.deg)
i_c, i_r, sep, _ = c.search_around_sky(
    b, 0.5 * u.arcsec
)  # i_c indexes the candidates... (argument first)
# astropy returns (idx_searcharound, idx_self): first indexes the ARGUMENT (b), second indexes c
i_rec, i_cand = i_c, i_r
rec = idx.iloc[i_rec].copy()
rec["cand_rank"] = cand["rank"].values[i_cand]
rec["sep_arcsec"] = sep.arcsec
rec["eligible"] = (
    rec.grating.astype(str).str.upper().str.startswith("PRISM")
    & (rec.grade >= 3)
    & (rec.z_best > 3)
)
rec["secure_lowz"] = (rec.grade >= 3) & (rec.z_best <= 3)
rec["secure_highz"] = (rec.grade >= 3) & (rec.z_best > 3)
agg = rec.groupby("cand_rank").agg(
    n_archive_records=("file", "size"),
    n_eligible_records=("eligible", "sum"),
    n_secure_lowz_records=("secure_lowz", "sum"),
    n_secure_highz_records=("secure_highz", "sum"),
    archive_best_grade=("grade", "max"),
    archive_gratings=("grating", lambda s: ";".join(sorted(set(map(str, s))))),
    archive_files=("file", lambda s: ";".join(sorted(s))),
)
zb = (
    rec[rec.grade >= 3]
    .sort_values("grade", ascending=False)
    .drop_duplicates("cand_rank")
    .set_index("cand_rank")
    .z_best.rename("archive_z_best_grade3")
)
cand = cand.merge(agg, left_on="rank", right_index=True, how="left").merge(
    zb, left_on="rank", right_index=True, how="left"
)
for col in ("n_archive_records", "n_eligible_records", "n_secure_lowz_records", "n_secure_highz_records"):
    cand[col] = cand[col].fillna(0).astype(int)
cand["status"] = np.where(
    cand.n_secure_lowz_records > 0,
    "secure low-z",
    np.where(
        cand.n_eligible_records > 0,
        "archive pending",
        np.where(cand.n_secure_highz_records > 0, "secure z>3 (V untested)", "untested"),
    ),
)
cand["archive_verdict"] = ""
rec.to_csv(os.path.join(OUT, f"candidate_archive_records{SUF}.csv"), index=False)

front = [
    "rank",
    "tier",
    "status",
    "ranking_score",
    "score_std",
    "score_percentile",
    "field",
    "id",
    "source_id",
    "ra",
    "dec",
    "sky_group",
    "sky_group_size",
    "mag_f444w",
    "snr_f444w",
    "r_h_arcsec",
    "z_phot",
    "ood_flag",
    "n_features_outside_positive_range",
    "disagreement_flag",
    "nearest_list",
    "nearest_list_sep_arcsec",
    "n_archive_records",
    "n_eligible_records",
    "n_secure_lowz_records",
    "n_secure_highz_records",
    "archive_best_grade",
    "archive_z_best_grade3",
    "archive_gratings",
    "archive_files",
    "archive_verdict",
    "in_census",
    "spec_tested",
    "spec_vshaped",
]
rules = [
    "sel_labbe23",
    "sel_kokorev24",
    "sel_kocevski24",
    "sel_perezgonzalez24",
    "sel_barro23",
    "sel_greene24",
    "sel_akins24",
]
cand[front + rules + feats].to_csv(
    os.path.join(OUT, f"candidates{SUF}.csv"), index=False
)
above[["rank"] if "rank" in above else []].shape  # noqa (no-op)
above[
    [
        "reason_excluded",
        "tier",
        "ranking_score",
        "field",
        "id",
        "source_id",
        "ra",
        "dec",
        "sky_group",
        "picked",
        "in_any_list",
        "in_hviding25_B1",
        "y",
        "mag_f444w",
    ]
].to_csv(os.path.join(OUT, f"candidates_ledger{SUF}.csv"), index=False)
counts = {
    "feature_set": fin["feature_set"],
    "rows_above_high_recall_threshold": int(len(above)),
    "rows_above_equal_burden_threshold": int((above.tier == "equal_burden").sum()),
    "excluded": above.reason_excluded.replace("", "candidate").value_counts().to_dict(),
    "candidates_total": int(len(cand)),
    "candidates_equal_burden_tier": int((cand.tier == "equal_burden").sum()),
    "candidates_high_recall_tier": int((cand.tier == "high_recall").sum()),
    "status_counts": cand.status.value_counts().to_dict(),
    "status_counts_equal_burden": cand[cand.tier == "equal_burden"]
    .status.value_counts()
    .to_dict(),
    "candidates_with_any_archive_record": int((cand.n_archive_records > 0).sum()),
    "eligible_records_to_refit": int(rec.eligible.sum()),
    "per_region_equal_burden": cand[cand.tier == "equal_burden"]
    .region.value_counts()
    .to_dict(),
    "ood_flagged_equal_burden": int(
        (cand.ood_flag & (cand.tier == "equal_burden")).sum()
    ),
    "mag_f444w_median_equal_burden": float(
        cand[cand.tier == "equal_burden"].mag_f444w.median()
    ),
    "in_census_but_never_listed": int(cand.in_census.astype(bool).sum()),
}
json.dump(counts, open(os.path.join(OUT, f"candidate_counts{SUF}.json"), "w"), indent=1)
print(json.dumps(counts, indent=1))
