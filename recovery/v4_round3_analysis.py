"""Round 3 additions: the spectroscopic redshift of every positive, and a per-catalogue
decomposition of the published-catalogue census.

Writes ONE new record, recovery/published_catalogues_extra.parquet. Nothing existing is
read for writing and no existing record is modified.

Two things go into that file, both keyed on ``source_id`` over every catalogue row.

1. REDSHIFT (director's T1). Each positive's spectroscopic redshift, taken in this order and
   labelled with which one was used:
     * ``list``    -- the redshift column of the published list the object is in
                     (Hviding et al. 2025 table A1 ``z``, Barro et al. 2025 ``zspec``,
                     de Graaff et al. 2026 ``zspec``), matched exactly as
                     recovery/build_labels.py matches those lists: each list row to its
                     nearest catalogue source within 0.5 arcsec. Where more than one list
                     carries an object the median is used; the spread is recorded.
     * ``archive`` -- otherwise the DAWN JWST Archive v4.4 secure redshift (grade >= 3)
                     within 0.5 arcsec, highest grade first, the same join
                     recovery/v4_candidates.py uses for the candidate ledger.
     * ``zphot``   -- otherwise the catalogue photometric redshift, a last resort that the
                     text must declare. In the run of record no positive needs it.

2. CENSUS (director's T3). Perger et al. 2025 (A&A 693, L2) says in Section 2 which
   catalogues its 919 objects come from, and its machine-readable table carries the ADS
   bibcode of the paper each object was first published in. Each of those 17 papers
   becomes a
   ``perger_<paper>`` membership flag here, so the compilation stops being one opaque flag
   and becomes a census that can be read catalogue by catalogue. The two catalogues Reviewer
   A asked about, Kokorev et al. 2024 and Akins et al. 2024, are both among the 17.
   ``eprint_kokorev2024`` and ``eprint_akins2024`` additionally match the coordinate excerpts
   printed in those two papers' own arXiv sources (arXiv:2401.09981 appendix table and
   arXiv:2406.10341 Table 2); both papers keep their full catalogues off arXiv and off
   VizieR, so those flags cover only the printed rows and are a partial direct check, not a
   full catalogue match.

Run from the repository root: python recovery/v4_round3_analysis.py
"""

import json
import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
V = "recovery"
D = os.path.join("inputs", "prior_art_catalogues")
EPRINT = os.path.join("inputs", "round3_eprints")

f = pd.read_parquet(
    os.path.join(V, "features.parquet"),
    columns=[
        "source_id",
        "field",
        "region",
        "id",
        "ra",
        "dec",
        "y",
        "y_strict",
        "picked",
        "in_support",
        "in_perger25",
        "in_kocevski24",
        "in_hviding25_A1",
        "in_barro25",
        "in_degraaff26",
        "z_phot",
    ],
)
a = SkyCoord(f.ra.values * u.deg, f.dec.values * u.deg)
out = pd.DataFrame({"source_id": f.source_id.values}).set_index("source_id")
counts = {}

# ------------------------------------------------------------------ 1. spectroscopic redshift
LISTS = {
    "hviding25_A1": ("hviding2025_zenodo_RUBIES-Hviding25-A1-lrds.fits", "z"),
    "barro25": (
        "barro2025_github_Barro25_LRD_NIRSpec_bestfit_properties.fits",
        "zspec",
    ),
    "degraaff26": (
        "degraaff2026_zenodo_17665942_v0.1_lrds_withdups_blackbody_eline_fits.fits",
        "zspec",
    ),
}
zl = pd.DataFrame(index=out.index)
for name, (fn, zc) in LISTS.items():
    t = Table.read(os.path.join(D, fn))
    ra = np.asarray(t["ra"], float)
    dec = np.asarray(t["dec"], float)
    z = np.asarray(t[zc], float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    b = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg)
    idx, sep, _ = b.match_to_catalog_sky(a)
    m = sep.arcsec < 0.5
    g = pd.DataFrame({"source_id": f.source_id.values[idx][m], "z": z[ok][m]})
    zl[name] = g.groupby("source_id").z.median().reindex(out.index)
    counts[name] = {
        "rows": int(ok.sum()),
        "matched_rows": int(m.sum()),
        "distinct_sources": int(g.source_id.nunique()),
    }

out["z_list"] = zl[list(LISTS)].median(axis=1, skipna=True)
out["z_list_spread"] = zl[list(LISTS)].max(axis=1) - zl[list(LISTS)].min(axis=1)
out["n_lists_with_z"] = zl[list(LISTS)].notna().sum(axis=1)

# DAWN JWST Archive v4.4, restricted to the positives (the only rows the paper quotes)
pos_i = np.flatnonzero(f.y.values == 1)
c = SkyCoord(f.ra.values[pos_i] * u.deg, f.dec.values[pos_i] * u.deg)
idx = pd.read_csv(
    os.path.join(D, "dja_v4.4_zenodo_dja_msaexp_emission_lines_v4.4.csv.gz"),
    usecols=["file", "root", "ra", "dec", "grating", "grade", "z_best", "version"],
    low_memory=False,
)
idx["grade"] = pd.to_numeric(idx.grade, errors="coerce")
idx["z_best"] = pd.to_numeric(idx.z_best, errors="coerce")
b = SkyCoord(idx.ra.values * u.deg, idx.dec.values * u.deg)
# same call form as v4_candidates.py: the first index runs over the argument, the second
# over self
i_rec, i_pos, sep, _ = c.search_around_sky(b, 0.5 * u.arcsec)
rec = idx.iloc[i_rec].copy()
rec["source_id"] = f.source_id.values[pos_i][i_pos]
sec = rec[(rec.grade >= 3) & rec.z_best.notna()]
best = (
    sec.sort_values(["grade", "z_best"], ascending=[False, False])
    .drop_duplicates("source_id")
    .set_index("source_id")
)
out["z_archive"] = best.z_best.reindex(out.index)
out["archive_grade"] = best.grade.reindex(out.index)
counts["dja_archive"] = {
    "records_near_a_positive": int(len(rec)),
    "secure_records": int(len(sec)),
    "positives_with_a_secure_z": int(sec.source_id.nunique()),
}

zp = pd.Series(f.z_phot.values, index=out.index)
out["z_spec"] = out.z_list.where(
    out.z_list.notna(), out.z_archive.where(out.z_archive.notna(), zp)
)
out["z_spec_source"] = np.where(
    out.z_list.notna(),
    "list",
    np.where(out.z_archive.notna(), "archive", np.where(zp.notna(), "zphot", "none")),
)
ispos = pd.Series(f.y.values == 1, index=out.index)
out.loc[~ispos, ["z_spec"]] = np.nan
out.loc[~ispos, ["z_spec_source"]] = ""
assert out.loc[ispos, "z_spec"].notna().all(), "a positive has no redshift of any kind"
counts["z_spec_source"] = out.loc[ispos, "z_spec_source"].value_counts().to_dict()
_both = ispos & out.z_list.notna() & out.z_archive.notna()
counts["list_vs_archive"] = {
    "n": int(_both.sum()),
    "median_abs_diff": round(
        float((out.z_list - out.z_archive)[_both].abs().median()), 5
    ),
    "max_abs_diff": round(float((out.z_list - out.z_archive)[_both].abs().max()), 5),
}

# ------------------------------------------------------------------ 2. catalogue census
p = os.path.join(D, "vizier_perger2025_JA+A_693_L2_stacked.tsv")
L = list(open(p, encoding="utf-8"))
h = [i for i, ln in enumerate(L) if ln.startswith("Name\tRAJ2000")][0]
cols = L[h].rstrip("\n").split("\t")
rows = [
    ln.rstrip("\n").split("\t")
    for ln in L[h + 3 :]
    if ln.strip() and not ln.startswith("#")
]
pg = pd.DataFrame(rows, columns=cols)
for col in cols:
    pg[col] = pg[col].str.strip()
pg = pg[pg.Name != ""].reset_index(drop=True)
assert len(pg) == 919, f"Perger table is {len(pg)} rows, not the published 919"
b = SkyCoord(
    pg.RAJ2000.astype(float).values * u.deg, pg.DEJ2000.astype(float).values * u.deg
)
i2, s2, _ = b.match_to_catalog_sky(a)
m2 = s2.arcsec < 0.5
pg["source_id"] = np.where(m2, f.source_id.values[i2], -1)

# the 17 originating papers Perger et al. Sect. 2 names, keyed by the bibcode their table
# carries per object
REF = {
    "2023A&A...677A.145U": "ubler2023",
    "2023arXiv231203065K": "kokorev2023",
    "2023ApJ...959...39H": "harikane2023",
    "2023Natur.616..266L": "labbe2023nat",
    "2023arXiv230801230M": "maiolino2023",
    "2023ApJ...957L...7K": "killi2023",
    "2023arXiv230607320L": "labbe2023",
    "2024ApJ...963..128B": "barro2024",
    "2024ApJ...963..129M": "matthee2024",
    "2024ApJ...964...39G": "greene2024",
    "2024ApJ...968....4P": "perezgonzalez2024",
    "2024ApJ...968...34W": "williams2024",
    "2024ApJ...968...38K": "kokorev2024",
    "2024ApJ...969L..13W": "wang2024",
    "2024arXiv240403576K": "kocevski2024",
    "2024arXiv240610341A": "akins2024",
    "2024Natur.628...57F": "furtak2024",
}
assert set(pg.Ref) == set(REF), "the Perger reference set moved"
pg["paper"] = pg.Ref.map(REF)
for pap in sorted(set(REF.values())):
    out["perger_" + pap] = out.index.isin(
        set(pg.loc[m2 & (pg.paper == pap), "source_id"])
    )
_pcols = [c for c in out.columns if c.startswith("perger_")]
out["in_perger25_recomputed"] = out[_pcols].any(axis=1)
assert (out.in_perger25_recomputed.values == f.in_perger25.astype(bool).values).all(), (
    "the recomputed Perger membership disagrees with the stored in_perger25 flag"
)
counts["perger_by_paper"] = {
    k: {
        "rows": int((pg.paper == k).sum()),
        "in_our_fields": int(((pg.paper == k) & m2).sum()),
    }
    for k in sorted(set(REF.values()))
}

# the coordinate excerpts printed in the two catalogues' own arXiv sources
EX = {
    "kokorev2024": ("kokorev2024_arxiv2401.09981_appendix_excerpt.csv", 19),
    "akins2024": ("akins2024_arxiv2406.10341_table2_excerpt.csv", 33),
}
for nm, (fn, n_expect) in EX.items():
    tab = pd.read_csv(os.path.join(EPRINT, fn))
    assert len(tab) == n_expect, f"{fn} is {len(tab)} rows, expected {n_expect}"
    bb = SkyCoord(tab.ra.values * u.deg, tab.dec.values * u.deg)
    i3, s3, _ = bb.match_to_catalog_sky(a)
    m3 = s3.arcsec < 0.5
    out["eprint_" + nm] = out.index.isin(set(f.source_id.values[i3][m3]))
    counts["eprint_" + nm] = {"rows": int(len(tab)), "matched": int(m3.sum())}

out.reset_index().to_parquet(
    os.path.join(V, "published_catalogues_extra.parquet"), index=False
)

# ------------------------------------------------------------------ 3. anchor test status
# Reviewer C's round-3 finding 4: six anchor macros are read from inputs/
# sources_v3.parquet, which the paper does not release, and the released `spec_tested` column
# gives a different split of the same anchors (4,348 against 4,838), so a reader cannot
# reproduce the sentence from the release alone. This exports the one column that decides it.
#
# `vshaped_spec` is False for an anchor either because the fit ran and the source failed the
# V-shape test, or because the fit was refused; `testable` separates the two. `elig` is
# whether the source holds a frozen eligible record at all. One row per labelled source,
# y in {0, 1}, so both the anchors and the positives are auditable from the released file.
s3 = pd.read_parquet(
    os.path.join("inputs", "sources_v3.parquet"),
    columns=["source_id", "field", "id", "elig", "testable", "vshaped_spec"],
).set_index("source_id")
lab = pd.read_parquet(os.path.join(V, "labels.parquet"), columns=["source_id", "y"])
lab = lab[lab.y.isin([0.0, 1.0])]
ats = lab.set_index("source_id").join(s3, how="left").reset_index()
assert ats.field.notna().all(), "a labelled source is missing from sources_v3"
ats = ats[["source_id", "field", "id", "elig", "testable", "vshaped_spec", "y"]]
ats["y"] = ats.y.astype(int)
ats.to_csv(os.path.join(V, "anchor_test_status.csv"), index=False)
_neg = ats[ats.y == 0]
_tt = _neg.testable.astype("boolean")
counts["anchor_test_status"] = {
    "rows": int(len(ats)),
    "anchors": int(len(_neg)),
    "positives": int((ats.y == 1).sum()),
    "anchors_tested": int(_tt.eq(True).fillna(False).sum()),
    "anchors_untested": int(_tt.eq(False).fillna(False).sum()),
}
assert (
    counts["anchor_test_status"]["anchors_tested"]
    + counts["anchor_test_status"]["anchors_untested"]
    == counts["anchor_test_status"]["anchors"]
), "the tested/untested split does not cover every anchor"

with open(os.path.join(V, "round3_analysis_counts.json"), "w", encoding="utf-8") as fh:
    json.dump(counts, fh, indent=2, default=float)
print(json.dumps(counts, indent=2, default=float))
