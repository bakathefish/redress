"""v4: what the archive says about the candidates versus the rules' own novel picks.

For the model's candidates and, as the comparison set, the union-of-seven selections that
are in no published list and carry no label (the rules' own novel picks, one per sky group),
count the sources with any DJA v4.4 record within 0.5 arcsec, with a secure (grade >= 3)
redshift at z <= 3, with a secure redshift at z > 3, and the catalog photo-z < 3 fraction.
These are selection rates on whatever the archive happened to observe, not purities.
Usage: python v4_archive_purity.py [suffix] -> archive_purity{SUF}.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "recovery"
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
cand = pd.read_csv(os.path.join(OUT, f"candidates{SUF}.csv"))
f = pd.read_parquet(
    os.path.join(OUT, "features.parquet"),
    columns=[
        "source_id",
        "ra",
        "dec",
        "picked",
        "in_any_list",
        "in_hviding25_B1",
        "y",
        "ambiguous",
        "in_support",
        "z_phot",
        "sky_group",
        "spec_vshaped",
    ],
)
union_novel = f[
    f.picked
    & ~f.in_any_list
    & ~f.in_hviding25_B1
    & f.y.isna()
    & ~f.ambiguous
    & f.in_support
    & ~f.spec_vshaped.fillna(False).astype(bool)
].drop_duplicates("sky_group")
idx = pd.read_csv(
    os.path.join(
        "inputs",
        "prior_art_catalogues",
        "dja_v4.4_zenodo_dja_msaexp_emission_lines_v4.4.csv.gz",
    ),
    usecols=["ra", "dec", "grating", "grade", "z_best"],
    low_memory=False,
)
idx["grade"] = pd.to_numeric(idx.grade, errors="coerce")
idx["z_best"] = pd.to_numeric(idx.z_best, errors="coerce")
b = SkyCoord(idx.ra.values * u.deg, idx.dec.values * u.deg)


def stats(df, name):
    a = SkyCoord(df.ra.values * u.deg, df.dec.values * u.deg)
    i_rec, i_src, sep, _ = a.search_around_sky(b, 0.5 * u.arcsec)
    rec = idx.iloc[i_rec].copy()
    rec["src"] = i_src
    sec = rec[rec.grade >= 3]
    lowz = set(sec[sec.z_best <= 3].src)
    highz = set(sec[sec.z_best > 3].src)
    return {
        "set": name,
        "n": int(len(df)),
        "with_any_archive_record": int(rec.src.nunique()),
        "secure_z_le_3": int(len(lowz)),
        "secure_z_gt_3": int(len(highz)),
        "secure_z_gt_3_only": int(len(highz - lowz)),
        "zphot_lt_3": int((df.z_phot < 3).sum()),
        "zphot_lt_3_pct": round(100 * float((df.z_phot < 3).mean()), 1),
        "f444w_median": round(float(df.mag_f444w.median()), 2)
        if "mag_f444w" in df
        else None,
    }


out = {
    "note": "selection rates on whatever the archive observed within 0.5 arcsec; the observed subsets are targeted, not random; not purities",
    "model_candidates_all": stats(cand, "model candidates, all tiers"),
    "model_candidates_equal_burden": stats(
        cand[cand.tier == "equal_burden"], "model candidates, equal-burden tier"
    ),
    "model_candidates_followup": stats(
        cand[cand.followup_tier.astype(bool)], "model candidates, follow-up tier"
    )
    if "followup_tier" in cand
    else None,
    "union_novel_picks": stats(
        union_novel, "union of seven, in no list, unlabelled, unique"
    ),
}
p = pd.read_parquet(
    os.path.join(OUT, "features.parquet"), columns=["y", "z_phot", "in_support"]
)
p = p[(p.y == 1) & p.in_support]
out["known_positives_zphot_lt_3"] = f"{int((p.z_phot < 3).sum())} of {len(p)}"
json.dump(out, open(os.path.join(OUT, f"archive_purity{SUF}.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
