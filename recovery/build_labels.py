"""v4 step 1: the label and feature table for the recovery classifier.

One row per catalogued source (sources_v3, 630,869) joined to its photometry
(training.parquet). Label column `y`: 1 = spectroscopic LRD (in Hviding 2025 A1 or B1,
Barro 2025 or de Graaff 2026 within 0.5 arcsec, or one of the paper's 32 strict positives);
0 = census source not V-shaped under the strict rule and in no published LRD list of any
kind; NaN = unlabelled (scored at deployment) or ambiguous (census non-V but in a
photometric LRD list; excluded from both classes). Also `y_strict` (the 32) and the
per-list membership flags, the seven rule flags and `picked`.
Reads only pinned inputs; writes recovery/labels.parquet and a counts JSON."""

import json
import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "recovery"
os.makedirs(OUT, exist_ok=True)

s = pd.read_parquet(
    os.path.join("inputs", "sources_v3.parquet"),
    columns=[
        "source_id",
        "field",
        "id",
        "ra",
        "dec",
        "positive",
        "picked",
        "vshaped_spec",
        "sel_labbe23",
        "sel_kokorev24",
        "sel_kocevski24",
        "sel_perezgonzalez24",
        "sel_barro23",
        "sel_greene24",
        "sel_akins24",
        "flux_radius",
        "mag_f444w",
        "r_star_v1_arcsec",
        "r_h_arcsec",
        "log_ratio_v1",
        "accessible_60",
        "z_phot",
        "covariates_complete",
    ],
)
t = pd.read_parquet(
    os.path.join("inputs", "training.parquet")
)
bands = ["f090w", "f115w", "f150w", "f200w", "f277w", "f356w", "f444w"]
keep = (
    ["field", "id"]
    + [f"f_{b}" for b in bands]
    + [f"e_{b}" for b in bands]
    + [
        "c_f090w_f115w",
        "c_f115w_f150w",
        "c_f150w_f200w",
        "c_f200w_f277w",
        "c_f277w_f356w",
        "c_f356w_f444w",
        "snr_f444w",
        "has_spec",
        "in_census",
        "spec_tested",
        "spec_vshaped",
    ]
)
d = s.merge(t[keep], on=["field", "id"], how="left")
assert len(d) == len(s)

a = SkyCoord(d.ra.values * u.deg, d.dec.values * u.deg)
D = os.path.join("inputs", "prior_art_catalogues")


def members_from_fits(path):
    tab = Table.read(path)
    rc = [c for c in tab.colnames if c.lower() in ("ra", "ra_deg", "right_ascension")][
        0
    ]
    dc = [c for c in tab.colnames if c.lower() in ("dec", "dec_deg", "declination")][0]
    ra = np.asarray(tab[rc], float)
    dec = np.asarray(tab[dc], float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    b = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg)
    idx, sep, _ = b.match_to_catalog_sky(a)
    m = sep.arcsec < 0.5
    return set(d.source_id.values[idx][m]), int(ok.sum())


spec_lists = {
    "hviding25_A1": os.path.join(D, "hviding2025_zenodo_RUBIES-Hviding25-A1-lrds.fits"),
    "hviding25_B1": os.path.join(D, "hviding2025_zenodo_RUBIES-Hviding25-B1-bl.fits"),
    "barro25": os.path.join(
        D, "barro2025_github_Barro25_LRD_NIRSpec_bestfit_properties.fits"
    ),
    "degraaff26": os.path.join(
        D, "degraaff2026_zenodo_17665942_v0.1_lrds_withdups_blackbody_eline_fits.fits"
    ),
}
counts = {}
spec_members = set()
for name, path in spec_lists.items():
    mem, n = members_from_fits(path)
    d[f"in_{name}"] = d.source_id.isin(mem)
    counts[name] = {"rows": n, "in_fields": len(mem)}
    spec_members |= mem

# photometric lists on disk (external_labels_raw: perger25, kocevski24, plus the two spectroscopic ones)
e = pd.read_csv(os.path.join(D, "external_labels_raw.csv"))
b = SkyCoord(e.ra.values * u.deg, e.dec.values * u.deg)
idx, sep, _ = b.match_to_catalog_sky(a)
e["source_id"] = np.where(sep.arcsec < 0.5, d.source_id.values[idx], None)
phot_members = set()
for name in ("perger25", "kocevski24"):
    mem = set(e[(e.src == name) & e.source_id.notna()].source_id)
    d[f"in_{name}"] = d.source_id.isin(mem)
    counts[name] = {"rows": int((e.src == name).sum()), "in_fields": len(mem)}
    phot_members |= mem

# positives: the lists that call themselves LRD catalogs (Hviding 2025 A1, Barro 2025, de Graaff 2026)
# plus our strict 32; Hviding table B1 ("broad Balmer line galaxies") is not an LRD list and is
# neither a positive nor a negative source (its members are excluded from the negatives).
d["in_spec_list"] = d.in_hviding25_A1 | d.in_barro25 | d.in_degraaff26
d["in_any_list"] = d.in_spec_list | d.source_id.isin(phot_members)
d["y_strict"] = d.positive.astype(bool)
pos = (d.in_spec_list | d.y_strict).astype(bool)
notv = d.vshaped_spec.astype("boolean").eq(False).fillna(False).astype(bool)
neg = notv & ~d.in_any_list
amb = (notv & d.in_any_list & ~pos) | (d.in_hviding25_B1 & ~pos)
d["y"] = np.where(pos, 1.0, np.where(neg, 0.0, np.nan))
d["ambiguous"] = amb

full = d[[f"f_{b}" for b in bands]].notna().all(axis=1) & d[
    [f"e_{b}" for b in bands]
].notna().all(axis=1)
d["bands_complete"] = full

counts["labels"] = {
    "positives": int(pos.sum()),
    "positives_strict32": int(d.y_strict.sum()),
    "positives_with_bands": int((pos & full).sum()),
    "negatives": int(neg.sum()),
    "negatives_with_bands": int((neg & full).sum()),
    "ambiguous_excluded": int(amb.sum()),
    "unlabelled": int(d.y.isna().sum() - amb.sum()),
    "positives_picked_by_any_rule": int((pos & d.picked).sum()),
    "positives_missed_by_all_rules": int((pos & ~d.picked).sum()),
    "strict32_missed": int((d.y_strict & ~d.picked).sum()),
    "deploy_rows_with_bands": int(full.sum()),
}
counts["per_field"] = (
    d[pos | neg]
    .assign(pos=pos[pos | neg].astype(int))
    .groupby("field")
    .agg(labelled=("y", "size"), positives=("pos", "sum"))
    .astype(int)
    .to_dict("index")
)
d.to_parquet(os.path.join(OUT, "labels.parquet"), index=False)
with open(os.path.join(OUT, "label_counts.json"), "w", encoding="utf-8") as fh:
    json.dump(counts, fh, indent=2)
print(json.dumps(counts, indent=2))
