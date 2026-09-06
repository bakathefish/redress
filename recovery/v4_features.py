"""v4 build step 1-3 in one script: label audit + denominators, the 30-feature table with
its manifest, and the five macroregion folds with 0.5-arcsec sky groups.

Design of record: the design round of 2026-09-04.

Inputs: recovery/labels.parquet (build_labels.py; 630,869 rows), the three spectroscopic
LRD lists (for the positional audit). No label is read while any feature constant is fitted:
the asinh softening per band is the median flux error over all band-complete rows, and the
stellar radius per field is the frozen catalogue r_star_v1 (label-free instrument nuisance).

Outputs (recovery/):
  features.parquet     one row per source with all 30 features (float32), flags and labels
  feature_manifest.json feature names, constants, support counts, fold check, table hash
  denominators.json    every denominator the write-up quotes (151/147/44/41/32/7, per region,
                       per catalog, union burden on the complete mask)
  label_audit.csv      one row per positive: matched-list separations and crowding counts
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "selections")
from redress.folds import sky_groups  # noqa: E402

OUT = "recovery"
BANDS = ["f090w", "f115w", "f150w", "f200w", "f277w", "f356w", "f444w"]
LAM_UM = {  # NIRCam pivot wavelengths, microns (JWST user documentation)
    "f090w": 0.901,
    "f115w": 1.154,
    "f150w": 1.501,
    "f200w": 1.990,
    "f277w": 2.786,
    "f356w": 3.563,
    "f444w": 4.421,
}
REGION = {
    "ceers-full": "CEERS",
    "gdn": "GOODS-N",
    "gds": "GOODS-S",
    "gds-sw": "GOODS-S",
    "ngdeep": "GOODS-S",
    "primer-cosmos-east": "COSMOS",
    "primer-cosmos-west": "COSMOS",
    "primer-uds-north": "UDS",
    "primer-uds-south": "UDS",
}
K = 2.5 / np.log(10.0)

d = pd.read_parquet(os.path.join(OUT, "labels.parquet"))
assert len(d) == 630869, len(d)
d["region"] = d.field.map(REGION)
assert d.region.notna().all()

# ---------------------------------------------------------------- sky groups (0.5 arcsec)
d["sky_group"] = sky_groups(d.ra.values, d.dec.values, 0.5).astype(np.int64)
gsize = d.groupby("sky_group").size()
d["sky_group_size"] = d.sky_group.map(gsize).astype(np.int32)
# fold unit test: no sky group may span two macroregions
span = d.groupby("sky_group").region.nunique()
n_cross = int((span > 1).sum())
assert n_cross == 0, f"{n_cross} sky groups cross macroregions"

# ---------------------------------------------------------------- features (label-free)
full = d.bands_complete.values.astype(bool)
F = np.column_stack([d[f"f_{b}"].values for b in BANDS]).astype(np.float64)
E = np.column_stack([d[f"e_{b}"].values for b in BANDS]).astype(np.float64)
soft = {b: float(np.nanmedian(E[full, i])) for i, b in enumerate(BANDS)}
S = np.array([soft[b] for b in BANDS])

with np.errstate(all="ignore"):
    M = -K * (np.arcsinh(F / (2.0 * S)) + np.log(S))  # asinh magnitudes
    SNR = F / np.where(E > 0, E, np.nan)
    ASNR = np.arcsinh(SNR)
    # magnitude uncertainty from the asinh derivative; weights capped at sigma = 0.01 mag
    SIG_M = K * E / np.sqrt(F * F + (2.0 * S) ** 2)
    W = 1.0 / np.clip(SIG_M, 0.01, None) ** 2


def wslope(cols):
    """Weighted least-squares slope of asinh magnitude against log10(lambda) over cols."""
    x = np.log10(np.array([LAM_UM[BANDS[c]] for c in cols]))
    y = M[:, cols]
    w = W[:, cols]
    ok = np.isfinite(y) & np.isfinite(w)
    w = np.where(ok, w, 0.0)
    y = np.where(ok, y, 0.0)
    n = ok.sum(axis=1)
    sw = w.sum(axis=1)
    with np.errstate(all="ignore"):
        xm = (w * x).sum(axis=1) / sw
        ym = (w * y).sum(axis=1) / sw
        num = (w * (x - xm[:, None]) * (y - ym[:, None])).sum(axis=1)
        den = (w * (x - xm[:, None]) ** 2).sum(axis=1)
        s = num / den
    s = np.where((n >= 2) & np.isfinite(s), s, 0.0)
    return s


X = pd.DataFrame(index=d.index)
for i, b in enumerate(BANDS):
    X[f"m_{b}"] = M[:, i]
for i, b in enumerate(BANDS):
    X[f"asnr_{b}"] = ASNR[:, i]
for b1, b2 in zip(BANDS[:-1], BANDS[1:]):
    X[f"c_{b1}_{b2}"] = X[f"m_{b1}"] - X[f"m_{b2}"]
X["c_f115w_f200w"] = X.m_f115w - X.m_f200w
X["c_f200w_f356w"] = X.m_f200w - X.m_f356w
X["c_f277w_f444w"] = X.m_f277w - X.m_f444w
X["slope_blue"] = wslope([0, 1, 2, 3])
X["slope_red"] = wslope([4, 5, 6])
X["v_curv"] = X.slope_red - X.slope_blue
rh = np.clip(d.r_h_arcsec.values.astype(np.float64), 0.004, None)  # 0.1 px floor
X["log_rh"] = np.log10(rh)
X["log_rh_over_rstar"] = np.log10(rh / d.r_star_v1_arcsec.values.astype(np.float64))
X["n_snr_ge1"] = np.nansum(SNR >= 1.0, axis=1).astype(np.float64)
X["n_snr_ge3"] = np.nansum(SNR >= 3.0, axis=1).astype(np.float64)
FEATURES = list(X.columns)
assert len(FEATURES) == 30, FEATURES

support = full & np.isfinite(X.values).all(axis=1) & (d.r_star_v1_arcsec.values > 0)
d["in_support"] = support

# ---------------------------------------------------------------- label audit (positives)
pos = d.y.values == 1
neg = d.y.values == 0
a = SkyCoord(d.ra.values * u.deg, d.dec.values * u.deg)
D = os.path.join("inputs", "prior_art_catalogues")
LISTS = {
    "hviding25_A1": "hviding2025_zenodo_RUBIES-Hviding25-A1-lrds.fits",
    "barro25": "barro2025_github_Barro25_LRD_NIRSpec_bestfit_properties.fits",
    "degraaff26": "degraaff2026_zenodo_17665942_v0.1_lrds_withdups_blackbody_eline_fits.fits",
}
sep_cols = {}
for name, fn in LISTS.items():
    tab = Table.read(os.path.join(D, fn))
    rc = [c for c in tab.colnames if c.lower() in ("ra", "ra_deg", "right_ascension")][
        0
    ]
    dc = [c for c in tab.colnames if c.lower() in ("dec", "dec_deg", "declination")][0]
    ra = np.asarray(tab[rc], float)
    dec = np.asarray(tab[dc], float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    b = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg)
    idx, sep, _ = a.match_to_catalog_sky(b)  # nearest list row for every source
    sep_cols[name] = np.where(d[f"in_{name}"].values, sep.arcsec, np.nan)
pidx = np.where(pos)[0]
pc = a[pidx]
# search_around_sky returns pairs; count per positive directly
ia, ip, s2, _ = pc.search_around_sky(a, 1.0 * u.arcsec)  # ia indexes a, ip indexes pc
n_within_05 = np.bincount(ip[s2.arcsec <= 0.5], minlength=len(pidx))
n_within_10 = np.bincount(ip, minlength=len(pidx))
audit = pd.DataFrame(
    {
        "source_id": d.source_id.values[pidx],
        "field": d.field.values[pidx],
        "id": d.id.values[pidx],
        "region": d.region.values[pidx],
        "ra": d.ra.values[pidx],
        "dec": d.dec.values[pidx],
        "sky_group": d.sky_group.values[pidx],
        "sky_group_size": d.sky_group_size.values[pidx],
        "n_sources_within_0p5": n_within_05,
        "n_sources_within_1p0": n_within_10,
        "sep_hviding25_A1": sep_cols["hviding25_A1"][pidx],
        "sep_barro25": sep_cols["barro25"][pidx],
        "sep_degraaff26": sep_cols["degraaff26"][pidx],
        "in_hviding25_A1": d.in_hviding25_A1.values[pidx],
        "in_barro25": d.in_barro25.values[pidx],
        "in_degraaff26": d.in_degraaff26.values[pidx],
        "y_strict": d.y_strict.values[pidx],
        "picked_by_union": d.picked.values[pidx],
        "bands_complete": full[pidx],
        "in_support": support[pidx],
        "mag_f444w": d.mag_f444w.values[pidx],
    }
)
audit["crowded_0p5"] = audit.n_sources_within_0p5 > 1
audit["n_catalogs"] = audit[["in_hviding25_A1", "in_barro25", "in_degraaff26"]].sum(
    axis=1
)
audit.to_csv(os.path.join(OUT, "label_audit.csv"), index=False)

# ---------------------------------------------------------------- denominators
picked = d.picked.values.astype(bool)
strict = d.y_strict.values.astype(bool)


def per(mask):
    return {
        r: int((mask & (d.region.values == r)).sum())
        for r in sorted(set(REGION.values()))
    }


den = {
    "positives_151": int(pos.sum()),
    "positives_in_support_147": int((pos & support).sum()),
    "positives_incomplete_bands": int((pos & ~support).sum()),
    "rule_missed_44": int((pos & ~picked).sum()),
    "rule_missed_in_support_41": int((pos & ~picked & support).sum()),
    "strict_32": int(strict.sum()),
    "strict_in_support": int((strict & support).sum()),
    "strict_missed_7": int((strict & ~picked).sum()),
    "strict_missed_in_support": int((strict & ~picked & support).sum()),
    "union_selects_of_151": int((pos & picked).sum()),
    "negatives_5397": int(neg.sum()),
    "negatives_in_support": int((neg & support).sum()),
    "ambiguous_excluded": int(d.ambiguous.values.sum()),
    "complete_rows": int(full.sum()),
    "support_rows": int(support.sum()),
    "union_selected_in_support": int((picked & support).sum()),
    "union_burden_fraction_in_support": float((picked & support).sum() / support.sum()),
    "positives_per_region": per(pos),
    "positives_in_support_per_region": per(pos & support),
    "rule_missed_in_support_per_region": per(pos & ~picked & support),
    "negatives_in_support_per_region": per(neg & support),
    "support_rows_per_region": per(support),
    "union_in_support_per_region": per(picked & support),
    "positives_per_catalog": {k: int((pos & d[f"in_{k}"].values).sum()) for k in LISTS},
    "positives_only_in_one_catalog": int(
        ((audit.n_catalogs == 1) & ~audit.y_strict).sum()
    ),
    "positives_crowded_0p5": int(audit.crowded_0p5.sum()),
    "positives_in_multi_row_sky_group": int((audit.sky_group_size > 1).sum()),
    "sky_groups_total": int(d.sky_group.nunique()),
    "sky_groups_crossing_regions": n_cross,
}

# ---------------------------------------------------------------- write
cols_keep = [
    "source_id",
    "field",
    "region",
    "id",
    "ra",
    "dec",
    "sky_group",
    "sky_group_size",
    "y",
    "y_strict",
    "picked",
    "ambiguous",
    "bands_complete",
    "in_support",
    "sel_labbe23",
    "sel_kokorev24",
    "sel_kocevski24",
    "sel_perezgonzalez24",
    "sel_barro23",
    "sel_greene24",
    "sel_akins24",
    "in_hviding25_A1",
    "in_hviding25_B1",
    "in_barro25",
    "in_degraaff26",
    "in_perger25",
    "in_kocevski24",
    "in_spec_list",
    "in_any_list",
    "mag_f444w",
    "snr_f444w",
    "z_phot",
    "r_h_arcsec",
    "r_star_v1_arcsec",
    "has_spec",
    "in_census",
    "spec_tested",
    "spec_vshaped",
]
out = d[cols_keep].copy()
for c in FEATURES:
    out[c] = X[c].astype(np.float32).values
out.to_parquet(os.path.join(OUT, "features.parquet"), index=False)
h = hashlib.sha256(open(os.path.join(OUT, "features.parquet"), "rb").read()).hexdigest()
manifest = {
    "features": FEATURES,
    "n_features": len(FEATURES),
    "asinh_softening_uJy_per_band": soft,
    "softening_rule": "median flux error over all band-complete rows (label-free)",
    "slope_weights": "inverse variance of the asinh magnitude, sigma floored at 0.01 mag; slope 0 when fewer than two finite bands",
    "size": "r_h_arcsec floored at 0.004 arcsec; r_star_v1_arcsec per field (frozen catalogue constant)",
    "regions": REGION,
    "sky_group_radius_arcsec": 0.5,
    "rows": int(len(out)),
    "support_rows": int(support.sum()),
    "features_parquet_sha256": h,
}
json.dump(manifest, open(os.path.join(OUT, "feature_manifest.json"), "w"), indent=2)
json.dump(den, open(os.path.join(OUT, "denominators.json"), "w"), indent=2)
print(json.dumps(den, indent=1))
print("softening:", soft)
print("features.parquet sha256", h)
