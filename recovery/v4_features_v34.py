"""v4 feature-set variant "v34": the 30 features plus four label-free compactness and shape
measures from the DJA photometric catalogs (extracted on the VM by vm_extract_shape.py from
the same catalog versions the pipeline used):

  conc_f444w   asinh-magnitude difference between the 0.36" and 0.50" diameter apertures in
               F444W (a concentration index; the LRD hallmark is compactness in the red)
  conc_f200w   the same in F200W (the blue side)
  log_r90_r20  log10 of the radius enclosing 90% of the light over the 20% radius
  axis_ratio   b_image / a_image

Writes recovery/features_v34.parquet and feature_manifest_v34.json. Rows whose new
features are not finite leave the support (reported)."""

import hashlib
import json
import os

import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "recovery"
K = 2.5 / np.log(10.0)

f = pd.read_parquet(os.path.join(OUT, "features.parquet"))
man = json.load(open(os.path.join(OUT, "feature_manifest.json")))
soft = man["asinh_softening_uJy_per_band"]
s = pd.read_csv(os.path.join(OUT, "v4_shape_columns.csv.gz"))
s["id"] = s.id.astype(np.int64)
cols = [
    "field",
    "id",
    "flux_radius_20",
    "flux_radius",
    "flux_radius_90",
    "a_image",
    "b_image",
    "f444w_flux_aper_0",
    "f444w_flux_aper_1",
    "f200w_flux_aper_0",
    "f200w_flux_aper_1",
]
s = s[cols].drop_duplicates(["field", "id"])
m = f.merge(s, on=["field", "id"], how="left", validate="one_to_one")
assert len(m) == len(f)
print("rows with shape columns:", int(m.flux_radius_20.notna().sum()), "of", len(m))


def amag(flux, band):
    sb = soft[band]
    return -K * (np.arcsinh(flux / (2.0 * sb)) + np.log(sb))


with np.errstate(all="ignore"):
    m["conc_f444w"] = amag(m.f444w_flux_aper_0.values, "f444w") - amag(
        m.f444w_flux_aper_1.values, "f444w"
    )
    m["conc_f200w"] = amag(m.f200w_flux_aper_0.values, "f200w") - amag(
        m.f200w_flux_aper_1.values, "f200w"
    )
    m["log_r90_r20"] = np.log10(
        np.clip(m.flux_radius_90.values, 0.5, None)
        / np.clip(m.flux_radius_20.values, 0.5, None)
    )
    m["axis_ratio"] = np.clip(m.b_image.values / m.a_image.values, 0.0, 1.0)
NEW = ["conc_f444w", "conc_f200w", "log_r90_r20", "axis_ratio"]
newok = np.isfinite(m[NEW].values).all(axis=1)
was = m.in_support.values.astype(bool)
m["in_support"] = was & newok
print(
    "support before",
    int(was.sum()),
    "after",
    int(m.in_support.sum()),
    "| positives in support before",
    int((was & (m.y == 1)).sum()),
    "after",
    int((m.in_support & (m.y == 1)).sum()),
)
for c in NEW:
    m[c] = m[c].astype(np.float32)
keep = [c for c in f.columns] + NEW
m[keep].to_parquet(os.path.join(OUT, "features_v34.parquet"), index=False)
h = hashlib.sha256(
    open(os.path.join(OUT, "features_v34.parquet"), "rb").read()
).hexdigest()
man2 = dict(man)
man2["features"] = man["features"] + NEW
man2["n_features"] = len(man2["features"])
man2["shape_columns_source"] = (
    "DJA fix_phot_apcorr catalogs (same versions as the pipeline), extracted on the staging VM by vm_extract_shape.py; apertures 0.36 and 0.50 arcsec diameter"
)
man2["support_rows"] = int(m.in_support.sum())
man2["features_parquet_sha256"] = h
man2["positives_in_support"] = int((m.in_support & (m.y == 1)).sum())
json.dump(man2, open(os.path.join(OUT, "feature_manifest_v34.json"), "w"), indent=2)
print(
    "positives conc_f444w median:",
    float(np.nanmedian(m.conc_f444w[m.y == 1])),
    "| all support median:",
    float(np.nanmedian(m.conc_f444w[m.in_support])),
)
print(
    "positives log_r90_r20 median:",
    float(np.nanmedian(m.log_r90_r20[m.y == 1])),
    "| all:",
    float(np.nanmedian(m.log_r90_r20[m.in_support])),
)
print("features_v34.parquet sha256", h)
