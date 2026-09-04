"""Runs ON THE AZURE VM (the staging VM) via az vm run-command: pulls the aperture and shape
columns the v4 feature table does not yet carry from the staged DJA photometric catalogs
(the same catalog versions the pipeline used), one CSV.gz for all nine fields, and uploads
it to the transfer container with the SAS URL given as argv[1].

Columns: id, ra, dec, flux_radius_20 / flux_radius / flux_radius_90 (pixels), a_image,
b_image, theta_image, kron_radius, kron_rcirc, and for F150W, F200W, F277W, F356W, F444W
the flux and error in the two catalog apertures (aper_0 = 0.36 arcsec, aper_1 = 0.50 arcsec
diameter), plus F410M aper_1 flux/error where the field has the band (information only)."""

import gzip
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

STAGING = "/home/worker/dja_staging"
OUT = os.environ.get("SHAPE_OUT", "v4_shape_columns.csv.gz")
CAT = {
    "ceers-full": "ceers-full-grizli-v7.4-fix_phot_apcorr.fits",
    "gdn": "gdn-grizli-v7.4-fix_phot_apcorr.fits",
    "gds": "gds-grizli-v7.2-fix_phot_apcorr.fits",
    "gds-sw": "gds-sw-grizli-v7.0-fix_phot_apcorr.fits",
    "ngdeep": "ngdeep-grizli-v7.2-fix_phot_apcorr.fits",
    "primer-cosmos-east": "primer-cosmos-east-grizli-v7.4-fix_phot_apcorr.fits",
    "primer-cosmos-west": "primer-cosmos-west-grizli-v7.4-fix_phot_apcorr.fits",
    "primer-uds-north": "primer-uds-north-grizli-v7.2-fix_phot_apcorr.fits",
    "primer-uds-south": "primer-uds-south-grizli-v7.2-fix_phot_apcorr.fits",
}
EXPECTED_ROWS = {
    "ceers-full": 80880,
    "gdn": 92463,
    "gds": 53846,
    "gds-sw": 48654,
    "ngdeep": 24761,
    "primer-cosmos-east": 75671,
    "primer-cosmos-west": 102814,
    "primer-uds-north": 82565,
    "primer-uds-south": 69215,
}
SHAPE = [
    "flux_radius_20",
    "flux_radius",
    "flux_radius_90",
    "a_image",
    "b_image",
    "theta_image",
    "kron_radius",
    "kron_rcirc",
]
BANDS = ["f150w", "f200w", "f277w", "f356w", "f444w"]
frames = []
for field, fn in CAT.items():
    path = os.path.join(STAGING, fn)
    with fits.open(path, memmap=True) as h:
        cols = set(h[1].columns.names)
        want = ["id", "ra", "dec"] + [c for c in SHAPE if c in cols]
        for b in BANDS + ["f410m"]:
            for k in ("flux", "fluxerr"):
                for i in (0, 1):
                    c = f"{b}_{k}_aper_{i}"
                    if c in cols:
                        want.append(c)
        data = h[1].data
        df = pd.DataFrame({c: np.array(data[c], dtype=np.float64) for c in want})
    df.insert(0, "field", field)
    df.insert(1, "catalog_file", fn)
    n_exp = EXPECTED_ROWS[field]
    print(
        f"{field}: {len(df)} rows (expected {n_exp}) {'OK' if len(df) == n_exp else 'MISMATCH'}, {len(want)} cols",
        flush=True,
    )
    frames.append(df)
allf = pd.concat(frames, ignore_index=True)
allf.to_csv(OUT, index=False, compression="gzip")
print(
    "rows", len(allf), "cols", allf.shape[1], "bytes", os.path.getsize(OUT), flush=True
)
if len(sys.argv) > 1:
    sas = sys.argv[1]
    url = sas.replace("?", "/v4_shape_columns.csv.gz?", 1) if "?" in sas else sas
    r = subprocess.run(
        [
            "curl",
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "-X",
            "PUT",
            "-H",
            "x-ms-blob-type: BlockBlob",
            "-T",
            OUT,
            url,
        ],
        capture_output=True,
        text=True,
    )
    print("upload http", r.stdout, r.stderr[-200:], flush=True)
