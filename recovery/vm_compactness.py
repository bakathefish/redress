"""Runs on the staging VM: the frozen aperture compactness of every candidate.

Uses the frozen producer of record, which is selections/redress/compactness.py in this
repository: forced centring at the
catalog position, exact-overlap circular apertures of 0.20 / 0.40 / 0.50 arcsec diameter on
the F444W science mosaic matched to the catalog version, no re-centroiding, an aperture that
leaves valid pixels makes the object unmeasurable. Mosaics missing from MOSAIC_DIR
are fetched from the DJA public store first (and removed afterwards if the disk is tight).

The producer's sha256 as it ran on the VM was c3134ff1...; the copy released here is the
same code with its header comments and its references to unreleased internal documents
rewritten, so it does not reproduce that hash. The run prints the hash of whatever it
actually loads, so the two can never silently diverge.

Usage (on the VM): venv/bin/python3 vm_compactness.py candidates.csv out.csv
"""

import hashlib
import os
import sys
import urllib.request

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

# The producer of record. On the VM it sat in PRODUCER_DIR next to the staged data; in
# this repository it is selections/redress/compactness.py, the same file, unchanged.
PRODUCER_DIR = os.environ.get(
    "PRODUCER_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "selections",
        "redress",
    ),
)
sys.path.insert(0, PRODUCER_DIR)
from compactness import compactness_columns, measure_apertures  # noqa: E402

M = os.environ.get("MOSAIC_DIR", "mosaics")
CUT = 121
VERSION = {
    "ceers-full": "v7.4",
    "gdn": "v7.4",
    "gds": "v7.2",
    "gds-sw": "v7.0",
    "ngdeep": "v7.2",
    "primer-cosmos-east": "v7.4",
    "primer-cosmos-west": "v7.4",
    "primer-uds-north": "v7.2",
    "primer-uds-south": "v7.2",
}
DJA = "https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7"


def fetch(field, kind):
    fn = f"{field}-grizli-{VERSION[field]}-f444w-clear_drc_{kind}.fits"
    path = os.path.join(M, fn)
    if os.path.exists(path):
        return path, False
    url = f"{DJA}/{fn}.gz"
    print("fetching", url, flush=True)
    gz = path + ".gz"
    urllib.request.urlretrieve(url, gz)
    os.system(f"gunzip -f '{gz}'")
    return path, True


cand = pd.read_csv(sys.argv[1])
print(
    "producer sha256:",
    hashlib.sha256(
        open(os.path.join(PRODUCER_DIR, "compactness.py"), "rb").read()
    ).hexdigest(),
    flush=True,
)
print(
    "candidates:", len(cand), "fields:", cand.field.value_counts().to_dict(), flush=True
)
out = []
for field, g in cand.groupby("field"):
    try:
        sci_path, dl1 = fetch(field, "sci")
        wht_path, dl2 = fetch(field, "wht")
    except Exception as exc:
        print(f"{field}: mosaic unavailable ({type(exc).__name__}: {exc})", flush=True)
        for r in g.itertuples():
            out.append(
                dict(
                    source_id=int(r.source_id),
                    field=field,
                    id=int(r.id),
                    measurable=False,
                    reason="mosaic unavailable",
                )
            )
        continue
    hs, hw = fits.open(sci_path, memmap=True), fits.open(wht_path, memmap=True)
    si = next(i for i, h in enumerate(hs) if h.data is not None and h.data.ndim == 2)
    wi = next(i for i, h in enumerate(hw) if h.data is not None and h.data.ndim == 2)
    w = WCS(hs[si].header)
    scale = float(np.abs(w.proj_plane_pixel_scales()[0].to(u.arcsec).value))
    ny, nx = hs[si].data.shape
    print(
        f'=== {field} shape={hs[si].data.shape} pixscale={scale:.4f}"/px n={len(g)}',
        flush=True,
    )
    for r in g.itertuples():
        rec = dict(
            source_id=int(r.source_id),
            field=field,
            id=int(r.id),
            mosaic=os.path.basename(sci_path),
            pixscale=scale,
        )
        pos = SkyCoord(ra=r.ra * u.deg, dec=r.dec * u.deg)
        try:
            px, py = w.world_to_pixel(pos)
        except Exception:
            out.append(dict(rec, measurable=False, reason="wcs failure"))
            continue
        if not (0 <= px < nx and 0 <= py < ny):
            out.append(dict(rec, measurable=False, reason="outside mosaic"))
            continue
        cs = Cutout2D(
            hs[si].data, pos, (CUT, CUT), wcs=w, mode="partial", fill_value=np.nan
        )
        cw = Cutout2D(
            hw[wi].data, pos, (CUT, CUT), wcs=w, mode="partial", fill_value=0.0
        )
        sci = np.array(cs.data, dtype=float)
        wht = np.array(cw.data, dtype=float)
        sci[~(wht > 0)] = np.nan
        if not np.isfinite(sci).any():
            out.append(dict(rec, measurable=False, reason="no coverage"))
            continue
        f = measure_apertures(sci, cs.wcs, [r.ra], [r.dec])
        c = compactness_columns(f)
        out.append(
            dict(
                rec,
                compactness_f444w=float(c["compactness_f444w"][0]),
                labbe_compactness=float(c["labbe_compactness"][0]),
                flux_d020=float(f["flux_d020"][0]),
                flux_d040=float(f["flux_d040"][0]),
                flux_d050=float(f["flux_d050"][0]),
                measurable=bool(f["measurable"][0]),
                reason=str(f["unmeasurable_reason"][0]),
            )
        )
    hs.close()
    hw.close()
    if dl1 or dl2:
        st = os.statvfs("/home")
        if st.f_bavail * st.f_frsize < 6e9:
            for p in (sci_path, wht_path):
                os.remove(p)
            print("removed fetched mosaics to free disk", flush=True)
pd.DataFrame(out).to_csv(sys.argv[2], index=False)
df = pd.DataFrame(out)
print(
    "measured:",
    int(df.measurable.sum()) if "measurable" in df else 0,
    "of",
    len(df),
    flush=True,
)
