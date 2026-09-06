"""Revision step 24: the Labbe et al. (2023) aperture compactness of the 151 positives.

The candidate catalog carries the frozen aperture compactness (step 14, circular apertures of
0.20, 0.40 and 0.50 arcsec DIAMETER on the F444W mosaic, forced centering at the catalog
position, exact-overlap pixel integration, no recentering, no background subtraction). The
positives did not, so the compactness fraction of the candidates could only be compared with
the positives on the catalog half-light radius. This script measures the same three apertures
for every positive on an F444W cutout from the DJA grizli cutout service (the v7 mosaics the
catalog photometry was measured on), so that the Labbe et al. (2023) and Akins et al. (2024)
compactness criteria can be read on the positives and the candidates alike.

Rules, as in step 14 and step 22: an aperture touching a non-finite or zero-weight pixel
(stored as an exact zero) makes the source unmeasurable, and so does a non-positive inner flux.
The mosaic version of each cutout is kept in the ``grizliv`` column.

Usage: python recovery/v4_positive_apertures.py  (resumable; cutouts cached under
inputs/positive_cutouts). Writes recovery/positive_apertures.csv.
"""

import os
import time
import urllib.request

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, aperture_photometry

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LABELS = os.path.join("inputs", "labels.parquet")  # not released (see RUN_MANIFEST.md)
CACHE = os.path.join("inputs", "positive_cutouts")
OUT = os.path.join("recovery", "positive_apertures.csv")
URL = (
    "https://grizli-cutout.herokuapp.com/thumb?ra={ra:.6f}&dec={dec:.6f}"
    "&size=3&filters=f444w-clear&output=fits"
)
DIAM = {"d020": 0.20, "d040": 0.40, "d050": 0.50}  # arcsec, DIAMETERS
os.makedirs(CACHE, exist_ok=True)

lab = pd.read_parquet(LABELS, columns=["source_id", "field", "id", "ra", "dec", "y"])
pos = lab[lab.y == 1].copy()
print("positives:", len(pos), flush=True)

rows, done = [], {}
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    done = {int(s): True for s in prev.source_id}
    rows = prev.to_dict("records")
t0, n = time.time(), 0
for r in pos.itertuples():
    if int(r.source_id) in done:
        continue
    fn = os.path.join(CACHE, f"{r.field}_{r.id}.fits")
    rec = dict(
        source_id=int(r.source_id), field=r.field, id=int(r.id), ra=r.ra, dec=r.dec
    )
    if not os.path.exists(fn):
        ok = False
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(URL.format(ra=r.ra, dec=r.dec), fn)
                ok = os.path.getsize(fn) > 5000
                if ok:
                    break
            except Exception:
                time.sleep(5 * (attempt + 1))
        if not ok:
            if os.path.exists(fn):
                os.remove(fn)
            rows.append(dict(rec, measurable=False, reason="cutout unavailable"))
            continue
    try:
        h = fits.open(fn)
        hd = next(x for x in h if x.data is not None and x.data.ndim == 2)
        w = WCS(hd.header)
        scale = float(np.abs(w.proj_plane_pixel_scales()[0].to(u.arcsec).value))
        sci = np.array(hd.data, dtype=float)
        sci[sci == 0.0] = np.nan
        px, py = w.world_to_pixel(SkyCoord(r.ra * u.deg, r.dec * u.deg))
        aps = {
            k: CircularAperture((px, py), r=0.5 * d / scale) for k, d in DIAM.items()
        }
        touched = aps["d050"].to_mask(method="exact").to_image(sci.shape) > 0
        inside = 0 <= px < sci.shape[1] and 0 <= py < sci.shape[0]
        if not inside or not np.isfinite(sci[touched]).all():
            rows.append(
                dict(
                    rec,
                    grizliv=hd.header.get("GRIZLIV"),
                    pixscale=scale,
                    measurable=False,
                    reason="aperture leaves valid footprint",
                )
            )
            continue
        f = {
            k: float(aperture_photometry(sci, ap, method="exact")["aperture_sum"][0])
            for k, ap in aps.items()
        }
        good = f["d020"] > 0
        rows.append(
            dict(
                rec,
                grizliv=hd.header.get("GRIZLIV"),
                pixscale=scale,
                flux_d020=f["d020"],
                flux_d040=f["d040"],
                flux_d050=f["d050"],
                labbe_compactness=f["d040"] / f["d020"] if good else np.nan,
                compactness_f444w=f["d020"] / f["d050"] if f["d050"] > 0 else np.nan,
                measurable=bool(good),
                reason="" if good else "non-positive inner flux",
            )
        )
    except Exception as exc:
        rows.append(dict(rec, measurable=False, reason=f"error {type(exc).__name__}"))
    n += 1
    if n % 25 == 0:
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"{n} measured, {len(rows)} total, {time.time() - t0:.0f} s", flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(
    "done",
    len(df),
    "measurable",
    int(df.measurable.sum()),
    "labbe compact(<1.7)",
    int((df.labbe_compactness < 1.7).sum()),
    "akins window",
    int(((df.compactness_f444w > 0.5) & (df.compactness_f444w <= 0.7)).sum()),
)
