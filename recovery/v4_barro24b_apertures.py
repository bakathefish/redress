"""Revision step 22: the Barro et al. (2024b, arXiv:2412.01887) comparator, colors and apertures.

Added at referee request after the benchmark of record was fixed. It is reported as an eighth
comparator next to the seven of ``redress.cuts.CUTS`` and is never folded into their union.

Criteria (their Sec. 2.2): (i) F200W-F444W > 1; (ii) F200W-F444W > F115W-F200W + 0.25;
(iii) F115W-F200W > -0.5; (iv) F444W < 27; and (their Sec. 2.3) the compactness
F_F444W(r=0.5")/F_F444W(r=0.2") < 1.5 in apertures of RADIUS 0.5" and 0.2" (their Fig. 2
caption; a stellar locus near 1.2 is consistent only with radii). Colors are AB from the DJA
catalog fluxes of the unreleased label table; a non-positive flux in any of the three bands
fails closed, because the paper states no upper-limit convention.

This script applies (i)-(iv) to every catalog row and writes ``barro24b_colors.parquet``, then
measures the two apertures for every survivor on an F444W cutout from the DJA grizli cutout
service (the v7 mosaics the catalog photometry was measured on, 0.05"/pixel), with forced
centering at the catalog position, exact-overlap circular apertures and no recentering. An
aperture touching a non-finite or zero-weight pixel (stored as an exact zero) makes the source
unmeasurable, and so does a non-positive inner flux. The mosaic version of each cutout is kept
in the ``grizliv`` column. Writes ``barro24b_apertures.csv``; the selection itself is applied by
``paper/referee_compute_2026_09_06.py`` through ``redress.cuts.barro24b.select``.

Usage: python recovery/v4_barro24b_apertures.py  (resumable; cutouts cached under inputs/barro24b_cutouts)
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
CACHE = os.path.join("inputs", "barro24b_cutouts")
OUT_COLORS = os.path.join("recovery", "barro24b_colors.parquet")
OUT_APER = os.path.join("recovery", "barro24b_apertures.csv")
URL = (
    "https://grizli-cutout.herokuapp.com/thumb?ra={ra:.6f}&dec={dec:.6f}"
    "&size=3&filters=f444w-clear&output=fits"
)
R_IN, R_OUT = 0.2, 0.5  # arcsec, RADII
os.makedirs(CACHE, exist_ok=True)


def ab(f):
    f = np.asarray(f, float)
    m = np.full(f.shape, np.nan)
    ok = f > 0
    m[ok] = 23.9 - 2.5 * np.log10(f[ok])
    return m


lab = pd.read_parquet(
    LABELS,
    columns=[
        "source_id",
        "field",
        "id",
        "ra",
        "dec",
        "y",
        "f_f115w",
        "f_f200w",
        "f_f444w",
        "picked",
    ],
)
m115, m200, m444 = ab(lab.f_f115w), ab(lab.f_f200w), ab(lab.f_f444w)
c1, c2 = m200 - m444, m115 - m200
col = (c1 > 1.0) & (c1 > c2 + 0.25) & (c2 > -0.5) & (m444 < 27.0)
col = np.where(np.isfinite(c1) & np.isfinite(c2) & np.isfinite(m444), col, False)
lab["barro24b_colors"] = col
lab["m444_ab"] = m444
lab["c_f200w_f444w_ab"] = c1
lab["c_f115w_f200w_ab"] = c2
lab[
    [
        "source_id",
        "field",
        "id",
        "barro24b_colors",
        "m444_ab",
        "c_f200w_f444w_ab",
        "c_f115w_f200w_ab",
    ]
].to_parquet(OUT_COLORS, index=False)
surv = lab[lab.barro24b_colors].copy()
print(
    "color+mag survivors:",
    len(surv),
    "of which positives:",
    int((surv.y == 1).sum()),
    flush=True,
)

rows, done = [], {}
if os.path.exists(OUT_APER):
    prev = pd.read_csv(OUT_APER)
    done = {int(s): True for s in prev.source_id}
    rows = prev.to_dict("records")
t0, n = time.time(), 0
for r in surv.itertuples():
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
        ap_in = CircularAperture((px, py), r=R_IN / scale)
        ap_out = CircularAperture((px, py), r=R_OUT / scale)
        touched = ap_out.to_mask(method="exact").to_image(sci.shape) > 0
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
        f_in = float(aperture_photometry(sci, ap_in, method="exact")["aperture_sum"][0])
        f_out = float(
            aperture_photometry(sci, ap_out, method="exact")["aperture_sum"][0]
        )
        rows.append(
            dict(
                rec,
                grizliv=hd.header.get("GRIZLIV"),
                pixscale=scale,
                f_r02=f_in,
                f_r05=f_out,
                ratio_r05_r02=f_out / f_in if f_in > 0 else np.nan,
                measurable=bool(f_in > 0),
                reason="" if f_in > 0 else "non-positive inner flux",
            )
        )
    except Exception as exc:
        rows.append(dict(rec, measurable=False, reason=f"error {type(exc).__name__}"))
    n += 1
    if n % 25 == 0:
        pd.DataFrame(rows).to_csv(OUT_APER, index=False)
        print(f"{n} measured, {len(rows)} total, {time.time() - t0:.0f} s", flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUT_APER, index=False)
print(
    "done",
    len(df),
    "measurable",
    int(df.measurable.sum()),
    "compact(<1.5)",
    int((df.ratio_r05_r02 < 1.5).sum()),
)
