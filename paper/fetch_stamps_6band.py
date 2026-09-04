"""Six-band cutouts for the objects that are already in the two figure manifests.

Round 3 presents each object as a strip of single-band panels plus the colour composite,
the way Akins et al. (2025) Fig. 4 and Labbe et al. (2025) Fig. 4 do. The three-band files
fetch_stamps.py wrote carry only F150W, F277W and F444W, so the three short-wavelength
bands and F356W have to be fetched.

Nothing about the object list changes. Every row here comes from
figures/stamps/manifest.csv or figures/manifest_extra.csv, at the coordinates those files
already record, from the same DJA cutout service with the same URL pattern, the same
size=5 radius and the same four-attempt retry as fetch_stamps.py. The existing three-band
files are never touched; the new ones are written beside them as
figures/stamps/<group>_<field>_<id>_6band.fits.

Run from the repository root: python paper/fetch_stamps_6band.py
"""

import os
import time
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "paper")
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(FIG, "stamps")

# the same service, the same radius, the same output; only the filter list is longer
BANDS6 = [
    "f115w-clear",
    "f150w-clear",
    "f200w-clear",
    "f277w-clear",
    "f356w-clear",
    "f444w-clear",
]
URL = (
    "https://grizli-cutout.herokuapp.com/thumb?ra={ra:.6f}&dec={dec:.6f}"
    "&size=5&filters=" + ",".join(BANDS6) + "&output=fits"
)
GAP = 4.0  # seconds between requests, as the brief sets


def valid(p, minsize=50000):
    try:
        with open(p, "rb") as fh:
            return fh.read(6) == b"SIMPLE" and os.path.getsize(p) > minsize
    except OSError:
        return False


def targets():
    """Every object already in the two manifests, keyed by the stamp name it owns."""
    rows = {}
    man = pd.read_csv(os.path.join(OUT, "manifest.csv"))
    for r in man.itertuples():
        rows[f"{r.group}_{r.field}_{r.id}"] = (float(r.ra), float(r.dec))
    ex = pd.read_csv(os.path.join(FIG, "manifest_extra.csv"))
    for r in ex.itertuples():
        # the stamp column names the three-band file this object already owns; the six-band
        # file takes the same stem, so fig_miss_anatomy reuses the gallery's "recovered_*"
        # objects and only the anatomy object adds a stem of its own
        stem = str(r.stamp)[: -len(".fits")]
        rows.setdefault(stem, (float(r.ra), float(r.dec)))
    return rows


def main():
    rows = targets()
    print(f"{len(rows)} objects, no new ones: {sorted(rows)}")
    got, missing = 0, []
    for i, (stem, (ra, dec)) in enumerate(sorted(rows.items())):
        dest = os.path.join(OUT, stem + "_6band.fits")
        if valid(dest):
            got += 1
            print(f"have  {stem}_6band.fits", flush=True)
            continue
        if i:
            time.sleep(GAP)
        url = URL.format(ra=ra, dec=dec)
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(url, dest)
                if valid(dest):
                    got += 1
                    print(
                        f"wrote {stem}_6band.fits  {os.path.getsize(dest) / 1e6:.2f} MB",
                        flush=True,
                    )
                    break
            except Exception as e:  # noqa: BLE001
                print("retry", stem, type(e).__name__, e, flush=True)
            time.sleep(3 * (attempt + 1))
        else:
            print("FAILED", stem, url, flush=True)
            missing.append(stem)
    print(f"\n{got} of {len(rows)} six-band stamps on disk")
    if missing:
        print("missing:", missing)


if __name__ == "__main__":
    main()
