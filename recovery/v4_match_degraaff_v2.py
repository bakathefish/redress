"""Match the released candidates against the 17 August 2026 version of the de Graaff et al.
(2026) LRD sample (Zenodo record 21977747, 181 spectra of 146 unique sources, `use_dG26`
flag) and add two columns to candidates.csv:

    in_degraaff26_v2   True where a `use_dG26` source lies within 0.5 arcsec
    zspec_degraaff26_v2  its published spectroscopic redshift (blank otherwise)

The labels, the anchors, the model and every count in the paper were frozen with version 0.1
of that release (Zenodo record 17665942, 20 November 2025, 116 unique sources), and this
script does not change any of them: it records which candidates a later version of one of
the three reference lists has since published as spectroscopic LRDs. Three candidates match
(ranks 78, 154 and 176); all three are marked `untested` because the archive index was frozen
with the lists.

Every other column of candidates.csv is written back byte for byte (the file is read and
written as text). Run from the repository root:

    python recovery/v4_match_degraaff_v2.py
"""

from __future__ import annotations

import os
import sys
import urllib.request

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = os.path.join(HERE, "candidates.csv")
URL = (
    "https://zenodo.org/records/21977747/files/"
    "deGraaff2026_mnras_lrds_withdups_blackbody_eline_fits.fits?download=1"
)
CACHE = os.path.join(HERE, "degraaff2026_zenodo_21977747_v2.fits")
MATCH_ARCSEC = 0.5


def main() -> int:
    if not os.path.exists(CACHE):
        with urllib.request.urlopen(URL, timeout=120) as r:
            data = r.read()
        with open(CACHE, "wb") as f:
            f.write(data)
    t = fits.open(CACHE)[1].data
    use = np.array([bool(x) for x in t["use_dG26"]])
    dg = SkyCoord(t["ra"][use] * u.deg, t["dec"][use] * u.deg)

    c = pd.read_csv(CANDIDATES, dtype=str, keep_default_na=False)
    cc = SkyCoord(c.ra.astype(float).values * u.deg, c.dec.astype(float).values * u.deg)
    idx, sep, _ = cc.match_to_catalog_sky(dg)
    m = sep.arcsec < MATCH_ARCSEC

    c["in_degraaff26_v2"] = np.where(m, "True", "False")
    z = np.array([""] * len(c), dtype=object)
    z[m] = ["%.5f" % v for v in t["zspec"][use][idx[m]]]
    c["zspec_degraaff26_v2"] = z
    c.to_csv(CANDIDATES, index=False, lineterminator="\n")

    print("de Graaff 2026 v2: %d rows, %d unique sources" % (len(t), use.sum()))
    print("candidates matched within %.1f arcsec: %d" % (MATCH_ARCSEC, m.sum()))
    print(
        c.loc[
            m, ["rank", "field", "id", "tier", "status", "zspec_degraaff26_v2"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
