"""Barro et al. 2024b (arXiv:2412.01887) — the two-color selection with an aperture compactness.

Added in revision as the eighth comparator, at referee request; it is NOT one of the seven
selections of the pre-stated benchmark in ``redress.cuts.CUTS`` and is registered separately
in ``redress.cuts.COMPARATORS``.

Verbatim (their Sec. 2.2, enumerated thresholds):
    (i)   F200W - F444W > 1
    (ii)  (F200W - F444W) > (F115W - F200W) + 0.25
    (iii) F115W - F200W > -0.5
    (iv)  F444W < 27
and the compactness criterion of their Sec. 2.3: "the flux ratio measured in two different
apertures (F_F444W(0.5'')/F_F444W(0.2'') < 1.5)". Their Fig. 2 caption states the apertures
as "apertures of radius r = 0.5'' and 0.2''", and a stellar locus near 1.2 (same caption) is
consistent only with radii, so the ratio this module consumes is F444W flux inside a 1.0''
DIAMETER aperture over the flux inside a 0.4'' DIAMETER aperture. That is a different pair of
apertures from the 0.4''/0.2'' diameters of labbe23 and kokorev24, so the frozen
``redress.compactness`` producer (diameters 0.20, 0.40, 0.50) cannot supply it; the paper
chain measures it with ``recovery/v4_barro24b_apertures.py`` on the DJA F444W mosaic cutouts
(forced centering at the catalog position, exact-overlap apertures, no recentering, an aperture
touching a non-finite pixel makes the source unmeasurable).

Deviations recorded here:
* The paper states no upper-limit convention for undetected bands; every color here is
  fail-closed (a non-positive flux in F115W, F200W or F444W fails criteria (i) to (iii)).
* The paper applied the selection to its own photometry of the MIRI-covered regions of PRIMER
  and JADES; this module applies the same thresholds to the DJA catalog photometry of the
  nine fields of the completeness test, so its output is "as re-implemented on DJA
  photometry", like the seven.
* The published sample also removed sources at mosaic edges and photometric-redshift z < 1
  sources by inspection (their Sec. 2.4 in the 2026 follow-up); neither step is implemented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from redress.cuts import _shared as sh

COLOR_F200W_F444W_MIN = 1.0  # (i), strict >
DIAGONAL_OFFSET = 0.25  # (ii), strict >
COLOR_F115W_F200W_MIN = -0.5  # (iii), strict >
M_F444W_MAX = 27.0  # (iv), strict <
COMPACT_RATIO_MAX = 1.5  # F444W(r=0.5")/F444W(r=0.2") < 1.5, strict <

_BANDS = ("f115w", "f200w", "f444w")


def select(phot: pd.DataFrame, aper_ratio_f444w_r05_r02) -> dict:
    """Apply Barro+24b: (i) & (ii) & (iii) & (iv) & compact. Fail-closed NaNs throughout."""
    sh.require_bands(phot, _BANDS, "barro24b")
    sh.require_complete_triples(phot, "barro24b")
    n = len(phot)
    c_long = sh.color(phot, "f200w", "f444w")  # F200W - F444W
    c_short = sh.color(phot, "f115w", "f200w")  # F115W - F200W
    red = sh.gt(c_long, COLOR_F200W_F444W_MIN)
    diagonal = sh.gt(c_long - c_short, DIAGONAL_OFFSET)
    blue_floor = sh.gt(c_short, COLOR_F115W_F200W_MIN)
    mag = sh.lt(sh.mags_ab(phot, "f444w"), M_F444W_MAX)
    ratio = sh.as_row_array(aper_ratio_f444w_r05_r02, n, "aper_ratio_f444w_r05_r02")
    compact = sh.between(ratio, 0.0, COMPACT_RATIO_MAX)
    compact_valid = np.isfinite(ratio) & (ratio > 0)
    colors = red & diagonal & blue_floor
    selected = colors & mag & compact
    return sh.result(
        selected,
        red=red,
        diagonal=diagonal,
        blue_floor=blue_floor,
        mag_gate=mag,
        colors=colors,
        compact=compact,
        compact_valid=compact_valid,
    )
