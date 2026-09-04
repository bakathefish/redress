"""Greene et al. 2024 (arXiv:2309.05714) — the UPDATED "v-shape" selection (§5.2).

Two selections live in that paper. Its §3.1 spectroscopic-TARGET selection is
the Labbé+23 cut verbatim (same red1/red2, same compactness, and the authority
that Labbé's apertures are DIAMETERS) — that one is :mod:`redress.cuts.labbe23`.
THIS module is the paper's §5.2 "Updated Photometric Selections" criterion.

Verbatim (§5.2, via CUT_DEFINITIONS_EXTRACT): "we start with the same
SNR(F444W) > 14 & m_F444W < 27.7 mag cuts as before. Then, we define a
'v-shape' color selection ...":

    v−shape: (−0.5 < F115W − F200W < 1.0) ∧ (F277W − F444W > 1.0)

"The main difference with respect to Labbé et al. (2023b) is using
F115W−F200W rather than F150W−F200W, and a blue limit to facilitate removing
brown dwarfs." The blue limit IS the Greene §5.1 brown-dwarf floor
(F115W−F200W > −0.5) folded into the window — no separate BD cut exists.

DELIBERATE ABSENCE: §5.2 imposes NO compactness criterion (the machine-
readable restatement carries none; the quick-reference table's "<1.7" applies
only to the §3.1 target set). The test suite proves a source with terrible
aperture concentration still selects here.
"""

from __future__ import annotations

import pandas as pd

from redress.cuts import _shared as sh

SNR_F444W_MIN = 14.0            # strict >, §5.2 ("same ... as before")
M_F444W_MAX = 27.7              # strict <, §5.2
VSHAPE_BLUE_LO = -0.5           # strict >: color must EXCEED the lower edge (r5h comment fix)
VSHAPE_BLUE_HI = 1.0            # strict <, window upper edge
COLOR_F277W_F444W_MIN = 1.0     # strict >

_BANDS = ("f115w", "f200w", "f277w", "f444w")


def select(phot: pd.DataFrame) -> dict:
    """Apply the Greene+24 §5.2 v-shape. Fail-closed NaNs everywhere."""
    sh.require_bands(phot, _BANDS, "greene24")
    detection = sh.gt(sh.snr(phot, "f444w"), SNR_F444W_MIN) & sh.lt(
        sh.mags_ab(phot, "f444w"), M_F444W_MAX
    )
    vshape_blue = sh.between(sh.color(phot, "f115w", "f200w"), VSHAPE_BLUE_LO, VSHAPE_BLUE_HI)
    vshape_red = sh.gt(sh.color(phot, "f277w", "f444w"), COLOR_F277W_F444W_MIN)
    return sh.result(
        detection & vshape_blue & vshape_red,
        detection=detection, vshape_blue=vshape_blue, vshape_red=vshape_red,
    )
