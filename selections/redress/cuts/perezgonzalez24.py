"""Pérez-González et al. 2024 (arXiv:2401.08782) — the SMILES two-color cut.

Verbatim (§2.1, via CUT_DEFINITIONS_EXTRACT): "We searched for NIRCam
SW-blue+LW-red sources defined by F277W−F444W > 1 mag and F150W−F200W < 0.5
mag colors, and magnitudes F444W ≤ 28 mag ... we found no need for
[a concentration criterion] after applying the F150W−F200W < 0.5 mag cut."
Brown-dwarf step: "we removed from the original sample the 4 sources with
color F115W−F150W < −0.5 mag" — NOTE the band pair: F115W−F150W, unlike
everyone else's F115W−F200W (extract watch-out #4).

INEQUALITY NOTE: the §2.1 body text prints "F444W ≤ 28 mag" (inclusive);
the Figure-1 caption prints "F444W < 28 mag". The selection DEFINITION in
the body governs — implemented as ≤, the one inclusive magnitude gate among
the seven cuts; the caption discrepancy is recorded here for the M0
fidelity note. Fiducial photometry in the paper is 0.25″-RADIUS PSF-matched
apertures (documentation for M0's photometry mapping; no code impact here).
The two proper-motion brown dwarfs removed by external identification are a
human/catalog step outside this photometric criterion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from redress.cuts import _shared as sh

COLOR_F277W_F444W_MIN = 1.0     # strict >, §2.1 verbatim
COLOR_F150W_F200W_MAX = 0.5     # strict <, §2.1 verbatim
M_F444W_MAX = 28.0              # INCLUSIVE ≤, §2.1 body text (see module note)
BD_COLOR_F115W_F150W_MIN = -0.5  # remove rows with color < −0.5 (strict removal)

_BANDS = ("f115w", "f150w", "f200w", "f277w", "f444w")


def select(phot: pd.DataFrame, known_pm_brown_dwarf=None) -> dict:
    """Apply Pérez-González+24. Fail-closed NaNs everywhere.

    The brown-dwarf retention criterion is the complement of the removal:
    a row is KEPT only when its F115W−F150W color is computable and ≥ −0.5 —
    a non-computable color cannot demonstrate it is not a brown dwarf
    (fail-closed, consistent with the package rule). r5j AUDIT SEPARATION:
    ``bd_measurable`` distinguishes failed-because-unmeasurable from
    failed-because-too-blue. r5j EXTERNAL STEP: the paper's removal of two
    PROPER-MOTION brown dwarfs (Hainline-identified) enters as the optional
    strict-boolean ``known_pm_brown_dwarf`` mask — visible when supplied,
    absent from the result when not. M0 gate: aperture provenance
    (0.25"-radius PSF-matched), the BD nondetection semantics, and the
    37 -> 31 count (incl. the 4 color-removed + 2 PM-removed) are decided by
    SMILES object-level reproduction under the frozen-mapping protocol.
    """
    sh.require_bands(phot, _BANDS, "perezgonzalez24")
    red = sh.gt(sh.color(phot, "f277w", "f444w"), COLOR_F277W_F444W_MIN)
    blue = sh.lt(sh.color(phot, "f150w", "f200w"), COLOR_F150W_F200W_MAX)
    mag = sh.le(sh.mags_ab(phot, "f444w"), M_F444W_MAX)
    bd_color = sh.color(phot, "f115w", "f150w")
    not_bd = sh.ge(bd_color, BD_COLOR_F115W_F150W_MIN)
    selected = red & blue & mag & not_bd
    extras: dict = {}
    if known_pm_brown_dwarf is not None:
        pm = sh.bool_mask(known_pm_brown_dwarf, len(phot), "known_pm_brown_dwarf")
        extras["pm_bd_retained"] = ~pm   # r5j2: True = KEPT (the name says what the flag means)
        selected = selected & ~pm
    return sh.result(
        selected,
        red_color=red, blue_color=blue, mag_gate=mag, bd_retention=not_bd,
        bd_measurable=np.isfinite(bd_color), **extras,
    )
