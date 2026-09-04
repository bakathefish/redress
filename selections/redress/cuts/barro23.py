"""Barro et al. 2023 (arXiv:2305.14418) — the single-color ERO selection.

Verbatim (§3.1, via CUT_DEFINITIONS_EXTRACT): "We identify extremely red
galaxies at high redshift using a single color cut F277W−F444W > 1.5 mag."
Sample limit: "We find 37 EROs in the CEERS field with F444W < 28 mag".

Deliberately NO compactness and NO SW-blue constraint — the paper drops the
additional color constraints, and the point-like nature is a RESULT there,
not a criterion ("all these EROs are unresolved" — outcome, never a cut).
The visual-inspection removal of hot pixels/fake objects is a human step the
re-implementation cannot and does not reproduce (M0 fidelity note). r5i: the
paper's 0.4″-aperture photometry mapping is decided by CEERS object-level
reproduction (37 EROs) at the M0 faithfulness gate, same standard as the
other cuts — with the r5i2 FREEZE PROTOCOL: the column/aperture mapping is
frozen from catalog METADATA (documented in the W1 prereg) BEFORE the CEERS
reproduction runs, so the mapping can never be tuned toward the published
count; visual-vetting removals are counted as explained discrepancies, never
absorbed into the mapping.

(The paper's separate, brighter "F150W ERO" contrast sample —
F150W−F444W > 2, F444W < 25 — is a different population and is NOT
implemented; documented here so nobody mistakes it for the LRD-like cut.)
"""

from __future__ import annotations

import pandas as pd

from redress.cuts import _shared as sh

COLOR_F277W_F444W_MIN = 1.5   # strict >, §3.1 verbatim
M_F444W_MAX = 28.0            # strict <, §3.1 verbatim

_BANDS = ("f277w", "f444w")


def select(phot: pd.DataFrame) -> dict:
    """Apply Barro+23: (F277W−F444W > 1.5) ∧ (m_F444W < 28). Fail-closed NaNs."""
    sh.require_bands(phot, _BANDS, "barro23")
    sh.require_complete_triples(phot, "barro23")   # r5i: no partially-schema'd tables
    red = sh.gt(sh.color(phot, "f277w", "f444w"), COLOR_F277W_F444W_MIN)
    mag = sh.lt(sh.mags_ab(phot, "f444w"), M_F444W_MAX)
    return sh.result(red & mag, red_color=red, mag_gate=mag)
