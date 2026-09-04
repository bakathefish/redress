"""Akins et al. 2024 (arXiv:2406.10341) — the COSMOS-Web selection (the plan's
seventh, "Akins wing" adjunct cut; plan-text resolution 2026-08-02).

Verbatim (§3.1, via CUT_DEFINITIONS_EXTRACT): "We select all sources with
S/N444 > 12 ... and m277 − m444 > 1.5", with compactness

    (Eq. 1)  C444 = f444(d = 0.2″) / f444(d = 0.5″)      [aperture DIAMETERS]

"We select objects with C444 > 0.5 ... We exclude objects with C444 > 0.7,
which tend to be imaging artifacts (hot pixels)."

BOUNDARY SEMANTICS (implemented from the two quoted sentences, NOT the
shorthand "0.5 < C444 < 0.7" — that sentence describes where point sources
reliably LIVE, not the cut): selected ⇔ (C444 > 0.5) ∧ ¬(C444 > 0.7), i.e.
the interval (0.5, 0.7] — a source at exactly 0.7 is NOT excluded (0.7 > 0.7
is false), a source at exactly 0.5 fails the > 0.5 selection. The test suite
pins both boundary rows; the 0.7-INCLUSIVE reading is decided finally by
published-selector reproduction at the M0 gate (r5k, frozen-mapping protocol).

C444 is exactly the DATA_CONTRACTS ``compactness_f444w`` column (0.2″/0.5″
DIAMETER apertures — the contract states the convention; changing the pair is
a schema bump). Note the DIRECTION: Akins's ratio is small/large (compact ⇒
LARGE value), inverted vs the Labbé/Kokorev large/small ratio (extract
watch-out #3).

Deliberately NO SW-blue color cut (COSMOS-Web F115W/F150W too shallow — the
paper says so). Brown-dwarf removal is an SED-χ²-grid comparison ("104
objects ... have brown dwarf model χ² less than the minimum galaxy/quasar
model χ² and are removed") — an EXTERNAL model product, supplied by the M0
runner as the optional ``is_brown_dwarf_sed`` mask; when absent the result
simply carries no ``bd_removal`` key (the step is visibly not-run, never
silently passed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from redress.cuts import _shared as sh

SNR_F444W_MIN = 12.0        # strict >, §3.1 verbatim
COLOR_F277W_F444W_MIN = 1.5  # strict >, §3.1 verbatim (m277 − m444 > 1.5)
C444_SELECT_MIN = 0.5       # strict > (selection sentence)
C444_EXCLUDE_ABOVE = 0.7    # exclude strictly-above (artifact sentence)

_BANDS = ("f277w", "f444w")


def select(phot: pd.DataFrame, is_brown_dwarf_sed=None) -> dict:
    """Apply Akins+24. Fail-closed NaNs; C444 from ``compactness_f444w``."""
    sh.require_bands(phot, _BANDS, "akins24")
    sh.require_complete_triples(phot, "akins24")
    if "compactness_f444w" not in phot.columns:
        raise ValueError("akins24: photometry table lacks compactness_f444w (C444)")
    detection = sh.gt(sh.snr(phot, "f444w"), SNR_F444W_MIN)
    red = sh.gt(sh.color(phot, "f277w", "f444w"), COLOR_F277W_F444W_MIN)
    c444 = phot["compactness_f444w"].to_numpy(dtype=float)
    # (C444 > 0.5) ∧ ¬(C444 > 0.7) — composed from finite-gated comparators
    # only, so a NaN C444 fails the selection instead of slipping past the
    # negated artifact exclusion (see _shared design rule).
    compact = sh.gt(c444, C444_SELECT_MIN) & ~sh.gt(c444, C444_EXCLUDE_ABOVE)
    # r5k three-state audit: invalid photometry vs extended vs hot-pixel
    # artifact are DIFFERENT facts; M0 reports each cause
    compact_valid = np.isfinite(c444) & (c444 > 0)
    compact_artifact = sh.gt(c444, C444_EXCLUDE_ABOVE)
    selected = detection & red & compact
    extras: dict = {}
    if is_brown_dwarf_sed is not None:
        # r5g (found on labbe23, same class here): strict mask validation —
        # np.asarray(bool) coerces NaN/"False" to True
        bd = sh.bool_mask(is_brown_dwarf_sed, len(phot), "is_brown_dwarf_sed")
        extras["bd_retained"] = ~bd    # r5k: True = KEPT (pg24 convention)
        selected = selected & ~bd
    out = sh.result(
        selected, detection=detection, red_color=red, compact=compact,
        compact_valid=compact_valid, compact_artifact=compact_artifact, **extras,
    )
    # r5k: the M0 runner must REFUSE to publish a final catalog from a result
    # whose SED brown-dwarf step never ran (ledgered runner rule)
    out["sed_bd_applied"] = is_brown_dwarf_sed is not None
    return out
