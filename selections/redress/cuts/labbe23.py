"""Labbé et al. 2023 (arXiv:2306.07320) — the ORIGINAL compact-red selection.

Verbatim (§2.2 / Table 1, via CUT_DEFINITIONS_EXTRACT): "Starting with sources
that are well-detected in the F444W band, with SNR(444) > 14 & m444 < 27.7
mag, we select sources that are (red1 | red2) & compact, where"

    red1 = (F115W−F150W < 0.8) ∧ (F200W−F277W > 0.7) ∧ (F200W−F356W > 1.0)
    red2 = (F150W−F200W < 0.8) ∧ (F277W−F356W > 0.7) ∧ (F277W−F444W > 1.0)
    compact = f444(0.4″)/f444(0.2″) < 1.7

Apertures are DIAMETERS (0.4″ and 0.2″) — the authority is Greene+24 §3.1
restating the identical cut "measured within a 0.4″ diameter aperture"
(extract watch-out; do not confuse with Barro+24's radius convention). The
ratio direction is large/small: compact ⇒ SMALL value.

This composite is the paper's PARENT sample (40 sources). The Main sample
(26) additionally requires PSF-dominance in a 2-D GALFIT F444W fit — an
external morphology product the M0 runner may supply as ``psf_dominated``;
when given, the result carries a ``main`` key = selected ∧ psf_dominated.
Labbé states no upper-limit substitution rule, so non-computable colors fail
closed (package rule); the aperture-ratio fluxes are not in the harmonized
schema, so the ratio arrives as the explicit ``aper_ratio_f444w_04_02``
argument (W1 maps it from the source catalogs' aperture photometry, with its
own fidelity note). M0 PARITY GATE (r5g owner question, council-resolved):
reproducing the published 40-Parent / 26-Main counts against the
provenance-locked UNCOVER catalog is an M0 faithfulness-gate item, NOT a
merge gate — real catalog data cannot enter the code-only merge stage
(compliance boundary); every object-level discrepancy will be explained or
counted against fidelity there. EXTERNAL-MASK CONTRACT (r5g2): the
``psf_dominated`` mask means Labbé's Table-2 note "F444W_Sers > F444W_PSF"
from the 2-D GALFIT fit — the M0 adapter that computes it owes its own
boundary/validity test against that definition (ledgered); this module only
guarantees strict-boolean handling. PHOTOMETRY CONVENTION (r5g2): colors are
catalog PSF-photometry and the SNR/mag gate is measured in a 0.32″ aperture
per Greene's §3.1 restatement — the W1 column-mapping carries a
convention-parity check (ledgered), since the harmonized table stores one
flux per band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from redress.cuts import _shared as sh

SNR_F444W_MIN = 14.0
M_F444W_MAX = 27.7
RED1 = (("f115w", "f150w", "lt", 0.8), ("f200w", "f277w", "gt", 0.7), ("f200w", "f356w", "gt", 1.0))
RED2 = (("f150w", "f200w", "lt", 0.8), ("f277w", "f356w", "gt", 0.7), ("f277w", "f444w", "gt", 1.0))
COMPACT_RATIO_MAX = 1.7     # f444(0.4″ diam)/f444(0.2″ diam) < 1.7, strict

_BANDS = ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w")


def _branch(phot: pd.DataFrame, spec) -> np.ndarray:
    out = None
    for blue, red, op, thr in spec:
        col = sh.color(phot, blue, red)
        if op == "lt":
            term = sh.lt(col, thr)
        elif op == "gt":
            term = sh.gt(col, thr)
        else:  # r5g2: an unknown operator must never silently become gt
            raise ValueError(f"unknown color operator {op!r} in criterion spec")
        out = term if out is None else (out & term)
    return out


def select(phot: pd.DataFrame, aper_ratio_f444w_04_02, psf_dominated=None) -> dict:
    """Apply Labbé+23 Parent = (red1 ∨ red2) ∧ compact behind the SNR/mag gate."""
    sh.require_bands(phot, _BANDS, "labbe23")
    n = len(phot)
    detection = sh.gt(sh.snr(phot, "f444w"), SNR_F444W_MIN) & sh.lt(
        sh.mags_ab(phot, "f444w"), M_F444W_MAX
    )
    red1 = _branch(phot, RED1)
    red2 = _branch(phot, RED2)
    ratio = sh.as_row_array(aper_ratio_f444w_04_02, n, "aper_ratio_f444w_04_02")
    # r5e (found on kokorev24, shared fix): flux ratios must be strictly
    # positive — 0/negative is pathological photometry, never "compact"
    compact = sh.between(ratio, 0.0, COMPACT_RATIO_MAX)
    # r5g (ported from kokorev24): invalid photometry vs genuinely extended
    # are DIFFERENT audit facts — M0 reports them separately
    compact_valid = np.isfinite(ratio) & (ratio > 0)
    selected = detection & (red1 | red2) & compact
    extras: dict = {}
    if psf_dominated is not None:
        psf = sh.bool_mask(psf_dominated, n, "psf_dominated")
        extras["main"] = selected & psf
    return sh.result(
        selected, detection=detection, red1=red1, red2=red2, compact=compact,
        compact_valid=compact_valid, **extras
    )
