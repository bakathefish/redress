"""Kokorev et al. 2024 (arXiv:2401.09981) — the blank-fields census selection.

Verbatim (§3.1, via CUT_DEFINITIONS_EXTRACT): SNR/mag gate "consistent with
the UNCOVER selection" (> 14σ in F444W, brighter than 27.7 AB), then

    red1 = (F115W−F150W < 0.8) ∧ (F200W−F277W > 0.7) ∧ (F200W−F356W > 1.0)
    red2 = (F150W−F200W < 0.8) ∧ (F277W−F356W > 0.6) ∧ (F277W−F444W > 0.7)
    compact = f_F444W(0.4″)/f_F444W(0.2″) < 1.7
    bd_removal = F115W − F200W > −0.5

⚠ THE CRITICAL DIFFERENCE (extract watch-out #1): Kokorev's red2 LOOSENS two
Labbé thresholds — F277W−F356W > 0.6 (Labbé 0.7) and F277W−F444W > 0.7
(Labbé 1.0). red1 is identical. The test suite pins a row that passes
Kokorev's red2 and fails Labbé's.

DETECTION SEMANTICS (verbatim): "every object has to be detected (> 3σ) in
at least one band per color to make the selection meaningful. In case of a
non-detection we use the 2σ upper limits, but only if the 'brighter' band in
the color is detected." Implemented per color from FLUXES:

* both bands ≥ 3σ → measured-magnitude color;
* exactly one band ≥ 3σ → substitute the undetected band's 2σ upper-limit
  flux, ``max(f, 0) + 2e`` (needs coverage there), PROVIDED the detected
  band's flux exceeds that substituted limit — the operational reading of
  "the 'brighter' band in the color is detected". r5e (review, accepted):
  this is the paper's PLUG-IN HEURISTIC faithfully emulated, NOT
  conservative inference — a true flux far below its limit can flip a
  ``lt``-criterion from pass to fail (counterexample in the verdict), so no
  bound claim is made. TWO PROVISIONAL CHOICES, both decided by the M0
  FAITHFULNESS GATE (reproduce Kokorev's published candidate list;
  preregister the best-reproducing convention): (a) the limit formula —
  ``max(f,0)+2e`` vs ``2e`` vs ``f+2e`` vs survey depth (the paper does not
  define it); (b) the "brighter band" reading above. GOVERNANCE (r5e2 owner
  question, council-resolved): the selection PROTOCOL is preregistered in the
  W1 prereg BEFORE any reproduction runs — candidate conventions, selection
  metric (published-list agreement), and the exact list/field used — the
  selection runs ONCE, and the winner is frozen for all M0 reporting; anything
  else would be convention-tuning on the reporting data;
* neither band ≥ 3σ (or the limit not constructible) → the color is
  unusable and its criterion FAILS for that row.

The rule applies uniformly to all five red-branch colors and the BD color
(the paper scopes it to "the color criteria"). Post-selection size vetting
(pysersic fits) and SED-based cleaning are downstream refinements outside
this photometric criterion (M0 fidelity note; final census 334 → 260 there).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from redress import conventions
from redress.cuts import _shared as sh

SNR_F444W_MIN = 14.0
M_F444W_MAX = 27.7
RED1 = (("f115w", "f150w", "lt", 0.8), ("f200w", "f277w", "gt", 0.7), ("f200w", "f356w", "gt", 1.0))
RED2 = (("f150w", "f200w", "lt", 0.8), ("f277w", "f356w", "gt", 0.6), ("f277w", "f444w", "gt", 0.7))
BD_COLOR_F115W_F200W_MIN = -0.5   # strict >, adopted from Greene+23
COMPACT_RATIO_MAX = 1.7
DETECTION_SIGMA = 3.0
UPPER_LIMIT_SIGMA = 2.0
#: r5e3: the censoring policy travels WITH every result (provenance for M0
#: outputs; the ID changes if the faithfulness gate selects another convention).
CENSORING_POLICY = "kokorev24-ul-v1-provisional: limit=max(f,0)+2e; brighter-band-detected rule"

_BANDS = ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w")


def _color_with_upper_limits(phot: pd.DataFrame, blue: str, red: str):
    """One color under Kokorev's per-color detection rule; NaN = unusable.

    Returns ``(color, used_limit)`` — r5e3: per-row provenance of whether a
    2σ substitution produced this color (False for measured/unusable).
    """
    fb = phot[f"f_{blue}_ujy"].to_numpy(dtype=float)
    eb = phot[f"e_{blue}_ujy"].to_numpy(dtype=float)
    cb = sh.cov_flags(phot, blue)
    fr = phot[f"f_{red}_ujy"].to_numpy(dtype=float)
    er = phot[f"e_{red}_ujy"].to_numpy(dtype=float)
    cr = sh.cov_flags(phot, red)

    det_b = cb & np.isfinite(fb) & np.isfinite(eb) & (eb > 0) & (fb / np.where(eb > 0, eb, np.inf) > DETECTION_SIGMA)
    det_r = cr & np.isfinite(fr) & np.isfinite(er) & (er > 0) & (fr / np.where(er > 0, er, np.inf) > DETECTION_SIGMA)

    out = np.full(fb.shape, np.nan)
    used = np.zeros(fb.shape, dtype=bool)

    # both detected: measured color (fluxes > 0 is implied by > 3σ detection)
    both = det_b & det_r
    if np.any(both):
        out[both] = conventions.ab_mag_from_fnu_ujy(fb[both]) - conventions.ab_mag_from_fnu_ujy(fr[both])

    # exactly one detected: 2σ upper-limit substitution for the other band,
    # valid only when the detected band is the brighter of the two
    only_b = det_b & ~det_r & cr & np.isfinite(fr) & np.isfinite(er) & (er > 0)
    if np.any(only_b):
        lim_r = np.maximum(fr[only_b], 0.0) + UPPER_LIMIT_SIGMA * er[only_b]
        ok = fb[only_b] > lim_r
        idx = np.flatnonzero(only_b)[ok]
        lim = lim_r[ok]
        out[idx] = conventions.ab_mag_from_fnu_ujy(fb[idx]) - conventions.ab_mag_from_fnu_ujy(lim)
        used[idx] = True

    only_r = det_r & ~det_b & cb & np.isfinite(fb) & np.isfinite(eb) & (eb > 0)
    if np.any(only_r):
        lim_b = np.maximum(fb[only_r], 0.0) + UPPER_LIMIT_SIGMA * eb[only_r]
        ok = fr[only_r] > lim_b
        idx = np.flatnonzero(only_r)[ok]
        lim = lim_b[ok]
        out[idx] = conventions.ab_mag_from_fnu_ujy(lim) - conventions.ab_mag_from_fnu_ujy(fr[idx])
        used[idx] = True

    return out, used


def _branch(phot: pd.DataFrame, spec):
    out, any_limit = None, None
    for blue, red, op, thr in spec:
        col, used = _color_with_upper_limits(phot, blue, red)
        term = sh.lt(col, thr) if op == "lt" else sh.gt(col, thr)
        out = term if out is None else (out & term)
        any_limit = used if any_limit is None else (any_limit | used)
    return out, any_limit


def select(phot: pd.DataFrame, aper_ratio_f444w_04_02) -> dict:
    """Apply Kokorev+24: (red1 ∨ red2) ∧ compact ∧ bd_removal behind the gate."""
    sh.require_bands(phot, _BANDS, "kokorev24")
    n = len(phot)
    detection = sh.gt(sh.snr(phot, "f444w"), SNR_F444W_MIN) & sh.lt(
        sh.mags_ab(phot, "f444w"), M_F444W_MAX
    )
    red1, ul1 = _branch(phot, RED1)
    red2, ul2 = _branch(phot, RED2)
    bd_color, ul_bd = _color_with_upper_limits(phot, "f115w", "f200w")
    bd = sh.gt(bd_color, BD_COLOR_F115W_F200W_MIN)
    ratio = sh.as_row_array(aper_ratio_f444w_04_02, n, "aper_ratio_f444w_04_02")
    # r5e: a flux ratio must be POSITIVE to mean anything — 0/negative values
    # are pathological photometry, not compact sources (strict 0 < r < 1.7)
    compact = sh.between(ratio, 0.0, COMPACT_RATIO_MAX)
    # r5e3 auditability: compact_valid separates PATHOLOGICAL ratios (<=0/NaN,
    # photometry problem) from genuinely EXTENDED sources (valid ratio >= 1.7)
    compact_valid = np.isfinite(ratio) & (ratio > 0)
    out = sh.result(
        detection & (red1 | red2) & compact & bd,
        detection=detection, red1=red1, red2=red2, compact=compact,
        compact_valid=compact_valid, bd_removal=bd,
        # r5e4: PER-BRANCH substitution provenance — an aggregate flag lied
        # when an irrelevant branch used a limit; now each criterion answers
        # for itself and M0 can report exactly which decisions leaned on limits
        used_upper_limit_red1=ul1, used_upper_limit_red2=ul2,
        used_upper_limit_bd=ul_bd,
    )
    out["censoring_policy"] = CENSORING_POLICY
    return out
