"""Kocevski et al. 2024 (arXiv:2404.03576) — the continuum-slope selection.

The only cut on SLOPES, not fixed colors (extract watch-out #5). Verbatim
(§3.1, via CUT_DEFINITIONS_EXTRACT): β defined by f_λ ∝ λ^β via

    (Eq. 1)  m_i = −2.5 (β + 2) log(λ_i) + c

fit by χ² to the observed AB magnitudes — exactly
:func:`redress.conventions.beta_from_mag_regression` (same convention module
that owns Eq. 2, the color↔β conversion). Primary criteria (§3.1 summary,
verbatim): "(i) SNR_F444W > 12  (ii) β_opt > 0  (iii) −2.8 < β_UV < −0.37
(iv) r_h < 1.5 r_h,stars", plus the line-emission-boost conditions
"(v) β_{F277W−F356W} > −1 (only at z < 8), (vi) β_{F277W−F410M} > −1 (when
F410M available)". The −2.8 floor inside (iii) IS the brown-dwarf cut
(β_UV < −2.8 ↔ F115W−F200W < −0.5, Greene's cut in slope space).

Bands per Table 2, keyed on the source's redshift:

    2 < z < 3.25   UV: F606W F814W F115W    opt: F150W F200W F277W
    3.25 < z < 4.75 UV: F814W F115W F150W    opt: F200W F277W F356W
    4.75 < z < 8    UV: F115W F150W F200W    opt: F277W F356W F444W
    z > 8           UV: F150W F200W F277W    opt: F356W F444W

PUBLISHED-PAPER INCONSISTENCY (r5f council catch, source PDF verified
2026-08-03): the paper's formal criteria print "beta_{F277W-F356W} > -1" and
"beta_{F277W-F410M} > -1", but its own sentence "The equivalent color cuts
are F277W - F356W > 0.53 and F277W - F410M > 0.84" corresponds via its
Eq. 2 to beta > 0, NOT beta > -1 (0.53/0.84 are the beta = 0 colors; the
beta = -1 colors are ~ 0.28/0.42). The FORMAL criterion (beta > -1) governs
this faithful implementation; both variants are preregistered for the M0
faithfulness gate. r5f6 oracle refinement (review counter, accepted): the
aggregate 539 -> 341 count alone is NOT decisive — the preregistered
standard is OBJECT-LEVEL agreement against the published 341-source list;
if the exact input catalog + rejected-object mask cannot be reproduced,
BOTH variants are reported as sensitivity results, neither suppressed.

FAITHFUL-MODE DECISIONS (r5f, replacing the earlier conventions):

* Bin boundaries: the paper's intervals are STRICT OPEN ("2 < z < 3.25",
  "z > 8"); exact boundary redshifts (2, 3.25, 4.75, 8) are UNASSIGNED by
  the text and FAIL CLOSED here - no half-open convention, no z=8 hybrid.
  Any boundary-assignment sensitivity belongs to a separately reported M0
  variant, never the baseline.
* Slopes REQUIRE the bin's full listed band set (three, or the predefined
  z>8 optical pair), all covered with usable photometry - a missing band
  fails closed. The earlier 2-of-3 fallback is REMOVED from the baseline
  (r5f: it would preferentially rescue missing-band objects and inflate
  completeness); it may return only as a named M0 sensitivity variant.
  ``n_bands_uv/opt`` still audit what was available per row.
* Condition (v) applies only where z < 8 (strict). Condition (vi)
  applicability follows F410M COVERAGE (r5f: a covered NEGATIVE flux is
  valid photometry, not "unavailable"): uncovered -> vacuous by the paper's
  own escape; covered but unmeasurable -> FAIL CLOSED, never fail-open.
* Sizes must be PHYSICAL: r_h > 0 and r_h,stars > 0 are required for the
  size criterion (r5f: a negative radius must never pass). The M0 runner
  owner-confirms both arrays share the SExtractor F444W half-light
  convention - unit agreement alone is insufficient (recorded).
* Pivot wavelengths are required only for bands this table can actually
  use: the bin-table bands plus F277W/F356W, plus F410M only when the
  table carries an F410M column.
* The stellar half-light-radius locus r_h,stars(m) is measured from stars
  per field at M0 — it arrives as the caller-supplied per-row array
  ``r_h_stars_arcsec``; r_h comes from the schema's
  ``flux_radius_f444w_arcsec``. NaN in either → fail closed.
* Pivot wavelengths are NEVER hardcoded (DATA_CONTRACTS §3): the caller
  supplies ``pivots_angstrom`` for every band this cut touches; a missing
  needed pivot raises loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from redress import conventions
from redress.cuts import _shared as sh

SNR_F444W_MIN = 12.0
BETA_OPT_MIN = 0.0             # strict >, criterion (ii)
BETA_UV_LO = -2.8              # strict <  on the left, criterion (iii)
BETA_UV_HI = -0.37             # strict <  on the right, criterion (iii)
RH_STELLAR_FACTOR = 1.5        # r_h < 1.5 · r_h,stars, strict, criterion (iv)
BETA_LINEBOOST_MIN = -1.0      # strict >, criteria (v)/(vi)
#: r5f5 (review counter, accepted): the printed-equivalents variant applies the
#: paper's LITERAL color numbers — the printed 0.53/0.84 embed the AUTHORS'
#: pivots, so re-deriving them through our pivots would test our wavelengths,
#: not their sentence. Under the printed variant these color cuts replace the
#: beta conversion entirely.
PRINTED_COLOR_356_MIN = 0.53
PRINTED_COLOR_410_MIN = 0.84
LINEBOOST_Z_MAX = 8.0          # (v) applies only at z < 8, strict

#: (z_lo, z_hi) STRICT-OPEN intervals (exact boundaries fail closed), bands — Table 2 verbatim.
Z_BINS = (
    (2.0, 3.25, ("f606w", "f814w", "f115w"), ("f150w", "f200w", "f277w")),
    (3.25, 4.75, ("f814w", "f115w", "f150w"), ("f200w", "f277w", "f356w")),
    (4.75, 8.0, ("f115w", "f150w", "f200w"), ("f277w", "f356w", "f444w")),
    (8.0, np.inf, ("f150w", "f200w", "f277w"), ("f356w", "f444w")),
)

_ALL_BANDS = ("f606w", "f814w", "f115w", "f150w", "f200w", "f277w", "f356w", "f410m", "f444w")


def _fit_beta_rows(phot, z, bands_by_bin, pivots):
    """Per-row slope over the bin's available bands; (beta, n_bands_used)."""
    n = len(phot)
    mags = {b: sh.mags_ab(phot, b) for b in _ALL_BANDS if f"f_{b}_ujy" in phot.columns}
    errs = {}
    for b, m_b in mags.items():
        f = phot[f"f_{b}_ujy"].to_numpy(dtype=float)
        e = phot[f"e_{b}_ujy"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            errs[b] = np.where(np.isfinite(m_b), 2.5 / np.log(10) * (e / f), np.nan)
    beta = np.full(n, np.nan)
    err = np.full(n, np.nan)
    used = np.zeros(n, dtype=int)
    for i in range(n):
        bands = bands_by_bin[i]
        if bands is None:
            continue
        ms, es, ls = [], [], []
        for b in bands:
            if b in mags and np.isfinite(mags[b][i]) and np.isfinite(errs[b][i]) and errs[b][i] > 0:
                ms.append(mags[b][i])
                es.append(errs[b][i])
                ls.append(pivots[b])
        used[i] = len(ms)
        # r5f FAITHFUL MODE: the paper's full listed band set or nothing
        if len(ms) == len(bands):
            beta[i], err[i] = conventions.beta_from_mag_regression(
                np.array(ls), np.array(ms), np.array(es)
            )
    # r5f5 uncertainty audit: errors are sigma-propagated ("unscaled" WLS
    # covariance), which IS defined at N=2 — a residual-based error would be
    # NaN there. The returned semantics follow conventions.beta_from_mag_
    # regression's documented behavior; flagged for co-review consent.
    return beta, err, used


def select(
    phot: pd.DataFrame,
    z_phot,
    r_h_stars_arcsec,
    pivots_angstrom,
    lineboost_beta_min: float = BETA_LINEBOOST_MIN,
) -> dict:
    """Apply Kocevski+24 (i)–(vi). Fail-closed except (vi)'s coverage escape.

    ``lineboost_beta_min`` names the published-inconsistency variants
    (r5f2): −1.0 = the FORMAL criteria (the baseline); 0.0 = the paper's
    printed "equivalent color cuts" (0.53/0.84 are the β=0 colors). The M0
    faithfulness gate discriminates them OBJECT-LEVEL against Kocevski+24's
    published 341-source list; only with that object mask in hand does a
    single preregistered winner report — WITHOUT the published object mask,
    M0 EMITS BOTH NAMED VARIANTS regardless of aggregate-count agreement
    (sweep triage fold 4, cutdefs-verify-r1: "only the winner reports" here
    contradicted this module's own stronger object-level-or-report-both
    policy above; an aggregate 539→341 match cannot adjudicate which
    variant the authors ran). The deciding oracle for
    the equivalence claim is the in-test Eq-2 conversion (hand-derived from
    the U2-pinned convention functions), plus the M0 reproduction as the
    empirical oracle (r5f2 owner question, council-resolved).
    """
    # skeptic review r1: every band this cut reads UNCONDITIONALLY is
    # declared here — f277w/f356w previously crashed with a raw KeyError on a
    # contract-valid table lacking them, instead of the clean named refusal.
    sh.require_bands(phot, ("f444w", "f277w", "f356w"), "kocevski24")
    sh.require_complete_triples(phot, "kocevski24")
    # r5f4: bools are NOT variant selectors (False == 0.0 in Python — a
    # boolean would silently pick the printed-equivalents variant)
    if isinstance(lineboost_beta_min, (bool, np.bool_)) or lineboost_beta_min not in (-1.0, 0.0):
        raise ValueError(
            f"lineboost_beta_min must be one of the PREREGISTERED variants -1.0 "
            f"(formal criteria) or 0.0 (printed equivalents), got {lineboost_beta_min!r}"
        )
    n = len(phot)
    z = sh.as_row_array(z_phot, n, "z_phot")
    uv_bands_row: list = [None] * n
    opt_bands_row: list = [None] * n
    needed = set()
    in_table_early = np.zeros(n, dtype=bool)
    any_z_below8 = False
    for lo, hi, uv, opt in Z_BINS:
        # r5f faithful mode: STRICT OPEN - exact boundaries fail closed
        in_bin = np.isfinite(z) & (z > lo) & (z < hi)
        in_table_early |= in_bin
        if np.any(in_bin):
            needed |= set(uv) | set(opt)
            if lo < LINEBOOST_Z_MAX:
                any_z_below8 = True
        for i in np.flatnonzero(in_bin):
            uv_bands_row[i] = uv
            opt_bands_row[i] = opt
    # r5f5 pivot locality: demand only what THIS table's rows can use — the
    # occupied bins' bands, plus the (v)/(vi) conversion bands only when the
    # FORMAL variant will actually convert colors to betas there.
    if lineboost_beta_min == -1.0:
        if any_z_below8:
            needed |= {"f277w", "f356w"}
        # r5f7: an all-uncovered F410M column converts nothing — demand the
        # pivot only where a conversion can actually happen
        if "f_f410m_ujy" in phot.columns and (sh.cov_flags(phot, "f410m") & in_table_early).any():
            needed |= {"f277w", "f410m"}
    missing = [b for b in needed if b not in pivots_angstrom]
    if missing:
        raise ValueError(f"kocevski24: pivots_angstrom lacks bands {sorted(missing)}")

    beta_uv, beta_uv_err, n_uv = _fit_beta_rows(phot, z, uv_bands_row, pivots_angstrom)
    beta_opt, beta_opt_err, n_opt = _fit_beta_rows(phot, z, opt_bands_row, pivots_angstrom)

    detection = sh.gt(sh.snr(phot, "f444w"), SNR_F444W_MIN)
    opt_red = sh.gt(beta_opt, BETA_OPT_MIN)
    uv_window = sh.between(beta_uv, BETA_UV_LO, BETA_UV_HI)

    if "flux_radius_f444w_arcsec" not in phot.columns:
        raise ValueError("kocevski24: photometry table lacks flux_radius_f444w_arcsec")
    r_h = phot["flux_radius_f444w_arcsec"].to_numpy(dtype=float)
    r_stars = sh.as_row_array(r_h_stars_arcsec, n, "r_h_stars_arcsec")
    with np.errstate(invalid="ignore"):
        # r5f: radii must be PHYSICAL (> 0) - a negative radius never passes
        size_ok = (
            np.isfinite(r_h) & np.isfinite(r_stars)
            & (r_h > 0) & (r_stars > 0)
            & (r_h < RH_STELLAR_FACTOR * r_stars)
        )

    printed_variant = lineboost_beta_min == 0.0
    # (v): applied only at z < 8. FORMAL variant: two-band beta > -1 via our
    # pivots. PRINTED variant (r5f5): the paper's literal color number.
    c_356 = sh.color(phot, "f277w", "f356w")
    # r5f6: audit betas convert only when their pivots exist (the printed
    # variant's literal colors never need them) — NaN audit, never KeyError
    if "f277w" in pivots_angstrom and "f356w" in pivots_angstrom:
        beta_v = np.array(
            [
                conventions.beta_from_color(c, pivots_angstrom["f277w"], pivots_angstrom["f356w"])
                if np.isfinite(c) else np.nan
                for c in c_356
            ]
        )
    else:
        beta_v = np.full(n, np.nan)
    # r5f8: applicability is a statement about rows the TABLE covers — an
    # out-of-table redshift has no bin, so no condition "applies" to it
    in_table = np.array([b is not None for b in uv_bands_row])
    applies_v = in_table & (z < LINEBOOST_Z_MAX)
    crit_v = sh.gt(c_356, PRINTED_COLOR_356_MIN) if printed_variant else sh.gt(beta_v, lineboost_beta_min)
    lineboost_356 = np.where(applies_v, crit_v, True)

    # (vi): applicability follows F410M COVERAGE (r5f - a covered negative
    # flux is valid photometry, not "unavailable"): uncovered rows get the
    # paper's vacuous escape; covered-but-unmeasurable rows FAIL CLOSED.
    if "f_f410m_ujy" in phot.columns:
        applies_vi = sh.cov_flags(phot, "f410m") & in_table
        c_410 = sh.color(phot, "f277w", "f410m")
        if "f277w" in pivots_angstrom and "f410m" in pivots_angstrom:
            beta_vi = np.array(
                [
                    conventions.beta_from_color(c, pivots_angstrom["f277w"], pivots_angstrom["f410m"])
                    if np.isfinite(c) else np.nan
                    for c in c_410
                ]
            )
        else:
            beta_vi = np.full(n, np.nan)
        crit_vi = sh.gt(c_410, PRINTED_COLOR_410_MIN) if printed_variant else sh.gt(beta_vi, lineboost_beta_min)
        lineboost_410 = np.where(applies_vi, crit_vi, True)
    else:
        applies_vi = np.zeros(n, dtype=bool)
        beta_vi = np.full(n, np.nan)
        lineboost_410 = np.ones(n, dtype=bool)

    selected = detection & opt_red & uv_window & size_ok & lineboost_356 & lineboost_410
    out = sh.result(
        selected,
        detection=detection, beta_opt_red=opt_red, beta_uv_window=uv_window,
        size=size_ok, lineboost_356=lineboost_356, lineboost_410=lineboost_410,
        beta_uv=beta_uv, beta_opt=beta_opt,
        beta_uv_err=beta_uv_err, beta_opt_err=beta_opt_err,
        n_bands_uv=n_uv, n_bands_opt=n_opt,
        beta_lineboost_356=beta_v, beta_lineboost_410=beta_vi,
        lineboost_356_applies=applies_v, lineboost_410_applies=applies_vi,
    )
    out["lineboost_variant"] = (
        "formal-beta>-1" if lineboost_beta_min == -1.0 else "printed-colors>0.53/0.84"
    )
    return out
