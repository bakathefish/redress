"""The continuum slope: linear-space power-law fit with identifiability and edge guards.

VERSION 3. Version 1 fitted log10(f), which requires f > 0 and is a one-sided cut on noise;
version 2 moved to linear space and fixed that. review round S1 then found that version 2
still reported a slope in cases where NO SLOPE EXISTS, and quantified it: on 90 pure-noise
pixels, 43.6% of fits pinned at the grid edge and about 4.6% still satisfied
`beta - 2*sigma > 0`. That is a false-positive channel straight into the candidate list, and
it explains the objects surviving version 2 with beta_opt near 3, 4 and 7 -- values that in
f_lambda mean f_nu proportional to lambda^5 to lambda^9, which no real continuum does.

THE FIVE GUARDS, each answering a specific defect.

  1. IDENTIFIABILITY. When the continuum amplitude is consistent with zero, every beta
     predicts the same (zero) spectrum and the profile is choosing whichever power law best
     correlates with noise. The fit is refused unless the amplitude is DETECTED:
     a / sigma_a >= MIN_CONTINUUM_SNR, with sigma_a from the same weighted design. This is
     the guard that matters most, because it is the null under which everything else fails.

  2. NO EDGE MINIMA. A minimum at the end of the beta grid is not a minimum; it only says
     the profile is still falling. Version 2 returned the clipped edge value and built an
     interval around it, with the delta-chi-squared measured from a point that was not the
     optimum. Edge minima are now refused outright.

  3. ONE-SIDED PROFILE BOUNDS, NOT +/- 2 SIGMA. The chi-squared profile is asymmetric, and
     collapsing it to a single symmetric sigma is anti-conservative on precisely the side
     each criterion uses. The optical criterion needs the LOWER bound of beta_opt and the UV
     criterion needs the UPPER bound of beta_UV, so both are computed directly at
     delta-chi-squared = 4 (the two-sigma equivalent for one parameter) and returned.

  4. A MISSING BOUND IS A REFUSAL. If the profile never crosses on the side a criterion
     needs, that side is unconstrained and the criterion cannot be evaluated. Version 2
     substituted the other side's width, which let an unconstrained tail pass on borrowed
     uncertainty.

  5. CORRELATED PIXELS. NIRSpec PRISM is undersampled and adjacent pixels are not
     independent, so a diagonal chi-squared understates sigma by up to sqrt(2). The bounds
     are therefore also computed from a BLOCK BOOTSTRAP over residuals in blocks of
     BLOCK_PIX pixels, which preserves short-range correlation, and the conservative
     (wider) of the two is used. The inflation factor is returned so it can be reported as a
     measured quantity rather than assumed.

WHAT MAY BE CLAIMED. This estimator "removes the demonstrated log-space truncation bias and
refuses non-identifiable fits". It is NOT claimed to be unbiased -- a nonlinear maximum
likelihood estimate is not finite-sample unbiased, and wavelength-dependent calibration or
background residuals would bias any estimator. review round S1 made that wording correction
and it is adopted.
"""

import numpy as np

BETA_GRID = np.linspace(-8.0, 8.0, 641)
MIN_CONTINUUM_SNR = 3.0  # guard 1: amplitude must be detected before a slope is claimed
MIN_PIX = 12  # raised from 8; see MIN_SPAN_DEX for the real information test
MIN_SPAN_DEX = 0.08  # a slope needs LEVER ARM, not merely pixel count
BLOCK_PIX = 2  # ~one PRISM resolution element; preserves short-range noise
N_BOOT = 200
MIN_BOOT_INTERIOR = 0.6      # below this the slope is not stable under resampling
MAX_INFLATION = 2.0          # sqrt(2) for two-pixel correlation, with headroom
DCHI2_1SIG = 1.0
DCHI2_2SIG = 4.0  # the one-parameter two-sigma equivalent


def _profile(P, PP, w, y):
    """chi2(beta) and amplitude(beta), vectorised over the whole grid.

    Profiling the linear amplitude out gives
        chi2(b) = y'Wy - (y'Wp_b)^2 / (p_b'Wp_b),  a(b) = (y'Wp_b) / (p_b'Wp_b)
    so the entire profile costs two matrix-vector products against precomputed powers.
    """
    A = (w * y) @ P
    B = w @ PP
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.sum(w * y * y)) - (A * A) / B
        amp = A / B
    return chi2, amp, B


def _cross(grid, chi2, i0, level, direction):
    """First beta on one side of the minimum where chi2 rises by `level`, interpolated.

    Returns nan when the profile never crosses inside the grid, which is the honest answer:
    that side is unconstrained.
    """
    cmin = chi2[i0]
    rng = range(i0 + 1, len(grid)) if direction > 0 else range(i0 - 1, -1, -1)
    prev = i0
    for i in rng:
        if not np.isfinite(chi2[i]):
            return np.nan
        if chi2[i] - cmin >= level:
            c0, c1 = chi2[prev] - cmin, chi2[i] - cmin
            if c1 == c0:
                return float(grid[i])
            f = (level - c0) / (c1 - c0)
            return float(grid[prev] + f * (grid[i] - grid[prev]))
        prev = i
    return np.nan


def fit_powerlaw(lam, f, e, lo, hi, mask=None):
    """Fit f = a * lam^beta over [lo, hi) in LINEAR flux space, with the five guards.

    Returns a dict. `ok` is False whenever the slope is not a measurement, with `reason`
    saying which guard refused it. `beta_lo2` / `beta_hi2` are the DIRECT one-sided
    delta-chi-squared = 4 bounds after correlated-noise inflation, and are what the
    selection criteria must use -- never beta +/- 2*sigma.
    """
    bad = {
        "ok": False,
        "beta": np.nan,
        "beta_lo2": np.nan,
        "beta_hi2": np.nan,
        "amp": np.nan,
        "amp_snr": np.nan,
        "n": 0,
        "span_dex": np.nan,
        "snr_pix": np.nan,
        "infl": np.nan,
        "reason": "",
    }
    m = (lam >= lo) & (lam < hi) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    if mask is not None:
        m = m & mask
    n = int(m.sum())
    if n < MIN_PIX:
        return dict(bad, n=n, reason=f"only {n} usable pixels")

    x, y, s = lam[m], f[m], e[m]
    span = float(np.log10(x.max()) - np.log10(x.min()))
    if span < MIN_SPAN_DEX:
        return dict(
            bad, n=n, span_dex=span, reason=f"wavelength span only {span:.3f} dex"
        )

    xn = x / np.median(x)  # exactly invariant for beta; see review round S1
    w = 1.0 / (s * s)
    P = xn[:, None] ** BETA_GRID[None, :]
    PP = P * P
    chi2, amp, B = _profile(P, PP, w, y)
    if not np.isfinite(chi2).any():
        return dict(bad, n=n, span_dex=span, reason="profile is not finite")

    i0 = int(np.nanargmin(chi2))
    if i0 == 0 or i0 == len(BETA_GRID) - 1:  # guard 2
        return dict(
            bad,
            n=n,
            span_dex=span,
            beta=float(BETA_GRID[i0]),
            reason="minimum at the beta-grid edge: unconstrained, not measured",
        )

    beta = float(BETA_GRID[i0])
    a = float(amp[i0])
    sig_a = float(1.0 / np.sqrt(B[i0])) if B[i0] > 0 else np.inf
    amp_snr = a / sig_a if np.isfinite(sig_a) and sig_a > 0 else np.nan
    snr_pix = float(np.median(y / s))
    if not np.isfinite(amp_snr) or amp_snr < MIN_CONTINUUM_SNR:  # guard 1
        return dict(
            bad,
            n=n,
            span_dex=span,
            beta=beta,
            amp=a,
            amp_snr=amp_snr,
            snr_pix=snr_pix,
            reason=f"continuum not detected (amplitude S/N {amp_snr if np.isfinite(amp_snr) else -1:.2f} < {MIN_CONTINUUM_SNR:.1f})",
        )

    lo2 = _cross(BETA_GRID, chi2, i0, DCHI2_2SIG, -1)  # guard 3
    hi2 = _cross(BETA_GRID, chi2, i0, DCHI2_2SIG, +1)
    lo1 = _cross(BETA_GRID, chi2, i0, DCHI2_1SIG, -1)
    hi1 = _cross(BETA_GRID, chi2, i0, DCHI2_1SIG, +1)

    # guard 5: block bootstrap over residuals, preserving short-range pixel correlation
    model = a * xn**beta
    resid = y - model
    nb = max(2, n // BLOCK_PIX)
    blocks = np.array_split(np.arange(n), nb)
    rng = np.random.default_rng(20260828 + n)
    draws = np.full(N_BOOT, np.nan)
    for k in range(N_BOOT):
        pick = rng.integers(0, len(blocks), len(blocks))
        rr = np.concatenate([resid[blocks[j]] for j in pick])
        rr = rr[:n] if rr.size >= n else np.resize(rr, n)
        c2, _am, _b = _profile(P, PP, w, model + rr)
        if np.isfinite(c2).any():
            draws[k] = float(np.nanargmin(c2))

    # A resample whose minimum PINS AT THE GRID EDGE did not measure a slope; it only says
    # that resample was unconstrained. Such draws carry no information about the width of
    # the distribution of MEASURED slopes, and including them is what produced a nonsense
    # median inflation of 4.17 in the UV window on the first run -- while the optical
    # window, where pins are rarer, read a believable 1.06. They are excluded and counted,
    # and if they dominate, the fit is refused as unstable rather than quietly widened.
    fin = draws[np.isfinite(draws)].astype(int)
    interior = fin[(fin > 0) & (fin < len(BETA_GRID) - 1)]
    frac_interior = interior.size / float(N_BOOT)
    if frac_interior < MIN_BOOT_INTERIOR:
        return dict(
            bad,
            n=n,
            span_dex=span,
            beta=beta,
            amp=a,
            amp_snr=amp_snr,
            snr_pix=snr_pix,
            reason="unstable under resampling (%.0f%% of bootstraps unconstrained)"
            % (100 * (1 - frac_interior)),
        )
    bd = BETA_GRID[interior]
    # a ROBUST scale: even among interior draws the distribution has heavy tails, and a
    # standard deviation would let a handful of far excursions set the whole interval
    boot_sig = float(1.4826 * np.median(np.abs(bd - np.median(bd))))
    prof_sig = (
        0.5 * ((hi1 - beta) + (beta - lo1))
        if np.isfinite(hi1) and np.isfinite(lo1)
        else (
            hi1 - beta
            if np.isfinite(hi1)
            else (beta - lo1 if np.isfinite(lo1) else np.nan)
        )
    )
    infl = (boot_sig / prof_sig) if (np.isfinite(prof_sig) and prof_sig > 0) else np.nan
    # Two-pixel correlation can widen sigma by at most about sqrt(2), and longer-range
    # structure by a little more. A value far above that is the bootstrap failing, not the
    # noise being correlated, so the factor is capped and the cap is a recorded column.
    k_infl = float(np.clip(infl, 1.0, MAX_INFLATION)) if np.isfinite(infl) else 1.0
    if np.isfinite(lo2):
        lo2 = beta - k_infl * (beta - lo2)
    if np.isfinite(hi2):
        hi2 = beta + k_infl * (hi2 - beta)

    return {
        "ok": True,
        "beta": beta,
        "beta_lo2": lo2,
        "beta_hi2": hi2,
        "amp": a,
        "amp_snr": amp_snr,
        "n": n,
        "span_dex": span,
        "snr_pix": snr_pix,
        "infl": k_infl,
        "reason": "",
    }
