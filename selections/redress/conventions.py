"""The project-wide spectral-slope convention, and every conversion through it.

Why this exists (the build plan, row 14): the LRD literature mixes two power-law
conventions with different signs, and a silent mixup flips "red" and "blue":

* ``f_lambda ∝ lambda^beta``  (beta_lambda — THIS PROJECT'S FIXED CONVENTION)
* ``f_nu ∝ nu^alpha``          (alpha_nu — common in AGN papers)

Derivation of the mapping (kept here so the owner can re-derive it cold):
``f_nu = f_lambda * lambda^2 / c``, so ``f_lambda ∝ lambda^beta`` implies
``f_nu ∝ lambda^(beta+2) ∝ nu^-(beta+2)``, hence::

    alpha_nu = -(beta_lambda + 2)          beta_lambda = -(alpha_nu + 2)

Anchor point: flat-f_nu (alpha = 0, zero AB color in ALL bands) is beta = -2.

AB color of a power law between pivot wavelengths lam_blue < lam_red::

    m_blue - m_red = -2.5*log10(f_nu(blue)/f_nu(red)) = 2.5*(beta+2)*log10(lam_red/lam_blue)

so beta > -2 means positive (red) AB color, beta < -2 means negative (blue).

AB magnitude convention: this project uses the exact-23.9 microjansky form,
``m_AB = 23.9 - 2.5*log10(f_nu / uJy)`` (the true 3631-Jy zero point differs by
6e-5 mag; stated once here, ignored thereafter).

Literature conversion ledger (filled as each paper's cuts are ingested in
``redress.cuts``; every entry quotes the paper's own convention sentence):

=====================  ==========================  =========================
Paper                  Their convention            Converted how
=====================  ==========================  =========================
(populated at U3 — each cut module documents its source quote and the
conversion applied, unit-tested against a hand-computed oracle)
=====================  ==========================  =========================
"""

from __future__ import annotations

import numpy as np
from astropy import constants as const
from astropy import units as u

#: The one project-wide convention. Everything converts TO this.
CONVENTION = "f_lambda proportional to lambda**beta_lambda"

_AB_ZP_UJY = 23.9  # exact-23.9 microjansky convention (see module docstring)


# ---------------------------------------------------------------- slope algebra

def alpha_nu_from_beta_lambda(beta: float) -> float:
    """``f_nu ∝ nu^alpha`` exponent equivalent to ``f_lambda ∝ lambda^beta``."""
    return -(beta + 2.0)


def beta_lambda_from_alpha_nu(alpha: float) -> float:
    """``f_lambda ∝ lambda^beta`` exponent equivalent to ``f_nu ∝ nu^alpha``."""
    return -(alpha + 2.0)


# ---------------------------------------------------------------- color <-> slope

def _coerce_wavelength_pair(lam_blue, lam_red) -> tuple[np.ndarray, np.ndarray]:
    """Return the pair as plain float arrays in ONE common unit, loudly.

    Council round 2: ``np.asarray(quantity)`` silently STRIPPED astropy units, so
    ``2*u.um`` vs ``20000*u.AA`` (the same physical wavelength) gave different
    answers. Rules now: Quantities are both-or-neither; when given, the red side
    is converted into the blue side's unit before any ratio; a plain/Quantity mix
    raises instead of guessing.
    """
    qb, qr = isinstance(lam_blue, u.Quantity), isinstance(lam_red, u.Quantity)
    if qb != qr:
        raise TypeError(
            "mixed Quantity and plain wavelengths: pass BOTH with units or BOTH plain "
            "(a stripped unit is a silent wrong answer)"
        )
    if qb:
        lam_red = lam_red.to_value(lam_blue.unit)
        lam_blue = lam_blue.to_value(lam_blue.unit)
    lam_blue = np.asarray(lam_blue, dtype=float)
    lam_red = np.asarray(lam_red, dtype=float)
    if np.any(~np.isfinite(lam_blue)) or np.any(~np.isfinite(lam_red)):
        raise ValueError("non-finite wavelengths")
    if np.any(lam_blue <= 0) or np.any(lam_red <= 0):
        raise ValueError("wavelengths must be positive")
    if np.any(lam_red <= lam_blue):
        raise ValueError(
            f"need lam_blue < lam_red elementwise (got {lam_blue!r}, {lam_red!r}); "
            "a color is blue-band minus red-band in this project"
        )
    return lam_blue, lam_red


def color_from_beta(beta: float, lam_blue, lam_red):
    """AB color ``m(lam_blue) - m(lam_red)`` of a ``lambda^beta`` power law.

    Wavelengths: plain numbers in the SAME unit, or astropy Quantities (any
    length units, converted safely). Scalar or array ``lam_red`` broadcasts.
    """
    lam_blue, lam_red = _coerce_wavelength_pair(lam_blue, lam_red)
    out = 2.5 * (beta + 2.0) * np.log10(lam_red / lam_blue)
    return out.item() if np.ndim(out) == 0 else out


def beta_from_color(color_mag: float, lam_blue, lam_red) -> float:
    """Invert :func:`color_from_beta`: the beta implied by one AB color."""
    lam_blue, lam_red = _coerce_wavelength_pair(lam_blue, lam_red)
    return float(color_mag / (2.5 * np.log10(lam_red / lam_blue)) - 2.0)


def beta_from_mag_regression(lam, mags, mag_errs=None) -> tuple[float, float]:
    """Fit beta_lambda from >=2 bands of AB photometry of an assumed power law.

    Model (from the module docstring): ``m = -2.5*(beta+2)*log10(lam) + C``, a
    straight line in ``log10(lam)``. Weighted least squares when ``mag_errs``
    is given (weights ``1/sigma``, parameter covariance taken as sigma-defined,
    i.e. NOT rescaled by residuals). Returns ``(beta, beta_err)``.

    Inputs must be finite — masking missing bands is the CALLER'S job at the
    ingest layer, so silence never hides a dropped band here.
    """
    if isinstance(lam, u.Quantity):
        lam = lam.to_value(lam.unit)  # only log-SPACING matters; any one unit is fine
    lam = np.asarray(lam, dtype=float)
    mags = np.asarray(mags, dtype=float)
    if lam.shape != mags.shape or lam.ndim != 1:
        raise ValueError("lam and mags must be 1-D arrays of equal length")
    if lam.size < 2:
        raise ValueError("need at least 2 bands to fit a slope")
    if np.any(~np.isfinite(lam)) or np.any(~np.isfinite(mags)):
        raise ValueError("non-finite input: mask missing bands at the ingest layer, not here")
    if np.any(lam <= 0):
        raise ValueError("wavelengths must be positive")
    x = np.log10(lam)
    if np.ptp(x) == 0:
        raise ValueError("all wavelengths identical: no leverage to fit a slope")

    if mag_errs is not None:
        mag_errs = np.asarray(mag_errs, dtype=float)
        if mag_errs.shape != mags.shape or np.any(~np.isfinite(mag_errs)) or np.any(mag_errs <= 0):
            raise ValueError("mag_errs must be positive, finite, same shape as mags")
        coef, cov = np.polyfit(x, mags, 1, w=1.0 / mag_errs, cov="unscaled")
        slope = float(coef[0])
        slope_err = float(np.sqrt(cov[0, 0]))
    else:
        # Council round 2: np.polyfit(cov=True) divides by N-deg-2 and so returns
        # nothing usable at N=3, though the textbook OLS error EXISTS for N>2.
        # Transparent manual OLS instead: slope = Sxy/Sxx; var(slope) = s^2/Sxx
        # with s^2 = RSS/(N-2). N=2 is an exact fit (no residual information):
        # error is honestly NaN there.
        xbar, ybar = x.mean(), mags.mean()
        sxx = float(np.sum((x - xbar) ** 2))
        slope = float(np.sum((x - xbar) * (mags - ybar)) / sxx)
        if lam.size > 2:
            resid = mags - (slope * (x - xbar) + ybar)
            s2 = float(np.sum(resid**2)) / (lam.size - 2)
            slope_err = float(np.sqrt(s2 / sxx))
        else:
            slope_err = float("nan")

    beta = -slope / 2.5 - 2.0             # slope = -2.5*(beta+2)
    return beta, slope_err / 2.5


# ---------------------------------------------------------------- AB magnitudes

def ab_mag_from_fnu_ujy(fnu_ujy: float) -> float:
    """AB magnitude of a flux density in microjansky (exact-23.9 convention)."""
    fnu_ujy = np.asarray(fnu_ujy, dtype=float)
    if np.any(fnu_ujy <= 0):
        raise ValueError("flux must be positive; encode non-detections as upper limits, not <=0 fluxes")
    out = _AB_ZP_UJY - 2.5 * np.log10(fnu_ujy)
    return out.item() if np.ndim(out) == 0 else out


def fnu_ujy_from_ab_mag(mag_ab: float) -> float:
    """Flux density in microjansky of an AB magnitude (exact-23.9 convention)."""
    out = 10.0 ** (-0.4 * (np.asarray(mag_ab, dtype=float) - _AB_ZP_UJY))
    return out.item() if np.ndim(out) == 0 else out


# ---------------------------------------------------------------- f_nu <-> f_lambda

def flambda_from_fnu(fnu: u.Quantity, lam: u.Quantity) -> u.Quantity:
    """``f_lambda = f_nu * c / lambda^2`` (own algebra; astropy only carries units)."""
    return (fnu * const.c / lam**2).to(u.erg / u.s / u.cm**2 / u.AA)


def fnu_from_flambda(flam: u.Quantity, lam: u.Quantity) -> u.Quantity:
    """``f_nu = f_lambda * lambda^2 / c`` (own algebra; astropy only carries units)."""
    return (flam * lam**2 / const.c).to(u.Jy)
