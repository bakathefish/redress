
import numpy as np
import pytest
from astropy import units as u
from hypothesis import given, settings
from hypothesis import strategies as st

from redress import conventions as conv

finite_beta = st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------- slope algebra

def test_flat_fnu_is_beta_minus_2():
    # f_nu flat (alpha = 0)  <=>  f_lambda ∝ lambda^-2. The classic anchor point.
    assert conv.beta_lambda_from_alpha_nu(0.0) == pytest.approx(-2.0)
    assert conv.alpha_nu_from_beta_lambda(-2.0) == pytest.approx(0.0)


def test_flat_flambda_is_alpha_minus_2():
    # f_lambda flat (beta = 0)  <=>  f_nu ∝ nu^-2.
    assert conv.alpha_nu_from_beta_lambda(0.0) == pytest.approx(-2.0)


@given(beta=finite_beta)
def test_alpha_beta_involution(beta):
    assert conv.beta_lambda_from_alpha_nu(conv.alpha_nu_from_beta_lambda(beta)) == pytest.approx(beta)


# ---------------------------------------------------------------- color <-> slope

def test_flat_fnu_has_zero_ab_color_in_any_bands():
    # AB magnitudes are defined on f_nu, so a flat-f_nu source (beta = -2) has
    # ZERO AB color between ANY two bands. This invariant catches sign errors cold.
    for lam1, lam2 in [(1.15, 4.44), (0.9, 1.5), (2.77, 3.56)]:
        assert conv.color_from_beta(-2.0, lam1, lam2) == pytest.approx(0.0)


def test_red_slope_gives_positive_color():
    # beta > -2 (redder than flat f_nu) must give blue-minus-red color > 0.
    assert conv.color_from_beta(0.0, 2.786, 4.421) > 0


def test_color_known_value_by_hand():
    # Hand oracle: color = 2.5 * (beta + 2) * log10(lam2/lam1).
    # beta = 0, lam2/lam1 = 2  ->  color = 2.5 * 2 * log10(2) = 5 * 0.3010299957 = 1.50515.
    assert conv.color_from_beta(0.0, 1.0, 2.0) == pytest.approx(1.5051499783, abs=1e-9)


@given(beta=finite_beta,
       lam1=st.floats(min_value=0.3, max_value=5.0),
       ratio=st.floats(min_value=1.05, max_value=30.0))
@settings(max_examples=100)
def test_color_beta_round_trip(beta, lam1, ratio):
    lam2 = lam1 * ratio
    color = conv.color_from_beta(beta, lam1, lam2)
    assert conv.beta_from_color(color, lam1, lam2) == pytest.approx(beta, abs=1e-9)


def test_color_unit_invariance_with_quantity_inputs():
    # council round 2 deciding test: the same physical wavelength in different
    # units must give the same answer (np.asarray used to strip units silently).
    a = conv.color_from_beta(1.3, 1 * u.um, 2 * u.um)
    b = conv.color_from_beta(1.3, 1 * u.um, 20000 * u.AA)
    assert a == pytest.approx(b, rel=1e-12)
    assert a == pytest.approx(conv.color_from_beta(1.3, 1.0, 2.0), rel=1e-12)
    assert conv.beta_from_color(a, 1 * u.um, 20000 * u.AA) == pytest.approx(1.3, abs=1e-9)


def test_mixed_quantity_and_plain_wavelengths_rejected():
    with pytest.raises(TypeError):
        conv.color_from_beta(0.0, 1 * u.um, 2.0)
    with pytest.raises(TypeError):
        conv.color_from_beta(0.0, 1.0, 2 * u.um)


def test_color_rejects_bad_wavelengths():
    with pytest.raises(ValueError):
        conv.color_from_beta(0.0, 2.0, 2.0)   # equal wavelengths: no slope defined
    with pytest.raises(ValueError):
        conv.color_from_beta(0.0, 3.0, 2.0)   # blue/red swapped
    with pytest.raises(ValueError):
        conv.color_from_beta(0.0, -1.0, 2.0)  # non-physical wavelength


# ---------------------------------------------------------------- beta regression

def test_beta_regression_recovers_exact_power_law():
    # Synthetic photometry of a pure power law must return beta exactly (no noise).
    beta_true = 1.7
    lam = np.array([1.15, 1.50, 2.00, 2.79, 3.56, 4.42])
    # color(ref, lam) = m(ref) - m(lam), so m(lam) = ZP - color(ref, lam); the
    # constant offset drops out of the slope fit. (First draft wrote ZP + color —
    # the fit returned -beta-4 and caught the sign error, which is this module's job.)
    mags = 24.0 - conv.color_from_beta(beta_true, 0.5, lam)
    beta_fit, beta_err = conv.beta_from_mag_regression(lam, mags)
    assert beta_fit == pytest.approx(beta_true, abs=1e-9)
    assert beta_err == pytest.approx(0.0, abs=1e-6)


def test_beta_regression_weights_downweight_outliers():
    beta_true = -0.5
    lam = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
    mags = 22.0 - conv.color_from_beta(beta_true, 0.5, lam)
    mags_bad = mags.copy()
    mags_bad[2] += 5.0  # gross outlier
    errs = np.array([0.02, 0.02, 10.0, 0.02, 0.02])  # outlier carries ~no weight
    beta_fit, _ = conv.beta_from_mag_regression(lam, mags_bad, mag_errs=errs)
    assert beta_fit == pytest.approx(beta_true, abs=1e-3)


def test_beta_regression_error_scales_with_mag_error():
    lam = np.array([1.0, 2.0, 4.0])
    mags = 25.0 - conv.color_from_beta(0.0, 0.5, lam)
    _, err_small = conv.beta_from_mag_regression(lam, mags, mag_errs=np.full(3, 0.01))
    _, err_big = conv.beta_from_mag_regression(lam, mags, mag_errs=np.full(3, 0.1))
    assert err_big == pytest.approx(10 * err_small, rel=1e-6)


def test_beta_regression_three_band_unweighted_has_finite_error():
    # council round 2 deciding test: 3 noisy bands leave 1 residual degree of
    # freedom, so a finite OLS slope error EXISTS and must be returned (the old
    # polyfit-based branch returned NaN at N=3 because of its N-deg-2 divisor).
    lam = np.array([1.0, 2.0, 4.0])
    mags = 25.0 - conv.color_from_beta(0.3, 0.5, lam) + np.array([0.01, -0.02, 0.01])
    beta_fit, beta_err = conv.beta_from_mag_regression(lam, mags)
    assert np.isfinite(beta_err) and beta_err > 0
    assert beta_fit == pytest.approx(0.3, abs=0.2)


def test_beta_regression_accepts_quantity_wavelengths():
    lam_um = np.array([1.15, 2.0, 4.4])
    mags = 24.0 - conv.color_from_beta(-1.0, 0.5, lam_um)
    b_plain, _ = conv.beta_from_mag_regression(lam_um, mags)
    b_um, _ = conv.beta_from_mag_regression(lam_um * u.um, mags)
    b_aa, _ = conv.beta_from_mag_regression(lam_um * 1e4 * u.AA, mags)
    assert b_plain == pytest.approx(b_um, abs=1e-12)
    # unit only shifts log10(lam) by a constant; the slope (hence beta) is unchanged
    assert b_plain == pytest.approx(b_aa, abs=1e-9)


def test_beta_regression_input_validation():
    with pytest.raises(ValueError):
        conv.beta_from_mag_regression(np.array([1.0]), np.array([24.0]))  # <2 points
    with pytest.raises(ValueError):
        conv.beta_from_mag_regression(np.array([1.0, 1.0]), np.array([24.0, 25.0]))  # no leverage
    with pytest.raises(ValueError):
        conv.beta_from_mag_regression(np.array([1.0, 2.0]), np.array([24.0, np.nan]))  # NaN: caller masks


# ---------------------------------------------------------------- AB magnitudes

def test_ab_zero_points():
    # Definition anchors: 3631 Jy  ->  m_AB = 0;  1 microJy  ->  m_AB = 23.9 (both exact
    # under the 23.9 convention; the true zero point 2.5*log10(3631e6) = 23.90006 differs
    # by 6e-5 mag — the module must state which it uses; we pin the exact-23.9 form).
    assert conv.ab_mag_from_fnu_ujy(1.0) == pytest.approx(23.9)
    assert conv.fnu_ujy_from_ab_mag(23.9) == pytest.approx(1.0)
    assert conv.ab_mag_from_fnu_ujy(3631.0e6) == pytest.approx(0.0, abs=1e-3)


@given(m=st.floats(min_value=10, max_value=35))
def test_ab_round_trip(m):
    assert conv.ab_mag_from_fnu_ujy(conv.fnu_ujy_from_ab_mag(m)) == pytest.approx(m)


def test_ab_rejects_nonpositive_flux():
    with pytest.raises(ValueError):
        conv.ab_mag_from_fnu_ujy(0.0)
    with pytest.raises(ValueError):
        conv.ab_mag_from_fnu_ujy(-3.0)


# ---------------------------------------------------------------- f_nu <-> f_lambda

def test_flambda_fnu_against_astropy_oracle():
    # Independent oracle: astropy's spectral_density equivalency.
    lam = 5500 * u.AA
    fnu = 3631 * u.Jy
    expected = fnu.to(u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(lam))
    got = conv.flambda_from_fnu(fnu, lam)
    assert got.unit.is_equivalent(u.erg / u.s / u.cm**2 / u.AA)
    assert got.to_value(expected.unit) == pytest.approx(expected.value, rel=1e-10)


def test_fnu_flambda_round_trip():
    lam = 2.0 * u.um
    flam = 1e-19 * u.erg / u.s / u.cm**2 / u.AA
    back = conv.fnu_from_flambda(conv.flambda_from_fnu(conv.fnu_from_flambda(flam, lam), lam), lam)
    assert back.to_value(u.uJy) == pytest.approx(
        conv.fnu_from_flambda(flam, lam).to_value(u.uJy), rel=1e-12
    )
