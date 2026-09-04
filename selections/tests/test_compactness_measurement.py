"""The compactness producer, tested against cases with EXACT analytic answers.

WHY THESE TESTS AND NOT PICTURES OF GALAXIES. The quantity being produced is a ratio of
two aperture fluxes, and the two ways to get it wrong are silent: confusing DIAMETER with
RADIUS, and integrating pixels by centre rather than by area. Both produce plausible
numbers. Neither shows up as a crash.

A UNIFORM image makes both catchable exactly. On a constant field, flux is proportional
to aperture AREA, so the ratio of two aperture fluxes is the ratio of their areas and has
a closed form:

    f(d=0.20") / f(d=0.50")  =  (0.10/0.25)^2  =  0.16          <- Akins C444
    f(d=0.40") / f(d=0.20")  =  (0.20/0.10)^2  =  4.00          <- Labbe/Kokorev

If the code treated those numbers as radii instead of diameters the ratios would be
unchanged (both scale together) — so a second test fixes the ABSOLUTE scale against
pi*r^2 in pixels, which radius-vs-diameter confusion cannot survive.

The protocol these tests enforce is frozen and owner-signed at
the author's frozen compactness measurement protocol. Every rule below quotes the section it
comes from, because a producer that drifts from the signed protocol is exactly the
failure the signature exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS

pytest.importorskip("photutils")

from redress.compactness import (
    APERTURE_DIAMETERS_ARCSEC,
    compactness_columns,
    measure_apertures,
)

PIXEL_SCALE_ARCSEC = 0.04  # DJA v7 sampling; the code must READ this, not assume it


def _wcs(pixel_scale_arcsec: float = PIXEL_SCALE_ARCSEC, size: int = 201) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [size / 2 + 0.5, size / 2 + 0.5]
    w.wcs.cdelt = [-pixel_scale_arcsec / 3600.0, pixel_scale_arcsec / 3600.0]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _uniform(size: int = 201, value: float = 1.0) -> np.ndarray:
    return np.full((size, size), value, dtype=float)


def _centre(w: WCS, size: int = 201) -> tuple[float, float]:
    ra, dec = w.wcs_pix2world([[size / 2 - 0.5, size / 2 - 0.5]], 0)[0]
    return float(ra), float(dec)


# --------------------------------------------------------------- the exact analytic cases


def test_uniform_field_gives_the_exact_area_ratios() -> None:
    """[protocol 3.4, 3.5] The whole point: on a constant field the flux ratio IS the area
    ratio, so both the diameter convention and the exact-area integration are pinned by a
    closed-form number rather than by a regression baseline."""
    w = _wcs()
    ra, dec = _centre(w)
    out = measure_apertures(_uniform(), w, [ra], [dec])
    cols = compactness_columns(out)

    assert cols["compactness_f444w"][0] == pytest.approx(0.16, rel=1e-3)
    assert cols["labbe_compactness"][0] == pytest.approx(4.00, rel=1e-3)


def test_absolute_aperture_areas_pin_diameter_not_radius() -> None:
    """The ratios above survive a diameter/radius mix-up because both terms scale together.
    The ABSOLUTE flux does not: on a unit field the flux in an aperture is its area in
    pixels, pi*(d/2/scale)^2. Treating 0.20 as a radius would quadruple it."""
    w = _wcs()
    ra, dec = _centre(w)
    out = measure_apertures(_uniform(), w, [ra], [dec])

    for name, diameter in APERTURE_DIAMETERS_ARCSEC.items():
        radius_px = (diameter / 2.0) / PIXEL_SCALE_ARCSEC
        assert out[f"flux_{name}"][0] == pytest.approx(
            np.pi * radius_px**2, rel=2e-3
        ), f'{name}: diameter {diameter}" must be a DIAMETER'


def test_pixel_scale_is_read_from_the_wcs_not_assumed() -> None:
    """[protocol 3.4] "not assumed to be 0.04"/pixel; the DJA products are distributed at
    more than one sampling and the scale is read from each mosaic." At half the pixel
    scale an aperture covers four times the pixels."""
    fine = _wcs(pixel_scale_arcsec=PIXEL_SCALE_ARCSEC / 2)
    ra, dec = _centre(fine)
    out = measure_apertures(_uniform(), fine, [ra], [dec])
    radius_px = (APERTURE_DIAMETERS_ARCSEC["d050"] / 2.0) / (PIXEL_SCALE_ARCSEC / 2)
    assert out["flux_d050"][0] == pytest.approx(np.pi * radius_px**2, rel=2e-3)


def test_exact_integration_beats_centre_counting() -> None:
    """[protocol 3.5] At a 0.10" radius the 0.2" aperture is only five pixels across, so
    whole-pixel counting is a first-order error rather than a rounding detail. Exact
    geometric overlap has no integer-pixel signature; centre-counting does."""
    w = _wcs()
    ra, dec = _centre(w)
    flux = measure_apertures(_uniform(), w, [ra], [dec])["flux_d020"][0]
    assert abs(flux - round(flux)) > 1e-6, "an exact area is not an integer pixel count"


# ------------------------------------------------------------------- forced centring


def test_centring_is_forced_at_the_given_position() -> None:
    """[protocol 3.3] Forced photometry at the catalogue position, with no re-centroiding.
    A bright pixel deliberately placed away from the aperture centre must NOT pull the
    aperture onto itself."""
    w = _wcs()
    ra, dec = _centre(w)
    image = np.zeros((201, 201))
    image[100, 100] = 1.0  # at the requested centre
    image[100, 140] = 1000.0  # far brighter, well outside every aperture

    out = measure_apertures(image, w, [ra], [dec])
    assert out["flux_d050"][0] == pytest.approx(1.0, rel=1e-6), (
        "a re-centroiding implementation would capture the bright decoy"
    )


# ----------------------------------------------------------------- refusals, per 3.7/3.9


def test_nan_inside_an_aperture_is_unmeasurable_not_zero() -> None:
    """[protocol 3.7] "An aperture that is not fully on valid science pixels yields NO
    compactness value — the object is recorded as unmeasurable rather than given a number
    computed from partial coverage." """
    w = _wcs()
    ra, dec = _centre(w)
    image = _uniform()
    image[100, 103] = np.nan  # inside the 0.5" aperture (radius 6.25 px)

    out = measure_apertures(image, w, [ra], [dec])
    assert not bool(out["measurable"][0])
    assert "nan" in str(out["unmeasurable_reason"][0]).lower()
    assert np.isnan(out["flux_d050"][0])


def test_aperture_off_the_footprint_is_unmeasurable() -> None:
    """[protocol 3.9] An aperture leaving the image is refused rather than silently
    truncated, which would understate the numerator and manufacture compactness."""
    w = _wcs()
    ra, dec = w.wcs_pix2world([[2.0, 2.0]], 0)[0]  # 2 px from the corner
    out = measure_apertures(_uniform(), w, [float(ra)], [float(dec)])
    assert not bool(out["measurable"][0])
    assert "footprint" in str(out["unmeasurable_reason"][0]).lower()


def test_non_positive_denominator_yields_no_ratio() -> None:
    """[protocol 3.9] "the measurement refuses ... when the denominator is non-positive."
    A negative sky patch must not produce a ratio with a sign nobody can interpret."""
    w = _wcs()
    ra, dec = _centre(w)
    out = measure_apertures(_uniform(value=-1.0), w, [ra], [dec])
    cols = compactness_columns(out)
    assert np.isnan(cols["compactness_f444w"][0])
    assert np.isnan(cols["labbe_compactness"][0])


def test_unmeasurable_never_reads_as_a_satisfied_cut() -> None:
    """[protocol 3.7] "Unmeasurable is not 'fails the cut'." The ratio is NaN so that any
    comparison against a threshold is False in both directions — an absent measurement can
    never pass a cut and can never be recorded as having failed one."""
    w = _wcs()
    ra, dec = _centre(w)
    image = _uniform()
    image[100, 100] = np.nan

    cols = compactness_columns(measure_apertures(image, w, [ra], [dec]))
    value = cols["compactness_f444w"][0]
    assert np.isnan(value)
    assert not (value > 0.5)
    assert not (value <= 0.5)


# ------------------------------------------------------- recorded-but-not-applied fields


def test_background_annulus_is_recorded_and_not_applied() -> None:
    """[protocol 3.6] "Recorded per object but NOT applied." A pedestal must show up in the
    recorded annulus value AND in the fluxes, because subtracting it here would be a free
    parameter the signed protocol does not grant."""
    w = _wcs()
    ra, dec = _centre(w)
    out = measure_apertures(_uniform(value=3.0), w, [ra], [dec])

    assert out["bkg_annulus_median"][0] == pytest.approx(3.0, rel=1e-6)
    radius_px = (APERTURE_DIAMETERS_ARCSEC["d050"] / 2.0) / PIXEL_SCALE_ARCSEC
    assert out["flux_d050"][0] == pytest.approx(3.0 * np.pi * radius_px**2, rel=2e-3)


def test_no_aperture_correction_is_applied() -> None:
    """[protocol 3.8] "None applied inside the ratio." On a uniform field an
    aperture-corrected flux would exceed the geometric area; a raw sum equals it."""
    w = _wcs()
    ra, dec = _centre(w)
    out = measure_apertures(_uniform(), w, [ra], [dec])
    radius_px = (APERTURE_DIAMETERS_ARCSEC["d020"] / 2.0) / PIXEL_SCALE_ARCSEC
    assert out["flux_d020"][0] == pytest.approx(np.pi * radius_px**2, rel=2e-3)


def test_many_positions_in_one_call_keep_row_order() -> None:
    """Rows are returned in the order given. A silent reorder would mis-attach every
    measured ratio to the wrong object, which no downstream check would catch."""
    w = _wcs()
    ra0, dec0 = _centre(w)
    edge_ra, edge_dec = w.wcs_pix2world([[2.0, 2.0]], 0)[0]

    out = measure_apertures(
        _uniform(), w, [ra0, float(edge_ra), ra0], [dec0, float(edge_dec), dec0]
    )
    assert len(out) == 3
    assert bool(out["measurable"][0])
    assert not bool(out["measurable"][1])
    assert bool(out["measurable"][2])


# ------------------------------------------------------------------ the compactness pair


def test_the_two_ratios_point_in_opposite_directions() -> None:
    """The direction trap the repo flags: Akins's C444 is small-over-large so COMPACT means
    a LARGE value, while the Labbe/Kokorev ratio is large-over-small so compact means a
    SMALL value. A point source must move them opposite ways relative to a flat field."""
    w = _wcs()
    ra, dec = _centre(w)

    flat = compactness_columns(measure_apertures(_uniform(), w, [ra], [dec]))
    point = np.zeros((201, 201))
    point[100, 100] = 1.0
    peaked = compactness_columns(measure_apertures(point, w, [ra], [dec]))

    assert peaked["compactness_f444w"][0] > flat["compactness_f444w"][0]
    assert peaked["labbe_compactness"][0] < flat["labbe_compactness"][0]


# ------------------------------------------- review round 6 regression tests (A2, A3, A5)


def test_unmeasurable_object_yields_no_ratio_from_its_surviving_apertures() -> None:
    """[review round 6, A5 — the real defect] A NaN in the ring covered by the 0.5"
    aperture but NOT the 0.4" one leaves d020 and d040 perfectly measured. The old code
    marked the row unmeasurable and STILL returned labbe_compactness = d040/d020 = 4.0.

    Protocol 3.7 declares unmeasurability at the level of the OBJECT, so an object the
    producer has refused cannot carry a ratio out of the same call."""
    w = _wcs()
    ra, dec = _centre(w)
    image = _uniform()
    image[100, 106] = np.nan  # 6 px out: inside d050 (6.25 px) and outside d040 (5 px)

    out = measure_apertures(image, w, [ra], [dec])
    cols = compactness_columns(out)

    assert not bool(out["measurable"][0])
    assert "d050" in str(out["unmeasurable_reason"][0])
    assert np.isfinite(out["flux_d020"][0]) and np.isfinite(out["flux_d040"][0])
    assert np.isnan(cols["compactness_f444w"][0])
    assert np.isnan(cols["labbe_compactness"][0]), (
        "a refused object must not return the ratio its surviving apertures allow"
    )


def test_partial_footprint_loss_also_refuses_the_surviving_ratio() -> None:
    """[review round 6, A5] The same defect reached through the footprint path rather than
    the NaN path: place the object so only d050 leaves the mosaic."""
    w = _wcs()
    ra, dec = w.wcs_pix2world([[5.0, 100.0]], 0)[0]

    out = measure_apertures(_uniform(), w, [float(ra)], [float(dec)])
    cols = compactness_columns(out)

    assert not bool(out["measurable"][0])
    assert np.isfinite(out["flux_d020"][0]) and np.isfinite(out["flux_d040"][0])
    assert np.isnan(cols["labbe_compactness"][0])


def test_the_frozen_diameters_cannot_be_overridden_by_a_caller() -> None:
    """[review round 6, A2] The producer had a public diameters_arcsec parameter, so any
    caller could replace the signed apertures without a protocol amendment. Changing an
    aperture is a signature event, not an argument."""
    import inspect

    params = set(inspect.signature(measure_apertures).parameters)
    assert "diameters_arcsec" not in params
    assert params == {"image", "wcs", "ra", "dec"}


def test_local_recentroiding_is_also_excluded() -> None:
    """[review round 6] The original centring test used a decoy 40 px away, which a
    SMALL-window re-centroiding implementation would ignore and still pass. This decoy
    sits 3 px out: inside d040 and d050, outside d020. Any nudge toward it pulls the
    decoy into the numerator and inflates compactness."""
    w = _wcs()
    ra, dec = _centre(w)
    image = np.zeros((201, 201))
    image[100, 100] = 1.0
    image[100, 103] = 1000.0

    out = measure_apertures(image, w, [ra], [dec])
    assert out["flux_d020"][0] == pytest.approx(1.0, rel=1e-6), (
        "the 0.2\" numerator must not drift onto a neighbour 3 px away"
    )
    assert out["flux_d040"][0] == pytest.approx(1001.0, rel=1e-6)


def test_the_integration_method_that_ran_is_recorded() -> None:
    """[review round 6, test finding] No numerical test separates exact overlap from
    sufficiently fine subpixel quadrature — subpixels=4096 passes every closed-form test
    in this file, because the two converge. The producer therefore RECORDS the method it
    used, so a silent substitution shows up in the output rather than nowhere."""
    w = _wcs()
    ra, dec = _centre(w)
    out = measure_apertures(_uniform(), w, [ra], [dec])
    assert out["integration_method"][0] == "exact"


def test_the_annulus_uses_exact_membership_not_centre_membership() -> None:
    """[review round 6, A3] The annulus used method="center" while protocol 3.5 fixes
    "exact" and expressly says NOT "center". A median hides that, because both return the
    same value on a uniform field — so the discriminator is MEMBERSHIP, which the
    producer now reports as bkg_annulus_npix. Exact overlap admits every pixel the
    annulus clips at all, so it must admit strictly more pixels than centre-counting."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from photutils.aperture import SkyCircularAnnulus

    from redress.compactness import BACKGROUND_ANNULUS_ARCSEC

    w = _wcs()
    ra, dec = _centre(w)
    out = measure_apertures(_uniform(), w, [ra], [dec])

    inner, outer = BACKGROUND_ANNULUS_ARCSEC
    ann = SkyCircularAnnulus(
        SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
        r_in=inner * u.arcsec,
        r_out=outer * u.arcsec,
    ).to_pixel(w)
    n_centre = int((ann.to_mask(method="center").to_image((201, 201)) > 0).sum())
    n_exact = int((ann.to_mask(method="exact").to_image((201, 201)) > 0).sum())

    assert n_exact > n_centre, "the two methods must actually differ for this to test"
    assert int(out["bkg_annulus_npix"][0]) == n_exact
