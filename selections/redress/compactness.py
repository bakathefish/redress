"""The compactness producer: the aperture photometry the public catalogues do not carry.

WHY THIS EXISTS. The DJA catalogues measure per-band aperture photometry at exactly two
diameters, 0.36" and 0.50". Three of the seven published selections need a 0.2" aperture
that is simply not there:

    Labbe+23    f444(0.4")/f444(0.2") < 1.7      needs 0.2" AND 0.4"
    Kokorev+24  as Labbe                          needs 0.2" AND 0.4"
    Akins+24    0.5 < f444(0.2")/f444(0.5") <= 0.7   needs 0.2"

Decision D-2 ruled that the missing photometry is MEASURED rather than that anybody's cut
is redefined, so each published criterion stays as published. `DATA_CONTRACTS.md` line 57
fixes the DEFINITION of `compactness_f444w` and not its source, so measuring satisfies the
frozen contract.

EVERY CHOICE IN THIS FILE IS FROZEN ELSEWHERE. Centring, apertures, pixel integration,
background, masking, aperture corrections and failure rules are fixed by the owner-signed
the author's frozen compactness measurement protocol, because each of them moves the ratio and a
free parameter chosen after seeing a result is a tuned parameter. Each function below
quotes the section it implements. **Changing any of them is a protocol amendment requiring
a dated disclosure and a new signature, not a code change.**

REUSED, NOT REBUILT (owner directive 2026-08-27): the photometry is `photutils`, the sky
geometry is `astropy`. Both are cited in the write-up's Software section. Akins's own
aperture code (`hollisakins/aperx`, `hollisakins/aperate`) is the method reference, since
the point is to reproduce HIS measurement rather than a lookalike.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from photutils.aperture import (
    SkyCircularAnnulus,
    SkyCircularAperture,
    aperture_photometry,
)

# [protocol 3.4] DIAMETERS in arcsec — the convention DATA_CONTRACTS line 57 fixes and
# Greene+24 3.1 confirms for the Labbe family. The names are deliberately unambiguous:
# `d020` is a 0.20" DIAMETER, never a radius.
APERTURE_DIAMETERS_ARCSEC: dict[str, float] = {
    "d020": 0.20,
    "d040": 0.40,
    "d050": 0.50,
}

# [protocol 3.6] Recorded, never applied. RADII in arcsec.
BACKGROUND_ANNULUS_ARCSEC: tuple[float, float] = (0.5, 1.0)

# [protocol 3.5] Exact geometric overlap. At a 0.10" radius an aperture is only a few
# pixels across, so whole-pixel counting is a first-order error rather than a rounding
# detail, and "exact" removes the sampling parameter instead of setting it.
_INTEGRATION = "exact"


def _fully_inside(bbox, shape: tuple[int, int]) -> bool:
    ny, nx = shape
    return bbox.ixmin >= 0 and bbox.iymin >= 0 and bbox.ixmax <= nx and bbox.iymax <= ny


def _annulus_median(
    image: np.ndarray, coord: SkyCoord, wcs: WCS
) -> tuple[float, int]:
    """[protocol 3.6] The local background, RECORDED so its influence is measurable after
    the fact, and NOT subtracted, because subtracting it here would be a free parameter
    the signed protocol does not grant."""
    inner, outer = BACKGROUND_ANNULUS_ARCSEC
    annulus = SkyCircularAnnulus(
        coord, r_in=inner * u.arcsec, r_out=outer * u.arcsec
    ).to_pixel(wcs)
    # [review round 6, A3] This was method="center". Protocol 3.5 fixes "exact" and
    # expressly says NOT "center", and recording-rather-than-applying the annulus does
    # not exempt a produced output from the protocol: the code cannot grant itself an
    # exemption the signed text does not contain. For a MEDIAN the weights are
    # irrelevant and only membership changes, so this is a conformance fix that cannot
    # move any ratio.
    mask = annulus.to_mask(method=_INTEGRATION)
    weights = mask.to_image(image.shape)
    if weights is None:
        return float("nan"), 0
    pixels = image[weights > 0]
    pixels = pixels[np.isfinite(pixels)]
    # The pixel COUNT is reported alongside the median because the median alone cannot
    # reveal which pixels were members, and membership is exactly what the integration
    # method decides. Without it, a silent switch back to "center" is untestable.
    if not pixels.size:
        return float("nan"), 0
    return float(np.median(pixels)), int(pixels.size)


def measure_apertures(
    image: np.ndarray,
    wcs: WCS,
    ra,
    dec,
) -> pd.DataFrame:
    """Aperture fluxes at the frozen diameters, one row per position, IN THE ORDER GIVEN.

    [protocol 3.3] Centring is FORCED at the supplied sky position. No re-centroiding, no
    PSF-fit centre, no brightest-pixel snap. Re-centroiding would introduce a per-object
    decision that can be tuned and would decouple the measurement from the catalogue rows
    every other column comes from; forced photometry is the choice with no free parameter
    left in it.

    [protocol 3.7, 3.9] An aperture that is not fully on valid science pixels yields NO
    value. The object is recorded as unmeasurable, with a reason, rather than being given
    a number computed from partial coverage — an aperture silently truncated at an image
    edge understates the numerator and manufactures compactness.
    """
    # [review round 6, A2] There is deliberately NO diameters_arcsec override. A public
    # parameter that replaces the frozen apertures would let any caller change the
    # measurement without a protocol amendment, which is the whole thing the freeze
    # exists to prevent. Changing an aperture is a signature event, not an argument.
    diameters = APERTURE_DIAMETERS_ARCSEC
    coords = SkyCoord(ra=np.atleast_1d(ra) * u.deg, dec=np.atleast_1d(dec) * u.deg)

    rows: list[dict] = []
    for coord in coords:
        row: dict = {"ra": float(coord.ra.deg), "dec": float(coord.dec.deg)}
        reason: str | None = None
        fluxes: dict[str, float] = {}

        for name, diameter in diameters.items():
            aperture = SkyCircularAperture(
                coord, r=(diameter / 2.0) * u.arcsec
            ).to_pixel(wcs)

            if not _fully_inside(aperture.bbox, image.shape):
                reason = reason or f"aperture {name} leaves the mosaic footprint"
                fluxes[name] = float("nan")
                continue

            mask = aperture.to_mask(method=_INTEGRATION)
            weights = mask.to_image(image.shape)
            if weights is None:
                reason = reason or f"aperture {name} leaves the mosaic footprint"
                fluxes[name] = float("nan")
                continue
            if np.any(~np.isfinite(image[weights > 0])):
                reason = reason or f"aperture {name} covers a NaN or non-finite pixel"
                fluxes[name] = float("nan")
                continue

            # photutils does the integration; this module only decides WHICH aperture.
            table = aperture_photometry(image, aperture, method=_INTEGRATION)
            fluxes[name] = float(table["aperture_sum"][0])

        for name, value in fluxes.items():
            row[f"flux_{name}"] = value
        row["bkg_annulus_median"], row["bkg_annulus_npix"] = _annulus_median(
            image, coord, wcs
        )
        # [review round 6, test finding] No numerical test can separate exact overlap
        # from sufficiently fine subpixel quadrature -- they converge. Recording the
        # method that ran makes the choice auditable instead of merely asserted.
        row["integration_method"] = _INTEGRATION
        row["measurable"] = reason is None
        row["unmeasurable_reason"] = reason
        rows.append(row)

    return pd.DataFrame(rows)


def compactness_columns(fluxes: pd.DataFrame) -> pd.DataFrame:
    """The two published ratios, derived from the measured fluxes.

    THE DIRECTIONS ARE OPPOSITE, and that is the single easiest thing to get backwards:

      * `compactness_f444w` is Akins's C444 = f(0.2")/f(0.5"), SMALL over LARGE, so a
        COMPACT source has a LARGE value. This is the DATA_CONTRACTS line 57 column.
      * `labbe_compactness` is the Labbe/Kokorev ratio = f(0.4")/f(0.2"), LARGE over
        SMALL, so a COMPACT source has a SMALL value (their cut is `< 1.7`).

    [protocol 3.9] A non-positive denominator yields NaN rather than a ratio whose sign
    nobody can interpret. NaN is also what an unmeasurable object carries, and that is
    load-bearing: every comparison against a threshold is False in BOTH directions, so an
    absent measurement can never read as a satisfied cut and can never be recorded as
    having failed one.
    """

    # [review round 6, A5 — a real defect, not ceremony] This previously derived each
    # ratio from its own two fluxes alone. An object with a NaN in the ring covered by
    # the 0.5" aperture but not the 0.4" one, or positioned so only the 0.5" aperture
    # leaves the mosaic, was recorded measurable=False and STILL returned a number for
    # `labbe_compactness`. The signed protocol declares unmeasurability at the level of
    # the OBJECT (3.7: "the object is recorded as unmeasurable"), so an object the
    # producer has already refused cannot carry a ratio out of the same call.
    #
    # This is deliberately the strict reading. It discards Labbe ratios that were in
    # themselves well measured, which loses objects. That is the safe direction — it can
    # only ever remove a candidate, never manufacture one — and per-ratio validity would
    # be a NEW rule the owner did not sign. Relaxing it is an amendment, not a patch.
    measurable = fluxes["measurable"].to_numpy(dtype=bool)

    def ratio(numerator: str, denominator: str) -> np.ndarray:
        num = fluxes[f"flux_{numerator}"].to_numpy(dtype=float)
        den = fluxes[f"flux_{denominator}"].to_numpy(dtype=float)
        out = np.full(num.shape, np.nan, dtype=float)
        usable = measurable & np.isfinite(num) & np.isfinite(den) & (den > 0)
        out[usable] = num[usable] / den[usable]
        return out

    return pd.DataFrame(
        {
            "compactness_f444w": ratio("d020", "d050"),
            "labbe_compactness": ratio("d040", "d020"),
            "measurable": fluxes["measurable"].to_numpy(),
        }
    )
