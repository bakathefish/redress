
import numpy as np
import pandas as pd
import pytest

from redress import contracts, conventions, cuts
from redress.cuts import _shared as sh
from redress.cuts import (
    akins24,
    barro23,
    greene24,
    kocevski24,
    kokorev24,
    labbe23,
    perezgonzalez24,
)

#: test pivots (Å) — self-consistent within the tests; real pivots are computed
#: from provenance-locked filter curves at W1 (DATA_CONTRACTS §3).
PIVOTS = {
    "f606w": 5920.0, "f814w": 8060.0, "f115w": 11540.0, "f150w": 15010.0,
    "f200w": 19890.0, "f277w": 27620.0, "f356w": 35680.0, "f410m": 40820.0,
    "f444w": 44040.0,
}


def make_phot(rows, bands):
    """Contract-valid photometry table from per-row {mags, snr, flux, ...} specs."""
    n = len(rows)
    data = {
        "object_id": [f"obj{i}" for i in range(n)],
        "field": ["ceers"] * n,
        "ra_deg": [10.0 + 0.001 * i for i in range(n)],
        "dec_deg": [2.0] * n,
        "source_catalog": ["synthetic-v0"] * n,
        "compactness_f444w": [float(r.get("compactness", np.nan)) for r in rows],
        "flux_radius_f444w_arcsec": [float(r.get("flux_radius", np.nan)) for r in rows],
        "nearest_neighbor_arcsec": [np.nan] * n,
    }
    for b in bands:
        f_col, e_col, c_col = [], [], []
        for r in rows:
            mags = r.get("mags", {})
            flux = r.get("flux", {})
            snr = r.get("snr", 50.0)
            snr_b = snr.get(b, 50.0) if isinstance(snr, dict) else snr
            if b in flux:
                f, e = flux[b]
                f_col.append(float(f)), e_col.append(float(e)), c_col.append(True)
            elif b in mags:
                f = conventions.fnu_ujy_from_ab_mag(mags[b])
                f_col.append(f), e_col.append(f / snr_b), c_col.append(True)
            else:
                f_col.append(np.nan), e_col.append(np.nan), c_col.append(False)
        data[f"f_{b}_ujy"] = f_col
        data[f"e_{b}_ujy"] = e_col
        data[f"cov_{b}"] = c_col
    df = pd.DataFrame(data)
    contracts.validate_photometry_table(df)   # the cuts consume CONTRACT-VALID tables
    return df


def power_law_mags(beta, anchor_band, anchor_mag, bands):
    """AB mags of an f_lambda ∝ lambda^beta power law through an anchor point."""
    c0 = anchor_mag + 2.5 * (beta + 2.0) * np.log10(PIVOTS[anchor_band])
    return {b: c0 - 2.5 * (beta + 2.0) * np.log10(PIVOTS[b]) for b in bands}


# ------------------------------------------------------------------- registry

def test_registry_has_the_plans_seven_cuts():
    assert set(cuts.CUTS) == {
        "labbe23", "kokorev24", "kocevski24", "perezgonzalez24",
        "barro23", "greene24", "akins24",
    }
    assert all(callable(f) for f in cuts.CUTS.values())


# ------------------------------------------------------------------- Labbé 2023

def test_labbe23_thresholds_verbatim():
    # §2.2: "SNR(444) > 14 & m444 < 27.7"; red1/red2 displayed equations;
    # compact = f444(0.4″)/f444(0.2″) < 1.7 (diameters, per Greene §3.1).
    assert labbe23.SNR_F444W_MIN == 14.0 and labbe23.M_F444W_MAX == 27.7
    assert labbe23.RED1 == (
        ("f115w", "f150w", "lt", 0.8), ("f200w", "f277w", "gt", 0.7), ("f200w", "f356w", "gt", 1.0)
    )
    assert labbe23.RED2 == (
        ("f150w", "f200w", "lt", 0.8), ("f277w", "f356w", "gt", 0.7), ("f277w", "f444w", "gt", 1.0)
    )
    assert labbe23.COMPACT_RATIO_MAX == 1.7


def test_labbe23_selects_both_branches_and_rejects_neither():
    rows = [
        # red1-only: 115−150=0.5, 200−277=0.8, 200−356=1.1; red2 fails (150−200=0.9)
        {"mags": {"f115w": 27.9, "f150w": 27.4, "f200w": 26.5, "f277w": 25.7,
                  "f356w": 25.4, "f444w": 25.2}, "compactness": np.nan},
        # red2-only: 150−200=0.5, 277−356=0.8, 277−444=1.2; red1 fails (115−150=1.2)
        {"mags": {"f115w": 29.2, "f150w": 28.0, "f200w": 27.5, "f277w": 27.2,
                  "f356w": 26.4, "f444w": 26.0}},
        # flat SED: neither branch
        {"mags": {b: 26.0 for b in ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w")}},
    ]
    phot = make_phot(rows, ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    out = labbe23.select(phot, aper_ratio_f444w_04_02=[1.2, 1.2, 1.2])
    assert out["red1"].tolist() == [True, False, False]
    assert out["red2"].tolist() == [False, True, False]
    assert out["selected"].tolist() == [True, True, False]


def test_labbe23_boundaries_gate_and_main_sample():
    # exact-threshold rows FAIL (strict inequalities as printed)
    base = {"f115w": 27.9, "f150w": 27.4, "f200w": 26.5, "f277w": 25.7,
            "f356w": 25.4, "f444w": 25.2}
    rows = [
        {"mags": base},                                        # clean pass
        {"mags": base, "snr": {"f444w": 14.0}},                # SNR == 14 -> fail
        {"mags": {**base, "f444w": 27.7}},                     # m444 == 27.7 -> fail
        {"mags": base},                                        # ratio == 1.7 -> fail
    ]
    phot = make_phot(rows, ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    out = labbe23.select(phot, [1.2, 1.2, 1.2, 1.7], psf_dominated=[True, True, True, True])
    assert out["selected"].tolist() == [True, False, False, False]
    # Main = Parent ∧ PSF-dominance; flipping psf drops the row from main only
    out2 = labbe23.select(phot, [1.2, 1.2, 1.2, 1.7], psf_dominated=[False, True, True, True])
    assert out2["selected"].tolist() == [True, False, False, False]
    assert out2["main"].tolist() == [False, False, False, False]


def test_labbe23_missing_band_fails_closed():
    # red1-passing mags but F356W uncovered -> red1 cannot be demonstrated
    rows = [{"mags": {"f115w": 27.9, "f150w": 27.4, "f200w": 26.5, "f277w": 25.7,
                      "f444w": 25.2}}]
    phot = make_phot(rows, ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    out = labbe23.select(phot, [1.2])
    assert not out["red1"][0] and not out["selected"][0]


# ----------------------------------------------------------------- Kokorev 2024

def test_kokorev24_thresholds_verbatim_and_red2_differs_from_labbe():
    # §3.1: red1 identical to Labbé; red2 LOOSENED to 0.6 / 0.7 (the extract's
    # #1 watch-out); bd_removal = F115W−F200W > −0.5; same SNR/mag/compact gate.
    assert kokorev24.RED1 == labbe23.RED1
    assert kokorev24.RED2 == (
        ("f150w", "f200w", "lt", 0.8), ("f277w", "f356w", "gt", 0.6), ("f277w", "f444w", "gt", 0.7)
    )
    assert kokorev24.BD_COLOR_F115W_F200W_MIN == -0.5
    assert (kokorev24.SNR_F444W_MIN, kokorev24.M_F444W_MAX) == (14.0, 27.7)
    assert (kokorev24.DETECTION_SIGMA, kokorev24.UPPER_LIMIT_SIGMA) == (3.0, 2.0)

    # the discriminating row: 277−356 = 0.65 and 277−444 = 0.8 pass Kokorev's
    # red2 and FAIL Labbé's (0.65 < 0.7; 0.8 < 1.0); red1 fails in both.
    rows = [{"mags": {"f115w": 27.9, "f150w": 27.8, "f200w": 27.3, "f277w": 26.8,
                      "f356w": 26.15, "f444w": 26.0}}]
    phot = make_phot(rows, ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    k = kokorev24.select(phot, [1.2])
    lab = labbe23.select(phot, [1.2])
    assert k["selected"][0] and k["red2"][0] and not k["red1"][0]
    assert not lab["selected"][0] and not lab["red2"][0]


def test_kokorev24_upper_limit_semantics():
    # the quoted rule, pinned at the color level: ">3σ in at least one band per
    # color; 2σ upper limits, but only if the 'brighter' band ... is detected"
    rows = [
        # (i) both detected: measured color
        {"flux": {"f277w": (10.0, 0.2), "f444w": (60.0, 1.2)}},
        # (ii) blue undetected (0.1±1.0), red detected and brighter than the
        #      2σ limit (60 > 0.1+2·1.0 = 2.1): substitution valid
        {"flux": {"f277w": (0.1, 1.0), "f444w": (60.0, 1.2)}},
        # (iii) red detected but FAINTER than the blue band's 2σ limit
        #       (1.5 < 2.1): the "brighter band" clause refuses -> unusable
        {"flux": {"f277w": (0.1, 1.0), "f444w": (1.5, 0.3)}},
        # (iv) neither detected -> unusable
        {"flux": {"f277w": (0.1, 1.0), "f444w": (0.2, 1.0)}},
        # (v) undetected band uncovered -> no limit constructible -> unusable
        {"flux": {"f444w": (60.0, 1.2)}},
    ]
    phot = make_phot(rows, ("f277w", "f444w"))
    col, _used = kokorev24._color_with_upper_limits(phot, "f277w", "f444w")
    m = conventions.ab_mag_from_fnu_ujy
    assert col[0] == pytest.approx(m(10.0) - m(60.0))
    assert col[1] == pytest.approx(m(0.1 + 2.0 * 1.0) - m(60.0))
    assert np.isnan(col[2]) and np.isnan(col[3]) and np.isnan(col[4])


# ---------------------------------------------------------------- Kocevski 2024

def test_kocevski24_thresholds_and_band_table_verbatim():
    assert kocevski24.SNR_F444W_MIN == 12.0
    assert kocevski24.BETA_OPT_MIN == 0.0
    assert (kocevski24.BETA_UV_LO, kocevski24.BETA_UV_HI) == (-2.8, -0.37)
    assert kocevski24.RH_STELLAR_FACTOR == 1.5
    assert kocevski24.BETA_LINEBOOST_MIN == -1.0 and kocevski24.LINEBOOST_Z_MAX == 8.0
    assert kocevski24.Z_BINS == (
        (2.0, 3.25, ("f606w", "f814w", "f115w"), ("f150w", "f200w", "f277w")),
        (3.25, 4.75, ("f814w", "f115w", "f150w"), ("f200w", "f277w", "f356w")),
        (4.75, 8.0, ("f115w", "f150w", "f200w"), ("f277w", "f356w", "f444w")),
        (8.0, np.inf, ("f150w", "f200w", "f277w"), ("f356w", "f444w")),
    )


_KOC_BANDS = ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w")


def _koc_row(beta_uv=-1.0, beta_opt=0.5, flux_radius=0.08, snr=50.0):
    mags = power_law_mags(beta_uv, "f150w", 26.0, ("f115w", "f150w", "f200w"))
    mags.update(power_law_mags(beta_opt, "f444w", 25.0, ("f277w", "f356w", "f444w")))
    return {"mags": mags, "flux_radius": flux_radius, "snr": snr}


def test_kocevski24_recovers_slopes_and_selects_at_z5():
    rows = [
        _koc_row(),                                   # all criteria pass
        _koc_row(beta_opt=-0.3),                      # (ii) fails: optical not red
        _koc_row(beta_uv=-3.0),                       # (iii) fails: brown-dwarf floor
        _koc_row(beta_uv=-0.2),                       # (iii) fails: not blue enough
        _koc_row(flux_radius=0.13),                   # (iv) fails: 0.13 !< 1.5·0.08
        _koc_row(snr=5.0),                            # (i) fails
    ]
    phot = make_phot(rows, _KOC_BANDS)
    out = kocevski24.select(
        phot, z_phot=[5.0] * 6, r_h_stars_arcsec=[0.08] * 6, pivots_angstrom=PIVOTS
    )
    # the fits recover the KNOWN slopes (mags lie exactly on the power laws)
    assert out["beta_uv"][0] == pytest.approx(-1.0, abs=1e-6)
    assert out["beta_opt"][0] == pytest.approx(0.5, abs=1e-6)
    assert out["selected"].tolist() == [True, False, False, False, False, False]
    assert not out["beta_opt_red"][1]
    assert not out["beta_uv_window"][2] and not out["beta_uv_window"][3]
    assert not out["size"][4] and not out["detection"][5]


def test_kocevski24_out_of_table_redshift_fails_closed():
    phot = make_phot([_koc_row()], _KOC_BANDS)
    out = kocevski24.select(phot, [1.5], [0.08], PIVOTS)
    assert np.isnan(out["beta_uv"][0]) and not out["selected"][0]


def test_kocevski24_lineboost_veto_and_z8_skip():
    # z=5: 277−356 color 0.1 implies beta ≈ −1.64 < −1 -> (v) refuses; raising
    # m356 to make the color 0.5 (beta ≈ −0.20) restores selection.
    uv = power_law_mags(-1.0, "f150w", 26.0, ("f115w", "f150w", "f200w"))
    fail_row = {"mags": {**uv, "f277w": 26.0, "f356w": 25.9, "f444w": 24.6}, "flux_radius": 0.08}
    pass_row = {"mags": {**uv, "f277w": 26.0, "f356w": 25.5, "f444w": 24.6}, "flux_radius": 0.08}
    phot = make_phot([fail_row, pass_row], _KOC_BANDS)
    out = kocevski24.select(phot, [5.0, 5.0], [0.08, 0.08], PIVOTS)
    assert out["beta_opt_red"].tolist() == [True, True]      # the 3-band slope is red
    assert out["lineboost_356"].tolist() == [False, True]    # the 2-band guard decides
    assert out["selected"].tolist() == [False, True]

    # z=8.5: the SAME too-blue 277−356 color is exempt — "(v) only at z < 8";
    # UV = 150/200/277 and opt = 356/444 per the z>8 table row.
    uv_hi = power_law_mags(-1.0, "f200w", 26.0, ("f150w", "f200w", "f277w"))
    opt_hi = power_law_mags(0.5, "f444w", 25.2, ("f356w", "f444w"))
    row = {"mags": {**uv_hi, **opt_hi}, "flux_radius": 0.08}
    phot_hi = make_phot([row], _KOC_BANDS)
    out_hi = kocevski24.select(phot_hi, [8.5], [0.08], PIVOTS)
    assert out_hi["beta_uv"][0] == pytest.approx(-1.0, abs=1e-6)
    color_356 = (uv_hi["f277w"]) - (opt_hi["f356w"])
    assert conventions.beta_from_color(color_356, PIVOTS["f277w"], PIVOTS["f356w"]) < -1.0
    assert out_hi["lineboost_356"][0] and out_hi["selected"][0]


def test_kocevski24_f410m_availability_escape():
    base = _koc_row()
    bands = _KOC_BANDS + ("f410m",)
    # (a) F410M column exists but this row is uncovered there -> (vi) vacuous
    phot_a = make_phot([base], bands)
    out_a = kocevski24.select(phot_a, [5.0], [0.08], PIVOTS)
    assert out_a["lineboost_410"][0] and out_a["selected"][0]
    # (b) covered, with 277−410M = 0.2 -> beta ≈ −1.53 < −1 -> (vi) refuses
    bad = dict(base)
    bad["mags"] = {**base["mags"], "f410m": base["mags"]["f277w"] - 0.2}
    phot_b = make_phot([bad], bands)
    out_b = kocevski24.select(phot_b, [5.0], [0.08], PIVOTS)
    assert not out_b["lineboost_410"][0] and not out_b["selected"][0]


def test_kocevski24_faithful_mode_requires_the_full_triplet():
    # r5f: a covered-redshift row missing one required triplet band FAILS the
    # faithful baseline (the old 2-of-3 fallback would have rescued it and
    # inflated completeness); the audit arrays still record what was there.
    uv = power_law_mags(-1.0, "f814w", 26.5, ("f814w", "f115w"))
    opt = power_law_mags(0.5, "f277w", 25.5, ("f150w", "f200w", "f277w"))
    extra = power_law_mags(0.5, "f277w", 25.5, ("f356w",))
    row = {"mags": {**uv, **opt, **extra, "f444w": 25.0}, "flux_radius": 0.08}
    phot = make_phot([row], ("f606w", "f814w", "f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    out = kocevski24.select(phot, [2.5], [0.08], PIVOTS)
    assert out["n_bands_uv"][0] == 2 and out["n_bands_opt"][0] == 3
    assert np.isnan(out["beta_uv"][0]) and not out["selected"][0]


def test_kocevski24_exact_bin_boundaries_fail_closed():
    # r5f: the paper's intervals are strict open — z exactly 2, 3.25, 4.75, 8
    # are UNASSIGNED and fail closed (all four values binary-exact floats).
    rows = [_koc_row() for _ in range(4)]
    phot = make_phot(rows, _KOC_BANDS)
    out = kocevski24.select(phot, [2.0, 3.25, 4.75, 8.0], [0.08] * 4, PIVOTS)
    assert np.all(np.isnan(out["beta_uv"]))
    assert not out["selected"].any()


def test_kocevski24_negative_radius_and_exact_size_equality_fail():
    # r5f: physical-domain guard (r_h = -0.01 must fail) and exact size
    # equality via binary-exact floats (r_stars 0.25 -> 1.5*0.25 = 0.375
    # exactly; r_h = 0.375 is NOT < 0.375).
    rows = [_koc_row(flux_radius=-0.01), _koc_row(flux_radius=0.375), _koc_row(flux_radius=0.374)]
    phot = make_phot(rows, _KOC_BANDS)
    out = kocevski24.select(phot, [5.0] * 3, [0.25] * 3, PIVOTS)
    assert out["size"].tolist() == [False, False, True]


def test_kocevski24_covered_f410m_nondetection_fails_closed():
    # r5f deciding test: cov_f410m=True with a finite NEGATIVE flux is valid
    # photometry — condition (vi) APPLIES and cannot be measured -> fail
    # closed, never the vacuous escape.
    base = _koc_row()
    base["flux"] = {"f410m": (-1.0, 1.0)}
    phot = make_phot([base], _KOC_BANDS + ("f410m",))
    out = kocevski24.select(phot, [5.0], [0.08], PIVOTS)
    assert not out["lineboost_410"][0] and not out["selected"][0]


def test_kocevski24_no_f410m_column_needs_no_f410m_pivot():
    # r5f: the pivot API demands only what the table can use.
    phot = make_phot([_koc_row()], _KOC_BANDS)
    pivots = {k: v for k, v in PIVOTS.items() if k != "f410m"}
    out = kocevski24.select(phot, [5.0], [0.08], pivots)
    assert out["selected"][0]


def test_kocevski24_snr_equality_fails():
    # r5f: SNR == 12 exactly fails the strict gate (pure division, no
    # round-trip).
    row = _koc_row(snr={"f444w": 12.0})
    phot = make_phot([row], _KOC_BANDS)
    out = kocevski24.select(phot, [5.0], [0.08], PIVOTS)
    assert not out["detection"][0]


def test_kocevski24_beta_boundaries_exact_via_monkeypatch(monkeypatch):
    # r5f: beta boundary strictness with EXACT values — bypass regression
    # round-trips by monkeypatching the slope fitter. beta_opt = 0 fails (ii);
    # beta_uv = -0.37 and -2.8 fail (iii); interior values pass.
    phot = make_phot([_koc_row()], _KOC_BANDS)

    cases = [
        (0.0, -1.0, "beta_opt_red", False),
        (0.5, -0.37, "beta_uv_window", False),
        (0.5, -2.8, "beta_uv_window", False),
        (0.5, -1.0, "selected", True),
    ]
    for beta_opt_val, beta_uv_val, key, expect in cases:
        def fake_fit(phot_, z_, bands_by_bin, pivots, _o=beta_opt_val, _u=beta_uv_val):
            n = len(phot_)
            is_uv = any(b is not None and "f115w" in b for b in bands_by_bin)
            val = _u if is_uv else _o
            return np.full(n, val), np.full(n, 0.01), np.full(n, 3, dtype=int)
        monkeypatch.setattr(kocevski24, "_fit_beta_rows", fake_fit)
        out = kocevski24.select(phot, [5.0], [0.08], PIVOTS)
        assert bool(out[key][0]) == expect, (beta_opt_val, beta_uv_val, key)


def test_kocevski24_missing_pivot_raises():
    # r5f re-scope: the f410m pivot is demanded ONLY when the table carries
    # the band (see test_kocevski24_no_f410m_column_needs_no_f410m_pivot for
    # the counterpart); with the column present, its absence raises loudly.
    base = _koc_row()
    base["mags"]["f410m"] = 25.0            # COVERED row (r5f7: an uncovered
    # column demands no pivot, so the demand-test needs real coverage)
    phot = make_phot([base], _KOC_BANDS + ("f410m",))
    bad_pivots = {k: v for k, v in PIVOTS.items() if k != "f410m"}
    with pytest.raises(ValueError, match="f410m"):
        kocevski24.select(phot, [5.0], [0.08], bad_pivots)


# --------------------------------------------------------- Pérez-González 2024

def test_perezgonzalez24_thresholds_verbatim():
    assert perezgonzalez24.COLOR_F277W_F444W_MIN == 1.0
    assert perezgonzalez24.COLOR_F150W_F200W_MAX == 0.5
    assert perezgonzalez24.M_F444W_MAX == 28.0          # ≤, §2.1 body text
    assert perezgonzalez24.BD_COLOR_F115W_F150W_MIN == -0.5


def test_perezgonzalez24_rows():
    rows = [
        # pass: 277−444 = 1.2, 150−200 = 0.3, m444 = 27.9 (deeper than Labbé's
        # 27.7 — the paper's "~0.5 mag deeper" remark), 115−150 = 0.0
        {"mags": {"f115w": 28.6, "f150w": 28.6, "f200w": 28.3, "f277w": 29.1, "f444w": 27.9}},
        # boundary: 277−444 exactly 1.0 -> strict > fails
        {"mags": {"f115w": 28.0, "f150w": 28.0, "f200w": 27.7, "f277w": 28.5, "f444w": 27.5}},
        # brown dwarf: 115−150 = −0.6 -> removed
        {"mags": {"f115w": 27.4, "f150w": 28.0, "f200w": 27.7, "f277w": 28.7, "f444w": 27.5}},
    ]
    phot = make_phot(rows, ("f115w", "f150w", "f200w", "f277w", "f444w"))
    out = perezgonzalez24.select(phot)
    assert out["selected"].tolist() == [True, False, False]
    assert not out["red_color"][1] and not out["bd_retention"][2]
    # NOTE the ≤-vs-< distinction at m444 = 28 exactly is pinned structurally
    # (the module uses the le comparator; §2.1 body text vs Fig-1 caption
    # discrepancy documented in the module docstring) — a float round-trip
    # boundary row would test IEEE rounding, not the paper.


# --------------------------------------------------------------- Barro 2023

def test_barro23_thresholds_and_rows():
    assert barro23.COLOR_F277W_F444W_MIN == 1.5
    assert barro23.M_F444W_MAX == 28.0
    rows = [
        {"mags": {"f277w": 29.1, "f444w": 27.5}},   # 1.6 > 1.5 -> pass
        {"mags": {"f277w": 29.0, "f444w": 27.5}},   # exactly 1.5 -> strict > fails
        {"mags": {"f277w": 30.0, "f444w": 28.4}},   # 1.6 but m444 too faint
    ]
    phot = make_phot(rows, ("f277w", "f444w"))
    out = barro23.select(phot)
    assert out["selected"].tolist() == [True, False, False]
    assert not out["red_color"][1] and not out["mag_gate"][2]


# --------------------------------------------------------------- Greene 2024

def test_greene24_thresholds_verbatim():
    assert greene24.SNR_F444W_MIN == 14.0 and greene24.M_F444W_MAX == 27.7
    assert (greene24.VSHAPE_BLUE_LO, greene24.VSHAPE_BLUE_HI) == (-0.5, 1.0)
    assert greene24.COLOR_F277W_F444W_MIN == 1.0


def test_greene24_vshape_rows_and_no_compactness():
    rows = [
        # pass — and give it an ABSURDLY extended concentration to prove §5.2
        # imposes no compactness (the quick-reference's "<1.7" is §3.1-only)
        {"mags": {"f115w": 26.8, "f200w": 26.5, "f277w": 27.3, "f444w": 26.0},
         "compactness": 5.0},
        # blue window lower edge: 115−200 exactly −0.5 -> fails
        {"mags": {"f115w": 26.0, "f200w": 26.5, "f277w": 27.3, "f444w": 26.0}},
        # blue window upper edge: 115−200 exactly 1.0 -> fails
        {"mags": {"f115w": 27.5, "f200w": 26.5, "f277w": 27.3, "f444w": 26.0}},
        # red arm too weak: 277−444 = 0.9
        {"mags": {"f115w": 26.8, "f200w": 26.5, "f277w": 26.9, "f444w": 26.0}},
    ]
    phot = make_phot(rows, ("f115w", "f200w", "f277w", "f444w"))
    out = greene24.select(phot)
    assert out["selected"].tolist() == [True, False, False, False]
    assert not out["vshape_blue"][1] and not out["vshape_blue"][2]
    assert not out["vshape_red"][3]


# ---------------------------------------------------------------- Akins 2024

def test_akins24_thresholds_verbatim():
    assert akins24.SNR_F444W_MIN == 12.0
    assert akins24.COLOR_F277W_F444W_MIN == 1.5
    assert (akins24.C444_SELECT_MIN, akins24.C444_EXCLUDE_ABOVE) == (0.5, 0.7)


def test_akins24_c444_boundary_semantics_from_the_quoted_sentences():
    # "select objects with C444 > 0.5" + "exclude objects with C444 > 0.7":
    # 0.7 exactly is KEPT (not >0.7), 0.5 exactly FAILS, 0.71 excluded, and a
    # NaN C444 must fail despite the negated exclusion term (the _shared
    # anti-NaN-negation rule).
    good = {"f277w": 29.2, "f444w": 27.0}                # color 2.2, snr 50
    rows = [
        {"mags": good, "compactness": 0.6},
        {"mags": good, "compactness": 0.7},
        {"mags": good, "compactness": 0.5},
        {"mags": good, "compactness": 0.71},
        {"mags": good},                                   # NaN C444
        {"mags": {"f277w": 28.0, "f444w": 27.0}, "compactness": 0.6},  # color 1.0 -> fail
    ]
    phot = make_phot(rows, ("f277w", "f444w"))
    out = akins24.select(phot)
    assert out["selected"].tolist() == [True, True, False, False, False, False]
    assert "bd_retained" not in out and out["sed_bd_applied"] is False


def test_akins24_sed_brown_dwarf_mask():
    good = {"f277w": 29.2, "f444w": 27.0}
    phot = make_phot([{"mags": good, "compactness": 0.6}] * 2, ("f277w", "f444w"))
    out = akins24.select(phot, is_brown_dwarf_sed=[True, False])
    assert out["selected"].tolist() == [False, True]
    assert out["bd_retained"].tolist() == [False, True]


# REMOVED FOR THE PUBLIC TREE: test_akins24_interval_single_form_across_docs
# asserted that two statements of the Akins C444 interval inside the author's
# document of transcribed paper quotes agreed with each other. That document is
# not part of this repository, so the test has nothing to read. The half of it
# that pins behaviour -- C444 == 0.7 is KEPT by the implementation -- is asserted
# in test_akins24_c444_boundary_is_inclusive below.


def test_akins24_c444_boundary_is_inclusive():
    """The upper end of the Akins C444 window is inclusive: 0.7 is KEPT."""
    good = {"f277w": 29.2, "f444w": 27.0}
    phot = make_phot([{"mags": good, "compactness": 0.7}], ("f277w", "f444w"))
    assert akins24.select(phot)["selected"].tolist() == [True]


def test_simple_cuts_fail_closed_when_a_required_band_is_uncovered():
    # a would-pass row, with one required band uncovered per cut
    barro_phot = make_phot([{"mags": {"f444w": 27.5}}], ("f277w", "f444w"))
    assert not barro23.select(barro_phot)["selected"][0]

    pg_phot = make_phot(
        [{"mags": {"f115w": 28.6, "f200w": 28.3, "f277w": 29.1, "f444w": 27.9}}],
        ("f115w", "f150w", "f200w", "f277w", "f444w"),
    )
    assert not perezgonzalez24.select(pg_phot)["selected"][0]

    greene_phot = make_phot(
        [{"mags": {"f200w": 26.5, "f277w": 27.3, "f444w": 26.0}}],
        ("f115w", "f200w", "f277w", "f444w"),
    )
    assert not greene24.select(greene_phot)["selected"][0]


def test_missing_columns_raise_loudly():
    phot = make_phot([{"mags": {"f444w": 26.0}}], ("f444w",))
    with pytest.raises(ValueError, match="lacks required columns"):
        barro23.select(phot)
    with pytest.raises(ValueError, match="aligned to the table"):
        labbe_phot = make_phot(
            [{"mags": {b: 26.0 for b in ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w")}}],
            ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"),
        )
        labbe23.select(labbe_phot, aper_ratio_f444w_04_02=[1.2, 1.2])


# ------------------------------------------------- r5e counters (kokorev24)

def test_kokorev24_compactness_rejects_pathological_ratios():
    # r5e deciding test: -1, 0, NaN, exactly 1.7 all FAIL; only 1.69 passes.
    good = {"f115w": 27.9, "f150w": 27.8, "f200w": 27.3, "f277w": 26.8,
            "f356w": 26.15, "f444w": 26.0}
    phot = make_phot([{"mags": good}] * 5, ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    out = kokorev24.select(phot, [-1.0, 0.0, np.nan, 1.7, 1.69])
    assert out["compact"].tolist() == [False, False, False, False, True]
    assert out["selected"].tolist() == [False, False, False, False, True]
    lab = labbe23.select(phot, [-1.0, 0.0, np.nan, 1.7, 1.69])   # shared fix
    assert lab["compact"].tolist() == [False, False, False, False, True]


def test_shared_row_array_refuses_misaligned_series():
    # r5e deciding test: a shuffled-index Series must refuse, never silently
    # attach morphology to the wrong objects.
    good = {"f115w": 27.9, "f150w": 27.8, "f200w": 27.3, "f277w": 26.8,
            "f356w": 26.15, "f444w": 26.0}
    phot = make_phot([{"mags": good}] * 2, ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    reversed_series = pd.Series([1.2, 5.0], index=[1, 0])
    with pytest.raises(ValueError, match="non-trivial index"):
        kokorev24.select(phot, reversed_series)
    ok = kokorev24.select(phot, pd.Series([1.2, 1.2]))           # trivial index fine
    assert ok["compact"].tolist() == [True, True]


def test_kokorev24_censoring_both_orientations_and_exact_3sigma():
    # r5e deciding test: the only_b branch (BLUE detected, red undetected) and
    # exact-3sigma equality (strict >), pinned at the color level.
    m = conventions.ab_mag_from_fnu_ujy
    rows = [
        # blue detected (60 uJy), red undetected (0.1 +/- 1.0): red limit
        # 2.1 uJy; blue 60 > 2.1 -> substitution valid, color = m(60)-m(2.1)
        {"flux": {"f150w": (60.0, 1.2), "f200w": (0.1, 1.0)}},
        # exact 3 sigma is NOT detected (strict >): both bands at snr == 3.0
        {"flux": {"f150w": (3.0, 1.0), "f200w": (3.0, 1.0)}},
        # near-3sigma positive nondetection pins the max(f,0)+2e formula:
        # f = 2.9, e = 1.0 -> limit 4.9 (NOT 2.0): distinguishes conventions
        {"flux": {"f150w": (60.0, 1.2), "f200w": (2.9, 1.0)}},
    ]
    phot = make_phot(rows, ("f150w", "f200w"))
    col, _used = kokorev24._color_with_upper_limits(phot, "f150w", "f200w")
    assert col[0] == pytest.approx(m(60.0) - m(0.1 + 2.0 * 1.0))
    assert np.isnan(col[1])
    assert col[2] == pytest.approx(m(60.0) - m(2.9 + 2.0 * 1.0))


def test_kokorev24_boundary_equalities_fail_strictly():
    # just-below-threshold rows FAIL (strictness direction); exact float
    # equality is not asserted through mag round-trips (IEEE noise, not
    # science) — strictness at the boundary is pinned by the gt/lt
    # comparator choice asserted in the threshold-constants test.
    base = {"f115w": 27.9, "f150w": 27.8, "f200w": 27.3, "f277w": 26.8,
            "f356w": 26.15, "f444w": 26.0}
    below_red2 = dict(base, f356w=26.201)     # 277-356 = 0.599 < 0.6 -> fails
    below_bd = dict(base, f115w=26.799)       # 115-200 = -0.501 < -0.5 -> BD fails
    phot = make_phot(
        [{"mags": base}, {"mags": below_red2}, {"mags": below_bd}],
        ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"),
    )
    out = kokorev24.select(phot, [1.2, 1.2, 1.2])
    assert out["selected"].tolist() == [True, False, False]
    assert not out["red2"][1] and not out["bd_removal"][2]


def test_kokorev24_end_to_end_selection_through_an_upper_limit():
    # end-to-end THROUGH a substitution: F150W undetected, F200W detected —
    # red2's lt-color uses the brightest-possible F150W (the 2-sigma limit),
    # its smallest color, and the row still selects. Also pins the physics
    # the module taught this test's author: a gt-criterion (red color) can
    # NEVER pass via a limit on its red band, because the brighter-band rule
    # forces the substituted limit fainter than the detected blue band.
    m = conventions.ab_mag_from_fnu_ujy
    f150_und = (0.001, 0.02)                   # limit = 0.041 uJy
    row = {
        "mags": {"f115w": 27.35, "f200w": 27.3, "f277w": 26.8,
                 "f356w": 26.15, "f444w": 26.0},
        "flux": {"f150w": f150_und},
    }
    phot = make_phot([row], ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    out = kokorev24.select(phot, [1.2])
    lim = max(f150_und[0], 0.0) + 2.0 * f150_und[1]
    f200 = conventions.fnu_ujy_from_ab_mag(27.3)
    assert f200 > lim                          # substitution valid by the rule
    color_150_200 = m(lim) - 27.3
    assert color_150_200 < 0.8                 # limit version passes the lt cut
    assert out["red2"][0] and out["selected"][0]


def test_kokorev24_detection_gate_wiring():
    # r5e2 deciding test: the GATE WIRING, not just the constants. SNR is a
    # pure division (no round-trip), so equality is exactly testable: 14.0
    # must FAIL (strict >). Magnitudes cross mag<->flux round-trips, so the
    # wiring direction is pinned just across the boundary (27.69 passes,
    # 27.71 fails); exact-27.7 strictness is pinned by the lt comparator +
    # constants test, not by IEEE luck.
    good = {"f115w": 27.0, "f150w": 26.9, "f200w": 26.4, "f277w": 25.9,
            "f356w": 25.25, "f444w": 25.1}
    rows = [
        {"mags": good, "snr": {"f444w": 14.0}},                  # equality fails
        {"mags": good, "snr": {"f444w": 14.5}},                  # just above passes
        {"mags": dict(good, f444w=27.71), "snr": 50.0},          # too faint fails
        {"mags": dict(good, f444w=27.69), "snr": 50.0},          # just bright enough
    ]
    phot = make_phot(rows, ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    out = kokorev24.select(phot, [1.2] * 4)
    assert out["detection"].tolist() == [False, True, False, True]


def test_kokorev24_negative_flux_upper_limit_clips_at_zero():
    # r5e2 deciding test: undetected flux -1 +/- 1 -> limit is max(f,0)+2e
    # = 2.0, NOT f+2e = 1.0. Pins the retained clipping policy (itself
    # provisional pending the M0 faithfulness gate).
    m = conventions.ab_mag_from_fnu_ujy
    rows = [{"flux": {"f150w": (60.0, 1.2), "f200w": (-1.0, 1.0)}}]
    phot = make_phot(rows, ("f150w", "f200w"))
    col, _used = kokorev24._color_with_upper_limits(phot, "f150w", "f200w")
    assert col[0] == pytest.approx(m(60.0) - m(2.0))


def test_comparator_strictness_is_exact_at_thresholds():
    # r5e3 mutation-killer: every criterion composes from these comparators
    # (the _shared design rule), so exact-equality strictness is pinned HERE
    # with pure floats — no mag round-trips involved.
    from redress.cuts import _shared as shd
    assert not shd.gt(np.array([0.6]), 0.6)[0]
    assert not shd.lt(np.array([0.8]), 0.8)[0]
    assert shd.le(np.array([28.0]), 28.0)[0]
    assert shd.ge(np.array([-0.5]), -0.5)[0]
    assert not shd.between(np.array([1.7]), 0.0, 1.7)[0]
    assert not shd.between(np.array([0.0]), 0.0, 1.7)[0]


def test_kokorev24_auditability_fields():
    # r5e3: invalid-vs-extended is distinguishable, substitutions are flagged
    # per row, and the censoring policy ships inside the result.
    good = {"f115w": 27.9, "f150w": 27.8, "f200w": 27.3, "f277w": 26.8,
            "f356w": 26.15, "f444w": 26.0}
    rows = [
        {"mags": good},                                        # measured, compact
        {"mags": good},                                        # EXTENDED (valid 5.0)
        {"mags": good},                                        # INVALID (-1.0)
        {"mags": {k: v for k, v in good.items() if k != "f150w"},
         "flux": {"f150w": (0.001, 0.02)}},                    # via substitution
    ]
    phot = make_phot(rows, ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    out = kokorev24.select(phot, [1.2, 5.0, -1.0, 1.2])
    assert out["compact"].tolist() == [True, False, False, True]
    assert out["compact_valid"].tolist() == [True, True, False, True]
    # r5e4 per-branch provenance: ONLY red2 leaned on a substitution (its
    # 150-200 term); red1's 115-150 attempt is INVALID there (the detected
    # F115W is fainter than the F150W limit) so no red1 flag — the exact
    # irrelevant-branch honesty the aggregate flag could not express.
    assert out["used_upper_limit_red2"].tolist() == [False, False, False, True]
    assert out["used_upper_limit_red1"].tolist() == [False, False, False, False]
    assert out["used_upper_limit_bd"].tolist() == [False, False, False, False]
    assert "provisional" in out["censoring_policy"]


def test_kokorev24_selector_wiring_is_strict_with_exact_colors(monkeypatch):
    # r5e4 mutation-killer at the SELECTOR level: bypass mag round-trips by
    # monkeypatching the color engine to exact float values — a gt-term at
    # exactly its threshold and an lt-term at exactly its threshold must both
    # FAIL through the real _branch wiring (an lt->le mutation dies here).
    good = {"f115w": 27.9, "f150w": 27.8, "f200w": 27.3, "f277w": 26.8,
            "f356w": 26.15, "f444w": 26.0}
    phot = make_phot([{"mags": good}] * 1, ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    exact = {
        ("f115w", "f150w"): 0.1, ("f200w", "f277w"): 0.7,   # gt-term AT 0.7 -> red1 fails
        ("f200w", "f356w"): 1.5,
        ("f150w", "f200w"): 0.8,                            # lt-term AT 0.8 -> red2 fails
        ("f277w", "f356w"): 0.65, ("f277w", "f444w"): 0.8,
        ("f115w", "f200w"): 0.6,
    }
    def fake(phot_, blue, red):
        n = len(phot_)
        return np.full(n, exact[(blue, red)]), np.zeros(n, dtype=bool)
    monkeypatch.setattr(kokorev24, "_color_with_upper_limits", fake)
    out = kokorev24.select(phot, [1.2])
    assert not out["red1"][0] and not out["red2"][0] and not out["selected"][0]
    # r5e5: BD call-site strictness with an exact color — AT -0.5 must fail
    exact[("f115w", "f200w")] = -0.5
    out2 = kokorev24.select(phot, [1.2])
    assert not out2["bd_removal"][0]


def test_kokorev24_policy_string_is_bound_to_the_implementation():
    # r5e5: the policy metadata cannot drift from the code silently — the
    # exact string is pinned, and its formula clause matches the behavior the
    # negative-flux test proves (max(f,0)+2e).
    assert kokorev24.CENSORING_POLICY == (
        "kokorev24-ul-v1-provisional: limit=max(f,0)+2e; brighter-band-detected rule"
    )
    assert f"{kokorev24.UPPER_LIMIT_SIGMA:.0f}e" in kokorev24.CENSORING_POLICY.replace("2e", f"{kokorev24.UPPER_LIMIT_SIGMA:.0f}e")


def test_kokorev24_exact_magnitude_gate_via_monkeypatch(monkeypatch):
    # r5e5: the m444 call site at EXACTLY 27.7 — bypass flux round-trips by
    # monkeypatching the mag engine; strict < must fail at equality.
    good = {"f115w": 27.9, "f150w": 27.8, "f200w": 27.3, "f277w": 26.8,
            "f356w": 26.15, "f444w": 26.0}
    phot = make_phot([{"mags": good}], ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    real_mags = sh.mags_ab

    def fake_mags(p, band):
        out = real_mags(p, band)
        if band == "f444w":
            out = np.full_like(out, 27.7)
        return out
    monkeypatch.setattr(kokorev24.sh, "mags_ab", fake_mags)
    out = kokorev24.select(phot, [1.2])
    assert not out["detection"][0]


def test_kocevski24_inconsistency_variants_diverge_and_are_named():
    # r5f2 deciding test: the published-inconsistency variants are executable
    # BY NAME and diverge on a row whose two-band beta sits in (-1, 0): the
    # formal baseline (-1) selects it, the printed-equivalents variant (0)
    # rejects it. Plus the Eq-2 conversion oracle, hand-derived in-test:
    # the printed 0.53 corresponds to beta ~ -0.1 (NOT -1), and the beta=-1
    # color is ~ 0.28 — the published sentence is internally inconsistent.
    uv = power_law_mags(-1.0, "f150w", 26.0, ("f115w", "f150w", "f200w"))
    row = {"mags": {**uv, "f277w": 26.0, "f356w": 25.5, "f444w": 24.6}, "flux_radius": 0.08}
    phot = make_phot([row], _KOC_BANDS)
    formal = kocevski24.select(phot, [5.0], [0.08], PIVOTS)
    printed = kocevski24.select(phot, [5.0], [0.08], PIVOTS, lineboost_beta_min=0.0)
    b356 = formal["beta_lineboost_356"][0]
    assert -1.0 < b356 < 0.0
    assert formal["selected"][0] and not printed["selected"][0]
    assert conventions.beta_from_color(0.53, PIVOTS["f277w"], PIVOTS["f356w"]) > -0.2
    assert conventions.beta_from_color(0.278, PIVOTS["f277w"], PIVOTS["f356w"]) == pytest.approx(-1.0, abs=0.01)


def test_kocevski24_malformed_schema_triple_raises():
    # r5f2 deciding test: a band column present without its full f/e/cov
    # triple is a malformed schema, refused loudly (a fully absent band stays
    # a legitimate fail-closed coverage gap).
    phot = make_phot([_koc_row()], _KOC_BANDS).drop(columns=["cov_f150w"])
    with pytest.raises(ValueError, match="incomplete f/e/cov"):
        kocevski24.select(phot, [5.0], [0.08], PIVOTS)


def test_kocevski24_lineboost_audit_outputs_and_exact_minus_one(monkeypatch):
    # r5f2: the two-band betas and applicability flags ship in the result;
    # and beta exactly at the threshold fails the strict > (exact via a
    # monkeypatched conversion, no color round-trip).
    base = _koc_row()
    phot = make_phot([base], _KOC_BANDS)
    out = kocevski24.select(phot, [5.0], [0.08], PIVOTS)
    assert out["lineboost_356_applies"][0] and not out["lineboost_410_applies"][0]
    assert np.isfinite(out["beta_lineboost_356"][0]) and np.isnan(out["beta_lineboost_410"][0])

    monkeypatch.setattr(kocevski24.conventions, "beta_from_color", lambda c, l1, l2: -1.0)
    out2 = kocevski24.select(phot, [5.0], [0.08], PIVOTS)
    assert not out2["lineboost_356"][0] and not out2["selected"][0]


# ----------------------------------------- skeptic review round 1 (kocevski24)

def test_kocevski24_missing_unconditional_bands_refuse_cleanly():
    # skeptic CONFIRMED-DEFECT: a contract-valid table without f277w/f356w hit
    # a raw KeyError; now the named refusal covers every unconditionally-read
    # band. The missing-morphology column refusal is pinned alongside.
    phot = make_phot([{"mags": {"f444w": 25.0}}], ("f444w",))
    with pytest.raises(ValueError, match="lacks required columns"):
        kocevski24.select(phot, [5.0], [0.08], PIVOTS)
    phot2 = make_phot([_koc_row()], _KOC_BANDS).drop(columns=["flux_radius_f444w_arcsec"])
    with pytest.raises(ValueError, match="flux_radius"):
        kocevski24.select(phot2, [5.0], [0.08], PIVOTS)


def test_kocevski24_weighted_fit_is_pinned_by_noncollinear_row():
    # skeptic SUSPICION closed: every earlier fixture was COLLINEAR, so any
    # weighting (or none) fit it — the sigma_m = (2.5/ln10)(e/f) weights were
    # mutation-survivable. This row is non-collinear with one high-noise band;
    # the expected beta comes from the CLOSED-FORM weighted least squares
    # (independent of polyfit), and the unweighted OLS slope differs by a
    # margin — so dropping or corrupting the weights now fails loudly.
    # the HIGH-NOISE band sits at an ENDPOINT (a mid-lever point moves the
    # intercept, not the slope — first fixture attempt, kept as a lesson)
    mags = {"f115w": 26.5, "f150w": 26.2, "f200w": 25.0}
    snrs = {"f115w": 100.0, "f150w": 100.0, "f200w": 5.0}
    opt = power_law_mags(0.5, "f444w", 25.0, ("f277w", "f356w", "f444w"))
    row = {"mags": {**mags, **opt}, "snr": {**snrs}, "flux_radius": 0.08}
    phot = make_phot([row], _KOC_BANDS)
    out = kocevski24.select(phot, [5.0], [0.08], PIVOTS)

    x = np.log10([PIVOTS["f115w"], PIVOTS["f150w"], PIVOTS["f200w"]])
    y = np.array([mags[b] for b in ("f115w", "f150w", "f200w")])
    sig = 2.5 / np.log(10) / np.array([snrs[b] for b in ("f115w", "f150w", "f200w")])
    w = 1.0 / sig**2
    Sw, Swx, Swy = w.sum(), (w * x).sum(), (w * y).sum()
    Swxx, Swxy = (w * x * x).sum(), (w * x * y).sum()
    slope_w = (Sw * Swxy - Swx * Swy) / (Sw * Swxx - Swx**2)
    beta_expected = -slope_w / 2.5 - 2.0

    xb, yb = x.mean(), y.mean()
    slope_ols = ((x - xb) * (y - yb)).sum() / ((x - xb) ** 2).sum()
    beta_ols = -slope_ols / 2.5 - 2.0

    assert abs(beta_expected - beta_ols) > 0.05      # the test can discriminate
    assert out["beta_uv"][0] == pytest.approx(beta_expected, abs=1e-6)


def test_kocevski24_variant_param_is_gated_and_stamped():
    # r5f3: only the two preregistered variants execute; the result names
    # which one ran.
    phot = make_phot([_koc_row()], _KOC_BANDS)
    with pytest.raises(ValueError, match="PREREGISTERED"):
        kocevski24.select(phot, [5.0], [0.08], PIVOTS, lineboost_beta_min=-0.5)
    out = kocevski24.select(phot, [5.0], [0.08], PIVOTS)
    assert out["lineboost_variant"] == "formal-beta>-1"


def test_kocevski24_every_bin_has_an_end_to_end_pass(monkeypatch):
    # r5f3: all four Table-2 bins demonstrated selecting end-to-end (earlier
    # tests only passed bins 3 and 4). HST bands feed the low-z bins.
    all_bands = ("f606w", "f814w", "f115w", "f150w", "f200w", "f277w", "f356w", "f444w")
    rows = []
    for zc, (lo, hi, uv, opt) in zip((2.5, 4.0, 5.0, 8.5), kocevski24.Z_BINS):
        mags = power_law_mags(-1.0, uv[0], 26.0, uv)
        # the red law must ALSO cover f277w/f356w so condition (v)'s two-band
        # beta rides the same slope — but must NEVER overwrite a UV band
        # (bin 4's UV list contains f277w; overwriting it contaminated the
        # UV fit to beta=-2.44 in the first fixture attempt)
        red = power_law_mags(0.5, opt[-1], 25.0, tuple(set(opt) | {"f277w", "f356w"}))
        mags.update({k: v for k, v in red.items() if k not in uv})
        for b in all_bands:
            mags.setdefault(b, 25.5)
        rows.append({"mags": mags, "flux_radius": 0.08})
    phot = make_phot(rows, all_bands)
    out = kocevski24.select(phot, [2.5, 4.0, 5.0, 8.5], [0.08] * 4, PIVOTS)
    assert out["beta_uv"] == pytest.approx([-1.0] * 4, abs=1e-6)
    assert out["selected"].tolist() == [True, True, True, True]


def test_kocevski24_f410m_exact_minus_one_fails(monkeypatch):
    # r5f3: condition (vi) at beta exactly -1 fails the strict > (exact via
    # monkeypatched conversion; F410M covered so the condition applies).
    base = _koc_row()
    base["mags"]["f410m"] = 25.0
    phot = make_phot([base], _KOC_BANDS + ("f410m",))
    monkeypatch.setattr(kocevski24.conventions, "beta_from_color", lambda c, l1, l2: -1.0)
    out = kocevski24.select(phot, [5.0], [0.08], PIVOTS)
    assert out["lineboost_410_applies"][0] and not out["lineboost_410"][0]
    assert not out["selected"][0]


def test_kocevski24_variant_gate_rejects_bools_and_stamps_both(monkeypatch):
    # r5f4: False == 0.0 in Python — a boolean must NOT silently select the
    # printed-equivalents variant; and both variant stamps are asserted.
    phot = make_phot([_koc_row()], _KOC_BANDS)
    with pytest.raises(ValueError, match="PREREGISTERED"):
        kocevski24.select(phot, [5.0], [0.08], PIVOTS, lineboost_beta_min=False)
    with pytest.raises(ValueError, match="PREREGISTERED"):
        kocevski24.select(phot, [5.0], [0.08], PIVOTS, lineboost_beta_min=True)
    out = kocevski24.select(phot, [5.0], [0.08], PIVOTS, lineboost_beta_min=0.0)
    assert out["lineboost_variant"] == "printed-colors>0.53/0.84"


def test_kocevski24_covered_f410m_threshold_triplet(monkeypatch):
    # r5f4: the covered-AND-passing F410M branch was mutation-survivable —
    # pin the strict > with a triplet of exact betas around the threshold.
    base = _koc_row()
    base["mags"]["f410m"] = 25.0
    phot = make_phot([base], _KOC_BANDS + ("f410m",))
    for beta_val, expect in ((-0.9, True), (-1.0, False), (-1.1, False)):
        monkeypatch.setattr(
            kocevski24.conventions, "beta_from_color", lambda c, l1, l2, _b=beta_val: _b
        )
        out = kocevski24.select(phot, [5.0], [0.08], PIVOTS)
        assert bool(out["lineboost_410"][0]) == expect, beta_val
        assert out["lineboost_410_applies"][0]


def test_kocevski24_fits_consume_only_the_bins_bands():
    # r5f4 antagonistic-band test: poison a band OUTSIDE the z=5 bin's lists
    # (f606w) with a wild value — the fitted betas must be bit-identical, or
    # the fit is consuming bands the paper does not use at that redshift.
    clean = _koc_row()
    poisoned = _koc_row()
    poisoned["mags"] = {**poisoned["mags"], "f606w": 12.0}
    bands = ("f606w",) + _KOC_BANDS
    out_clean = kocevski24.select(make_phot([clean], bands), [5.0], [0.08], PIVOTS)
    out_poison = kocevski24.select(make_phot([poisoned], bands), [5.0], [0.08], PIVOTS)
    assert out_clean["beta_uv"][0] == out_poison["beta_uv"][0]
    assert out_clean["beta_opt"][0] == out_poison["beta_opt"][0]
    assert out_poison["selected"][0]


def test_kocevski24_printed_variant_uses_literal_colors():
    # r5f5 deciding test: a 277-356 color of 0.5301 passes the PRINTED cut
    # (literal > 0.53) even though our-pivot beta is below 0 — and fails the
    # formal beta>-1 variant only if beta <= -1 (here beta ~ -0.09, so formal
    # passes too; the DIVERGENT case is color 0.30: formal beta ~ -0.92 > -1
    # passes, printed 0.30 < 0.53 fails).
    uv = power_law_mags(-1.0, "f150w", 26.0, ("f115w", "f150w", "f200w"))
    def row_with_color(c):
        m277 = 26.0
        opt = {"f277w": m277, "f356w": m277 - c}
        opt["f444w"] = m277 - c - 0.9
        return {"mags": {**uv, **opt}, "flux_radius": 0.08}
    phot = make_phot([row_with_color(0.5301), row_with_color(0.30)], _KOC_BANDS)
    formal = kocevski24.select(phot, [5.0, 5.0], [0.08, 0.08], PIVOTS)
    printed = kocevski24.select(phot, [5.0, 5.0], [0.08, 0.08], PIVOTS, lineboost_beta_min=0.0)
    assert conventions.beta_from_color(0.5301, PIVOTS["f277w"], PIVOTS["f356w"]) < 0.0
    assert printed["lineboost_356"].tolist() == [True, False]
    assert formal["lineboost_356"].tolist() == [True, True]


def test_kocevski24_pivot_locality_z5_only_table():
    # r5f5 deciding test: a z=5 table needs only its six relevant pivots.
    phot = make_phot([_koc_row()], _KOC_BANDS)
    local = {k: PIVOTS[k] for k in ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w")}
    out = kocevski24.select(phot, [5.0], [0.08], local)
    assert out["selected"][0]


def test_kocevski24_beta_errors_are_audited():
    # r5f5: three-band fits carry finite sigma-propagated errors; the z>8
    # two-band optical fit ALSO carries a finite error because the weights
    # define an unscaled covariance at N=2 (documented semantics, flagged
    # for co-review — a residual-based error would be NaN there).
    out5 = kocevski24.select(make_phot([_koc_row()], _KOC_BANDS), [5.0], [0.08], PIVOTS)
    assert np.isfinite(out5["beta_uv_err"][0]) and np.isfinite(out5["beta_opt_err"][0])
    uv_hi = power_law_mags(-1.0, "f200w", 26.0, ("f150w", "f200w", "f277w"))
    opt_hi = power_law_mags(0.5, "f444w", 25.2, ("f356w", "f444w"))
    row = {"mags": {**uv_hi, **opt_hi}, "flux_radius": 0.08}
    out8 = kocevski24.select(make_phot([row], _KOC_BANDS), [8.5], [0.08], PIVOTS)
    assert out8["n_bands_opt"][0] == 2 and np.isfinite(out8["beta_opt_err"][0])


def test_nullable_coverage_is_a_named_refusal_never_fail_open():
    # r5f6 deciding test (review): NaN in a coverage column coerced to True
    # under to_numpy(bool) — a silent FAIL-OPEN on the fail-closed column.
    # Now: named schema refusal, everywhere coverage is read.
    phot = make_phot([{"mags": {"f277w": 29.1, "f444w": 27.5}}], ("f277w", "f444w"))
    phot_bad = phot.copy()
    phot_bad["cov_f277w"] = [np.nan]
    with pytest.raises(ValueError, match="cov_f277w contains nulls"):
        barro23.select(phot_bad)
    koc = make_phot([_koc_row()], _KOC_BANDS)
    koc["cov_f444w"] = [np.nan]
    with pytest.raises(ValueError, match="cov_f444w contains nulls"):
        kocevski24.select(koc, [5.0], [0.08], PIVOTS)


def test_kocevski24_printed_variant_low_z_needs_no_lineboost_pivots():
    # r5f6 deciding test: z=2.5 printed variant with exactly its six slope
    # pivots (no f356w) succeeds with NaN audit beta — never a KeyError.
    all_bands = ("f606w", "f814w", "f115w", "f150w", "f200w", "f277w", "f356w", "f444w")
    uv = power_law_mags(-1.0, "f606w", 26.0, ("f606w", "f814w", "f115w"))
    red = power_law_mags(0.5, "f277w", 25.0, ("f150w", "f200w", "f277w", "f356w"))
    mags = {**uv, **red, "f444w": 25.5}
    row = {"mags": mags, "flux_radius": 0.08}
    phot = make_phot([row], all_bands)
    local = {k: PIVOTS[k] for k in ("f606w", "f814w", "f115w", "f150w", "f200w", "f277w")}
    out = kocevski24.select(phot, [2.5], [0.08], local, lineboost_beta_min=0.0)
    assert np.isnan(out["beta_lineboost_356"][0])
    assert out["lineboost_variant"] == "printed-colors>0.53/0.84"
    assert out["selected"][0]


def test_kocevski24_r5f7_gates():
    # np.bool_(False) is NOT a Python bool — the r5f4 gate missed it (review
    # r5f7); and an all-uncovered F410M column demands no pivot.
    phot = make_phot([_koc_row()], _KOC_BANDS)
    with pytest.raises(ValueError, match="PREREGISTERED"):
        kocevski24.select(phot, [5.0], [0.08], PIVOTS, lineboost_beta_min=np.bool_(False))
    base = _koc_row()
    phot2 = make_phot([base], _KOC_BANDS + ("f410m",))   # column exists, row uncovered
    local = {k: PIVOTS[k] for k in ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w")}
    out = kocevski24.select(phot2, [5.0], [0.08], local)
    assert out["selected"][0] and not out["lineboost_410_applies"][0]


def test_kocevski24_r5f8_applicability_domain_and_z8_f410m_veto(monkeypatch):
    # r5f8: (1) out-of-table rows carry NO applicability claims in the audit;
    # (2) at z>8 a covered-but-too-blue F410M vetoes via condition (vi) — the
    # paper's "only the second condition is imposed" path, end to end.
    out_row = _koc_row()
    phot = make_phot([out_row], _KOC_BANDS)
    out = kocevski24.select(phot, [1.5], [0.08], PIVOTS)
    assert not out["lineboost_356_applies"][0] and not out["selected"][0]

    uv_hi = power_law_mags(-1.0, "f200w", 26.0, ("f150w", "f200w", "f277w"))
    opt_hi = power_law_mags(0.5, "f444w", 25.2, ("f356w", "f444w"))
    good = {"mags": {**uv_hi, **opt_hi, "f410m": uv_hi["f277w"] - 1.0}, "flux_radius": 0.08}
    bad = {"mags": {**uv_hi, **opt_hi, "f410m": uv_hi["f277w"] + 0.2}, "flux_radius": 0.08}
    phot2 = make_phot([good, bad], _KOC_BANDS + ("f410m",))
    out2 = kocevski24.select(phot2, [8.5, 8.5], [0.08, 0.08], PIVOTS)
    assert out2["lineboost_410_applies"].tolist() == [True, True]
    assert out2["lineboost_410"].tolist() == [True, False]
    assert out2["selected"].tolist() == [True, False]


def test_kocevski24_r5f9_pivot_locality_and_non_mutation():
    # r5f9: (1) F410M covered ONLY on an out-of-table row demands no pivot;
    # (2) the input table is bit-identical after select (non-mutation pin).
    base = _koc_row()
    outside = _koc_row()
    outside["mags"]["f410m"] = 25.0            # covered, but z out of table
    phot = make_phot([base, outside], _KOC_BANDS + ("f410m",))
    local = {k: PIVOTS[k] for k in ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w")}
    before = phot.copy(deep=True)
    out = kocevski24.select(phot, [5.0, 1.5], [0.08, 0.08], local)
    assert out["selected"].tolist() == [True, False]
    pd.testing.assert_frame_equal(phot, before)


# ---------------------------------------- labbe23 r5g fix batch

def test_labbe23_psf_mask_and_bd_mask_refuse_coercible_garbage():
    # r5g deciding test: NaN, the STRING "False", and a reversed-index Series
    # must refuse — np.asarray(bool) would have admitted all three as True.
    base = {"f115w": 27.9, "f150w": 27.4, "f200w": 26.5, "f277w": 25.7,
            "f356w": 25.4, "f444w": 25.2}
    phot = make_phot([{"mags": base}], ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    for bad in ([np.nan], ["False"], [1]):
        with pytest.raises(ValueError, match="strictly-boolean"):
            labbe23.select(phot, [1.2], psf_dominated=bad)
    two = make_phot([{"mags": base}] * 2, ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    with pytest.raises(ValueError, match="non-trivial index"):
        labbe23.select(two, [1.2, 1.2], psf_dominated=pd.Series([True, False], index=[1, 0]))
    akins_phot = make_phot([{"mags": {"f277w": 29.2, "f444w": 27.0}, "compactness": 0.6}], ("f277w", "f444w"))
    with pytest.raises(ValueError, match="strictly-boolean"):
        akins24.select(akins_phot, is_brown_dwarf_sed=[np.nan])


def test_labbe23_compact_valid_separates_invalid_from_extended():
    # r5g deciding test, ported audit semantics: [-1, NaN, 1.8, 1.2] ->
    # compact_valid [F,F,T,T], compact [F,F,F,T].
    base = {"f115w": 27.9, "f150w": 27.4, "f200w": 26.5, "f277w": 25.7,
            "f356w": 25.4, "f444w": 25.2}
    phot = make_phot([{"mags": base}] * 4, ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    out = labbe23.select(phot, [-1.0, np.nan, 1.8, 1.2])
    assert out["compact_valid"].tolist() == [False, False, True, True]
    assert out["compact"].tolist() == [False, False, False, True]


def test_labbe23_six_exact_color_boundaries_fail(monkeypatch):
    # r5g deciding test: each of the six color criteria at EXACT equality
    # fails through the real branch wiring (a lt->le / gt->ge mutation dies);
    # the all-passing baseline proves the direction on the epsilon side.
    base = {"f115w": 27.9, "f150w": 27.4, "f200w": 26.5, "f277w": 25.7,
            "f356w": 25.4, "f444w": 25.2}
    phot = make_phot([{"mags": base}], ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    passing = {
        ("f115w", "f150w"): 0.5, ("f200w", "f277w"): 0.8, ("f200w", "f356w"): 1.1,
        ("f150w", "f200w"): 0.5, ("f277w", "f356w"): 0.8, ("f277w", "f444w"): 1.2,
    }
    thresholds = {
        ("f115w", "f150w"): 0.8, ("f200w", "f277w"): 0.7, ("f200w", "f356w"): 1.0,
        ("f150w", "f200w"): 0.8, ("f277w", "f356w"): 0.7, ("f277w", "f444w"): 1.0,
    }
    def run(colors):
        def fake(phot_, blue, red):
            return np.full(len(phot_), colors[(blue, red)])
        monkeypatch.setattr(labbe23.sh, "color", fake)
        return labbe23.select(phot, [1.2])
    out_pass = run(passing)
    assert out_pass["red1"][0] and out_pass["red2"][0] and out_pass["selected"][0]
    branch_of = {
        ("f115w", "f150w"): "red1", ("f200w", "f277w"): "red1", ("f200w", "f356w"): "red1",
        ("f150w", "f200w"): "red2", ("f277w", "f356w"): "red2", ("f277w", "f444w"): "red2",
    }
    for pair, thr in thresholds.items():
        exact = dict(passing)
        exact[pair] = thr
        out = run(exact)
        # the OR structure keeps `selected` alive via the OTHER branch — the
        # exact-equality strictness is asserted on the branch itself
        assert not out[branch_of[pair]][0], (pair, thr)
        assert out["red2" if branch_of[pair] == "red1" else "red1"][0]


def test_labbe23_unknown_operator_raises():
    # r5g2: a typo'd operator in a criterion spec must refuse, never
    # silently become gt.
    base = {"f115w": 27.9, "f150w": 27.4, "f200w": 26.5, "f277w": 25.7,
            "f356w": 25.4, "f444w": 25.2}
    phot = make_phot([{"mags": base}], ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    with pytest.raises(ValueError, match="unknown color operator"):
        labbe23._branch(phot, (("f115w", "f150w", "le", 0.8),))


def test_labbe23_skeptic_minor_pins():
    # skeptic review (labbe, CLEAN PASS) minor items: no-psf call emits NO
    # main key; and the input table is bit-identical after select.
    base = {"f115w": 27.9, "f150w": 27.4, "f200w": 26.5, "f277w": 25.7,
            "f356w": 25.4, "f444w": 25.2}
    phot = make_phot([{"mags": base}], ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    before = phot.copy(deep=True)
    out = labbe23.select(phot, [1.2])
    assert "main" not in out
    pd.testing.assert_frame_equal(phot, before)


def test_aperture_orientation_cross_module_exhibit():
    # skeptic review finding 8 (high impact): the SAME point source feeds
    # labbe23 as large/small (f04/f02) and akins24 as small/large (f02/f05).
    # A point source passes BOTH under correct orientations — and the exhibit
    # shows labbe SILENTLY passes even under the inverted value, which is why
    # the W1 orientation assertion (ledgered) is load-bearing, not decorative.
    f02, f04, f05 = 1.0, 1.5, 1.8            # point-source aperture growth
    labbe_ratio = f04 / f02                   # 1.5  -> compact (<1.7)
    akins_c444 = f02 / f05                    # 0.556 -> compact (0.5, 0.7]
    base = {"f115w": 27.9, "f150w": 27.4, "f200w": 26.5, "f277w": 25.7,
            "f356w": 25.4, "f444w": 25.2}
    phot_l = make_phot([{"mags": base}], ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    assert labbe23.select(phot_l, [labbe_ratio])["compact"][0]
    phot_a = make_phot([{"mags": {"f277w": 29.2, "f444w": 27.0}, "compactness": akins_c444}],
                       ("f277w", "f444w"))
    assert akins24.select(phot_a)["compact"][0]
    # the silent-risk exhibit: the INVERTED labbe input still passes
    assert labbe23.select(phot_l, [1.0 / labbe_ratio])["compact"][0]
    # while akins fed the inverted value fails loudly-by-value
    phot_a2 = make_phot([{"mags": {"f277w": 29.2, "f444w": 27.0}, "compactness": 1.0 / akins_c444}],
                        ("f277w", "f444w"))
    assert not akins24.select(phot_a2)["compact"][0]


def test_greene24_r5h_gate_and_red_equality(monkeypatch):
    # r5h: detection-gate behavior (SNR == 14 exact fails; magnitude wiring
    # direction) and red-arm equality through the real selector.
    good = {"f115w": 26.8, "f200w": 26.5, "f277w": 27.3, "f444w": 26.0}
    rows = [
        {"mags": good, "snr": {"f444w": 14.0}},
        {"mags": good, "snr": {"f444w": 14.5}},
        # magnitude rows shift f277w in lockstep so the red color stays 1.3
        # (the gate is the ONLY thing under test in these two rows)
        {"mags": dict(good, f444w=27.71, f277w=29.01), "snr": 50.0},
        {"mags": dict(good, f444w=27.69, f277w=28.99), "snr": 50.0},
    ]
    phot = make_phot(rows, ("f115w", "f200w", "f277w", "f444w"))
    out = greene24.select(phot)
    assert out["detection"].tolist() == [False, True, False, True]
    # r5h2: assert SELECTED too — a mutant dropping detection from the final
    # conjunction must die here, not just in the audit column
    assert out["selected"].tolist() == [False, True, False, True]

    one = make_phot([{"mags": good}], ("f115w", "f200w", "f277w", "f444w"))
    exact = {("f115w", "f200w"): 0.3, ("f277w", "f444w"): 1.0}
    def fake(phot_, blue, red):
        return np.full(len(phot_), exact[(blue, red)])
    monkeypatch.setattr(greene24.sh, "color", fake)
    out2 = greene24.select(one)
    assert not out2["vshape_red"][0] and not out2["selected"][0]
    exact[("f277w", "f444w")] = 1.0 + 1e-9
    out3 = greene24.select(one)
    assert out3["vshape_red"][0] and out3["selected"][0]


def test_greene24_exact_magnitude_gate_via_monkeypatch(monkeypatch):
    # r5h2: m444 EXACTLY 27.7 fails the strict < (mag engine faked to bypass
    # flux round-trips, kokorev pattern).
    good = {"f115w": 26.8, "f200w": 26.5, "f277w": 27.3, "f444w": 26.0}
    phot = make_phot([{"mags": good}], ("f115w", "f200w", "f277w", "f444w"))
    real = sh.mags_ab
    def fake(p, band):
        out = real(p, band)
        return np.full_like(out, 27.7) if band == "f444w" else out
    monkeypatch.setattr(greene24.sh, "mags_ab", fake)
    out = greene24.select(phot)
    assert not out["detection"][0] and not out["selected"][0]


def test_barro23_r5i_schema_and_exact_boundary(monkeypatch):
    # r5i: partial triples refuse; m444 EXACTLY 28.0 fails the strict <.
    phot = make_phot([{"mags": {"f277w": 29.1, "f444w": 27.5}}], ("f277w", "f444w"))
    # an ORPHAN stem (not in barro's required set) exercises the triple
    # guard; a missing REQUIRED column hits require_bands first by design
    broken = phot.copy()
    broken["f_f150w_ujy"] = [1.0]
    with pytest.raises(ValueError, match="incomplete f/e/cov"):
        barro23.select(broken)
    real = sh.mags_ab
    def fake(p, band):
        out = real(p, band)
        return np.full_like(out, 28.0) if band == "f444w" else out
    monkeypatch.setattr(barro23.sh, "mags_ab", fake)
    out = barro23.select(phot)
    assert not out["mag_gate"][0] and not out["selected"][0]


def test_barro23_skeptic_boundary_row_and_symmetric_coverage():
    # skeptic review (barro): real-photometry row at m444 EXACTLY 28.0 (its
    # probe verified the round-trip is exact) — kills the lt->le mutant the
    # monkeypatch test also guards; plus the symmetric f444w-uncovered row.
    rows = [
        {"mags": {"f277w": 29.7, "f444w": 28.0}},   # color 1.7 red; mag AT 28 -> reject
        {"mags": {"f277w": 29.1}},                   # f444w uncovered -> reject
    ]
    phot = make_phot(rows, ("f277w", "f444w"))
    out = barro23.select(phot)
    assert out["selected"].tolist() == [False, False]
    assert not out["mag_gate"][0]


def test_barro23_non_mutation():
    # r5i2: the input table is bit-identical after select.
    phot = make_phot([{"mags": {"f277w": 29.1, "f444w": 27.5}}], ("f277w", "f444w"))
    before = phot.copy(deep=True)
    barro23.select(phot)
    pd.testing.assert_frame_equal(phot, before)


def test_perezgonzalez24_r5j_audit_masks_and_boundaries(monkeypatch):
    # r5j: bd_measurable separates unmeasurable from too-blue; the PM mask is
    # a visible strict-boolean step; and every boundary is mutation-pinned
    # exactly (colors via faked engine, mag via faked mags: le KEEPS 28.0).
    good = {"f115w": 28.6, "f150w": 28.6, "f200w": 28.3, "f277w": 29.1, "f444w": 27.9}
    rows = [
        {"mags": good},
        {"mags": {k: v for k, v in good.items() if k != "f115w"}},   # BD unmeasurable
    ]
    phot = make_phot(rows, ("f115w", "f150w", "f200w", "f277w", "f444w"))
    out = perezgonzalez24.select(phot)
    assert out["bd_measurable"].tolist() == [True, False]
    assert out["bd_retention"].tolist() == [True, False]
    assert "pm_bd_retained" not in out
    out2 = perezgonzalez24.select(phot, known_pm_brown_dwarf=[True, False])
    assert out2["selected"].tolist() == [False, False] and out2["pm_bd_retained"].tolist() == [False, True]
    with pytest.raises(ValueError, match="strictly-boolean"):
        perezgonzalez24.select(phot, known_pm_brown_dwarf=[np.nan, 1.0])

    one = make_phot([{"mags": good}], ("f115w", "f150w", "f200w", "f277w", "f444w"))
    exact = {("f277w", "f444w"): 1.0 + 1e-9, ("f150w", "f200w"): 0.5 - 1e-9,
             ("f115w", "f150w"): -0.5}
    def fake(p_, blue, red):
        return np.full(len(p_), exact[(blue, red)])
    monkeypatch.setattr(perezgonzalez24.sh, "color", fake)
    ok = perezgonzalez24.select(one)
    assert ok["selected"][0]                        # -0.5 EXACT passes (>= inclusive)
    for pair, val in ((("f277w", "f444w"), 1.0), (("f150w", "f200w"), 0.5),
                      (("f115w", "f150w"), -0.5 - 1e-9)):
        trial = dict(exact); trial[pair] = val
        exact_backup = exact.copy(); exact.update(trial)
        out3 = perezgonzalez24.select(one)
        assert not out3["selected"][0], (pair, val)
        exact.clear(); exact.update(exact_backup)

    real = sh.mags_ab
    def fake_mag(p_, band):
        outm = real(p_, band)
        return np.full_like(outm, 28.0) if band == "f444w" else outm
    monkeypatch.setattr(perezgonzalez24.sh, "color", lambda p_, b, r: np.full(len(p_), {("f277w","f444w"):1.2,("f150w","f200w"):0.3,("f115w","f150w"):0.0}[(b,r)]))
    monkeypatch.setattr(perezgonzalez24.sh, "mags_ab", fake_mag)
    out4 = perezgonzalez24.select(one)
    assert out4["mag_gate"][0] and out4["selected"][0]     # 28.0 KEPT: <= inclusive


def test_perezgonzalez24_just_above_28_fails(monkeypatch):
    # r5j2: nextafter(28, +inf) must FAIL the inclusive gate.
    good = {"f115w": 28.6, "f150w": 28.6, "f200w": 28.3, "f277w": 29.1, "f444w": 27.9}
    one = make_phot([{"mags": good}], ("f115w", "f150w", "f200w", "f277w", "f444w"))
    real = sh.mags_ab
    val = np.nextafter(28.0, np.inf)
    def fake(p_, band):
        outm = real(p_, band)
        return np.full_like(outm, val) if band == "f444w" else outm
    monkeypatch.setattr(perezgonzalez24.sh, "mags_ab", fake)
    out = perezgonzalez24.select(one)
    assert not out["mag_gate"][0] and not out["selected"][0]


def test_akins24_r5k_hardening(monkeypatch):
    # r5k: three-state compactness audit; sed_bd_applied runner flag; exact
    # S/N==12 and color==1.5 strictness through the selector; triples guard;
    # non-mutation.
    good = {"f277w": 29.2, "f444w": 27.0}
    rows = [{"mags": good, "compactness": c} for c in (np.nan, 0.4, 0.75, 0.6)]
    phot = make_phot(rows, ("f277w", "f444w"))
    before = phot.copy(deep=True)
    out = akins24.select(phot)
    assert out["compact_valid"].tolist() == [False, True, True, True]
    assert out["compact_artifact"].tolist() == [False, False, True, False]
    assert out["compact"].tolist() == [False, False, False, True]
    assert out["sed_bd_applied"] is False
    out2 = akins24.select(phot, is_brown_dwarf_sed=[False] * 4)
    assert out2["sed_bd_applied"] is True
    pd.testing.assert_frame_equal(phot, before)

    one = make_phot([{"mags": good, "compactness": 0.6, "snr": {"f444w": 12.0}}], ("f277w", "f444w"))
    out_snr = akins24.select(one)
    # skeptic r1: assert SELECTED too — a mutant dropping detection from the
    # final conjunction must die here, not just in the audit column (the
    # greene24 lesson, repeated by this suite and now closed here as well)
    assert not out_snr["detection"][0] and not out_snr["selected"][0]
    two = make_phot([{"mags": good, "compactness": 0.6}], ("f277w", "f444w"))
    monkeypatch.setattr(akins24.sh, "color", lambda p_, b, r: np.full(len(p_), 1.5))
    assert not akins24.select(two)["red_color"][0]

    broken = make_phot([{"mags": good, "compactness": 0.6}], ("f277w", "f444w"))
    broken["f_f150w_ujy"] = [1.0]
    with pytest.raises(ValueError, match="incomplete f/e/cov"):
        akins24.select(broken)


def test_result_contract_row_aligned_except_declared_metadata():
    # r5k2 generic-consumer test: for EVERY cut, every result value is a
    # row-aligned array unless its key is in the declared METADATA_KEYS
    # scalar namespace — a consumer can iterate rows safely by skipping them.
    base6 = {"f115w": 27.9, "f150w": 27.4, "f200w": 26.5, "f277w": 25.7,
             "f356w": 25.4, "f444w": 25.2}
    b6 = ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w")
    two6 = make_phot([{"mags": base6, "compactness": 0.6, "flux_radius": 0.08}] * 2, b6)
    two2 = make_phot([{"mags": {"f277w": 29.2, "f444w": 27.0}, "compactness": 0.6}] * 2,
                     ("f277w", "f444w"))
    pg6 = make_phot([{"mags": {"f115w": 28.6, "f150w": 28.6, "f200w": 28.3,
                               "f277w": 29.1, "f444w": 27.9}}] * 2,
                    ("f115w", "f150w", "f200w", "f277w", "f444w"))
    results = [
        labbe23.select(two6, [1.2, 1.2]),
        kokorev24.select(two6, [1.2, 1.2]),
        kocevski24.select(two6, [5.0, 5.0], [0.08, 0.08], PIVOTS),
        perezgonzalez24.select(pg6),
        barro23.select(two2),
        greene24.select(two6),
        akins24.select(two2),
    ]
    for res in results:
        for k, v in res.items():
            if k in sh.METADATA_KEYS:
                assert np.ndim(v) == 0 or isinstance(v, (str, bool))
            else:
                assert np.asarray(v).shape[0] == 2, k


# ------------------------------------------------- combinator truth tables (design D5, r15)

def test_kokorev24_combinator_truth_table(monkeypatch):
    # r15 (review D5): token equality cannot pin Boolean STRUCTURE — this
    # truth table toggles every top-level arm of
    #   detection & (red1 | red2) & compact & bd_removal
    # independently, so any AND<->OR mutant of the combinator dies by name.
    good = {"f115w": 27.9, "f150w": 27.8, "f200w": 27.3, "f277w": 26.8,
            "f356w": 26.15, "f444w": 26.0}
    phot = make_phot([{"mags": good}], ("f115w", "f150w", "f200w", "f277w", "f356w", "f444w"))
    R1_PASS = {("f115w", "f150w"): 0.5, ("f200w", "f277w"): 0.9, ("f200w", "f356w"): 1.2}
    R1_FAIL = {("f115w", "f150w"): 0.5, ("f200w", "f277w"): 0.5, ("f200w", "f356w"): 1.2}
    R2_PASS = {("f150w", "f200w"): 0.5, ("f277w", "f356w"): 0.7, ("f277w", "f444w"): 0.8}
    R2_FAIL = {("f150w", "f200w"): 0.9, ("f277w", "f356w"): 0.7, ("f277w", "f444w"): 0.8}
    BD_PASS = {("f115w", "f200w"): 0.6}
    BD_FAIL = {("f115w", "f200w"): -0.6}

    def run(colors, ratio):
        def fake(phot_, blue, red):
            n = len(phot_)
            return np.full(n, colors[(blue, red)]), np.zeros(n, dtype=bool)
        monkeypatch.setattr(kokorev24, "_color_with_upper_limits", fake)
        return kokorev24.select(phot, [ratio])

    r1_only = run({**R1_PASS, **R2_FAIL, **BD_PASS}, 1.2)
    assert r1_only["red1"][0] and not r1_only["red2"][0] and r1_only["selected"][0]

    r2_only = run({**R1_FAIL, **R2_PASS, **BD_PASS}, 1.2)
    assert r2_only["red2"][0] and not r2_only["red1"][0] and r2_only["selected"][0]

    neither = run({**R1_FAIL, **R2_FAIL, **BD_PASS}, 1.2)
    assert not neither["selected"][0]

    compact_fail = run({**R1_PASS, **R2_PASS, **BD_PASS}, 2.4)   # 2.4 >= 1.7 extended
    assert not compact_fail["selected"][0]

    bd_fail = run({**R1_PASS, **R2_PASS, **BD_FAIL}, 1.2)
    assert not bd_fail["bd_removal"][0] and not bd_fail["selected"][0]


def test_kocevski24_combinator_arm_toggles():
    # r15 (review D5): a selected row flips to rejected when EITHER the size
    # arm or the red-slope arm is broken, independently — AND->OR mutants of
    # the top-level conjunction die here (atop the existing exact-boundary
    # and variant-divergence killers).
    base = make_phot([_koc_row()], _KOC_BANDS)
    assert kocevski24.select(base, [5.0], [0.08], PIVOTS)["selected"][0]

    extended = make_phot([_koc_row(flux_radius=0.5)], _KOC_BANDS)
    assert not kocevski24.select(extended, [5.0], [0.08], PIVOTS)["selected"][0]

    blue_opt = make_phot([_koc_row(beta_opt=-2.0)], _KOC_BANDS)
    assert not kocevski24.select(blue_opt, [5.0], [0.08], PIVOTS)["selected"][0]
