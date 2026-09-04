"""The label engine, version 3: guarded linear-space fits and one-sided profile bounds.

WHAT CHANGED. Version 2 fitted in linear flux space, which removed the log-space truncation
bias. review round S1 showed that it still returned a slope where no slope is identifiable
(43.6% of pure-noise fits pin at the grid edge; ~4.6% pass a nominal 2-sigma test), and that
collapsing an asymmetric chi-squared profile into one symmetric sigma is anti-conservative
on exactly the side each criterion uses. Version 3 uses slopefit v3, which refuses
non-identifiable and edge fits and returns DIRECT one-sided delta-chi-squared = 4 bounds
inflated for pixel correlation by a block bootstrap.

THE CRITERIA ARE NOW WRITTEN ON THE BOUNDS, not on beta +/- 2*sigma:

    c_uv   : the UPPER bound of beta_UV  is below -0.2
    c_opt  : the LOWER bound of beta_opt is above 0
    c_diff : lower(beta_opt) - upper(beta_UV) > 0.5

The third was previously a comparison of point estimates with no uncertainty at all, which
the review flagged; using the two one-sided bounds makes it conservative in the same direction as
the other two, at the cost of rejecting marginal cases. That cost is the right one to pay
for a sample this small.

Every earlier estimator is still computed and recorded per spectrum, so the size of each
correction stays a measured column: log-space (v1), unguarded linear (v2), guarded (v3).
"""

import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from slopefit import fit_powerlaw  # noqa: E402

OUT = os.path.join(HERE, "census_spectra")
BREAK, UV_LO, OPT_HI = 3645.0, 1250.0, 7000.0
LINES = (
    1216.0,
    1549.0,
    1640.0,
    1909.0,
    2798.0,
    3727.0,
    3869.0,
    4102.0,
    4340.0,
    4861.0,
    4959.0,
    5007.0,
    5876.0,
    6548.0,
    6563.0,
    6584.0,
    6716.0,
    6731.0,
)
KMS, C = 5000.0, 299792.458


def line_mask(lam):
    k = np.ones_like(lam, dtype=bool)
    for l0 in LINES:
        k &= np.abs(lam - l0) > l0 * KMS / C
    return k


def fetch(row):
    dest = os.path.join(OUT, row.file)
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        return dest
    for _ in range(3):
        try:
            urllib.request.urlretrieve(row.url, dest + ".part")
            os.replace(dest + ".part", dest)
            return dest
        except Exception:
            pass
    return None


def log_space_fit(lam, f, e, lo, hi, mask):
    """v1's estimator, kept only to measure how far it moved."""
    m = (
        (lam >= lo)
        & (lam < hi)
        & np.isfinite(f)
        & np.isfinite(e)
        & (e > 0)
        & (f > 0)
        & mask
    )
    if m.sum() < 8:
        return np.nan
    x, y = np.log10(lam[m]), np.log10(f[m])
    w = 1.0 / np.clip((e[m] / f[m]) / np.log(10), 1e-6, None) ** 2
    X = np.vstack([x, np.ones_like(x)]).T
    try:
        cov = np.linalg.inv(X.T @ (w[:, None] * X))
    except np.linalg.LinAlgError:
        return np.nan
    return float((cov @ (X.T @ (w * y)))[0])


def verdict(uv, op):
    """The three criteria, written on one-sided bounds."""
    if not (uv["ok"] and op["ok"]):
        return dict(
            testable=False, c_uv=False, c_opt=False, c_diff=False, v_shaped=False
        )
    c1 = bool(np.isfinite(uv["beta_hi2"]) and uv["beta_hi2"] < -0.2)
    c2 = bool(np.isfinite(op["beta_lo2"]) and op["beta_lo2"] > 0.0 and op["amp"] > 0)
    c3 = bool(
        np.isfinite(op["beta_lo2"])
        and np.isfinite(uv["beta_hi2"])
        and (op["beta_lo2"] - uv["beta_hi2"]) > 0.5
    )
    return dict(
        testable=True, c_uv=c1, c_opt=c2, c_diff=c3, v_shaped=bool(c1 and c2 and c3)
    )


def classify(path, z):
    try:
        with fits.open(path, memmap=False) as h:
            d = h[1].data
            w_um = np.asarray(d["wave"], float)
            fl = np.asarray(d["flux"], float)
            er = np.asarray(d["err"], float)
    except Exception as exc:
        return dict(ok=False, testable=False, note="unreadable: " + type(exc).__name__)
    lam = 1e4 * w_um / (1.0 + z)
    flam, elam = fl / w_um**2, er / w_um**2
    km = line_mask(lam)
    uv = fit_powerlaw(lam, flam, elam, UV_LO, BREAK, mask=km)
    op = fit_powerlaw(lam, flam, elam, BREAK, OPT_HI, mask=km)
    v = verdict(uv, op)
    opm = (
        (lam >= BREAK)
        & (lam < OPT_HI)
        & np.isfinite(flam)
        & np.isfinite(elam)
        & (elam > 0)
        & km
    )
    snr_int = (
        float(np.sqrt(np.sum((flam[opm] / elam[opm]) ** 2))) if opm.sum() else np.nan
    )
    return dict(
        ok=True,
        note="",
        beta_uv=uv["beta"],
        beta_uv_hi2=uv["beta_hi2"],
        uv_ok=uv["ok"],
        uv_reason=uv["reason"],
        uv_amp_snr=uv["amp_snr"],
        uv_infl=uv["infl"],
        beta_opt=op["beta"],
        beta_opt_lo2=op["beta_lo2"],
        opt_ok=op["ok"],
        opt_reason=op["reason"],
        opt_amp_snr=op["amp_snr"],
        opt_infl=op["infl"],
        n_uv=uv["n"],
        n_opt=op["n"],
        snr_opt_pix=op["snr_pix"],
        snr_opt_int=snr_int,
        beta_uv_logspace=log_space_fit(lam, flam, elam, UV_LO, BREAK, km),
        beta_opt_logspace=log_space_fit(lam, flam, elam, BREAK, OPT_HI, km),
        **v,
    )


def main():
    cat = pd.read_csv(os.path.join(HERE, "census_list.csv"))
    print("census: %d spectra" % len(cat), flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        paths = list(pool.map(fetch, list(cat.itertuples())))
    rows = []
    for r, p in zip(cat.itertuples(), paths):
        base = dict(ra=r.ra, dec=r.dec, z=r.z, field=r.field, root=r.root, file=r.file)
        rows.append(
            dict(base, ok=False, testable=False, note="download failed")
            if p is None
            else dict(base, **classify(p, r.z))
        )
        if len(rows) % 250 == 0:
            print("  %d/%d" % (len(rows), len(cat)), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "census_vshape3.csv"), index=False)

    t = df[df["testable"].fillna(False) == True]
    print("\n=== census v3 (guarded linear-space fits, one-sided bounds) ===")
    print("spectra                        : %d" % len(df))
    print("BOTH windows identifiable      : %d" % len(t))
    print("V-shaped                       : %d" % int(t["v_shaped"].sum()))
    if len(t):
        print("V-shaped fraction of testable  : %.4f" % t["v_shaped"].mean())
        for f_, g in t.groupby("field"):
            print(
                "  %-14s %d/%d = %.4f"
                % (f_, int(g["v_shaped"].sum()), len(g), g["v_shaped"].mean())
            )
        print(
            "by criterion: c_uv %d  c_opt %d  c_diff %d"
            % (int(t.c_uv.sum()), int(t.c_opt.sum()), int(t.c_diff.sum()))
        )
        print(
            "median correlation inflation of sigma: uv %.2f  opt %.2f"
            % (float(np.nanmedian(t["uv_infl"])), float(np.nanmedian(t["opt_infl"])))
        )

    print("\n=== why spectra were refused (the guards doing their job) ===")
    for col in ("uv_reason", "opt_reason"):
        vc = df[col].fillna("").replace("", np.nan).dropna()
        if not len(vc):
            continue
        short = (
            vc.str.replace(r"\(.*\)", "", regex=True)
            .str.replace(r"only \d+ usable pixels", "too few pixels", regex=True)
            .str.replace(r"span only [\d.]+ dex", "span too small", regex=True)
        )
        print("  %s:" % col)
        for k, n in short.value_counts().head(6).items():
            print("     %5d  %s" % (n, k.strip()))

    print("\n=== the three estimators, side by side ===")
    for name, col in (
        ("v1 log-space", "beta_opt_logspace"),
        ("v3 guarded", "beta_opt"),
    ):
        v = df[col].astype(float)
        print(
            "  %-14s beta_opt > 0 in %d spectra (median %.2f)"
            % (name, int((v > 0).sum()), float(np.nanmedian(v)))
        )


if __name__ == "__main__":
    main()
