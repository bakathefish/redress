"""Shared machinery for the re-implemented cuts (fail-closed by construction).

Design rule: every criterion in every cut composes from the finite-gated
comparators here (:func:`gt`, :func:`lt`, :func:`between`, :func:`ge`,
:func:`le`), never from raw numpy comparisons. Raw ``a > x`` is already False
for NaN — but its NEGATION (``~(a > x)``, as in Akins's artifact exclusion)
would let a NaN row PASS. Finite-gating at the comparator level makes that
mistake unrepresentable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from redress import conventions


def require_bands(phot: pd.DataFrame, bands: tuple[str, ...], who: str) -> None:
    """Loudly demand the f_/e_/cov_ column triple for every band a cut reads."""
    missing = [
        col
        for b in bands
        for col in (f"f_{b}_ujy", f"e_{b}_ujy", f"cov_{b}")
        if col not in phot.columns
    ]
    if missing:
        raise ValueError(f"{who}: photometry table lacks required columns {missing}")


def mags_ab(phot: pd.DataFrame, band: str) -> np.ndarray:
    """AB magnitudes for one band, NaN where not computable (fail-closed feed).

    NaN when: not covered, flux missing, or flux <= 0 (a magnitude of a
    non-positive flux does not exist; the cut criteria treat NaN as FAIL —
    except where a paper states an upper-limit rule, which that module
    implements itself from fluxes). Positive fluxes convert through the
    ONE zero-point in ``redress.conventions`` (exact-23.9).
    """
    f = phot[f"f_{band}_ujy"].to_numpy(dtype=float)
    cov = cov_flags(phot, band)
    out = np.full(f.shape, np.nan)
    ok = cov & np.isfinite(f) & (f > 0)
    if np.any(ok):
        out[ok] = conventions.ab_mag_from_fnu_ujy(f[ok])
    return out


def snr(phot: pd.DataFrame, band: str) -> np.ndarray:
    """Flux / error for one band; NaN where uncovered or the error is unusable."""
    f = phot[f"f_{band}_ujy"].to_numpy(dtype=float)
    e = phot[f"e_{band}_ujy"].to_numpy(dtype=float)
    cov = cov_flags(phot, band)
    out = np.full(f.shape, np.nan)
    ok = cov & np.isfinite(f) & np.isfinite(e) & (e > 0)
    out[ok] = f[ok] / e[ok]
    return out


def color(phot: pd.DataFrame, blue: str, red: str) -> np.ndarray:
    """AB color m_blue − m_red; NaN propagates from either band (fail-closed)."""
    return mags_ab(phot, blue) - mags_ab(phot, red)


def gt(a, threshold: float) -> np.ndarray:
    """a > threshold, False wherever a is non-finite."""
    a = np.asarray(a, dtype=float)
    return np.isfinite(a) & (a > threshold)


def lt(a, threshold: float) -> np.ndarray:
    """a < threshold, False wherever a is non-finite."""
    a = np.asarray(a, dtype=float)
    return np.isfinite(a) & (a < threshold)


def le(a, threshold: float) -> np.ndarray:
    """a <= threshold, False wherever a is non-finite."""
    a = np.asarray(a, dtype=float)
    return np.isfinite(a) & (a <= threshold)


def ge(a, threshold: float) -> np.ndarray:
    """a >= threshold, False wherever a is non-finite."""
    a = np.asarray(a, dtype=float)
    return np.isfinite(a) & (a >= threshold)


def between(a, lo: float, hi: float) -> np.ndarray:
    """lo < a < hi (both strict), False wherever a is non-finite."""
    a = np.asarray(a, dtype=float)
    return np.isfinite(a) & (a > lo) & (a < hi)


def as_row_array(x, n: int, name: str) -> np.ndarray:
    """Coerce a caller-supplied per-row quantity to a float array of length n.

    r5e (review counter, accepted): a pandas Series with a shuffled index would
    silently attach morphology to the WRONG objects under positional
    conversion. A Series is accepted only with the trivial 0..n-1 index;
    anything else must be aligned by the caller and passed as
    ``.to_numpy()`` — misalignment becomes a loud refusal, not a wrong
    catalog.
    """
    if isinstance(x, pd.Series):
        if not x.index.equals(pd.RangeIndex(n)):
            raise ValueError(
                f"{name} is a pandas Series with a non-trivial index — align it to the "
                "table first and pass .to_numpy() (positional use would silently "
                "misattach rows)"
            )
        x = x.to_numpy()
    a = np.asarray(x, dtype=float)
    if a.shape != (n,):
        raise ValueError(f"{name} must be a 1-D array aligned to the table ({n} rows)")
    return a


#: r5k2: the RESULT CONTRACT — every value is a row-aligned array EXCEPT the
#: scalar metadata keys named here; generic consumers iterate rows over
#: everything else and read these as run-level provenance.
METADATA_KEYS = frozenset({"censoring_policy", "lineboost_variant", "sed_bd_applied"})


def result(selected: np.ndarray, **criteria) -> dict[str, np.ndarray]:
    """Assemble the audit-trail return value; every array row-aligned."""
    n = selected.shape[0]
    for k, v in criteria.items():
        if np.asarray(v).shape[0] != n:
            raise ValueError(f"criterion {k!r} is not row-aligned")
    out = {"selected": selected}
    out.update(criteria)
    return out


def require_complete_triples(phot: pd.DataFrame, who: str) -> None:
    """r5f2 (review): a PARTIAL band-column triple is a malformed schema.

    A band absent entirely is legitimate (the survey lacks it; rows fail
    closed) — but f_x without cov_x/e_x would crash or, worse, mislead
    mid-run. Any band stem present in ANY of its three forms must be present
    in ALL three.
    """
    stems = set()
    for c in phot.columns:
        if c.startswith("f_") and c.endswith("_ujy") or c.startswith("e_") and c.endswith("_ujy"):
            stems.add(c[2:-4])
        elif c.startswith("cov_"):
            stems.add(c[4:])
    broken = [
        b for b in sorted(stems)
        if not all(k in phot.columns for k in (f"f_{b}_ujy", f"e_{b}_ujy", f"cov_{b}"))
    ]
    if broken:
        raise ValueError(f"{who}: malformed schema — incomplete f/e/cov triples for {broken}")


def cov_flags(phot: pd.DataFrame, band: str) -> np.ndarray:
    """Coverage column as a STRICTLY validated boolean array (r5f6, review).

    ``to_numpy(dtype=bool)`` coerces NaN — and any truthy garbage — to True,
    which is a silent FAIL-OPEN on the one column whose whole job is
    fail-closed gating. Coverage must be non-null and purely boolean; anything
    else is a named schema refusal, never an inferred True.
    """
    col = phot[f"cov_{band}"]
    if col.isna().any():
        raise ValueError(f"cov_{band} contains nulls — coverage cannot be inferred (fail-closed)")
    vals = col.unique()
    if not all(isinstance(v, (bool, np.bool_)) for v in vals):
        raise ValueError(f"cov_{band} must be strictly boolean, found {vals[:4]!r}")
    return col.to_numpy(dtype=bool)


def bool_mask(x, n: int, name: str) -> np.ndarray:
    """Caller-supplied boolean mask, STRICTLY validated (r5g, review).

    ``np.asarray(x, dtype=bool)`` maps NaN and non-empty strings ("False"!)
    to True — a silent fail-open on external-product masks (PSF dominance,
    SED brown-dwarf flags). Rules: values must be exactly booleans; a pandas
    Series needs the trivial 0..n-1 index (else align and pass .to_numpy());
    length must match the table.
    """
    if isinstance(x, pd.Series):
        if not x.index.equals(pd.RangeIndex(n)):
            raise ValueError(
                f"{name} is a pandas Series with a non-trivial index — align it to the "
                "table first and pass .to_numpy()"
            )
        x = x.to_numpy()
    arr = np.asarray(x)
    flat = arr.ravel()
    if arr.shape != (n,) or not all(isinstance(v, (bool, np.bool_)) for v in flat):
        raise ValueError(
            f"{name} must be a length-{n} strictly-boolean mask (NaN/strings/numbers "
            "refuse rather than coerce)"
        )
    return arr.astype(bool)
