"""Machine enforcement of the frozen data contracts (schema version 1).

Nothing in the pipeline consumes a table that has not passed these validators,
and Worker X's replication package builds against the DOC alone — so the doc and
this module are tested against each other: every violation class the doc names
must raise ``ContractError`` here, with the offending column/row named.

Design choice: validators COLLECT all violations and raise once with the full
list, so a broken ingest shows every problem in one round instead of
whack-a-mole. Sentinel policy: ingest converts survey sentinels; anything that
still looks like one here (|flux| in the classic -99/-999 family) is a hard
error, because a sentinel that survives to modeling poisons everything after it.
"""

from __future__ import annotations

import pandas as pd

SCHEMA_VERSION = 1

#: Canonical band keys (DATA_CONTRACTS §3). Pivot wavelengths are deliberately
#: NOT here — filter curves get provenance-locked at W1, pivots computed in-repo.
BANDS_NIRCAM = ("f090w", "f115w", "f150w", "f200w", "f277w", "f356w", "f410m", "f444w")
BANDS_MIRI = ("f770w", "f1800w")
BANDS_ACS = ("f435w", "f606w", "f814w")
BANDS_WFC3 = ("f105w", "f125w", "f140w", "f160w")
ALL_BANDS = BANDS_NIRCAM + BANDS_MIRI + BANDS_ACS + BANDS_WFC3

#: Canonical field slugs (DATA_CONTRACTS §4). `cosmos-web` is deployment-only.
FIELD_SLUGS = frozenset(
    {"ceers", "primer-cosmos", "primer-uds", "goods-n", "goods-s", "egs-mega", "cosmos-web"}
)

#: DATA_CONTRACTS §5 — the §8-row-4 leakage fence, consumed by
#: ``splits.assert_no_forbidden_features`` wherever a training matrix is built.
FORBIDDEN_DETECTOR_FEATURES = (
    "z_spec",
    "label_*",
    "targeting_*",
    "grism_*",
    "spec_*",
    "fwhm_*",
    "ew_*",
    "line_*",
)

_REQUIRED_PHOT_COLUMNS = ("object_id", "field", "ra_deg", "dec_deg", "source_catalog")
_REQUIRED_LABEL_COLUMNS = (
    "field",
    "object_id",
    "label_lrd",
    "label_source",
    "z_spec",
    "targeting_program",
    "grism_unbiased",
)
#: The classic survey sentinels; ingest must have converted these to NaN+cov.
_SENTINELS = (-99.0, -999.0, 99.0)


class ContractError(Exception):
    """A table violates DATA_CONTRACTS.md. Message lists every violation found."""


def _raise_if(violations: list[str], table_name: str) -> None:
    if violations:
        msg = f"{table_name} violates DATA_CONTRACTS v{SCHEMA_VERSION} ({len(violations)} problem(s)):\n"
        raise ContractError(msg + "\n".join(f"  - {v}" for v in violations))


def bands_in(df: pd.DataFrame) -> list[str]:
    """Band keys that appear in the table (via their ``f_<band>_ujy`` columns)."""
    return [c[len("f_") : -len("_ujy")] for c in df.columns if c.startswith("f_") and c.endswith("_ujy")]


def validate_photometry_table(df: pd.DataFrame, allowed_fields: set[str] | None = None) -> None:
    """Raise ``ContractError`` unless ``df`` satisfies DATA_CONTRACTS §1 exactly."""
    v: list[str] = []
    allowed = FIELD_SLUGS if allowed_fields is None else set(allowed_fields)

    for col in _REQUIRED_PHOT_COLUMNS:
        if col not in df.columns:
            v.append(f"missing required column '{col}'")
    if v:
        _raise_if(v, "photometry table")  # identity columns gate everything else

    bands = bands_in(df)
    if not bands:
        v.append("no photometry band columns at all (need at least one f_<band>_ujy)")
    for band in bands:
        if band not in ALL_BANDS:
            v.append(f"unknown band '{band}' (not in the DATA_CONTRACTS §3 registry)")
        for col in (f"e_{band}_ujy", f"cov_{band}"):
            if col not in df.columns:
                v.append(f"band '{band}' incomplete: missing '{col}' (need f/e/cov triplet)")
    if v:
        _raise_if(v, "photometry table")

    # identity + geometry
    if df["object_id"].isna().any():
        v.append("object_id contains NaN")
    dup = df.duplicated(subset=["field", "object_id"])
    if dup.any():
        v.append(f"duplicate (field, object_id) rows: {df.loc[dup, 'object_id'].tolist()[:5]}")
    bad_field = ~df["field"].isin(allowed)
    if bad_field.any():
        v.append(f"unknown field slug(s): {sorted(df.loc[bad_field, 'field'].unique().tolist())}")
    ra = pd.to_numeric(df["ra_deg"], errors="coerce")
    dec = pd.to_numeric(df["dec_deg"], errors="coerce")
    if ra.isna().any() or ((ra < 0) | (ra >= 360)).any():
        v.append("ra_deg has NaN or values outside [0, 360)")
    if dec.isna().any() or ((dec < -90) | (dec > 90)).any():
        v.append("dec_deg has NaN or values outside [-90, 90]")

    # per-band: the one missingness rule + error positivity + sentinel scan
    for band in bands:
        f = pd.to_numeric(df[f"f_{band}_ujy"], errors="coerce")
        e = pd.to_numeric(df[f"e_{band}_ujy"], errors="coerce")
        if not pd.api.types.is_bool_dtype(df[f"cov_{band}"]):
            # council r4b: astype(bool) coerces the STRING "False" to True — a
            # silent flip from uncovered to covered. Require a real bool dtype.
            v.append(f"cov_{band} must be bool dtype (got {df[f'cov_{band}'].dtype})")
            continue
        cov = df[f"cov_{band}"]
        if (cov & (f.isna() | e.isna())).any():
            v.append(f"cov_{band}=True but flux/error NaN (covered means measured)")
        if (~cov & (f.notna() | e.notna())).any():
            v.append(f"cov_{band}=False but flux/error present (uncovered means NaN)")
        if (cov & e.notna() & (e <= 0)).any():
            v.append(f"e_{band}_ujy has non-positive values where covered")
        if f.isin(_SENTINELS).any():
            v.append(f"f_{band}_ujy contains sentinel values ({_SENTINELS}); ingest must convert them")

    _raise_if(v, "photometry table")


def validate_label_table(df: pd.DataFrame, allowed_fields: set[str] | None = None) -> None:
    """Raise ``ContractError`` unless ``df`` satisfies DATA_CONTRACTS §2 exactly."""
    v: list[str] = []
    allowed = FIELD_SLUGS if allowed_fields is None else set(allowed_fields)
    for col in _REQUIRED_LABEL_COLUMNS:
        if col not in df.columns:
            v.append(f"missing required column '{col}'")
    if v:
        _raise_if(v, "label table")

    # council r4b: join keys must be NaN-free (a NaN key poisons every join and
    # leakage guard downstream), and label fields obey the same canonical
    # registry photometry does.
    if df["field"].isna().any() or df["object_id"].isna().any():
        v.append("join keys (field, object_id) contain NaN")
    else:
        bad_field = ~df["field"].isin(allowed)
        if bad_field.any():
            v.append(f"unknown field slug(s): {sorted(df.loc[bad_field, 'field'].unique().tolist())}")
    dup = df.duplicated(subset=["field", "object_id"])
    if dup.any():
        v.append(f"duplicate (field, object_id) rows: {df.loc[dup, 'object_id'].tolist()[:5]}")
    if not df["label_lrd"].isin([1, 0, -1]).all():
        bad = sorted(set(df["label_lrd"].unique()) - {1, 0, -1})
        v.append(f"label_lrd outside {{1, 0, -1}}: {bad}")
    # council r4b: a non-numeric z_spec must FAIL, not silently coerce to
    # "unknown redshift" — NaN means "no measurement", never "bad entry".
    z = pd.to_numeric(df["z_spec"], errors="coerce")
    if (df["z_spec"].notna() & z.isna()).any():
        v.append("z_spec contains non-numeric entries (bad data must raise, not coerce to NaN)")
    if (z.notna() & (z < 0)).any():
        v.append("z_spec has negative values")
    if not pd.api.types.is_bool_dtype(df["grism_unbiased"]):
        v.append(f"grism_unbiased must be bool dtype (got {df['grism_unbiased'].dtype})")
    _raise_if(v, "label table")
