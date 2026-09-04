"""Field-level splits, cross-field dedup, and loud leakage guards.

Why this exists (build plan, rows 3-4, M1): with ~370 positives, the two classic
self-deceptions are (a) the same physical object appearing on both sides of a
train/test split via overlapping catalogs, and (b) spectrum-derived information
leaking into the photometric detector whose truth labels COME from spectra.
The rules: deduplicate by sky position BEFORE splitting, split BY FIELD (train on
N-1 fields, evaluate on the held-out field — the ExoMiner "split by object, never
by signal" pattern at field granularity), and make every violation a raised
``LeakageError``, never a warning.

Geometry: positions become unit vectors on the sphere; neighbor search uses a
cKDTree on 3-D chord distance (``chord = 2 sin(theta/2)``), which is exact and
immune to RA wraparound and pole pathologies. Our haversine separation is tested
against astropy's SkyCoord to milliarcsecond agreement.

Dedup grouping is friends-of-friends (union-find): if A~B and B~C, then {A,B,C}
form one group even when A and C are farther than the radius — pinned by test so
the behavior can't drift silently. Canonical group label = lowest member index.
WHICH duplicate row to keep (deeper field? better SNR?) is ingest-layer POLICY;
this module only guarantees the grouping and the guard rails.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterator

import numpy as np
from scipy.spatial import cKDTree


class LeakageError(Exception):
    """A split or feature set violates the §8 leakage rules. Always fatal."""


# ---------------------------------------------------------------- geometry

def _check_radec(ra, dec) -> tuple[np.ndarray, np.ndarray]:
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    if ra.shape != dec.shape or ra.ndim != 1:
        raise ValueError("ra and dec must be 1-D arrays of equal length")
    if np.any(~np.isfinite(ra)) or np.any(~np.isfinite(dec)):
        raise ValueError("non-finite coordinates: fix or drop rows at ingest, not here")
    if np.any(np.abs(dec) > 90):
        raise ValueError("dec outside [-90, 90] degrees")
    return ra, dec


def angular_sep_arcsec(ra1, dec1, ra2, dec2):
    """Great-circle separation in arcsec (haversine — stable at small angles).

    ``theta = 2 asin(sqrt(sin^2(dDec/2) + cos dec1 cos dec2 sin^2(dRA/2)))``
    """
    ra1, dec1, ra2, dec2 = (np.radians(np.asarray(x, dtype=float)) for x in (ra1, dec1, ra2, dec2))
    s_dec = np.sin((dec2 - dec1) / 2.0) ** 2
    s_ra = np.sin((ra2 - ra1) / 2.0) ** 2
    h = s_dec + np.cos(dec1) * np.cos(dec2) * s_ra
    theta = 2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    out = np.degrees(theta) * 3600.0
    return out.item() if np.ndim(out) == 0 else out


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.column_stack(
        (np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec))
    )


# ---------------------------------------------------------------- dedup

def dedup_groups(ra, dec, radius_arcsec: float) -> np.ndarray:
    """Friends-of-friends group id per object (label = lowest member index).

    Pairs closer than ``radius_arcsec`` are linked; links are transitive.
    Chord-distance KD-tree on unit vectors: exact, wrap-safe, O(N log N).

    LEAKAGE-SPLIT TOOL ONLY (design D1 tooling boundary, r14): transitive
    FoF deliberately OVER-merges, which is conservative-safe for holdout
    splits (over-merging only makes splits stricter) and WRONG for M0
    counting, where one low-resolution bridge row would chain two resolved
    sources into one family. The M0 deduplicator is a SEPARATE blend-aware
    implementation (contract-v2 deliverable, design D1); an AST-level
    import guard in tests/test_splits.py fails any src module that imports
    this one.
    """
    ra, dec = _check_radec(ra, dec)
    if radius_arcsec <= 0:
        raise ValueError(f"radius_arcsec must be positive, got {radius_arcsec}")
    xyz = _unit_vectors(ra, dec)
    chord = 2.0 * np.sin(np.radians(radius_arcsec / 3600.0) / 2.0)
    pairs = cKDTree(xyz).query_pairs(r=chord, output_type="ndarray")

    parent = np.arange(ra.size)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    for a, b in pairs:
        ra_, rb_ = find(int(a)), find(int(b))
        if ra_ != rb_:
            # union by smaller root so the final label is the lowest member index
            lo, hi = (ra_, rb_) if ra_ < rb_ else (rb_, ra_)
            parent[hi] = lo

    return np.array([find(i) for i in range(ra.size)])


# ---------------------------------------------------------------- field splits

def _check_fields(fields) -> np.ndarray:
    f = np.asarray(fields)
    if f.ndim != 1 or f.size == 0:
        raise ValueError("fields must be a non-empty 1-D array of field labels")
    return f


def field_holdout_masks(fields, holdout: str) -> tuple[np.ndarray, np.ndarray]:
    """Boolean ``(train_mask, test_mask)``: test = the held-out field, train = the rest."""
    f = _check_fields(fields)
    if holdout not in f:
        raise ValueError(f"holdout field {holdout!r} not present in fields")
    test = f == holdout
    return ~test, test


def leave_one_field_out(fields) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(holdout_field, train_mask, test_mask)`` for every distinct field."""
    f = _check_fields(fields)
    for holdout in np.unique(f):
        train, test = field_holdout_masks(f, str(holdout))
        yield str(holdout), train, test


# ---------------------------------------------------------------- leakage guards

def assert_no_group_leakage(group_ids, train_mask, test_mask) -> None:
    """Raise ``LeakageError`` if any dedup group has members on BOTH sides.

    This is the automated §8-row-4 test: after dedup-then-split, a physical
    object (= one friends-of-friends group) must live entirely in train or
    entirely in test.
    """
    g = np.asarray(group_ids)
    train = np.asarray(train_mask, dtype=bool)
    test = np.asarray(test_mask, dtype=bool)
    if not (g.shape == train.shape == test.shape) or g.ndim != 1:
        raise ValueError("group_ids, train_mask, test_mask must be 1-D and equal length")
    if np.any(train & test):
        raise LeakageError("train and test masks are not disjoint")
    leaky = np.intersect1d(np.unique(g[train]), np.unique(g[test]))
    if leaky.size:
        raise LeakageError(
            f"{leaky.size} dedup group(s) span both sides of the split "
            f"(first offenders: {leaky[:5].tolist()}) — same physical object in train AND test"
        )


def assert_no_forbidden_features(feature_names, forbidden) -> None:
    """Raise ``LeakageError`` if any feature name matches a forbidden pattern.

    ``forbidden`` entries are exact names or ``fnmatch`` globs (e.g. ``spec_*``).
    The actual forbidden list for the detector lives in DATA_CONTRACTS; this is
    the mechanism that enforces it wherever a training matrix is assembled.
    """
    if isinstance(forbidden, (str, bytes)):
        # council round 3: a bare string iterates as CHARACTERS, silently matching
        # nothing — the guard would pass while guarding nothing.
        raise TypeError("forbidden must be an iterable of patterns, not a bare string")
    names = list(feature_names)
    hits = [
        name
        for name in names
        if any(name == pat or fnmatch.fnmatch(name, pat) for pat in forbidden)
    ]
    if hits:
        raise LeakageError(
            f"forbidden (spectrum-derived) features in training matrix: {hits} — "
            "truth labels come from spectra; using them as inputs is circular"
        )
