
import ast
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from redress import splits as sp

# ---------------------------------------------------------------- angular separation

def test_separation_matches_astropy_oracle():
    rng = np.random.default_rng(3)
    ra1 = rng.uniform(0, 360, 50)
    dec1 = rng.uniform(-89, 89, 50)
    ra2 = ra1 + rng.uniform(-0.01, 0.01, 50)
    dec2 = np.clip(dec1 + rng.uniform(-0.01, 0.01, 50), -90, 90)
    ours = sp.angular_sep_arcsec(ra1, dec1, ra2, dec2)
    oracle = SkyCoord(ra1 * u.deg, dec1 * u.deg).separation(SkyCoord(ra2 * u.deg, dec2 * u.deg)).arcsec
    np.testing.assert_allclose(ours, oracle, rtol=0, atol=1e-3)  # agree to 1 mas


def test_separation_cos_dec_factor():
    # 1 arcsec of RA at dec=60 is only 0.5 arcsec on the sky (cos 60 = 0.5).
    sep_eq = sp.angular_sep_arcsec(10.0, 0.0, 10.0 + 1 / 3600, 0.0)
    sep_60 = sp.angular_sep_arcsec(10.0, 60.0, 10.0 + 1 / 3600, 60.0)
    assert sep_eq == pytest.approx(1.0, rel=1e-6)
    assert sep_60 == pytest.approx(0.5, rel=1e-4)


def test_separation_ra_wraparound():
    # 0.72 arcsec apart across the RA=0 line; a naive RA difference says ~360 deg.
    sep = sp.angular_sep_arcsec(359.9999, 0.0, 0.0001, 0.0)
    assert sep == pytest.approx(0.72, rel=1e-3)


def test_separation_antipodal():
    assert sp.angular_sep_arcsec(0.0, 0.0, 180.0, 0.0) == pytest.approx(180 * 3600, rel=1e-9)


# ---------------------------------------------------------------- dedup grouping

def test_dedup_pairs_and_singletons():
    #  A and B are 0.2" apart; C is degrees away.
    ra = np.array([150.0, 150.0 + 0.2 / 3600, 151.0])
    dec = np.array([2.0, 2.0, 2.0])
    g = sp.dedup_groups(ra, dec, radius_arcsec=0.5)
    assert g[0] == g[1] and g[2] != g[0]


def test_dedup_is_friends_of_friends():
    # A-B within radius, B-C within radius, A-C outside: transitively ONE group.
    # (Documented choice: union-find = friends-of-friends linking, the standard
    # behavior for cross-match grouping; the test pins it so it can't drift.)
    ra = np.array([150.0, 150.0 + 0.4 / 3600, 150.0 + 0.8 / 3600])
    dec = np.array([2.0, 2.0, 2.0])
    g = sp.dedup_groups(ra, dec, radius_arcsec=0.5)
    assert g[0] == g[1] == g[2]


def test_dedup_across_ra_wrap():
    ra = np.array([359.99995, 0.00005])   # 0.36" apart across the wrap
    dec = np.array([0.0, 0.0])
    g = sp.dedup_groups(ra, dec, radius_arcsec=0.5)
    assert g[0] == g[1]


def test_dedup_deterministic_and_labels_are_canonical():
    rng = np.random.default_rng(11)
    ra = rng.uniform(10, 11, 200)
    dec = rng.uniform(-1, 1, 200)
    g1 = sp.dedup_groups(ra, dec, radius_arcsec=1.0)
    g2 = sp.dedup_groups(ra, dec, radius_arcsec=1.0)
    np.testing.assert_array_equal(g1, g2)
    # group label = lowest member index, so labels are stable and interpretable
    for label in np.unique(g1):
        assert label == np.min(np.where(g1 == label))


def test_dedup_rejects_bad_input():
    with pytest.raises(ValueError):
        sp.dedup_groups(np.array([1.0]), np.array([1.0, 2.0]), radius_arcsec=1.0)
    with pytest.raises(ValueError):
        sp.dedup_groups(np.array([1.0]), np.array([1.0]), radius_arcsec=0.0)
    with pytest.raises(ValueError):
        sp.dedup_groups(np.array([np.nan]), np.array([1.0]), radius_arcsec=1.0)


# ---------------------------------------------------------------- field splits

FIELDS = np.array(["ceers", "ceers", "goodss", "goodss", "goodsn", "cosmos"])


def test_holdout_masks_partition():
    train, test = sp.field_holdout_masks(FIELDS, holdout="goodss")
    np.testing.assert_array_equal(test, FIELDS == "goodss")
    np.testing.assert_array_equal(train, FIELDS != "goodss")
    assert not np.any(train & test)


def test_holdout_unknown_field_raises():
    with pytest.raises(ValueError):
        sp.field_holdout_masks(FIELDS, holdout="not-a-field")


def test_leave_one_field_out_covers_every_field_once():
    folds = list(sp.leave_one_field_out(FIELDS))
    held = [h for h, _, _ in folds]
    assert sorted(held) == sorted(np.unique(FIELDS).tolist())
    for holdout, train, test in folds:
        assert np.all(FIELDS[test] == holdout)
        assert np.all(FIELDS[train] != holdout)
        assert np.all(train | test)          # every object lands on exactly one side
        assert not np.any(train & test)


# ---------------------------------------------------------------- leakage guards

def test_group_leakage_detected():
    # objects 0 and 1 are the same physical source (one dedup group) but sit on
    # opposite sides of the split -> MUST raise.
    groups = np.array([0, 0, 2, 3])
    train = np.array([True, False, True, False])
    test = ~train
    with pytest.raises(sp.LeakageError, match="group"):
        sp.assert_no_group_leakage(groups, train, test)


def test_group_leakage_clean_split_passes():
    groups = np.array([0, 0, 2, 3])
    train = np.array([True, True, False, False])
    test = ~train
    sp.assert_no_group_leakage(groups, train, test)  # no raise


def test_overlapping_masks_rejected():
    groups = np.array([0, 1])
    both = np.array([True, True])
    with pytest.raises(sp.LeakageError, match="disjoint"):
        sp.assert_no_group_leakage(groups, both, both)


def test_forbidden_features_detected():
    # §8 row 4: spectrum-derived quantities must never enter the detector's
    # training matrix — the truth labels COME from spectra (circularity).
    names = ["f150w_ujy", "f444w_ujy", "z_spec", "beta_opt"]
    with pytest.raises(sp.LeakageError, match="z_spec"):
        sp.assert_no_forbidden_features(names, forbidden=("z_spec", "fwhm_ha", "spec_*"))


def test_forbidden_features_glob_patterns():
    names = ["f150w_ujy", "spec_snr"]
    with pytest.raises(sp.LeakageError, match="spec_snr"):
        sp.assert_no_forbidden_features(names, forbidden=("spec_*",))


def test_forbidden_features_clean_passes():
    sp.assert_no_forbidden_features(["f150w_ujy", "beta_opt"], forbidden=("z_spec", "spec_*"))


def test_forbidden_as_bare_string_rejected():
    # council round 3 deciding test: a bare string iterates as characters and
    # guards NOTHING — must be an explicit config error.
    with pytest.raises(TypeError, match="bare string"):
        sp.assert_no_forbidden_features(["spec_snr"], forbidden="spec_*")


# ---------------------------------------------------------------- D1 tooling boundary (design r14)

def test_dedup_groups_docstring_pins_the_tooling_boundary():
    # Design D1 (r14): the transitive FoF must declare itself
    # leakage-split-only so it can never be mistaken for the M0 counting
    # deduplicator (whose blend-aware linkage is a separate deliverable).
    doc = sp.dedup_groups.__doc__ or ""
    assert "LEAKAGE-SPLIT TOOL ONLY" in doc
    assert "M0" in doc


def _fof_usage_offences(source: str) -> list[str]:
    """AST-level detector for dedup_groups usage (design D1 r14; FIXED r15 —
    The review: the r14 guard was overbroad, rejecting ANY from-splits import,
    and bypassable via `from redress import splits; splits.dedup_groups(...)`.
    The rule is dedup_groups-SPECIFIC: direct from-import of the symbol, or
    attribute access to it through any module binding. Safe symbols
    (angular_sep_arcsec, leakage guards) import freely."""
    offences: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[-1] == "splits":
                for alias in node.names:
                    if alias.name == "dedup_groups":
                        offences.append(f"from {node.module} import dedup_groups")
        elif isinstance(node, ast.Attribute) and node.attr == "dedup_groups":
            offences.append("attribute access .dedup_groups")
    return offences


def test_fof_guard_fixture_indirect_module_access_fails():
    # Review round r15's named check: the module-binding route must FAIL (the r14
    # guard missed it).
    src = (
        "from redress import splits\n"
        "def f(ra, dec):\n"
        "    return splits.dedup_groups(ra, dec, 0.5)\n"
    )
    assert _fof_usage_offences(src)


def test_fof_guard_fixture_safe_symbol_import_passes():
    # Review round r15's named check: an unrelated safe symbol from splits must
    # PASS — the rule is dedup_groups-specific, not module-wide.
    src = (
        "from redress.splits import angular_sep_arcsec\n"
        "x = angular_sep_arcsec(1.0, 2.0, 3.0, 4.0)\n"
    )
    assert _fof_usage_offences(src) == []


def test_fof_guard_fixture_direct_import_fails():
    src = "from redress.splits import dedup_groups\n"
    assert _fof_usage_offences(src)


def test_no_src_module_uses_the_fof_dedup():
    # Design D1 (r14/r15): EMPTY allowlist — no package module may reach the FoF
    # dedup by ANY route. Docstring mentions (calibrate.py documents the
    # dedup-group id namespace) never false-positive: strings are not AST
    # Attribute nodes.
    src_root = Path(__file__).resolve().parents[1] / "redress"
    offenders: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        if path.name == "splits.py":
            continue
        for offence in _fof_usage_offences(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.name}: {offence}")
    assert offenders == [], f"package modules using the FoF dedup (forbidden, design D1): {offenders}"
