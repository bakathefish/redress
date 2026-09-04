"""Tests for the fold machinery, written before the module that satisfies them.

fold machinery: sky-position grouping, outer leave-one-field-out, inner grouped k-fold
(THE_PLAN_2026-08-30.md S1).

WHAT IS BEING PINNED, and why each pin is here rather than left to inspection:

* GEOMETRY. Sky grouping is a great-circle question, and the two ways to get it wrong
  are both silent. A flat approximation that forgets ``cos(dec)`` calls two objects at
  dec=+60 twice as far apart as they are; one that forgets RA wraparound calls two
  objects across the RA=0 meridian 360 degrees apart. Both are pinned by construction
  below, with the separations verified against a formula written independently of the
  one the module uses (cross product / dot product via ``atan2``, against the module's
  chord-distance KD-tree).

* NO TRUNCATION. Review item 9 refuted the proof behind the pipeline's top-20,000 head
  bound on dedup. A grouping that quietly stops after N rows produces a candidate list
  whose tail was never deduplicated at all, so the no-cap property is pinned with a
  planted duplicate pair sitting at the very END of a 25,601-row input.

* PERMUTATION INVARIANCE. Every number this project reports is downstream of these
  folds. If the fold assignment depends on the order rows happened to arrive in, then
  so does the headline, and no reviewer can reproduce it. Pinned end to end: shuffle the
  rows, re-run, map back, demand the identical partition.

* GROUP INTEGRITY. The reason for grouping at all is that overlapping tiles put the same
  physical object in more than one catalogue row. A group split across a train/test
  boundary is the same object on both sides, which is leakage that inflates every score.
"""

from __future__ import annotations

import numpy as np
import pytest

from redress import folds

ARCSEC = 1.0 / 3600.0


# ---------------------------------------------------------------- independent geometry


def _unit(ra_deg, dec_deg) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    return np.stack(
        (np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)), -1
    )


def true_sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    """Great-circle separation in arcsec, by a DIFFERENT formula than the module's.

    ``theta = atan2(|v1 x v2|, v1 . v2)`` on unit vectors. The module under test reaches
    the same angle through a chord-distance KD-tree; agreement between two independent
    derivations is the point, so this helper must never be refactored into a call into
    ``redress.folds``. ``atan2`` of the cross-product norm stays well conditioned at the
    sub-arcsecond angles that decide every assertion here, where ``arccos`` would not.
    """
    v1, v2 = _unit(ra1, dec1), _unit(ra2, dec2)
    return float(
        np.degrees(np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2)))
        * 3600.0
    )


def delta_ra_for_separation(dec_deg: float, sep_arcsec: float) -> float:
    """The RA offset (degrees) that puts two points at declination ``dec_deg`` exactly
    ``sep_arcsec`` apart on the sky. Inverts the haversine at equal declination:
    ``sin(theta/2) = cos(dec) * sin(dRA/2)``."""
    theta = np.radians(sep_arcsec * ARCSEC)
    return float(
        np.degrees(2.0 * np.arcsin(np.sin(theta / 2.0) / np.cos(np.radians(dec_deg))))
    )


# ---------------------------------------------------------------- partition helpers


def partition_over(labels, back) -> frozenset:
    """The INDUCED PARTITION as a frozenset of frozensets of ORIGINAL row ids.

    ``back[i]`` is the original row now sitting at position ``i``. Comparing partitions
    rather than label arrays is the whole discipline: label VALUES are free to differ
    between runs, the partition is not.
    """
    groups: dict[int, set[int]] = {}
    for position, label in enumerate(np.asarray(labels)):
        groups.setdefault(int(label), set()).add(int(back[position]))
    return frozenset(frozenset(members) for members in groups.values())


def fold_sets(fold_list, back) -> list[frozenset]:
    """Each fold's TEST rows, expressed as original row ids, in fold order."""
    return [frozenset(int(back[i]) for i in test) for _, test in fold_list]


def planted_field(
    seed: int = 20260830, cells: tuple[int, int] = (15, 12), n_dup: int = 20
):
    """200 pseudo-random sources: 180 distinct sky positions plus 20 planted duplicates
    of them, which is the overlapping-tile situation the grouping exists for.

    The 180 bases are drawn uniformly INSIDE the cells of a 10 arcsec grid, jittered by
    at most 3 arcsec, so no two of them can land closer than 4 arcsec. That makes "180
    groups before the duplicates" a structural fact rather than a lucky draw: a plain
    uniform scatter over the same box gave a chance 0.5 arcsec pair on the first seed
    tried, and a fixture whose expected value depends on the seed teaches nothing when
    it fails. Each duplicate sits within 0.22 arcsec of its parent (both offsets bounded
    by 0.15 arcsec), so it must merge with that parent and can reach no other source.
    """
    n_ra, n_dec = cells
    n_base = n_ra * n_dec
    rng = np.random.default_rng(seed)
    step, jitter = 10.0 * ARCSEC, 3.0 * ARCSEC
    dec = (
        2.0
        + np.tile(np.arange(n_dec), n_ra) * step
        + rng.uniform(-jitter, jitter, n_base)
    )
    ra = 150.0 + (
        np.repeat(np.arange(n_ra), n_dec) * step + rng.uniform(-jitter, jitter, n_base)
    ) / np.cos(np.radians(dec))

    picks = rng.choice(n_base, size=n_dup, replace=False)
    nudge = 0.15 * ARCSEC
    dup_dec = dec[picks] + rng.uniform(-nudge, nudge, n_dup)
    dup_ra = ra[picks] + rng.uniform(-nudge, nudge, n_dup) / np.cos(
        np.radians(dec[picks])
    )
    return np.concatenate([ra, dup_ra]), np.concatenate([dec, dup_dec]), picks


# ================================================================ sky_groups: radius


def test_a_pair_inside_the_radius_shares_a_group():
    ra = np.array([150.0, 150.0])
    dec = np.array([2.0, 2.0 + 0.2 * ARCSEC])
    assert true_sep_arcsec(ra[0], dec[0], ra[1], dec[1]) == pytest.approx(0.2, abs=1e-6)
    labels = folds.sky_groups(ra, dec, radius_arcsec=0.5)
    assert labels[0] == labels[1]


def test_a_pair_outside_the_radius_does_not_share_a_group():
    ra = np.array([150.0, 150.0])
    dec = np.array([2.0, 2.0 + 1.0 * ARCSEC])
    assert true_sep_arcsec(ra[0], dec[0], ra[1], dec[1]) == pytest.approx(1.0, abs=1e-6)
    labels = folds.sky_groups(ra, dec, radius_arcsec=0.5)
    assert labels[0] != labels[1]


def test_friendship_is_transitive_along_a_chain():
    """A-B 0.4", B-C 0.4", A-C 0.8" is ONE group. Transitivity over-merges by design:
    for a held-out split, over-merging can only make the split stricter, while a missed
    link puts one physical object on both sides."""
    ra = np.array([150.0, 150.0, 150.0])
    dec = np.array([2.0, 2.0 + 0.4 * ARCSEC, 2.0 + 0.8 * ARCSEC])
    assert true_sep_arcsec(ra[0], dec[0], ra[1], dec[1]) == pytest.approx(0.4, abs=1e-6)
    assert true_sep_arcsec(ra[1], dec[1], ra[2], dec[2]) == pytest.approx(0.4, abs=1e-6)
    assert true_sep_arcsec(ra[0], dec[0], ra[2], dec[2]) == pytest.approx(0.8, abs=1e-6)
    labels = folds.sky_groups(ra, dec, radius_arcsec=0.5)
    assert labels[0] == labels[1] == labels[2]
    assert np.unique(labels).size == 1


# ================================================================ sky_groups: cos(dec)


def test_the_same_ra_offset_merges_at_dec_60_and_splits_at_dec_0():
    """The cos(dec) pin, both arms from ONE RA offset.

    At dec=+60 an RA offset of ~0.8 arcsec spans 0.4 arcsec of sky; at dec=0 the same
    offset spans the full ~0.8 arcsec. A grouping that compares raw RA differences would
    answer identically at both declinations and be wrong at one of them.
    """
    d_ra = delta_ra_for_separation(60.0, 0.4)
    assert d_ra == pytest.approx(0.8 * ARCSEC, rel=1e-6)

    hi_ra, hi_dec = np.array([150.0, 150.0 + d_ra]), np.array([60.0, 60.0])
    assert true_sep_arcsec(hi_ra[0], hi_dec[0], hi_ra[1], hi_dec[1]) == pytest.approx(
        0.4, abs=1e-6
    )
    hi = folds.sky_groups(hi_ra, hi_dec, radius_arcsec=0.5)
    assert hi[0] == hi[1], "0.4 arcsec at dec=+60 must merge"

    lo_ra, lo_dec = np.array([150.0, 150.0 + d_ra]), np.array([0.0, 0.0])
    assert true_sep_arcsec(lo_ra[0], lo_dec[0], lo_ra[1], lo_dec[1]) == pytest.approx(
        0.8, abs=1e-6
    )
    lo = folds.sky_groups(lo_ra, lo_dec, radius_arcsec=0.5)
    assert lo[0] != lo[1], (
        "the SAME RA offset at dec=0 is 0.8 arcsec and must not merge"
    )


def test_ra_wraparound_does_not_split_a_pair_across_the_meridian():
    """359.99995 and 0.00005 are 0.36 arcsec apart, not 360 degrees."""
    ra = np.array([359.99995, 0.00005])
    dec = np.array([0.0, 0.0])
    assert true_sep_arcsec(ra[0], dec[0], ra[1], dec[1]) == pytest.approx(
        0.36, abs=1e-6
    )
    labels = folds.sky_groups(ra, dec, radius_arcsec=0.5)
    assert labels[0] == labels[1]


def test_a_pair_straddling_the_pole_merges():
    """Two points 0.18 arcsec from the pole at opposite RA are 0.36 arcsec apart. The
    flat cos(dec) approximation gets this pair wrong (it reads ~0.56 arcsec and refuses
    to merge), which is why the module works in three dimensions instead."""
    ra = np.array([10.0, 190.0])
    dec = np.array([90.0 - 0.18 * ARCSEC, 90.0 - 0.18 * ARCSEC])
    assert true_sep_arcsec(ra[0], dec[0], ra[1], dec[1]) == pytest.approx(
        0.36, abs=1e-6
    )
    labels = folds.sky_groups(ra, dec, radius_arcsec=0.5)
    assert labels[0] == labels[1]


# ================================================================ sky_groups: no cap


def test_no_head_bound_a_duplicate_at_row_25600_still_merges():
    """Review item 9: the pipeline's top-20,000 head bound was refuted, so the tail must
    be deduplicated exactly like the head. 25,600 grid sources at 5 arcsec spacing (no
    chance collisions at 0.5) plus ONE planted duplicate at the very last row."""
    side = 160
    grid = np.arange(side) * (5.0 * ARCSEC)
    ra = 150.0 + np.repeat(grid, side)
    dec = 2.0 + np.tile(grid, side)
    ra = np.append(ra, ra[-1] + 0.2 * ARCSEC)
    dec = np.append(dec, dec[-1])
    assert ra.size == side * side + 1 > 20_000

    labels = folds.sky_groups(ra, dec, radius_arcsec=0.5)
    assert labels[-1] == labels[-2], "the duplicate past row 20,000 was not merged"
    assert np.unique(labels).size == side * side


# ================================================================ sky_groups: invariance


def test_the_induced_partition_survives_a_row_shuffle():
    ra, dec, picks = planted_field()
    n = ra.size
    base = folds.sky_groups(ra, dec, radius_arcsec=0.5)

    for offset, parent in enumerate(picks):
        assert base[180 + offset] == base[parent], "a planted duplicate failed to merge"
    assert np.unique(base).size == 180

    rng = np.random.default_rng(7)
    for _ in range(5):
        perm = rng.permutation(n)
        shuffled = folds.sky_groups(ra[perm], dec[perm], radius_arcsec=0.5)
        assert partition_over(shuffled, perm) == partition_over(base, np.arange(n))


def test_the_labels_themselves_are_canonical_under_a_row_shuffle():
    """Stronger than the partition rule the API promises, and relied on downstream: the
    label VALUES are permutation-equivariant, because groups are numbered by their
    lexicographically smallest (RA, Dec) member rather than by row index. Without this,
    shuffle-then-group-then-fold would reshuffle the folds."""
    ra, dec, _ = planted_field()
    base = folds.sky_groups(ra, dec, radius_arcsec=0.5)
    perm = np.random.default_rng(11).permutation(ra.size)
    assert np.array_equal(
        folds.sky_groups(ra[perm], dec[perm], radius_arcsec=0.5), base[perm]
    )


# ================================================================ sky_groups: guards


def test_sky_groups_rejects_bad_input():
    with pytest.raises(ValueError):
        folds.sky_groups([150.0, 150.1], [2.0], radius_arcsec=0.5)
    with pytest.raises(ValueError):
        folds.sky_groups([150.0, np.nan], [2.0, 2.0], radius_arcsec=0.5)
    with pytest.raises(ValueError):
        folds.sky_groups([150.0, 150.1], [2.0, 91.0], radius_arcsec=0.5)
    with pytest.raises(ValueError):
        folds.sky_groups([150.0, 150.1], [2.0, 2.0], radius_arcsec=0.0)
    with pytest.raises(ValueError):
        folds.sky_groups([[150.0]], [[2.0]], radius_arcsec=0.5)


def test_singletons_are_numbered_in_canonical_sky_order():
    ra = np.array([150.3, 150.1, 150.2])
    dec = np.array([2.0, 2.0, 2.0])
    assert folds.sky_groups(ra, dec).tolist() == [2, 0, 1]


# ================================================================ outer_lofo

FIELDS = np.array(["uds", "cosmos", "gdn", "cosmos", "uds", "cosmos", "gds"])


def test_outer_lofo_yields_one_fold_per_field_in_sorted_order():
    result = folds.outer_lofo(FIELDS)
    assert len(result) == 4
    held = [str(FIELDS[test[0]]) for _, test in result]
    assert held == ["cosmos", "gdn", "gds", "uds"] == sorted(set(FIELDS.tolist()))


def test_outer_lofo_test_folds_are_field_pure():
    for _, test in folds.outer_lofo(FIELDS):
        assert np.unique(FIELDS[test]).size == 1


def test_outer_lofo_covers_every_row_exactly_once():
    seen = np.concatenate([test for _, test in folds.outer_lofo(FIELDS)])
    assert np.array_equal(np.sort(seen), np.arange(FIELDS.size))


def test_outer_lofo_train_is_exactly_the_complement():
    for train, test in folds.outer_lofo(FIELDS):
        assert set(train.tolist()) | set(test.tolist()) == set(range(FIELDS.size))
        assert not set(train.tolist()) & set(test.tolist())
        assert np.array_equal(train, np.sort(train))
        assert train.dtype.kind == "i" and test.dtype.kind == "i"


def test_outer_lofo_handles_a_single_field_and_rejects_empties():
    single = folds.outer_lofo(np.array(["uds", "uds"]))
    assert len(single) == 1
    train, test = single[0]
    assert test.tolist() == [0, 1] and train.tolist() == []
    with pytest.raises(ValueError):
        folds.outer_lofo(np.array([], dtype=object))
    with pytest.raises(ValueError):
        folds.outer_lofo(np.array([["a"], ["b"]]))


# ================================================================ inner_grouped_kfold

GROUPS = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 9, 9, 9])


def test_inner_grouped_kfold_never_splits_a_group():
    for train, test in folds.inner_grouped_kfold(GROUPS, n_splits=4, seed=20260830):
        assert not set(GROUPS[train].tolist()) & set(GROUPS[test].tolist())


def test_inner_grouped_kfold_covers_every_row_exactly_once():
    result = folds.inner_grouped_kfold(GROUPS, n_splits=4, seed=20260830)
    assert len(result) == 4
    seen = np.concatenate([test for _, test in result])
    assert np.array_equal(np.sort(seen), np.arange(GROUPS.size))
    for train, test in result:
        assert test.size > 0
        assert np.array_equal(
            np.sort(np.concatenate([train, test])), np.arange(GROUPS.size)
        )


def test_inner_grouped_kfold_is_deterministic_for_a_fixed_seed():
    a = folds.inner_grouped_kfold(GROUPS, n_splits=4, seed=20260830)
    b = folds.inner_grouped_kfold(GROUPS, n_splits=4, seed=20260830)
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b, strict=True):
        assert np.array_equal(tr_a, tr_b) and np.array_equal(te_a, te_b)


def test_the_seed_actually_moves_the_tie_break():
    """Twelve singleton groups into four folds is all ties, so the seed is the only
    thing deciding the assignment. If two seeds could not disagree here, the seed would
    be decoration and repeated CV would repeat the same split."""
    singletons = np.arange(12)
    a = fold_sets(folds.inner_grouped_kfold(singletons, 4, seed=1), np.arange(12))
    b = fold_sets(folds.inner_grouped_kfold(singletons, 4, seed=2), np.arange(12))
    assert a != b


def test_inner_grouped_kfold_balances_by_placing_the_largest_group_first():
    """Descending group size onto the currently smallest fold. With sizes 5, 4, 3, 2 into
    two folds the greedy answer is {5, 2} and {4, 3} — seven rows each."""
    sized = np.array([0] * 5 + [1] * 4 + [2] * 3 + [3] * 2)
    result = folds.inner_grouped_kfold(sized, n_splits=2, seed=20260830)
    assert sorted(test.size for _, test in result) == [7, 7]


def test_inner_grouped_kfold_row_sets_survive_a_row_shuffle():
    ra, dec, _ = planted_field()
    n = ra.size
    group_ids = folds.sky_groups(ra, dec, radius_arcsec=0.5)
    base = folds.inner_grouped_kfold(group_ids, n_splits=5, seed=20260830)

    rng = np.random.default_rng(13)
    for _ in range(5):
        perm = rng.permutation(n)
        shuffled_groups = folds.sky_groups(ra[perm], dec[perm], radius_arcsec=0.5)
        shuffled = folds.inner_grouped_kfold(shuffled_groups, n_splits=5, seed=20260830)
        assert fold_sets(shuffled, perm) == fold_sets(base, np.arange(n))


def test_inner_grouped_kfold_accepts_non_integer_group_keys():
    keys = np.array(["g-c", "g-a", "g-a", "g-b", "g-c", "g-b"])
    for train, test in folds.inner_grouped_kfold(keys, n_splits=3, seed=5):
        assert not set(keys[train].tolist()) & set(keys[test].tolist())


def test_inner_grouped_kfold_rejects_impossible_split_counts():
    with pytest.raises(ValueError):
        folds.inner_grouped_kfold(GROUPS, n_splits=1, seed=0)
    with pytest.raises(ValueError):
        folds.inner_grouped_kfold(np.array([0, 0, 1]), n_splits=3, seed=0)
    with pytest.raises(ValueError):
        folds.inner_grouped_kfold(np.array([], dtype=int), n_splits=2, seed=0)
    with pytest.raises(ValueError):
        folds.inner_grouped_kfold(np.array([[0, 1], [1, 0]]), n_splits=2, seed=0)


# ================================================================ exclude_groups


def test_exclude_groups_removes_exactly_the_intended_rows():
    group_ids = np.array([0, 0, 1, 1, 2, 2, 3])
    train_idx = np.array([0, 1, 2, 4, 5, 6])
    kept = folds.exclude_groups(train_idx, group_ids, [1, 3])
    assert kept.tolist() == [0, 1, 4, 5]


def test_exclude_groups_preserves_order_and_never_mutates_its_input():
    group_ids = np.array([0, 1, 0, 1, 0])
    train_idx = np.array([4, 0, 3, 2, 1])
    kept = folds.exclude_groups(train_idx, group_ids, [1])
    # rows 1 and 3 are the group-1 rows; both go, and the survivors keep their order
    assert kept.tolist() == [4, 0, 2]
    assert train_idx.tolist() == [4, 0, 3, 2, 1]


def test_exclude_groups_is_a_no_op_for_absent_or_empty_hold_outs():
    group_ids = np.array([0, 0, 1, 1])
    train_idx = np.array([0, 2, 3])
    for held in ([], np.array([], dtype=int), [9, 42]):
        kept = folds.exclude_groups(train_idx, group_ids, held)
        assert kept.tolist() == train_idx.tolist()
        assert kept is not train_idx


def test_exclude_groups_can_empty_the_training_set():
    kept = folds.exclude_groups(np.array([0, 1]), np.array([7, 7]), [7])
    assert kept.size == 0 and kept.dtype.kind == "i"


def test_exclude_groups_rejects_out_of_range_and_non_integer_indices():
    group_ids = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        folds.exclude_groups(np.array([0, 3]), group_ids, [0])
    with pytest.raises(ValueError):
        folds.exclude_groups(np.array([-1]), group_ids, [0])
    with pytest.raises(ValueError):
        folds.exclude_groups(np.array([0.0, 1.0]), group_ids, [0])


def test_the_anchor_exclusion_rule_holds_where_a_sky_group_straddles_two_fields():
    """The rule S1 exists to enforce: an anchor whose sky group reaches into the held-out
    field must not enter fitting for that fold. Overlapping tiles make this real — the
    same source is catalogued once in each field — so plain leave-one-field-out leaves a
    copy of the held-out object in train, and only exclude_groups takes it out.
    """
    ra = np.array([150.0, 150.0 + 0.2 * ARCSEC, 150.5, 150.9, 151.4])
    dec = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    fields = np.array(["cosmos", "uds", "cosmos", "cosmos", "uds"])

    group_ids = folds.sky_groups(ra, dec, radius_arcsec=0.5)
    assert group_ids[0] == group_ids[1], "the straddling pair must be one group"

    ((train, test),) = [f for f in folds.outer_lofo(fields) if fields[f[1][0]] == "uds"]
    assert test.tolist() == [1, 4]
    assert 0 in train.tolist(), "raw LOFO leaves the held-out object's twin in train"

    cured = folds.exclude_groups(train, group_ids, np.unique(group_ids[test]))
    assert cured.tolist() == [2, 3]
    assert not set(group_ids[cured].tolist()) & set(group_ids[test].tolist())


# ================================================================ shadow_folds


def test_shadow_folds_gives_an_untested_duplicate_its_parents_fold():
    """The leak this function exists to close. An untested row 0.1 arcsec from a tested
    row is the SAME physical object seen a second time, so the fold that holds that
    tested row out must not be allowed to train on the copy."""
    ra = np.array([150.0, 150.0, 150.0])
    dec = np.array([2.0, 2.0 + 0.1 * ARCSEC, 2.0 + 10.0 * ARCSEC])
    assert true_sep_arcsec(ra[0], dec[0], ra[1], dec[1]) == pytest.approx(0.1, abs=1e-6)
    assert true_sep_arcsec(ra[1], dec[1], ra[2], dec[2]) == pytest.approx(9.9, abs=1e-6)

    shadow = folds.shadow_folds(
        ra, dec, np.array([True, False, True]), np.array([2, -1, 4]), 0.3
    )
    assert shadow.tolist() == [-2, 2, -2]


def test_shadow_folds_excludes_an_untested_row_that_shadows_two_folds():
    """A copy of two different objects held out by two different folds cannot be trained
    on by either, so it is excluded from every fold rather than assigned to one."""
    ra = np.array([150.0, 150.0, 150.0])
    dec = np.array([2.0, 2.0 + 0.2 * ARCSEC, 2.0 + 0.4 * ARCSEC])
    assert true_sep_arcsec(ra[0], dec[0], ra[1], dec[1]) == pytest.approx(0.2, abs=1e-6)
    assert true_sep_arcsec(ra[1], dec[1], ra[2], dec[2]) == pytest.approx(0.2, abs=1e-6)

    shadow = folds.shadow_folds(
        ra, dec, np.array([True, False, True]), np.array([1, -1, 3]), 0.3
    )
    assert shadow.tolist() == [-2, -3, -2]


def test_shadow_folds_leaves_a_distant_untested_row_free():
    """5 arcsec from everything: no tested row's twin, so every fold may train on it."""
    ra = np.array([150.0, 150.0])
    dec = np.array([2.0, 2.0 + 5.0 * ARCSEC])
    assert true_sep_arcsec(ra[0], dec[0], ra[1], dec[1]) == pytest.approx(5.0, abs=1e-6)

    shadow = folds.shadow_folds(
        ra, dec, np.array([True, False]), np.array([2, -1]), 0.3
    )
    assert shadow.tolist() == [-2, -1]


def test_shadow_folds_marks_every_tested_row_minus_two():
    """Tested rows are governed by ``fold_assignment``, not by shadowing, even when one
    sits 0.1 arcsec from a tested row of a different fold. Marking them -2 rather than
    running them through the shadow rules keeps the two channels from fighting."""
    ra = np.array([150.0, 150.0, 150.0])
    dec = np.array([2.0, 2.0 + 0.1 * ARCSEC, 2.0 + 10.0 * ARCSEC])
    shadow = folds.shadow_folds(
        ra, dec, np.array([True, True, True]), np.array([0, 3, 1]), 0.3
    )
    assert shadow.tolist() == [-2, -2, -2]


def test_shadow_folds_does_not_chain_through_untested_rows():
    """Pure matching against the TESTED rows, not friends-of-friends. Chaining would let
    a row two hops away inherit a fold, which shrinks every training set without making
    any fold safer, and would make the shadow depend on the untested rows' own geometry
    -- which no fold structure depends on."""
    ra = np.array([150.0, 150.0, 150.0])
    dec = np.array([2.0, 2.0 + 0.2 * ARCSEC, 2.0 + 0.4 * ARCSEC])
    assert true_sep_arcsec(ra[0], dec[0], ra[2], dec[2]) == pytest.approx(0.4, abs=1e-6)

    shadow = folds.shadow_folds(
        ra, dec, np.array([True, False, False]), np.array([2, -1, -1]), 0.3
    )
    assert shadow.tolist() == [-2, 2, -1], (
        "the second-hop row is 0.4 arcsec from the only tested row and must stay free"
    )


def test_shadow_folds_frees_a_row_whose_only_neighbours_carry_no_fold():
    """A tested row outside the split -- ``fold_assignment == -1`` -- holds nothing out,
    so its neighbour shadows nothing."""
    ra = np.array([150.0, 150.0])
    dec = np.array([2.0, 2.0 + 0.1 * ARCSEC])
    shadow = folds.shadow_folds(
        ra, dec, np.array([True, False]), np.array([-1, -1]), 0.3
    )
    assert shadow.tolist() == [-2, -1]


def test_shadow_folds_uses_the_same_great_circle_convention_as_sky_groups():
    """The radius means the same thing in both functions. Pinned where a flat
    approximation would disagree: one RA offset spans 0.2 arcsec at dec=+60 and
    0.4 arcsec at dec=0, so it must shadow at one declination and not at the other."""
    d_ra = delta_ra_for_separation(60.0, 0.2)
    tested = np.array([True, False])
    assignment = np.array([3, -1])

    hi_ra, hi_dec = np.array([150.0, 150.0 + d_ra]), np.array([60.0, 60.0])
    assert true_sep_arcsec(hi_ra[0], hi_dec[0], hi_ra[1], hi_dec[1]) == pytest.approx(
        0.2, abs=1e-6
    )
    assert folds.shadow_folds(hi_ra, hi_dec, tested, assignment, 0.3).tolist() == [
        -2,
        3,
    ]

    lo_ra, lo_dec = np.array([150.0, 150.0 + d_ra]), np.array([0.0, 0.0])
    assert true_sep_arcsec(lo_ra[0], lo_dec[0], lo_ra[1], lo_dec[1]) == pytest.approx(
        0.4, abs=1e-6
    )
    assert folds.shadow_folds(lo_ra, lo_dec, tested, assignment, 0.3).tolist() == [
        -2,
        -1,
    ]


def test_shadow_folds_is_permutation_equivariant():
    ra, dec, picks = planted_field()
    n = ra.size
    tested = np.zeros(n, bool)
    tested[:180] = True
    assignment = np.where(tested, np.arange(n) % 5, -1).astype(np.int64)

    base = folds.shadow_folds(ra, dec, tested, assignment, 0.3)
    assert int((base == -2).sum()) == 180
    assert int((base >= 0).sum()) == picks.size, "every planted duplicate must shadow"

    perm = np.random.default_rng(17).permutation(n)
    assert np.array_equal(
        folds.shadow_folds(ra[perm], dec[perm], tested[perm], assignment[perm], 0.3),
        base[perm],
    )


def test_shadow_folds_with_no_tested_rows_frees_everything():
    ra = np.array([150.0, 150.0])
    dec = np.array([2.0, 2.0 + 0.1 * ARCSEC])
    shadow = folds.shadow_folds(ra, dec, np.zeros(2, bool), np.array([-1, -1]), 0.3)
    assert shadow.tolist() == [-1, -1]
    empty = folds.shadow_folds(
        np.empty(0), np.empty(0), np.empty(0, bool), np.empty(0, int), 0.3
    )
    assert empty.tolist() == [] and empty.dtype.kind == "i"


def test_shadow_folds_rejects_bad_input():
    ra, dec = np.array([150.0, 150.1]), np.array([2.0, 2.0])
    tested, assignment = np.array([True, False]), np.array([0, -1])
    with pytest.raises(ValueError):
        folds.shadow_folds(ra, dec[:1], tested, assignment, 0.3)
    with pytest.raises(ValueError):
        folds.shadow_folds(ra, dec, tested[:1], assignment, 0.3)
    with pytest.raises(ValueError):
        folds.shadow_folds(ra, dec, tested, assignment[:1], 0.3)
    with pytest.raises(ValueError):
        folds.shadow_folds(ra, dec, tested, np.array([[0, -1]]), 0.3)
    with pytest.raises(ValueError):
        folds.shadow_folds(ra, dec, tested, np.array([0.5, -1.0]), 0.3)
    with pytest.raises(ValueError):
        folds.shadow_folds(ra, dec, tested, assignment, 0.0)
    with pytest.raises(ValueError):
        folds.shadow_folds(ra, np.array([2.0, np.nan]), tested, assignment, 0.3)


def brute_force_shadow(ra, dec, tested, assignment, radius_arcsec) -> np.ndarray:
    """The shadow rules read literally, one row at a time, against the INDEPENDENT
    separation formula at the top of this file.

    Deliberately the slowest possible implementation. The module reaches its answer
    through a KD-tree, a flattened neighbour list and a lexsort reduction to distinct
    (row, fold) pairs, and that reduction is where an off-by-one would hide: it would
    show up as a rare wrong fold on a row with several neighbours, which no hand-built
    three-row fixture reaches. This must never be refactored into a call into
    ``redress.folds``.
    """
    out = np.full(ra.size, -1, dtype=np.int64)
    for row in range(ra.size):
        if tested[row]:
            out[row] = -2
            continue
        near = {
            int(assignment[other])
            for other in range(ra.size)
            if tested[other]
            and assignment[other] >= 0
            and true_sep_arcsec(ra[row], dec[row], ra[other], dec[other])
            <= radius_arcsec
        }
        out[row] = -1 if not near else (near.pop() if len(near) == 1 else -3)
    return out


def test_shadow_folds_matches_a_brute_force_reading_of_its_own_rules():
    """Six dense random fields, checked row by row against the rules read literally.

    Dense on purpose: the sources are scattered with a 0.6 arcsec spread, so untested
    rows routinely have several tested neighbours across several folds and all six
    outcomes occur many times over. The assertion below records the value coverage, so a
    field that stopped exercising the conflict branch would fail rather than pass
    quietly.
    """
    rng = np.random.default_rng(4242)
    seen: dict[int, int] = {}
    for _ in range(6):
        n = 80
        ra = 150.0 + rng.normal(0.0, 0.6 * ARCSEC, n) / np.cos(np.radians(2.0))
        dec = 2.0 + rng.normal(0.0, 0.6 * ARCSEC, n)
        tested = rng.random(n) < 0.55
        assignment = np.where(tested, rng.integers(-1, 5, n), -1).astype(np.int64)

        got = folds.shadow_folds(ra, dec, tested, assignment, 0.3)
        assert np.array_equal(got, brute_force_shadow(ra, dec, tested, assignment, 0.3))
        for value in got.tolist():
            seen[value] = seen.get(value, 0) + 1

    assert set(seen) == {-3, -2, -1, 0, 1, 2, 3, 4}, (
        f"the random fields stopped exercising some branch: {sorted(seen)}"
    )
    assert min(seen.values()) >= 5, f"a branch was barely touched: {seen}"


# ================================================ inner_grouped_kfold: class balance

# Snapshot of the assignment the size-only rule produced BEFORE `positive_mask` was
# added, taken by running the pre-change module on this exact case. It is the regression
# proof for the no-mask path: the balanced branch must be reachable ONLY through the new
# argument, and a run without it must reproduce the old folds row for row.
SNAPSHOT_GROUPS_SEED, SNAPSHOT_FOLD_SEED = 20260830, 4242
SNAPSHOT_FOLD_OF_ROW = [
    2, 2, 0, 2, 3, 3, 0, 1, 2, 2, 3, 0, 1, 2, 3, 0, 2, 4, 1, 0, 3, 1, 4, 3, 2,
    1, 2, 2, 0, 1, 4, 0, 2, 0, 4, 1, 3, 2, 0, 2, 2, 4, 0, 3, 2, 0, 4, 2, 3, 4,
    0, 3, 0, 4, 3, 0, 3, 3, 4, 2, 3, 3, 2, 1, 2, 2, 2, 1, 1, 0, 1, 1, 1, 1, 2,
    4, 1, 0, 0, 2, 1, 4, 3, 3, 2, 0, 3, 1, 2, 0, 0, 0, 3, 2, 0, 0, 0, 3, 4, 4,
    4, 3, 1, 2, 2, 4, 1, 2, 0, 3, 4, 4, 3, 3, 4, 4, 4, 4, 3, 0, 1, 4, 3, 3, 2,
    1, 1, 4, 4, 0, 4, 4, 1, 0, 4, 3, 1, 4, 1, 4, 2, 4, 3, 1, 4, 3, 3, 3, 4, 0,
    2, 3, 1, 2, 3, 0, 1, 1, 0, 2, 1, 0, 4, 1, 2, 0, 1, 3, 0, 1, 1, 2, 0, 4, 2,
    0, 0, 4, 1, 3, 1, 3, 4, 0, 3, 4, 3, 1, 0, 0, 2, 0, 4, 2, 1, 4, 2, 3, 4, 1,
]  # fmt: skip


def snapshot_case() -> np.ndarray:
    return np.random.default_rng(SNAPSHOT_GROUPS_SEED).integers(0, 40, size=200)


def fold_of_row(fold_list, n: int) -> list[int]:
    row = np.full(n, -1, dtype=np.int64)
    for fold, (_, test) in enumerate(fold_list):
        row[test] = fold
    return row.tolist()


def positives_per_fold(fold_list, positive) -> list[int]:
    return [int(np.asarray(positive)[test].sum()) for _, test in fold_list]


def singleton_positive_case(n_pos: int = 20, n_neg: int = 200, seed: int = 20260830):
    """``n_pos`` positive singleton groups among ``n_neg`` negative ones. Every group is
    the same size, so the size-only rule has nothing to balance on and the positives
    land wherever the rng tie-break puts them, which is how a fold ends up holding far
    fewer than its share of the ~20 positives this project actually has.

    The positive groups are SCATTERED through the key order rather than taken as the
    first ``n_pos``. Contiguous positives cannot exhibit the fault: with equal-sized
    groups the size rule fills folds strictly round robin, so the first 20 groups would
    split 4-4-4-4-4 by arithmetic alone and the vacuity control below would fail.
    """
    group_ids = np.arange(n_pos + n_neg)
    positive = np.zeros(n_pos + n_neg, dtype=bool)
    positive[
        np.random.default_rng(seed).choice(group_ids.size, size=n_pos, replace=False)
    ] = True
    return group_ids, positive


def test_the_size_only_rule_spreads_the_positives_unevenly():
    """The vacuity control for the balance pins below: without ``positive_mask`` the
    20 positives DO land unevenly -- across seeds 0-19 the per-fold count runs from 2 to
    8 where perfect balance is 4 -- so a pass underneath means the mask did it."""
    group_ids, positive = singleton_positive_case()
    uneven = [
        positives_per_fold(folds.inner_grouped_kfold(group_ids, 5, seed), positive)
        for seed in range(20)
    ]
    assert any(counts != [4, 4, 4, 4, 4] for counts in uneven), (
        "the size-only rule balanced every seed -- the mask pins below prove nothing"
    )


def test_positive_mask_spreads_singleton_positives_exactly_evenly():
    group_ids, positive = singleton_positive_case()
    for seed in range(20):
        result = folds.inner_grouped_kfold(group_ids, 5, seed, positive_mask=positive)
        assert positives_per_fold(result, positive) == [4, 4, 4, 4, 4]


def test_positive_mask_starves_no_fold_when_the_positive_groups_are_uneven():
    """Positive counts 3, 2, 1, 1 over four folds, among twelve empty filler groups.

    Perfect balance is arithmetically OUT OF REACH here and the assertion says so: the
    three positives of the first group are one indivisible group, so some fold holds
    three while another holds at most one. What IS reachable, and what the plan's
    "class-balanced across folds" requires, is that NO fold is left with zero -- the
    four positive groups must land on four different folds.
    """
    group_ids = np.array([0] * 3 + [1] * 2 + [2] + [3] + list(range(4, 16)))
    positive = group_ids < 4
    assert int(positive.sum()) == 7
    assert np.bincount(group_ids[positive]).tolist() == [3, 2, 1, 1]

    for seed in range(20):
        result = folds.inner_grouped_kfold(group_ids, 4, seed, positive_mask=positive)
        counts = positives_per_fold(result, positive)
        assert min(counts) >= 1, f"seed {seed}: a fold holds no positives ({counts})"
        assert sorted(counts) == [1, 1, 2, 3], (
            f"seed {seed}: {counts} is not the balanced placement of 3, 2, 1, 1"
        )


def test_omitting_the_positive_mask_reproduces_the_previous_assignment_exactly():
    """The regression proof. Byte-identical to the pre-change module on a fixed random
    case, so nothing that already ran through these folds moves."""
    group_ids = snapshot_case()
    without = folds.inner_grouped_kfold(group_ids, 5, SNAPSHOT_FOLD_SEED)
    assert fold_of_row(without, group_ids.size) == SNAPSHOT_FOLD_OF_ROW

    explicit_none = folds.inner_grouped_kfold(
        group_ids, 5, SNAPSHOT_FOLD_SEED, positive_mask=None
    )
    assert fold_of_row(explicit_none, group_ids.size) == SNAPSHOT_FOLD_OF_ROW


def test_the_positive_mask_actually_changes_the_assignment():
    """If the masked and unmasked answers agreed on a case built to need balancing, the
    argument would be decoration."""
    group_ids, positive = singleton_positive_case()
    without = folds.inner_grouped_kfold(group_ids, 5, SNAPSHOT_FOLD_SEED)
    with_mask = folds.inner_grouped_kfold(
        group_ids, 5, SNAPSHOT_FOLD_SEED, positive_mask=positive
    )
    assert fold_of_row(without, group_ids.size) != fold_of_row(
        with_mask, group_ids.size
    )


def test_class_balanced_folds_are_deterministic_for_a_fixed_seed():
    group_ids, positive = singleton_positive_case()
    a = folds.inner_grouped_kfold(group_ids, 5, 20260830, positive_mask=positive)
    b = folds.inner_grouped_kfold(group_ids, 5, 20260830, positive_mask=positive)
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b, strict=True):
        assert np.array_equal(tr_a, tr_b) and np.array_equal(te_a, te_b)


def test_class_balanced_folds_keep_every_group_whole_and_every_row_once():
    group_ids = np.repeat(np.arange(30), 3)
    positive = np.zeros(group_ids.size, dtype=bool)
    positive[::7] = True
    result = folds.inner_grouped_kfold(group_ids, 5, 20260830, positive_mask=positive)
    seen = np.concatenate([test for _, test in result])
    assert np.array_equal(np.sort(seen), np.arange(group_ids.size))
    for train, test in result:
        assert test.size > 0
        assert not set(group_ids[train].tolist()) & set(group_ids[test].tolist())


def test_class_balanced_row_sets_survive_a_row_shuffle():
    """The balanced order reads only group keys, positive counts and sizes, all of them
    permutation-invariant, so a shuffled catalogue must fold identically."""
    group_ids = np.repeat(np.arange(30), 3)
    positive = np.zeros(group_ids.size, dtype=bool)
    positive[::7] = True
    base = folds.inner_grouped_kfold(group_ids, 5, 20260830, positive_mask=positive)

    rng = np.random.default_rng(19)
    for _ in range(5):
        perm = rng.permutation(group_ids.size)
        shuffled = folds.inner_grouped_kfold(
            group_ids[perm], 5, 20260830, positive_mask=positive[perm]
        )
        assert fold_sets(shuffled, perm) == fold_sets(base, np.arange(group_ids.size))


def test_an_all_positive_or_all_negative_mask_is_accepted():
    group_ids = np.repeat(np.arange(20), 2)
    for positive in (
        np.ones(group_ids.size, dtype=bool),
        np.zeros(group_ids.size, dtype=bool),
    ):
        result = folds.inner_grouped_kfold(
            group_ids, 5, 20260830, positive_mask=positive
        )
        seen = np.concatenate([test for _, test in result])
        assert np.array_equal(np.sort(seen), np.arange(group_ids.size))


def test_inner_grouped_kfold_rejects_a_misaligned_positive_mask():
    with pytest.raises(ValueError):
        folds.inner_grouped_kfold(
            GROUPS, 4, 0, positive_mask=np.ones(GROUPS.size - 1, dtype=bool)
        )
    with pytest.raises(ValueError):
        folds.inner_grouped_kfold(
            GROUPS, 4, 0, positive_mask=np.ones((2, GROUPS.size // 2), dtype=bool)
        )
