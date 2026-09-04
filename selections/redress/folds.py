"""fold machinery: sky-position grouping, outer leave-one-field-out, inner grouped
k-fold (THE_PLAN_2026-08-30.md S1).

THE TWO-LEVEL RULE, and why both levels exist.

    OUTER   leave-one-field-out. Every REPORTED estimate comes from a field the model
            has never seen. Fields differ in depth, filter set and PSF, so a
            same-field holdout measures interpolation, not transfer, and the number
            that reaches the poster must be the transferable one.

    INNER   grouped k-fold inside the outer training set. EVERY selection decision --
            hyperparameters, the z_phot feature call, the operating point, the
            calibration fit -- is made here and never touches the outer estimate. A
            decision made against the outer fold is a decision made against the number
            being reported, which is how a benchmark quietly becomes a fit.

The GROUP is a sky position, not a catalogue row. PRIMER tiles overlap, so one physical
source is catalogued more than once; splitting those rows across a train/test boundary
puts the same object on both sides and inflates every score. `sky_groups` collapses them
first, `inner_grouped_kfold` keeps each group whole, and `exclude_groups` enforces the
plan's anchor rule -- an anchor in a held-out field or sky group enters neither fitting,
propensity, tuning nor calibration for that fold.

THE UNTESTED CHANNEL is the other half of that, and the half a grouped split does not
reach on its own. The inner k-fold splits and holds out the TESTED rows, the only ones
carrying a label to score against, and every other row goes into every fold's training
bag -- including the untested second copy of a tested object in an overlapping tile.
Same source, same position, no spectrum of its own, no fold of its own, and the fold
holding the tested copy out fits on the twin and then scores it. `shadow_folds` marks
which fold's hold-out each untested row duplicates, so that fold alone drops it and the
others still get the row.

CLASS BALANCE. `inner_grouped_kfold` balances group SIZES by default, which is the wrong
quantity when the positives are rare: with of order twenty of them a size-balanced split
can hand one fold none, and a fold holding no positives scores nothing and calibrates on
nothing. Passing `positive_mask` places the positive-bearing groups first and spreads
them across folds, which is the plan's "class-balanced across folds"
(design note 2.2). Without that argument the size-only rule is
byte-identical to what it was before the argument existed, pinned by snapshot.

GEOMETRY: EXACT, NOT FLAT. mt-04 permitted the small-angle flat approximation, in which
positions become `(RA * cos(dec), dec)` and separations are Euclidean in that plane. It
is NOT used, because it is wrong in three places this project actually visits and wrong
silently in all of them:

  * per-point `cos(dec)` scaling. `x = RA * cos(dec)` differences pick up a spurious
    `RA * sin(dec) * d(dec)` term; at RA ~ 350 deg and dec ~ +60 that term reaches a few
    arcsec for a pair genuinely half an arcsec apart, i.e. larger than the merge radius
    itself.
  * a single global `cos(dec_ref)` instead. The census spans COSMOS (dec ~ +2), UDS
    (dec ~ -5) and GOODS-N (dec ~ +62); one reference declination misscales the third
    field by a factor of two.
  * RA wraparound and the poles. A raw RA difference calls a pair across the RA=0
    meridian 360 degrees apart.

So positions become unit vectors and neighbours are found by CHORD distance,
`chord = 2 sin(theta/2)`, on a `cKDTree` -- exact, wrap-safe, pole-safe, and the same
construction `redress.splits` already uses. The flat approximation would agree with it to
better than a milliarcsecond over a single PRIMER tile; it is declined because nothing
downstream would notice the day it stops being one tile.

DETERMINISM is a first-class requirement, not a nicety. Every published number is
downstream of these folds, so an ordering-dependent fold assignment makes the headline
unreproducible. Two properties are guaranteed and both are pinned by test:

  * `sky_groups` returns PERMUTATION-EQUIVARIANT labels. Groups are numbered by their
    lexicographically smallest `(RA, Dec)` member -- a property of the sky, not of row
    order -- so shuffling rows permutes the labels rather than renaming the groups. The
    weaker promise (identical PARTITION, possibly different label values) is what the
    API contract states; this stronger one is what makes shuffle-then-group-then-fold
    reproduce, and downstream code may rely on it.
  * `inner_grouped_kfold` consumes only group KEYS and SIZES, both permutation-
    invariant, and draws its tie-breaks from `numpy.random.default_rng(seed)` in a
    fixed order. Same inputs and same seed give the same folds regardless of row order.

NO CAP, ANYWHERE. Review item 9 refuted the proof behind the pipeline's top-20,000 head
bound on dedup. There is no head bound, no top-N truncation and no silent subsample in
this module; grouping runs over every row it is given.
"""

from __future__ import annotations

from itertools import chain

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_RADIUS_ARCSEC = 0.5


# ---------------------------------------------------------------- input checks


def _check_radec(ra_deg, dec_deg) -> tuple[np.ndarray, np.ndarray]:
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    if ra.ndim != 1 or dec.ndim != 1:
        raise ValueError(
            f"ra and dec must be 1-D, got shapes {ra.shape} and {dec.shape}"
        )
    if ra.shape != dec.shape:
        raise ValueError(
            f"ra and dec must be the same length, got {ra.size} and {dec.size}"
        )
    if not (np.all(np.isfinite(ra)) and np.all(np.isfinite(dec))):
        raise ValueError(
            "non-finite coordinates: drop or repair these rows at ingest, where the "
            "decision is recorded, not silently here"
        )
    if np.any(np.abs(dec) > 90.0):
        raise ValueError("dec outside [-90, 90] degrees")
    return ra, dec


def _check_1d_labels(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite labels, which cannot be grouped")
    return array


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.column_stack(
        (np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec))
    )


# ---------------------------------------------------------------- sky grouping


def sky_groups(
    ra_deg, dec_deg, radius_arcsec: float = DEFAULT_RADIUS_ARCSEC
) -> np.ndarray:
    """Friends-of-friends group label per row, one label per physical sky position.

    Two rows separated by at most ``radius_arcsec`` are linked, and linkage is
    TRANSITIVE: A~B and B~C put A, B and C in one group even when A and C are farther
    apart than the radius. Transitivity over-merges by design. For a held-out split that
    is the safe direction -- an over-merged group only makes the split stricter -- while
    a missed link puts one physical object on both sides of it. (Counting is the other
    direction and needs blend-aware linkage; this is not that tool.)

    Separations are EXACT great-circle angles. Positions become unit vectors and
    neighbours are found by chord distance ``2 sin(theta/2)`` on a ``cKDTree``, which is
    wrap-safe at RA=0 and correct at the poles. mt-04 permitted the small-angle flat
    approximation instead -- project to ``(RA * cos(dec), dec)`` and measure Euclidean
    distance there, correct to well under a milliarcsecond within one PRIMER tile -- and
    the module docstring records in full why it is declined: per-point ``cos(dec)``
    scaling carries a spurious ``RA * sin(dec) * d(dec)`` term that exceeds the merge
    radius at high RA and high declination, a single global reference declination
    misscales GOODS-N against COSMOS by a factor of two, and neither form survives the
    RA=0 meridian.

    Labels are ints in ``[0, n_groups)``, assigned in the canonical order of each
    group's lexicographically smallest ``(RA, Dec)`` member. That ordering is a property
    of the sky rather than of row order, so the labels are permutation-EQUIVARIANT:
    ``sky_groups(ra[p], dec[p]) == sky_groups(ra, dec)[p]``. The contract this API
    promises is only that the induced PARTITION is permutation-invariant; the stronger
    equivariance is what lets a shuffled catalogue reproduce identical downstream folds.

    NO CAP. Every row is grouped -- no head bound, no top-N truncation, no subsample.

    Raises ``ValueError`` on ragged, non-finite or out-of-range coordinates, and on a
    non-positive radius.
    """
    ra, dec = _check_radec(ra_deg, dec_deg)
    radius = float(radius_arcsec)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            f"radius_arcsec must be finite and positive, got {radius_arcsec!r}"
        )
    n = ra.size
    if n == 0:
        return np.empty(0, dtype=np.int64)

    chord = 2.0 * np.sin(np.radians(radius / 3600.0) / 2.0)
    pairs = cKDTree(_unit_vectors(ra, dec)).query_pairs(r=chord, output_type="ndarray")

    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    for left, right in pairs:
        a, b = find(int(left)), find(int(right))
        if a != b:
            lo, hi = (a, b) if a < b else (b, a)
            parent[hi] = lo

    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)

    # CANONICAL NUMBERING. Walk the rows in (RA, Dec) order and number each group the
    # first time it is met. Distinct groups cannot share a smallest member: two rows at
    # identical coordinates are zero arcsec apart and therefore already one group, so
    # the ordering is total and independent of the input row order.
    order = np.lexsort((dec, ra))
    roots_in_sky_order = roots[order]
    _, first_seen = np.unique(roots_in_sky_order, return_index=True)
    canonical = roots_in_sky_order[np.sort(first_seen)]
    relabel = np.empty(n, dtype=np.int64)
    relabel[canonical] = np.arange(canonical.size, dtype=np.int64)
    return relabel[roots]


# ---------------------------------------------------------------- outer folds


def outer_lofo(fields) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-field-out: one ``(train_idx, test_idx)`` per distinct field.

    Folds come back in sorted order of the unique field values, ``test_idx`` holds the
    rows of that field and ``train_idx`` holds every other row, both ascending. This is
    the OUTER level: it produces the reported estimate, so nothing tuned inside a fold
    may be chosen by looking at it.

    A single-field input yields one fold with an EMPTY training set rather than an
    error. That is the honest answer -- there is nothing to train on -- and it fails
    loudly at the fit rather than quietly here.
    """
    values = _check_1d_labels(fields, "fields")
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for value in np.unique(values):
        held = values == value
        out.append(
            (
                np.flatnonzero(~held).astype(np.int64, copy=False),
                np.flatnonzero(held).astype(np.int64, copy=False),
            )
        )
    return out


# ---------------------------------------------------------------- inner folds


def inner_grouped_kfold(
    group_ids, n_splits: int, seed: int, positive_mask=None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped k-fold: every member of a group lands in the SAME test fold.

    The assignment is greedy and balanced: groups are taken in order of DESCENDING SIZE,
    ties broken by ASCENDING canonical group key, and each is placed on the fold that is
    currently smallest. Largest-first is what keeps the folds even -- placing a big group
    last leaves nowhere for it to go without skewing one fold. When several folds are
    tied for smallest, the choice is drawn from ``numpy.random.default_rng(seed)``, so
    repeated cross-validation under different seeds explores genuinely different splits
    instead of repeating one.

    CLASS BALANCE, when ``positive_mask`` is given. Size is the wrong quantity to
    balance on when the positives are rare, and here they are: of order twenty in the
    whole modelable set, so a size-balanced split can hand one fold none at all, and a
    fold holding no positives scores nothing and calibrates on nothing. ``positive_mask``
    is a boolean array aligned ROW-WISE with ``group_ids``. Given it, the groups carrying
    at least one positive are placed FIRST -- descending positive count, ties by
    descending size, then ascending group key -- each onto the fold holding the fewest
    positives so far, ties broken by fewest rows and then by lowest fold index. The
    remaining zero-positive groups follow by exactly the size rule above. That is the
    plan's "class-balanced across folds" (design note 2.2).

    Group indivisibility still binds, and the guarantee is stated so as not to paper over
    it: a single group carrying three positives puts three in one fold whatever the
    ordering. What is promised is a deterministic greedy balance WITHOUT splitting a
    group, not equal counts (greedy placement is not globally optimal for every
    group-count distribution, and does not claim to be).

    ``positive_mask=None``, the default, is the size-only rule unchanged and
    BYTE-IDENTICAL to the version that predates this argument, pinned by test against a
    snapshot of that version's output: folds that moved under an unrelated new argument
    would silently move every number already computed from them. The rng is drawn only
    for size-rule ties, so with a mask given it is drawn fewer times, in the order the
    second phase reaches those ties; the first phase is fully determined and draws none.

    Determinism: only group KEYS, SIZES and per-group POSITIVE COUNTS enter the
    assignment, all three invariant under row permutation, so the same inputs and seed
    give the same folds however the rows are ordered. With ``n_splits <= n_groups`` every
    fold is non-empty, because the first ``n_splits`` placements each land on a distinct
    still-empty fold.

    ``group_ids`` may be any 1-D array of comparable labels -- the integer labels from
    :func:`sky_groups`, or field/source strings. Raises ``ValueError`` for
    ``n_splits < 2``, for ``n_splits`` greater than the number of distinct groups, and
    for a ``positive_mask`` that is not 1-D and the same length as ``group_ids``.
    """
    groups = _check_1d_labels(group_ids, "group_ids")
    splits = int(n_splits)
    if splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits!r}")

    keys, inverse = np.unique(groups, return_inverse=True)
    inverse = np.reshape(inverse, -1)
    if splits > keys.size:
        raise ValueError(
            f"n_splits={splits} exceeds the {keys.size} distinct group(s); a group is "
            "never split across folds, so there are not enough of them to fill one each"
        )

    sizes = np.bincount(inverse, minlength=keys.size)
    # `np.unique` already sorted the keys ascending, and a stable sort on the negated
    # sizes preserves that order within each size — descending size, ascending key.
    order = np.argsort(-sizes, kind="stable")

    rng = np.random.default_rng(seed)
    fold_of_group = np.full(keys.size, -1, dtype=np.int64)
    load = np.zeros(splits, dtype=np.int64)

    if positive_mask is not None:
        positives = _check_1d_labels(positive_mask, "positive_mask").astype(bool)
        if positives.size != groups.size:
            raise ValueError(
                "positive_mask must be the same length as group_ids, got "
                f"{positives.size} and {groups.size}"
            )
        per_group = np.bincount(inverse[positives], minlength=keys.size)
        bearing = np.flatnonzero(per_group)
        # `np.lexsort` reads its keys last-first: positive count descending, then size
        # descending, then group key ascending. The last key is total, so the order is
        # fully determined and this phase draws nothing from the rng.
        priority = bearing[np.lexsort((bearing, -sizes[bearing], -per_group[bearing]))]
        positive_load = np.zeros(splits, dtype=np.int64)
        for group in priority:
            chosen = int(np.lexsort((np.arange(splits), load, positive_load))[0])
            fold_of_group[group] = chosen
            positive_load[chosen] += per_group[group]
            load[chosen] += sizes[group]
        order = order[fold_of_group[order] < 0]

    for group in order:
        smallest = np.flatnonzero(load == load.min())
        chosen = int(
            smallest[0] if smallest.size == 1 else smallest[rng.integers(smallest.size)]
        )
        fold_of_group[group] = chosen
        load[chosen] += sizes[group]

    fold_of_row = fold_of_group[inverse]
    return [
        (
            np.flatnonzero(fold_of_row != fold).astype(np.int64, copy=False),
            np.flatnonzero(fold_of_row == fold).astype(np.int64, copy=False),
        )
        for fold in range(splits)
    ]


# ------------------------------------------------------------- the untested channel

# `shadow_folds` return values. Negative throughout, so a shadow can never be mistaken
# for a fold index, and each negative says something different about WHY a row carries
# no fold of its own.
SHADOW_FREE = -1  # no tested row within the radius: every fold may train on it
SHADOW_TESTED = -2  # a tested row: governed by fold_assignment, not by shadowing
SHADOW_CONFLICT = -3  # copies objects held out by two or more folds: usable by none


def shadow_folds(
    ra_deg,
    dec_deg,
    tested_mask,
    fold_assignment,
    radius_arcsec: float = 0.3,
) -> np.ndarray:
    """Which fold's hold-out each UNTESTED row duplicates, so no fold trains on the
    object it is scoring through the untested channel.

    THE LEAK THIS CLOSES. The inner k-fold splits and holds out only the TESTED rows --
    they are the only ones with a label to score against -- and every other row goes
    into every fold's training bag. But PRIMER tiles overlap, so one physical source is
    catalogued more than once, and a second copy of a tested object is very often
    UNTESTED: same sky position, no spectrum of its own. Nothing in the fold assignment
    keeps that copy out, so the fold that holds the tested row out still fits on its
    twin and then scores it. Grouping the tested rows by sky position does not touch
    this: the leak arrives through the rows the grouping never saw.

    THE ANSWER PER ROW, all four values negative-or-fold and never ambiguous:

    ``SHADOW_TESTED`` (-2)
        a tested row. Its fold comes from ``fold_assignment``; shadowing does not
        govern it, and keeping the two channels apart is what stops them fighting.
    ``SHADOW_FREE`` (-1)
        an untested row with no tested row inside ``radius_arcsec``, or whose only
        neighbours sit outside the split (``fold_assignment == -1``, holding nothing
        out). It duplicates nobody's hold-out and every fold may train on it.
    ``f >= 0``
        an untested row whose in-radius tested neighbours all belong to fold ``f``. It
        is a copy of what fold ``f`` holds out, so fold ``f`` alone must refuse it --
        the other folds still get the row, which matters when the modelable set is
        small enough that discarding it outright would cost real training data.
    ``SHADOW_CONFLICT`` (-3)
        an untested row within the radius of tested rows in TWO OR MORE folds. There is
        no fold it is safe for, so it is refused by all of them.

    MATCHING, NOT FRIENDS-OF-FRIENDS. Each untested row is matched against the TESTED
    rows only; nothing chains through a second untested row. Chaining would let a row
    two hops away inherit a fold, shrinking every training set without making any fold
    safer, and would make the answer depend on the untested rows' own geometry -- which
    no fold structure depends on. This function therefore reads the fold assignment and
    never perturbs it.

    GEOMETRY is :func:`sky_groups`' exactly: unit vectors and chord distance
    ``2 sin(theta/2)`` on a ``cKDTree``, so ``radius_arcsec`` means the same
    great-circle angle in both, wrap-safe at RA=0 and correct at the poles. The DEFAULT
    differs and deliberately: 0.3 arcsec, the pipeline's ``FOLD_RADIUS_ARCSEC`` and its
    cross-match radius, against 0.5 for ``sky_groups``. The pipeline passes the radius
    explicitly at every call, so the default is a fallback rather than the operating
    value.

    Raises ``ValueError`` on ragged, non-finite or out-of-range coordinates, on a
    non-integer or misaligned ``fold_assignment``, and on a non-positive radius.
    """
    ra, dec = _check_radec(ra_deg, dec_deg)
    tested = np.asarray(tested_mask, dtype=bool)
    assignment = np.asarray(fold_assignment)
    if tested.ndim != 1 or assignment.ndim != 1:
        raise ValueError(
            "tested_mask and fold_assignment must be 1-D, got shapes "
            f"{tested.shape} and {assignment.shape}"
        )
    if tested.shape != ra.shape or assignment.shape != ra.shape:
        raise ValueError(
            "ra, dec, tested_mask and fold_assignment must be the same length, got "
            f"{ra.size}, {dec.size}, {tested.size} and {assignment.size}"
        )
    if assignment.size and assignment.dtype.kind not in "iu":
        raise ValueError(
            "fold_assignment must hold integer fold indices, with -1 for a row outside "
            f"the split, got dtype {assignment.dtype}"
        )
    radius = float(radius_arcsec)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            f"radius_arcsec must be finite and positive, got {radius_arcsec!r}"
        )

    shadow = np.full(ra.size, SHADOW_FREE, dtype=np.int64)
    shadow[tested] = SHADOW_TESTED
    tested_rows = np.flatnonzero(tested)
    untested_rows = np.flatnonzero(~tested)
    if tested_rows.size == 0 or untested_rows.size == 0:
        return shadow

    vectors = _unit_vectors(ra, dec)
    chord = 2.0 * np.sin(np.radians(radius / 3600.0) / 2.0)
    neighbours = cKDTree(vectors[tested_rows]).query_ball_point(
        vectors[untested_rows], r=chord
    )

    counts = np.fromiter(
        (len(hit) for hit in neighbours), dtype=np.int64, count=untested_rows.size
    )
    flat = np.fromiter(
        chain.from_iterable(neighbours), dtype=np.int64, count=int(counts.sum())
    )
    owner = np.repeat(np.arange(untested_rows.size), counts)
    neighbour_fold = assignment[tested_rows[flat]].astype(np.int64, copy=False)

    # A neighbour outside the split holds nothing out, so it shadows nothing.
    inside = neighbour_fold >= 0
    owner, neighbour_fold = owner[inside], neighbour_fold[inside]

    # Reduce to DISTINCT (row, fold) pairs. One distinct fold means the row copies that
    # fold's hold-out and only that fold refuses it; two or more means no fold may have
    # it; none means every fold may.
    ordering = np.lexsort((neighbour_fold, owner))
    owner, neighbour_fold = owner[ordering], neighbour_fold[ordering]
    first = np.ones(owner.size, dtype=bool)
    first[1:] = (owner[1:] != owner[:-1]) | (neighbour_fold[1:] != neighbour_fold[:-1])
    distinct = np.bincount(owner[first], minlength=untested_rows.size)

    # Rows with no in-fold neighbour are never written here and keep SHADOW_FREE; rows
    # with two or more take whichever fold landed last and are then overwritten.
    resolved = np.full(untested_rows.size, SHADOW_FREE, dtype=np.int64)
    resolved[owner[first]] = neighbour_fold[first]
    resolved[distinct >= 2] = SHADOW_CONFLICT
    shadow[untested_rows] = resolved
    return shadow


# ---------------------------------------------------------------- anchor exclusion


def exclude_groups(train_idx, group_ids, held_out_group_ids) -> np.ndarray:
    """``train_idx`` with every row whose group is in ``held_out_group_ids`` removed.

    The plan's anchor rule, as a function. Leave-one-field-out alone is not enough when
    tiles overlap: the held-out field's sources are catalogued a second time under a
    neighbouring field, so a plain complement leaves a copy of the evaluation object in
    the training set. Passing the held-out fold's group ids through here removes it, and
    the same call keeps those anchors out of the fold's propensity model, hyperparameter
    tuning and calibration fit -- every stage, not only the fit.

    Row order is preserved and the input array is never modified. An empty or entirely
    absent hold-out returns a copy of ``train_idx`` unchanged.
    """
    idx = np.asarray(train_idx)
    if idx.ndim != 1:
        raise ValueError(f"train_idx must be 1-D, got shape {idx.shape}")
    if idx.size and idx.dtype.kind not in "iu":
        raise ValueError(
            f"train_idx must hold integer row indices, got dtype {idx.dtype}"
        )
    idx = idx.astype(np.int64, copy=True)

    groups = np.asarray(group_ids)
    if groups.ndim != 1:
        raise ValueError(f"group_ids must be 1-D, got shape {groups.shape}")
    if idx.size and (idx.min() < 0 or idx.max() >= groups.size):
        raise ValueError(
            f"train_idx values must index group_ids (0..{groups.size - 1}), got range "
            f"{idx.min()}..{idx.max()}"
        )

    held = np.asarray(list(held_out_group_ids))
    if held.size == 0 or idx.size == 0:
        return idx
    return idx[~np.isin(groups[idx], held)]
