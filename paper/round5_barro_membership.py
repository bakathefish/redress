"""Round 5, Reviewer A: the floor recounted against the Barro et al. photometric selection.

Reviewer A observed that the paper's floor (rule-missed, not in the Kocevski et al. 2024
catalogue, not in the Perger et al. 2025 compilation) treats the whole Barro spectroscopic
list as if it carried no photometric selection, when in fact Barro et al. assembled that
list by applying their own published photometric cut (Barro et al. 2024b) to about a
thousand photometric LRDs and then adding exactly seven compact spectroscopic LRDs by hand
because the photometric selection missed them. Every member of the list except those seven
is therefore inside a published photometric selection, and the floor has to be recounted
against it.

This script reads only

  recovery/round5/barro25_repo/            the public table, git clone of
                                          https://github.com/guillermobc/Barro25
  recovery/labels.parquet               the label table of record
  recovery/features.parquet             picked / in_support / region / list flags
  recovery/published_catalogues_extra.parquet   z_spec (round 3 record)
  recovery/oof_scores.parquet           sel_burden, the equal-burden learned selection
  the Barro et al. (2025) arXiv source (ms.tex)   the Barro manuscript, for the names
                                          of the seven manual additions only

and writes recovery/round5/barro_membership.json. It changes nothing. Every definition is
taken from paper/build_numbers.py at the line cited beside it, and every quantity
that file already publishes is re-derived here and asserted equal, so a drift in either
file fails this script rather than passing silently.

Run from the repository root:
    python paper/round5_barro_membership.py
"""

import json
import os

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
V = os.path.join(ROOT, "recovery")
REPO = os.path.join(ROOT, "recovery", "round5", "barro25_repo")
OUT = os.path.join(ROOT, "recovery", "round5", "barro_membership.json")

MATCH_ARCSEC = (
    0.5  # the brief's matching radius, and the one Barro et al. use themselves
)
# (ms.tex line 267, "a 0.5 matching radius"); it is also the radius
# recovery/build_labels.py used to set in_barro25 in the first place.

R = {}  # the result document


def note(k, v):
    R[k] = v
    return v


# =========================================================================================
# 1. the repository as downloaded
# =========================================================================================
def read_barro():
    """The Barro et al. spectroscopic table, both variants, as one frame.

    The two FITS files carry the same 118 sources in the same row order with identical ra
    and dec. They differ in one row and one respect only: row 32 has the sentinel DJA_ID
    '-99--99' and zspec -99 in the modified-blackbody file and the real values in the
    best-fit file. The identifier and redshift are therefore taken from the best-fit file
    wherever the blackbody file holds the sentinel, which is a repair of a single missing
    entry and not a change of source.
    """
    inv = []
    tabs = {}
    for fn in sorted(os.listdir(REPO)):
        p = os.path.join(REPO, fn)
        if not os.path.isfile(p):
            continue
        rec = {"file": fn, "bytes": os.path.getsize(p)}
        if fn.endswith(".fits"):
            with fits.open(p) as h:
                hdu = h[1]
                rec["rows"] = int(len(hdu.data))
                rec["columns"] = [c.name for c in hdu.columns]
                rec["n_columns"] = len(hdu.columns)
                tabs[fn] = {c.name: np.asarray(hdu.data[c.name]) for c in hdu.columns}
        inv.append(rec)
    note("repoInventory", inv)

    mbb = tabs["Barro25_LRD_NIRSpec_MBB_properties.fits"]
    bf = tabs["Barro25_LRD_NIRSpec_bestfit_properties.fits"]
    n = len(mbb["DJA_ID"])
    assert len(bf["DJA_ID"]) == n, "the two repository tables have different lengths"
    ida = [str(x).strip() for x in mbb["DJA_ID"]]
    idb = [str(x).strip() for x in bf["DJA_ID"]]
    za = np.asarray(mbb["zspec"], dtype=float)
    zb = np.asarray(bf["zspec"], dtype=float)
    assert np.allclose(np.asarray(mbb["ra"], float), np.asarray(bf["ra"], float)), (
        "the two repository tables disagree on ra"
    )
    assert np.allclose(np.asarray(mbb["dec"], float), np.asarray(bf["dec"], float)), (
        "the two repository tables disagree on dec"
    )

    sentinel = [i for i in range(n) if ida[i] == "-99--99" or za[i] == -99.0]
    note(
        "repoSentinelRows",
        [
            {
                "row": int(i),
                "mbb_dja_id": ida[i],
                "mbb_zspec": float(za[i]),
                "bestfit_dja_id": idb[i],
                "bestfit_zspec": float(zb[i]),
            }
            for i in sentinel
        ],
    )
    ids = [idb[i] if i in sentinel else ida[i] for i in range(n)]
    zs = np.array([zb[i] if i in sentinel else za[i] for i in range(n)])
    assert len(set(ids)) == n, "the repository table has a duplicate DJA_ID"
    assert (zs > 0).all(), "a repository row still has no redshift"

    b = pd.DataFrame(
        {
            "brow": np.arange(n),
            "dja_id": ids,
            "bra": np.asarray(mbb["ra"], float),
            "bdec": np.asarray(mbb["dec"], float),
            "bz": zs,
            "uv_opt_color": np.asarray(mbb["uv_opt_color"], float),
            "uv_slope": np.asarray(mbb["uv_slope"], float),
        }
    )
    note("repoRows", int(n))
    note("repoHasPhotometricSample", False)
    note(
        "repoManualAdditionFlagColumn",
        None,  # no column in either file flags the manual additions or the BLAGN class
    )
    return b


bar = read_barro()

# =========================================================================================
# 2. the tables of record, and the sky match
# =========================================================================================
lab = pd.read_parquet(
    os.path.join(V, "labels.parquet"),
    columns=["source_id", "field", "id", "ra", "dec", "y"],
)
L = lab[lab.y.notna()].reset_index(
    drop=True
)  # every labelled object: positives and anchors
note("labelledObjects", int(len(L)))

idx, sep, _ = SkyCoord(
    bar.bra.values * u.deg, bar.bdec.values * u.deg
).match_to_catalog_sky(SkyCoord(L.ra.values * u.deg, L.dec.values * u.deg))
bar["sep_arcsec"] = sep.arcsec
bar["lab_row"] = idx
hit = bar.sep_arcsec.values < MATCH_ARCSEC
bar["matched"] = hit
bar["source_id"] = np.where(hit, L.source_id.values[idx], -1)
note("repoRowsMatched", int(hit.sum()))
note("repoRowsUnmatched", int((~hit).sum()))
note("repoMatchMaxSepArcsec", round(float(bar.sep_arcsec[hit].max()), 4))
note("repoMatchMinUnmatchedSepArcsec", round(float(bar.sep_arcsec[~hit].min()), 1))
note(
    "repoUnmatchedRows",
    [
        {
            "row": int(r.brow),
            "dja_id": r.dja_id,
            "ra": float(r.bra),
            "dec": float(r.bdec),
            "zspec": round(float(r.bz), 4),
            "nearest_labelled_arcsec": round(float(r.sep_arcsec), 1),
        }
        for _, r in bar[~hit].iterrows()
    ],
)
assert bar.source_id[hit].duplicated().sum() == 0, (
    "two repository rows share one source"
)

# --- the assertion the brief asks for: the match reproduces in_barro25 -------------------
f = pd.read_parquet(os.path.join(V, "features.parquet"))
posAll = f[f.y == 1]  # build_numbers.py line 138: the 151 positives
note("nPos", int(len(posAll)))

flagged = set(
    int(s)
    for s in lab.loc[
        pd.read_parquet(os.path.join(V, "labels.parquet"), columns=["in_barro25"])
        .in_barro25.astype(bool)
        .values,
        "source_id",
    ]
)
matched_ids = set(int(s) for s in bar.source_id[hit])
note("nFlaggedInBarro25", len(flagged))
note("nMatchedToBarroTable", len(matched_ids))
note("flaggedNotMatched", sorted(flagged - matched_ids))
note("matchedNotFlagged", sorted(matched_ids - flagged))
assert flagged == matched_ids, (
    "the 0.5 arcsec match to the public Barro table does not reproduce in_barro25"
)
assert set(int(s) for s in posAll.source_id) >= matched_ids, (
    "a matched Barro row is not one of the 151 positives"
)

# =========================================================================================
# 3. the seven manual additions
# =========================================================================================
# There is no column in the repository that flags them, so they are resolved from the names
# Barro et al. give in the manuscript that points at this repository
# (the Barro et al. (2025) arXiv source (ms.tex)). Section 3.1, line 299, lists the
# seven and their redshifts; line 520 gives the DJA-style identifier for three of them
# ("1181-28074" for the Rosetta AGN, "RUBIES-UDS-154183" for The Cliff, "CAPERS-BH*-1")
# and line 550 names the last as "CAPERS-119334". Each anchor below is resolved against the
# downloaded table, never typed: an anchor is either an identifier token that must appear as
# the trailing field of exactly one DJA_ID, or a survey prefix plus a stated property, and
# in both cases the script asserts that exactly one row satisfies it.
MANUAL = [
    # (key, paper name, source of the anchor, kind, anchor, quoted z)
    (
        "blagn1",
        "RUBIES-BLAGN-1 (RUBIES-UDS-BLAGN1)",
        "wang24_lrd",
        "prefix_z",
        "rubies-uds",
        3.11,
    ),
    ("jades12402", "JADES-1180-12402", "setton24/williams23", "token", "12402", 3.19),
    (
        "rosetta",
        'the "Rosetta" AGN (1181-28074)',
        "juodzbalis24a",
        "token",
        "28074",
        2.26,
    ),
    ("uds144195", "RUBIES-UDS-144195", "hviding25", "token", "144195", 3.404),
    ("cliff", "The Cliff (RUBIES-UDS-154183)", "degraaff25", "token", "154183", 3.55),
    ("mombhstar", "MoM-BH*-1", "naidu25", "prefix_color", "mom", 7.76),
    (
        "capersbhstar",
        "CAPERS-BH*-1 (CAPERS-119334)",
        "taylor25",
        "token",
        "119334",
        9.29,
    ),
]
BHSTAR_COLOR = 4.0  # ms.tex line 529: "all three of the extreme BH*-like LRDs with
# [UV-to-optical colour] >~ 4"; used only to pick MoM-BH*-1 out of the
# four mom- rows, and asserted unique.
ZTOL_PREFIX = (
    0.05  # the tolerance on the manuscript's quoted redshift for the one anchor
)
# resolved by survey prefix and redshift

manual_rows = []
for key, name, cite, kind, anchor, zq in MANUAL:
    if kind == "token":
        cand = bar[bar.dja_id.str.endswith("-" + anchor)]
    elif kind == "prefix_z":
        c = bar[bar.dja_id.str.startswith(anchor)]
        cand = c[(c.bz - zq).abs() < ZTOL_PREFIX]
    elif kind == "prefix_color":
        c = bar[bar.dja_id.str.startswith(anchor)]
        cand = c[c.uv_opt_color >= BHSTAR_COLOR]
    assert len(cand) == 1, (
        f"the anchor for {name} matches {len(cand)} rows, not one: "
        + ", ".join(cand.dja_id.tolist())
    )
    r = cand.iloc[0]
    assert bool(r.matched), (
        f"{name} is not in the nine fields, so it cannot be a positive"
    )
    manual_rows.append(
        {
            "key": key,
            "name": name,
            "barro_citation": cite,
            "anchor_kind": kind,
            "anchor": anchor,
            "barro_row": int(r.brow),
            "dja_id": r.dja_id,
            "barro_zspec": round(float(r.bz), 4),
            "quoted_zspec": zq,
            "dz_quoted_minus_table": round(float(zq - r.bz), 4),
            "uv_opt_color": round(float(r.uv_opt_color), 3),
            "source_id": int(r.source_id),
            "sep_arcsec": round(float(r.sep_arcsec), 4),
        }
    )
manual_ids = set(m["source_id"] for m in manual_rows)
assert len(manual_ids) == len(MANUAL), (
    "two manual-addition anchors resolve to one source"
)
note("manualAdditions", manual_rows)
note("nManualAdditions", len(manual_rows))
note("nManualAdditionsInFields", len(manual_ids))
# the three the manuscript calls black hole stars must be the three reddest of the seven and
# must clear the colour the manuscript states for them
_bh = [m for m in manual_rows if m["key"] in ("cliff", "mombhstar", "capersbhstar")]
assert all(m["uv_opt_color"] >= BHSTAR_COLOR for m in _bh), (
    "a source identified as one of the three black hole stars is bluer than the manuscript "
    "says all three are"
)
note("bhStarColours", {m["key"]: m["uv_opt_color"] for m in _bh})

# =========================================================================================
# 4. the sets, exactly as build_numbers.py builds them
# =========================================================================================
P = posAll.set_index("source_id")
x = pd.read_parquet(os.path.join(V, "published_catalogues_extra.parquet")).set_index(
    "source_id"
)
P = P.join(
    x[["z_spec", "z_spec_source"]], how="left"
)  # build_numbers.py lines 1590-1596
assert len(P) == len(posAll), "the join changed the positive count"
assert P.z_spec.notna().all(), "a positive has no spectroscopic redshift"

oof = pd.read_parquet(os.path.join(V, "oof_scores.parquet")).set_index("source_id")

picked = P.picked.astype(bool)  # line 1200, the union
miss = ~picked  # line 1200, nRuleMissed
inK = P.in_kocevski24.astype(bool)  # line 583
inP = P.in_perger25.astype(bool)  # line 584
floor = miss & ~(inK | inP)  # lines 596-598 / 1215-1217
inSup = P.in_support.astype(bool)  # line 135
selb = oof["sel_burden"].reindex(P.index).eq(True)  # line 1327, modelRecallEnd
inB = P.in_barro25.astype(bool)
inH = P.in_hviding25_A1.astype(bool)
inD = P.in_degraaff26.astype(bool)
z = P.z_spec
region = P.region

isManual = pd.Series(P.index.isin(list(manual_ids)), index=P.index)
barroPhot = inB & ~isManual  # in Barro's published photometric selection
floorBarro = floor & ~barroPhot  # the corrected floor

# --- reproduce every published quantity these definitions feed --------------------------
NUM = json.load(open(os.path.join(ROOT, "paper", "numbers.json")))
CHECK = []


def check(name, got, key):
    want = NUM[key]
    CHECK.append(
        {
            "quantity": name,
            "numbers_json_key": key,
            "value": int(got),
            "published": int(want),
        }
    )
    assert int(got) == int(want), f"{name}: rebuilt {got}, numbers.json {want}"


check("positives", len(P), "nPos")
check("rule-missed", miss.sum(), "nRuleMissed")
check("union recovery end to end", picked.sum(), "unionRecallEnd")
check("union recovery in support", (picked & inSup).sum(), "unionRecallSupport")
check("floor", floor.sum(), "nNoSelection")
check("floor in GOODS-North", (floor & (region == "GOODS-N")).sum(), "nFloorInGOODSN")
check("Barro list members among the positives", inB.sum(), "barroListN")
check("learned recovery end to end", selb.sum(), "modelRecallEnd")
check("paired both", (picked & selb).sum(), "pairedBoth")
check("paired learned only", (selb & ~picked).sum(), "pairedModelOnly")
check("paired rules only end to end", (picked & ~selb).sum(), "pairedUnionOnlyEnd")
check("paired neither end to end", (~picked & ~selb).sum(), "pairedNeitherEnd")

SUBSETS = [
    ("All", pd.Series(True, index=P.index), None, None),
    ("ZgeFour", z >= 4.0, "nPosZgeFour", "floorZgeFour"),  # line 1666
    ("ZinRange", z > 3.0, "nPosZinRange", "floorZinRange"),  # line 1672
    (
        "TwoPlus",
        P[["in_hviding25_A1", "in_barro25", "in_degraaff26"]].astype(bool).sum(axis=1)
        >= 2,
        "nPosTwoPlus",
        "floorTwoPlus",
    ),  # lines 1692-1696
]
for tag, m, nkey, fkey in SUBSETS:
    if nkey:
        check("positives, " + tag, m.sum(), nkey)
        check("published floor, " + tag, (m & floor).sum(), fkey)
note("checks", CHECK)

# =========================================================================================
# 5. the numbers the brief asks for
# =========================================================================================
note("nPosInBarroPhot", int(barroPhot.sum()))
note("nPosInBarroPhotPct", round(100 * float(barroPhot.sum()) / len(P), 1))
# the repository holds the spectroscopic table only, so there is no photometric sample to
# match the remaining positives against; this stays null rather than zero
note("nPosInBarroPhotExtra", None)
note(
    "nPosInBarroPhotExtraNote",
    "not computable: the public repository contains the 118-row spectroscopic table only, "
    "not the roughly one thousand photometric LRDs of Barro et al. Section 3.1",
)
note("nRuleMissedInBarroPhot", int((miss & barroPhot).sum()))
note(
    "nRuleMissedInBarroPhotPct",
    round(100 * float((miss & barroPhot).sum()) / float(miss.sum()), 1),
)

for tag, m, _, _ in SUBSETS:
    n = int(m.sum())
    suffix = "" if tag == "All" else tag
    note("nPos" + (suffix or ""), n) if suffix else None
    note("floorBarro" + suffix, int((m & floorBarro).sum()))
    note(
        "floorBarro" + suffix + "Pct", round(100 * float((m & floorBarro).sum()) / n, 1)
    )
    note("floorPublished" + suffix, int((m & floor).sum()))
    note(
        "floorMovedByBarro" + suffix,
        int((m & floor).sum()) - int((m & floorBarro).sum()),
    )
    note("nPosDen" + (suffix or "All"), n)

note("floorBarroInGOODSN", int((floorBarro & (region == "GOODS-N")).sum()))
note("floorPublishedInGOODSN", int((floor & (region == "GOODS-N")).sum()))
note(
    "floorBarroByRegion",
    {str(k): int(v) for k, v in region[floorBarro].value_counts().sort_index().items()},
)

# --- the subset of positives selected on a spectroscopic LRD criterion -------------------
# Hviding et al. table A1 and the de Graaff et al. list are the two of the three that apply a
# criterion to the spectrum itself; the Barro list is the one assembled photometrically.
spec = inH | inD
sm = spec
sms = spec & inSup
S = {}
S["nPosSpecCrit"] = int(sm.sum())
S["nPosSpecCritSupport"] = int(sms.sum())
S["unionRecallEnd"] = int((sm & picked).sum())
S["unionRecallEndPct"] = round(100 * S["unionRecallEnd"] / S["nPosSpecCrit"], 1)
S["unionRecallSupport"] = int((sms & picked).sum())
S["unionRecallSupportPct"] = round(
    100 * S["unionRecallSupport"] / S["nPosSpecCritSupport"], 1
)
S["nRuleMissed"] = int((sm & miss).sum())
S["nRuleMissedPct"] = round(100 * S["nRuleMissed"] / S["nPosSpecCrit"], 1)
S["nRuleMissedSupport"] = int((sms & miss).sum())
S["floorPublished"] = int((sm & floor).sum())
S["floorPublishedPct"] = round(100 * S["floorPublished"] / S["nPosSpecCrit"], 1)
S["floorBarro"] = int((sm & floorBarro).sum())
S["floorBarroPct"] = round(100 * S["floorBarro"] / S["nPosSpecCrit"], 1)
S["modelRecallEnd"] = int((sm & selb).sum())
S["modelRecallEndPct"] = round(100 * S["modelRecallEnd"] / S["nPosSpecCrit"], 1)
S["modelRecallSupport"] = int((sms & selb).sum())
S["modelGainEnd"] = S["modelRecallEnd"] - S["unionRecallEnd"]
S["pairedBothEnd"] = int((sm & picked & selb).sum())
S["pairedLearnedOnlyEnd"] = int((sm & selb & ~picked).sum())
S["pairedRulesOnlyEnd"] = int((sm & picked & ~selb).sum())
S["pairedNeitherEnd"] = int((sm & ~picked & ~selb).sum())
S["pairedBothSupport"] = int((sms & picked & selb).sum())
S["pairedLearnedOnlySupport"] = int((sms & selb & ~picked).sum())
S["pairedRulesOnlySupport"] = int((sms & picked & ~selb).sum())
S["pairedNeitherSupport"] = int((sms & ~picked & ~selb).sum())
S["nInHvidingA1"] = int(inH.sum())
S["nInDeGraaff"] = int(inD.sum())
S["nInBoth"] = int((inH & inD).sum())
S["nAlsoInBarro"] = int((sm & inB).sum())
S["nAlsoInBarroPhot"] = int((sm & barroPhot).sum())
assert (
    S["pairedBothEnd"]
    + S["pairedLearnedOnlyEnd"]
    + S["pairedRulesOnlyEnd"]
    + S["pairedNeitherEnd"]
    == S["nPosSpecCrit"]
), "the end-to-end paired counts do not partition the subset"
assert (
    S["pairedBothSupport"]
    + S["pairedLearnedOnlySupport"]
    + S["pairedRulesOnlySupport"]
    + S["pairedNeitherSupport"]
    == S["nPosSpecCritSupport"]
), "the in-support paired counts do not partition the subset"
note("specCrit", S)

# =========================================================================================
# 6. addendum: the rule-missed positives split by the learned selection, and by Barro
# =========================================================================================
# Asked for after the first pass. The question behind it is whether the objects the learned
# selection rescues are the ones a published photometric selection already held, or the ones
# nothing held. `selb` is the same equal-burden out-of-fold flag that gives modelRecallEnd,
# and `modelRuleMissed` at build_numbers.py line 258 is the count of rule-missed positives it
# recovers, which is asserted here rather than assumed.
NUMB = {}
NUMB["nPos"] = int(len(P))
NUMB["nRuleMissed"] = int(miss.sum())
NUMB["nRuleMissedRecoveredByModel"] = int((miss & selb).sum())
NUMB["nRuleMissedNotRecoveredByModel"] = int((miss & ~selb).sum())
check(
    "rule-missed recovered by the learned selection",
    (miss & selb).sum(),
    "modelRuleMissed",
)
assert (
    NUMB["nRuleMissedRecoveredByModel"] + NUMB["nRuleMissedNotRecoveredByModel"]
    == NUMB["nRuleMissed"]
), "the learned-selection split does not partition the rule-missed positives"

NUMB["nPosInBarroPhot"] = int(barroPhot.sum())
NUMB["nRuleMissedInBarroPhot"] = int((miss & barroPhot).sum())
NUMB["nRuleMissedNotInBarroPhot"] = int((miss & ~barroPhot).sum())
# the two by two the addendum asks for
NUMB["nRuleMissedRecoveredInBarroPhot"] = int((miss & selb & barroPhot).sum())
NUMB["nRuleMissedRecoveredNotInBarroPhot"] = int((miss & selb & ~barroPhot).sum())
NUMB["nRuleMissedNotRecoveredInBarroPhot"] = int((miss & ~selb & barroPhot).sum())
NUMB["nRuleMissedNotRecoveredNotInBarroPhot"] = int((miss & ~selb & ~barroPhot).sum())
assert (
    NUMB["nRuleMissedRecoveredInBarroPhot"]
    + NUMB["nRuleMissedRecoveredNotInBarroPhot"]
    + NUMB["nRuleMissedNotRecoveredInBarroPhot"]
    + NUMB["nRuleMissedNotRecoveredNotInBarroPhot"]
    == NUMB["nRuleMissed"]
), "the two by two does not partition the rule-missed positives"
NUMB["nRuleMissedRecoveredInBarroPhotPct"] = round(
    100 * NUMB["nRuleMissedRecoveredInBarroPhot"] / NUMB["nRuleMissedRecoveredByModel"],
    1,
)
NUMB["nRuleMissedNotRecoveredInBarroPhotPct"] = round(
    100
    * NUMB["nRuleMissedNotRecoveredInBarroPhot"]
    / NUMB["nRuleMissedNotRecoveredByModel"],
    1,
)

NUMB["floorPublished"] = int(floor.sum())
NUMB["floorBarro"] = int(floorBarro.sum())
NUMB["floorBarroRecoveredByModel"] = int((floorBarro & selb).sum())
NUMB["floorBarroNotRecoveredByModel"] = int((floorBarro & ~selb).sum())
NUMB["floorPublishedRecoveredByModel"] = int((floor & selb).sum())
assert (
    NUMB["floorBarroRecoveredByModel"] + NUMB["floorBarroNotRecoveredByModel"]
    == NUMB["floorBarro"]
), "the learned-selection split does not partition the corrected floor"
# what is left once both the published photometric selections and the learned selection are
# given their due: rule-missed, outside every published photometric selection, and not
# recovered out of fold either
NUMB["floorBarroNotRecoveredByModelPct"] = round(
    100 * NUMB["floorBarroNotRecoveredByModel"] / NUMB["nPos"], 1
)
note("numbers", NUMB)

# --- the corrected floor, object by object ----------------------------------------------
LISTS = [
    ("in_hviding25_A1", "Hviding A1"),
    ("in_barro25", "Barro"),
    ("in_degraaff26", "de Graaff"),
]


def lists_holding(r):
    """The published lists that hold this object, named. Empty means it is one of the
    positives admitted by the fixed spectral test rather than by a list."""
    return [name for col, name in LISTS if bool(getattr(r, col))]


fl = P[floorBarro]
note(
    "floorBarroObjects",
    [
        {
            "source_id": int(sid),
            "field": r.field,
            "region": r.region,
            "catalogue_id": int(r.id),
            "ra": round(float(r.ra), 6),
            "dec": round(float(r.dec), 6),
            "z_spec": round(float(r.z_spec), 4),
            "z_spec_source": r.z_spec_source,
            "lists": lists_holding(r),
            "n_lists": len(lists_holding(r)),
            "in_hviding25_A1": bool(r.in_hviding25_A1),
            "in_barro25": bool(r.in_barro25),
            "in_degraaff26": bool(r.in_degraaff26),
            "is_barro_manual_addition": int(sid) in manual_ids,
            "in_support": bool(r.in_support),
            "learned_selected": bool(selb.loc[sid]),
        }
        for sid, r in fl.sort_values(["region", "z_spec"]).iterrows()
    ],
)
# the objects the Barro correction removes from the floor, for the report
rm = P[floor & barroPhot]
note(
    "floorRemovedByBarro",
    [
        {
            "source_id": int(sid),
            "field": r.field,
            "region": r.region,
            "catalogue_id": int(r.id),
            "z_spec": round(float(r.z_spec), 4),
            "in_hviding25_A1": bool(r.in_hviding25_A1),
            "in_degraaff26": bool(r.in_degraaff26),
        }
        for sid, r in rm.sort_values(["region", "z_spec"]).iterrows()
    ],
)

# --- round 5 close (director addendum): the rule-missed positives split by whether the
# learned selection recovers them at equal burden out of fold, against membership of the
# Barro et al. photometric selection and of the two matched catalogues. Same sets as above.
recovered = miss & selb
note("nRuleMissedRecovered", int(recovered.sum()))
note("nRuleMissedNotRecovered", int((miss & ~selb).sum()))
note("nRuleMissedRecoveredInBarroPhot", int((recovered & barroPhot).sum()))
note("nRuleMissedNotRecoveredInBarroPhot", int((miss & ~selb & barroPhot).sum()))
note("nRuleMissedRecoveredInMatchedCat", int((recovered & (inK | inP)).sum()))
note(
    "nRuleMissedRecoveredInAnyPublished",
    int((recovered & (barroPhot | inK | inP)).sum()),
)
note("floorBarroRecoveredByModel", int((floorBarro & selb).sum()))
note("floorBarroInSupport", int((floorBarro & inSup).sum()))
note("nPosInAnyMatchedSelection", int((barroPhot | inK | inP).sum()))
note("nPosInNoMatchedSelection", int((~(barroPhot | inK | inP)).sum()))
assert (
    R["nRuleMissedRecoveredInBarroPhot"] + R["nRuleMissedNotRecoveredInBarroPhot"]
    == R["nRuleMissedInBarroPhot"]
)
assert (
    R["nRuleMissedRecoveredInAnyPublished"] + R["floorBarroRecoveredByModel"]
    == R["nRuleMissedRecovered"]
)
assert R["nPosInAnyMatchedSelection"] + R["nPosInNoMatchedSelection"] == R["nPos"]
assert int((miss & ~(barroPhot | inK | inP)).sum()) == R["floorBarro"]

R["_provenance"] = {
    "repo_url": "https://github.com/guillermobc/Barro25",
    "repo_commit": "e70d8ac577a2e33680d9d27557b5b7ced598a900",
    "repo_path": os.path.relpath(REPO, ROOT).replace("\\", "/"),
    "match_radius_arcsec": MATCH_ARCSEC,
    "manual_addition_source": "the Barro et al. (2025) arXiv source (ms.tex) lines 299, 520, 529, 550",
    "definitions_from": "paper/build_numbers.py (line numbers cited in the source)",
}

with open(OUT, "w") as fh:
    json.dump(R, fh, indent=2, sort_keys=False)
print("wrote", os.path.relpath(OUT, ROOT))
print()
print("nPosInBarroPhot           ", R["nPosInBarroPhot"], "of", R["nPos"])
print("nRuleMissedInBarroPhot    ", R["nRuleMissedInBarroPhot"], "of", int(miss.sum()))
for tag in ("", "ZgeFour", "ZinRange", "TwoPlus"):
    t = tag or "All"
    print(
        "floor %-9s published %2d -> corrected %2d (%.1f%% of %d)"
        % (
            t,
            R["floorPublished" + tag],
            R["floorBarro" + tag],
            R["floorBarro" + tag + "Pct"],
            R["nPosDen" + (tag or "All")],
        )
    )
print("floorBarroInGOODSN        ", R["floorBarroInGOODSN"])
print("specCrit                  ", json.dumps(S))
