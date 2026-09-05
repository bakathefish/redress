"""Insert the round-5 macro block into paper/build_numbers.py (idempotent).

The block reads recovery/round5/subsets.json (written by paper/round5_subsets.py)
and, when present, recovery/round5/barro_membership.json (paper/round5_barro_membership.py),
checks every value that must agree with an existing macro, and adds the round-5 macros.
It is inserted immediately before the json.dump line that writes numbers.json.
"""

from __future__ import annotations

import os
import sys

BN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "paper",
    "build_numbers.py",
)
MARK = "# --- round 5 (publication rewrite) ---"

BLOCK = r"""
# --- round 5 (publication rewrite) ---
# Subset counts and tests computed by paper/round5_subsets.py and the corrected
# floor computed by paper/round5_barro_membership.py, both deterministic scripts
# over the tables of record (rulings: paper/DIRECTOR_round5.md). Every value
# that must agree with a macro above is asserted against it here.
_R5 = json.load(open(os.path.join(ROOT, "recovery", "round5", "subsets.json")))["numbers"]
assert _R5["hvA1Rows"] == N["listHvidingA"]
assert _R5["hvA1AllKocevski"] == N["kocevskiModuleOfHviding"]
assert _R5["hvA1AllKokorev"] == N["kokorevModuleOfHviding"]
assert _R5["hvA1AllBarro"] == N["barroModuleOfHviding"]
assert _R5["hvA1AllUnion"] == N["unionCatalogHvidingA"]
assert _R5["mcnemarModelUnionBCheck"] == N["mcnemarModelUnionB"]
assert _R5["mcnemarModelUnionCCheck"] == N["mcnemarModelUnionC"]
assert _R5["blReplicaMatchedRecallCheck"] == N["blReplicaMatchedRecall"]
assert _R5["blAblateSizeMatchedRecallCheck"] == N["blAblateSizeMatchedRecall"]
assert (
    _R5["mcnemarReplicaNoSizeB"] - _R5["mcnemarReplicaNoSizeC"]
    == N["blReplicaMatchedRecall"] - N["blAblateSizeMatchedRecall"]
)
assert _R5["unionSixSelected"] + _R5["unionSixSelectedFewer"] == N["unionSelected"]
# Hviding et al. (2025) table A1 objects at their own magnitude limit, F444W < 26.5
N["hvidingBrightN"] = _R5["hvA1BrightN"]
N["hvidingBrightKocevski"] = _R5["hvA1BrightKocevski"]
N["hvidingBrightKokorev"] = _R5["hvA1BrightKokorev"]
N["hvidingBrightBarro"] = _R5["hvA1BrightBarro"]
N["hvidingBrightUnion"] = _R5["hvA1BrightUnion"]
for _k in ("Kocevski", "Kokorev", "Barro", "Union"):
    N["hvidingBright%sPct" % _k] = round(100.0 * N["hvidingBright%s" % _k] / N["hvidingBrightN"], 1)
# positives brighter than F444W = 26, and on the footing of Pan et al. (2026)
N["nPosBrightTwentySix"] = _R5["nPosBrightTwentySix"]
N["unionRecallBrightTwentySix"] = _R5["unionRecallBrightTwentySix"]
N["nRuleMissedBrightTwentySix"] = _R5["nRuleMissedBrightTwentySix"]
N["modelRecallBrightTwentySix"] = _R5["modelRecallBrightTwentySix"]
N["unionRecallBrightTwentySixPct"] = round(100.0 * _R5["unionRecallBrightTwentySix"] / _R5["nPosBrightTwentySix"], 1)
N["modelRecallBrightTwentySixPct"] = round(100.0 * _R5["modelRecallBrightTwentySix"] / _R5["nPosBrightTwentySix"], 1)
N["nPosBrightPan"] = _R5["nPosBrightPan"]
N["unionRecallBrightPan"] = _R5["unionRecallBrightPan"]
N["modelRecallBrightPan"] = _R5["modelRecallBrightPan"]
N["unionRecallBrightPanPct"] = round(100.0 * _R5["unionRecallBrightPan"] / _R5["nPosBrightPan"], 1)
# the union without kocevski24, and with the published Kocevski catalogue in its place
N["unionSixSelected"] = _R5["unionSixSelected"]
N["unionSixSelectedFewer"] = _R5["unionSixSelectedFewer"]
N["unionSixRecallEnd"] = _R5["unionSixRecallEnd"]
N["unionSixRecallSupport"] = _R5["unionSixRecallSupport"]
N["unionSixAnchors"] = _R5["unionSixAnchors"]
N["modelAtUnionSixBurden"] = _R5["modelAtUnionSixBurden"]
N["unionKocCatSelected"] = _R5["unionKocCatSelected"]
N["unionKocCatRecallEnd"] = _R5["unionKocCatRecallEnd"]
# paired test, eight-bag replica against the no-size ablation, both at matched burden
N["mcnemarReplicaNoSizeB"] = _R5["mcnemarReplicaNoSizeB"]
N["mcnemarReplicaNoSizeC"] = _R5["mcnemarReplicaNoSizeC"]
N["mcnemarReplicaNoSizeP"] = pfmt(mcnemar_p(_R5["mcnemarReplicaNoSizeB"], _R5["mcnemarReplicaNoSizeC"]))
assert N["mcnemarReplicaNoSizeP"] == _R5["mcnemarReplicaNoSizeP"]
# the two paired tests added on Reviewer B's rulings M1 and M3, when the script has them
if "mcnemarAnchorHalfPrimaryB" in _R5:
    N["mcnemarAnchorHalfPrimaryB"] = _R5["mcnemarAnchorHalfPrimaryB"]
    N["mcnemarAnchorHalfPrimaryC"] = _R5["mcnemarAnchorHalfPrimaryC"]
    N["mcnemarAnchorHalfPrimaryP"] = pfmt(mcnemar_p(_R5["mcnemarAnchorHalfPrimaryB"], _R5["mcnemarAnchorHalfPrimaryC"]))
if "mcnemarNonCompactB" in _R5:
    N["mcnemarNonCompactB"] = _R5["mcnemarNonCompactB"]
    N["mcnemarNonCompactC"] = _R5["mcnemarNonCompactC"]
    N["mcnemarNonCompactP"] = pfmt(mcnemar_p(_R5["mcnemarNonCompactB"], _R5["mcnemarNonCompactC"]))
if "modelPooledAtUnionBurden" in _R5:
    N["modelPooledAtUnionBurden"] = _R5["modelPooledAtUnionBurden"]
# The Bonferroni family is every hypothesis test the paper prints a p value for. The list
# is kept here so that the count can be checked against the text; the blue-bin McNemar of
# Section 5.2 is printed as counts only and is not in it.
_R5_TESTS = [
    "mcnemar primary: learned vs union, in support",
    "mcnemar imitation vs union, matched",
    "mcnemar primary vs imitation, matched",
    "mcnemar primary vs imitation, rule-missed, matched",
    "mcnemar primary vs positives-against-anchors",
    "mcnemar replica vs no-size, matched",
    "fisher compactness, missed vs kept",
    "fisher V-shape pass, missed vs kept",
    "sign test, five regions",
    "mann-whitney colour",
    "mann-whitney magnitude",
    "mann-whitney colour on compact",
]
if "mcnemarAnchorHalfPrimaryB" in N:
    _R5_TESTS.append("mcnemar anchor-weight-0.5 vs primary, matched")
if "mcnemarNonCompactB" in N:
    _R5_TESTS.append("mcnemar learned vs union on the non-compact positives")
_R5_WORDS = {12: "twelve", 13: "thirteen", 14: "fourteen"}
N["bonferroniFamily"] = len(_R5_TESTS)
N["bonferroniFamilyWord"] = _R5_WORDS[len(_R5_TESTS)]
N["bonferroniHeadlineP"] = pfmt(min(1.0, len(_R5_TESTS) * mcnemar_p(N["mcnemarModelUnionB"], N["mcnemarModelUnionC"])))
# the corrected floor, once round5_barro_membership.py has run
_R5B_PATH = os.path.join(ROOT, "recovery", "round5", "barro_membership.json")
if os.path.exists(_R5B_PATH):
    _R5B = json.load(open(_R5B_PATH))
    _R5B = _R5B.get("numbers", _R5B)
    for _k, _v in _R5B.items():
        if isinstance(_v, (int, float, str)) and not _k.startswith("_"):
            N[_k] = _v
# --- end round 5 ---
"""


def main() -> int:
    with open(BN, encoding="utf-8") as f:
        text = f.read()
    if MARK in text:
        print("block already present; nothing to do")
        return 0
    anchor = 'json.dump(N, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=float)'
    if text.count(anchor) != 1:
        print("anchor not found exactly once")
        return 1
    text = text.replace(anchor, BLOCK.lstrip("\n") + "\n" + anchor)
    with open(BN, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("block inserted into", BN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
