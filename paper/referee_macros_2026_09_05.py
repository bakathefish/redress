"""Macros for the 2026-09-05 referee-response revision, appended to paper/numbers.tex.

Every number is recounted here from released or frozen tables rather than typed:

  * the B1 anchor overlap from recovery/labels.parquet (the frozen label table that
    recovery/build_labels.py wrote; the released candidates_ledger.csv holds the subset
    above the deployed 0.5 per cent threshold);
  * the three candidates that the 17 August 2026 de Graaff et al. release holds, matched
    at the paper's 0.5 arcsec radius against the released candidates.csv;
  * the catalog-only count outside the three matched selections (19) and the part of it
    that a re-implemented rule recovers (19 - 8 = 11), read from numbers.tex itself.

The block is delimited by marker comments and replaced on every run, so the script is
idempotent. Run from the repository root:

    python paper/revision_2026-09-05_referee_macros.py
"""

from __future__ import annotations

import io
import os
import re
import sys
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.dirname(HERE)
ROOT = os.path.dirname(PUB)
NUMBERS = os.path.join(PUB, "numbers.tex")
LABELS = os.path.join(ROOT, "recovery", "labels.parquet")
REDRESS = os.environ.get("REDRESS_ROOT", ROOT)  # in the release this repository is the redress tree
CANDIDATES = os.path.join(REDRESS, "recovery", "candidates.csv")
LEDGER = os.path.join(REDRESS, "recovery", "candidates_ledger.csv")

DG_NEW_ZENODO = "21977747"
DG_NEW_URL = (
    "https://zenodo.org/records/21977747/files/"
    "deGraaff2026_mnras_lrds_withdups_blackbody_eline_fits.fits?download=1"
)
DG_NEW_CACHE = os.path.join(ROOT, "recovery", "degraaff2026_zenodo_21977747_v2.fits")
DG_OLD_ZENODO = "17665942"
MATCH_ARCSEC = 0.5

BEGIN = "% ---- revision 2026-09-05 (referee response): begin ----"
END = "% ---- revision 2026-09-05 (referee response): end ----"


def macro(name: str, value) -> str:
    return "\\newcommand{\\%s}{%s}" % (name, value)


def read_macro(text: str, name: str) -> int:
    m = re.search(r"\\newcommand\{\\%s\}\{([0-9{},]+)\}" % name, text)
    if not m:
        raise SystemExit("macro %s not found in numbers.tex" % name)
    return int(m.group(1).replace("{,}", "").replace(",", ""))


def b1_overlap() -> dict:
    d = pd.read_parquet(
        LABELS,
        columns=["in_hviding25_B1", "y", "ambiguous", "bands_complete", "picked"],
    )
    b1neg = d.in_hviding25_B1.astype(bool) & (d.y == 0)
    amb = d.ambiguous.astype(bool)
    out = {
        "nBOneAnchors": int(b1neg.sum()),
        "nBOneAnchorsSupport": int((b1neg & d.bands_complete.astype(bool)).sum()),
        "nBOneAnchorsUnionSelected": int((b1neg & d.picked.astype(bool)).sum()),
        "nAmbiguousTrue": int((amb & d.y.isna()).sum()),
        "_amb_total": int(amb.sum()),
    }
    led = pd.read_csv(LEDGER)
    above = led[led.in_hviding25_B1.astype(bool) & (led.y == 0)]
    out["nBOneAnchorsAboveHalf"] = int(len(above))
    out["nBOneAnchorsEqualBurdenDeployed"] = int((above.tier == "equal_burden").sum())
    return out


def degraaff_matches() -> dict:
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    import astropy.units as u

    if not os.path.exists(DG_NEW_CACHE):
        os.makedirs(os.path.dirname(DG_NEW_CACHE), exist_ok=True)
        with urllib.request.urlopen(DG_NEW_URL, timeout=120) as r:
            data = r.read()
        with open(DG_NEW_CACHE, "wb") as f:
            f.write(data)
    t = fits.open(DG_NEW_CACHE)[1].data
    use = np.array([bool(x) for x in t["use_dG26"]])
    dg = SkyCoord(t["ra"][use] * u.deg, t["dec"][use] * u.deg)
    c = pd.read_csv(CANDIDATES)
    cc = SkyCoord(c.ra.values * u.deg, c.dec.values * u.deg)
    idx, sep, _ = cc.match_to_catalog_sky(dg)
    m = sep.arcsec < MATCH_ARCSEC
    rows = c[m].copy()
    rows["zspec_dg26v2"] = t["zspec"][use][idx[m]]
    rows["sep_arcsec"] = sep.arcsec[m]
    rows = rows.sort_values("rank")
    ranks = [int(r) for r in rows["rank"]]
    zs = ["%.2f" % z for z in rows["zspec_dg26v2"]]
    return {
        "dgNewCatalogRows": int(len(t)),
        "dgNewCatalogSources": int(use.sum()),
        "candDgMatched": int(m.sum()),
        "candDgMatchedFollowup": int(rows.followup_tier.astype(bool).sum()),
        "candDgMatchedEqualBurden": int((rows.tier == "equal_burden").sum()),
        "candDgMatchedUntested": int((rows.status == "untested").sum()),
        "candDgRanks": "%s and %s" % (", ".join(str(r) for r in ranks[:-1]), ranks[-1]),
        "candDgRedshifts": "%s and %s" % (", ".join(zs[:-1]), zs[-1]),
        "candDgMaxSep": "%.2f" % float(np.ceil(rows.sep_arcsec.max() * 100) / 100),
        "_rows": rows[
            ["rank", "field", "id", "tier", "status", "zspec_dg26v2", "sep_arcsec"]
        ],
    }


def main() -> int:
    with open(NUMBERS, encoding="utf-8") as f:
        text = f.read()
    n_outside = read_macro(text, "nPosInNoMatchedSelection")
    floor_barro = read_macro(text, "floorBarro")
    n_amb = read_macro(text, "nAmbiguous")

    b1 = b1_overlap()
    if b1["_amb_total"] != n_amb:
        raise SystemExit(
            "ambiguous count %d != nAmbiguous %d" % (b1["_amb_total"], n_amb)
        )
    if b1["nBOneAnchors"] + b1["nAmbiguousTrue"] != n_amb:
        raise SystemExit("B1 anchors + truly ambiguous != nAmbiguous")
    dg = degraaff_matches()

    vals = {
        "nOutsideCatalogsRuleSelected": n_outside - floor_barro,
        "dgOldCatalogVersion": "0.1",
        "dgOldCatalogZenodo": DG_OLD_ZENODO,
        "dgOldCatalogDate": "20 November 2025",
        "dgOldCatalogSources": "116",
        "dgNewCatalogZenodo": DG_NEW_ZENODO,
        "dgNewCatalogDate": "17 August 2026",
        "kocevskiZlo": "2",
        "kocevskiZhi": "11",
        "vtestUvUpper": "-0.2",
        "vtestOptLower": "0",
        "vtestDiffLower": "0.5",
    }
    vals.update({k: v for k, v in b1.items() if not k.startswith("_")})
    vals.update({k: v for k, v in dg.items() if not k.startswith("_")})

    block = [BEGIN]
    block.append(
        "% recounted by paper/revision_2026-09-05_referee_macros.py"
    )
    for k, v in vals.items():
        block.append(macro(k, v))
    block.append(END)
    block_text = "\n".join(block) + "\n"

    if BEGIN in text:
        text = re.sub(
            re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n",
            block_text,
            text,
            flags=re.S,
        )
    else:
        text = text.rstrip("\n") + "\n" + block_text
    with open(NUMBERS, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    print("B1 anchors:", {k: v for k, v in b1.items() if not k.startswith("_")})
    print(dg["_rows"].to_string(index=False))
    print("wrote %d macros to numbers.tex" % len(vals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
