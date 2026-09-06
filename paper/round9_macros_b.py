"""Round 9, supplement: the fitter constants of Appendix D, the two audit remainders of
Appendix E, and the per-field data table of Appendix A (tables/data_fields.tex).

Reads the released fitter (selections/spectra/{census3,slopefit}.py in the public clone),
the rule audit table (recovery/rule_audit_positives.csv) and recovery/data_table.json.
Idempotent: the macro block is replaced on every run. Run from the repository root."""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLONE = ROOT
sys.path.insert(0, os.path.join(CLONE, "selections", "spectra"))
import census3  # noqa: E402
import slopefit  # noqa: E402
import pandas as pd  # noqa: E402

BEGIN = "% ---- revision 2026-09-06 (astra r5, round 9, supplement): begin ----"
END = "% ---- revision 2026-09-06 (astra r5, round 9, supplement): end ----"

N = {}
N["vtestUvLo"] = int(census3.UV_LO)
N["vtestBreak"] = int(census3.BREAK)
N["vtestOptHi"] = int(census3.OPT_HI)
N["vtestLines"] = len(census3.LINES)
N["vtestLinesWord"] = {18: "eighteen"}[len(census3.LINES)]
N["vtestMaskKms"] = "{:,}".format(int(census3.KMS)).replace(",", "{,}")
N["vtestMinSnr"] = int(slopefit.MIN_CONTINUUM_SNR)
N["vtestMinPix"] = int(slopefit.MIN_PIX)
N["vtestMinSpanDex"] = slopefit.MIN_SPAN_DEX
N["vtestDchi"] = int(slopefit.DCHI2_2SIG)
N["vtestBootInterior"] = int(round(100 * slopefit.MIN_BOOT_INTERIOR))

a = pd.read_csv(os.path.join(CLONE, "recovery", "rule_audit_positives.csv"))
N["auditKokorevRecordOnly"] = int(
    ((a.rule == "kokorev24") & (a.outcome == "none")).sum()
)
N["auditKocevskiRecordOnly"] = int(
    ((a.rule == "kocevski24") & (a.outcome == "none")).sum()
)
N["auditKokorevBrownDwarf"] = int(
    ((a.rule == "kokorev24") & (a.outcome == "brown-dwarf color")).sum()
)
N["auditKocevskiLineBoost"] = int(
    ((a.rule == "kocevski24") & (a.outcome == "line-boost criterion")).sum()
)

D = json.load(open(os.path.join(ROOT, "recovery", "data_table.json")))
vals = [v["r_star_arcsec"] for v in D.values()]
N["rStarMinArcsec"] = "%.3f" % min(vals)
N["rStarMaxArcsec"] = "%.3f" % max(vals)
d444 = [v["depth5sig_f444w"] for v in D.values()]
N["depthFourFourFourMin"] = "%.1f" % min(d444)
N["depthFourFourFourMax"] = "%.1f" % max(d444)

lines = [BEGIN, "% source: paper/round9_macros_b.py"]
for p in (
    os.path.join(ROOT, "paper", "numbers.tex"),
    os.path.join(ROOT, "paper", "numbers.tex"),
):
    txt = open(p, encoding="utf-8").read()
    txt = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END) + "\n?", "", txt, flags=re.S)
    head = txt
    existing = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", head))
    out = list(lines)
    for k, v in N.items():
        cmd = "renewcommand" if k in existing else "newcommand"
        out.append("\\%s{\\%s}{%s}" % (cmd, k, v))
    out.append(END)
    open(p, "w", encoding="utf-8").write(
        txt.rstrip("\n") + "\n" + "\n".join(out) + "\n"
    )
    print("wrote", len(N), "macros to", os.path.relpath(p, ROOT))

# ---- the per-field data table. One row per field of the support; GOODS fields carry the
# median stellar radius of the five resolved fields (Section 4.3), so the column is printed
# with that note in the caption rather than repeated here.
REGION = {
    "ceers-full": "CEERS",
    "gdn": "GOODS-North",
    "gds": "GOODS-South",
    "gds-sw": "GOODS-South",
    "primer-cosmos-east": "COSMOS",
    "primer-cosmos-west": "COSMOS",
    "primer-uds-north": "UDS",
    "primer-uds-south": "UDS",
}
BANDS = ["f090w", "f115w", "f150w", "f200w", "f277w", "f356w", "f444w"]
rows = []
for f in REGION:
    v = D[f]
    rows.append(
        " & ".join(
            [
                f,
                REGION[f],
                "{:,}".format(v["band_complete_rows"]),
                "%.3f" % v["r_star_arcsec"],
            ]
            + ["%.1f" % v["depth5sig_" + b] for b in BANDS]
        )
        + " \\\\"
    )
tp = os.path.join(ROOT, "paper", "tables", "data_fields.tex")
open(tp, "w", encoding="utf-8").write("\n".join(rows) + "\n")
print("wrote", os.path.relpath(tp, ROOT), len(rows), "rows")
