"""Round 9, third figure pass: the visual defects found on the r6 build.

Each edit is an exact-string replacement in one of the figure builders or TikZ sources,
so a second run is a no-op and a changed builder fails loudly rather than silently.

  fig_recall_burden   the shuffled-label range was a box that the rule key's leader lines
                      ran through; it is now a vertical bar at the union's row count
  fig_colour_planes   the strip between the panels and the key was 12 mm too tall
  fig_regions         the y-axis label ran past the top of the canvas
  fig_confusion       8 mm of empty canvas under the matrices
  fig_candidates      panel (c) tick labels collided; the panel is wider and the labels
                      are single words on their own lines
  fig_robustness      the longest row label ran off the left edge
  fig_gallery_*       an odd last object in a two-column section is centered
  fig_architecture    the final-fit threshold label was clipped by its box
  fig_pipeline        two words were hyphenated across lines inside boxes

Run from the repository root: python paper/round9_figure_patch_c.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PR = os.path.join(ROOT, "paper")
FIGS = os.path.join(PR, "figures")

EDITS = {
    os.path.join(PR, "build_figures.py"): [
        # --- fig_recall_burden: vertical bar instead of a box
        (
            """ax.fill_between(
    [800, 1700],
    N["shuffledMin"],
    N["shuffledMax"],
    color=C_MISS,
    alpha=0.16,
    lw=0,
    label="shuffled labels (%d)" % N["shuffledN"],
    zorder=7.5,
)
""",
            """# Round 9c: the shuffled-label range is a vertical bar at the union's row count, not a
# box; the box sat under the rule key's leader lines.
ax.plot(
    [N["unionSelected"], N["unionSelected"]],
    [N["shuffledMin"], N["shuffledMax"]],
    color=C_MISS,
    alpha=0.55,
    lw=3.0,
    solid_capstyle="butt",
    label="shuffled labels (%d)" % N["shuffledN"],
    zorder=3.5,
)
""",
        ),
        # --- fig_colour_planes: shorter strip under the panels
        (
            "CP_W, CP_H = 178.0, 88.0\n",
            "CP_W, CP_H = 178.0, 79.0\n",
        ),
        (
            "CP_PH, CP_BOT = 60.0, 23.0  # panel height, and the height of the strip below the panels\n",
            "CP_PH, CP_BOT = 60.0, 15.5  # panel height, and the height of the strip below the panels\n",
        ),
        # --- fig_regions: two-line y label and room for it
        (
            "fig.subplots_adjust(left=0.07, right=0.975, top=0.985, bottom=0.20)\n",
            "fig.subplots_adjust(left=0.085, right=0.975, top=0.985, bottom=0.20)\n",
        ),
        (
            'ax.set_ylabel("LRDs recovered at the union\'s row count in the region")\n',
            'ax.set_ylabel("LRDs recovered at the union\'s\\nrow count in the region", linespacing=1.3)\n',
        ),
        # --- fig_confusion: trim the empty band under the matrices
        (
            "CONF_H = 56.0\n",
            "CONF_H = 48.0\n",
        ),
        (
            "AX_Y = 15.0 / CONF_H\n",
            "AX_Y = 9.0 / CONF_H\n",
        ),
        # --- fig_candidates: wider panel (c), one word per line
        (
            'fig, axes = plt.subplots(1, 3, figsize=(COL2, 62 * MM), layout="constrained")\n',
            'fig, axes = plt.subplots(\n    1, 3, figsize=(COL2, 62 * MM), layout="constrained", width_ratios=[1.0, 1.0, 1.3]\n)\n',
        ),
        (
            """sets = [
    ("all\\n%d" % N["candTotal"], cand),
    ("equal-burden\\n%d" % N["candEqualBurden"], eb),
    ("follow-up\\n%d" % N["followupN"], fu),
]
""",
            """sets = [
    ("all\\n%d" % N["candTotal"], cand),
    ("equal-burden\\ntier\\n%d" % N["candEqualBurden"], eb),
    ("follow-up\\ntier\\n%d" % N["followupN"], fu),
]
""",
        ),
        # --- fig_robustness: room for the longest row label
        (
            "ROB_LEFT = 48.0  # millimetres kept for the row labels and the group headings\n",
            "ROB_LEFT = 55.0  # millimetres kept for the row labels and the group headings\n",
        ),
        # --- galleries: center an odd last object
        (
            """                row_top = cursor + r * G_ROW
                x0 = G_MARGIN + c * (G_STRIP + G_PAIRGAP)
""",
            """                row_top = cursor + r * G_ROW
                x0 = G_MARGIN + c * (G_STRIP + G_PAIRGAP)
                if len(d) > 1 and len(d) % 2 == 1 and k == len(d) - 1:
                    # Round 9c: an odd last object is centered rather than left in a hole
                    x0 = G_MARGIN + 0.5 * (G_STRIP + G_PAIRGAP)
""",
        ),
    ],
    os.path.join(FIGS, "fig_architecture.tex"): [
        (
            "\\node[alab, text=picked, anchor=west, inner sep=0.3mm] at (140.3,12.4) {\\tBurden{} (final fit)};\n",
            "\\node[alab, text=picked, anchor=north east, inner sep=0.3mm] at (138.9,9.1) {\\tBurden{} (final fit)};\n",
        ),
    ],
    os.path.join(FIGS, "fig_pipeline.tex"): [
        (
            "  {\\matchedRuleMissed{} of the \\nRuleMissed{} rule-missed recovered}\n",
            "  {\\matchedRuleMissed{} of the \\nRuleMissed{} \\mbox{rule-missed} recovered}\n",
        ),
        (
            "\\node[cat, text width=37.9mm, minimum width=41.1mm, minimum height=21mm]\n  (t2d) at (135.5, 16.5) {candidate catalog: \\candTotal{} unlisted sources;\n",
            "\\node[cat, text width=39.9mm, minimum width=43.1mm, minimum height=21mm]\n  (t2d) at (133.4, 16.5) {candidate catalog: \\candTotal{} \\mbox{unlisted} sources;\n",
        ),
        (
            "\\draw[arr] (131.0, 16.5) -- (135.5, 16.5);\n",
            "\\draw[arr] (131.0, 16.5) -- (133.4, 16.5);\n",
        ),
    ],
}

bad = 0
for path, edits in EDITS.items():
    text = open(path, encoding="utf-8").read()
    for old, new in edits:
        if new in text and old not in text:
            print("already  ", os.path.basename(path), repr(old[:50]))
            continue
        if text.count(old) != 1:
            print("MISSING  ", os.path.basename(path), repr(old[:70]), text.count(old))
            bad += 1
            continue
        text = text.replace(old, new)
        print("patched  ", os.path.basename(path), repr(old[:50]))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
if bad:
    sys.exit("%d edit(s) did not match" % bad)
print("done")
