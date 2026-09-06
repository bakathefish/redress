"""Round 9, second figure pass: legend overflow on fig_recall_burden, the region label on
fig_scores, and the tier label of fig_pipeline. Idempotent. Run from the repository root."""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PR = os.path.join(ROOT, "paper")


def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        if old in s:
            s = s.replace(old, new, 1)
        elif new in s:
            continue
        else:
            raise SystemExit("anchor not found in %s:\n%s" % (path, old[:160]))
    open(path, "w", encoding="utf-8").write(s)
    print("patched", os.path.relpath(path, ROOT))


BF = os.path.join(PR, "build_figures.py")
patch(
    BF,
    [
        (
            'label="learned ranking, within-region ranks at one budget fraction"',
            'label="learned ranking (within-region ranks, one budget fraction)"',
        ),
        (
            'label="imitation of the rules, same construction"',
            'label="imitation of the rules, same construction"',
        ),
        (
            '    marker="x",\n    s=22,\n    facecolor="none",\n    edgecolor=C_MODEL,\n    lw=0.9,\n    zorder=6,\n    label="learned, 0.5% threshold",\n',
            '    marker="+",\n    s=30,\n    color=C_MODEL,\n    lw=0.9,\n    zorder=6,\n    label="learned, 0.5% threshold",\n',
        ),
        (
            'label="learned at the union of eight\'s regional counts"',
            'label="learned at the union of eight\'s row counts"',
        ),
        (
            'label="imitation at its own realized burden"',
            'label="imitation, own realized burden"',
        ),
        (
            "fig, ax = plt.subplots(figsize=(COL2, 92 * MM))\nfig.subplots_adjust(left=0.07, right=0.985, top=0.985, bottom=0.24)",
            "fig, ax = plt.subplots(figsize=(COL2, 98 * MM))\nfig.subplots_adjust(left=0.07, right=0.985, top=0.985, bottom=0.30)",
        ),
        (
            '    loc="lower center",\n    bbox_to_anchor=(0.5, 0.005),\n    ncol=4,\n    fontsize=BASE,\n    handletextpad=0.4,\n    columnspacing=1.0,\n    labelspacing=0.35,\n)\nsave(fig, "fig_recall_burden", 178.0)',
            '    loc="lower center",\n    bbox_to_anchor=(0.5, 0.005),\n    ncol=3,\n    fontsize=BASE,\n    handletextpad=0.4,\n    columnspacing=1.0,\n    labelspacing=0.35,\n)\nsave(fig, "fig_recall_burden", 178.0)',
        ),
        # fig_scores: region label as a title, threshold label on the free side of the line
        (
            '    _ax.text(0.985, 0.92, "%s: %s rows, %d comparison, %d LRDs" % (_r, "{:,}".format(len(_s)), len(_n), len(_p)), transform=_ax.transAxes, ha="right", va="top", fontsize=SMALL, color=K)\n    _ax.text(_tb + 0.012, 1.4, "threshold %.3f" % _tb, fontsize=SMALL, color=C_MISS, ha="left", va="bottom")\n',
            '    _ax.set_title("%s: %s rows, %d comparison objects, %d LRDs" % (_r, "{:,}".format(len(_s)), len(_n), len(_p)), fontsize=SMALL, loc="left", pad=2.0)\n    _ax.text(_tb - 0.012, 3.0e4, "threshold %.3f" % _tb, fontsize=SMALL, color=C_MISS, ha="right", va="top")\n',
        ),
        (
            "fig, axes = plt.subplots(5, 1, figsize=(COL1, 118 * MM), sharex=True)\nfig.subplots_adjust(left=0.155, right=0.975, top=0.975, bottom=0.075, hspace=0.12)",
            "fig, axes = plt.subplots(5, 1, figsize=(COL1, 124 * MM), sharex=True)\nfig.subplots_adjust(left=0.155, right=0.975, top=0.965, bottom=0.07, hspace=0.42)",
        ),
        (
            'axes[0].legend(loc="upper left", fontsize=SMALL, borderaxespad=0.4, handlelength=1.2)',
            'axes[0].legend(loc="upper center", fontsize=SMALL, borderaxespad=0.3, handlelength=1.2, ncol=3, columnspacing=0.8, handletextpad=0.4)',
        ),
    ],
)

FP = os.path.join(PR, "figures", "fig_pipeline.tex")
patch(
    FP,
    [
        (
            "\\node[tier] at (0, 16.5) {The learned ranking};",
            "\\node[tier, align=left] at (0, 16.5) {The learned\\\\ranking};",
        ),
    ],
)
print("done")
