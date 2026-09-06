"""Every figure of the recovery paper, drawn from the tables of record in recovery and
from paper/numbers.json. Nothing is typed in by hand; every count on a figure is
read from the same records the text quotes. Author labels carry the journal year of the
bibliography entry (paper/bib/references.bib), not the arXiv year.

Run from the repository root: python paper/build_figures.py
Writes paper/figures/fig_*.pdf and .png, plus figures/contact_sheet.png and
figures/font_report.txt.

fig_pipeline is not drawn in matplotlib: it is a TikZ standalone, figures/fig_pipeline.tex,
a two-tier "study at a glance" schematic that reads its numbers from numbers.tex. This
script runs pdflatex on it from inside the figures directory, so one command rebuilds all
eleven figures and the schematic cannot go stale against the numbers.

House style (MNRAS/ApJ): serif text through STIX with mathtext to match the LaTeX body,
8 pt everywhere at output size, 0.6 to 1.0 pt rules, black and gray as the base with one
blue and one vermillion accent, ticks inward on all four sides, bold panel letters instead
of titles.

Sizing rule, and the reason the round-1 figures printed at 5 to 6.5 pt: main.tex includes
every figure at \\columnwidth or \\textwidth, so LaTeX scales whatever it is given to that
width. Saving with bbox_inches="tight" cropped each canvas to an arbitrary width, and the
scale factor that followed shrank or grew the lettering by up to 20 percent in either
direction. Here every figure is laid out inside its exact final canvas and saved with no
bounding-box cropping, so the scale factor is 1 and the point sizes below are the point
sizes on the page.
"""

import json
import os
import shutil
import subprocess
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from astropy.io import fits as afits  # noqa: E402
from astropy.visualization import make_lupton_rgb  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "recovery")
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
N = json.load(open(os.path.join(HERE, "numbers.json")))
# round 9 (2026-09-06): the region-matched comparison is the primary result, and its
# numbers, the regional-budget curve and the per-object outcomes come from these records
N9 = json.load(open(os.path.join(ROOT, "recovery", "round9_numbers.json")))
CURVE = json.load(open(os.path.join(ROOT, "recovery", "regional_budget_curve.json")))
POUT = pd.read_csv(os.path.join(ROOT, "recovery", "positive_outcomes.csv")).set_index("source_id")
REG = ["CEERS", "GOODS-N", "GOODS-S", "COSMOS", "UDS"]

# ------------------------------------------------------------------ sizes
MM = 1.0 / 25.4
COL1 = 84.0 * MM  # single column, as main.tex includes it (\columnwidth)
COL2 = 178.0 * MM  # two columns, as main.tex includes it (\textwidth)
BASE = 8.0  # nothing on any figure is smaller than this ...
SMALL = 7.5  # ... except these, which the brief allows: per-bar counts,
#              stamp labels, threshold labels, cell percentages.

# ------------------------------------------------------------------ house style
plt.rcParams.update(
    {
        # serif body matching the LaTeX paper, without a LaTeX dependency at runtime
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "text.usetex": False,
        # 8 pt at output size, and output size is final size
        "font.size": BASE,
        "axes.labelsize": BASE,
        "axes.titlesize": BASE,
        "legend.fontsize": BASE,
        "xtick.labelsize": BASE,
        "ytick.labelsize": BASE,
        # thin rules
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.6,
        "grid.linewidth": 0.4,
        # ticks inward on all four sides
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # all four spines, no grid, no frame around legends
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.45,
        "legend.labelspacing": 0.30,
        "legend.borderpad": 0.2,
        "legend.borderaxespad": 0.4,
        "legend.columnspacing": 1.1,
        # vector PDF with real glyphs, 300 dpi raster companion
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": None,  # never crop: the canvas IS the final size
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# base is black and gray; exactly two accents, used the same way on every figure
K = "#000000"
C_UNION = "#666666"  # the seven published rules
C_MODEL = "#1a5a8a"  # accent 1, the learned selection
C_MISS = "#c0532a"  # accent 2, missed by every rule / shuffled labels
C_CAND = "#e0a06a"  # tint of accent 2, unlisted candidates
C_NEG = "#a0a0a0"  # spectroscopic non-LRD anchors
C_CAT = "#dcdcdc"  # the catalog as a whole
C_ALT = "#b0b0b0"  # robustness variants
C_LOCO = "#666666"  # leave-one-list-out runs
GREY_TXT = "#7f7f7f"  # secondary lettering (threshold and literature labels)

# ------------------------------------------------------------------ the seven rules
# One categorical palette for the seven published selections, used identically wherever a
# rule is drawn: its point in fig_recall_burden, its threshold line and shaded region in
# fig_colour_planes, its bar in the fig_miss_anatomy ladder. The set is Paul Tol's "muted"
# qualitative scheme (Tol 2021, SRON/EPS/TN/09-002 issue 3.2), which is designed to stay
# distinguishable under deuteranopia, protanopia and tritanopia; none of the seven is close
# to the two reserved accents #1a5a8a and #c0532a. A rule's color is carried redundantly by
# its own dash pattern, so a greyscale print still separates the lines.
RULE_COLOUR = {
    "labbe23": "#332288",  # indigo
    "kokorev24": "#88CCEE",  # cyan
    "kocevski24": "#44AA99",  # teal
    "perezgonzalez24": "#117733",  # green
    "barro23": "#999933",  # olive
    "greene24": "#CC6677",  # rose
    "akins24": "#AA4499",  # purple
}
RULE_DASH = {
    "labbe23": (0, (5.0, 2.0)),
    "kokorev24": (0, (1.6, 1.6)),
    "kocevski24": (0, (5.0, 1.6, 1.2, 1.6)),
    "perezgonzalez24": (0, (3.0, 1.6)),
    "barro23": (0, (7.0, 2.0)),
    "greene24": (0, (2.4, 1.4, 1.0, 1.4)),
    "akins24": (0, (4.0, 1.4, 1.0, 1.4, 1.0, 1.4)),
}


def _dark(hexcol, target=0.16):
    """The same hue at a luminance a reader can take as 8 pt text on white.

    Tol's muted set is designed for filled areas and markers, where a light cyan or olive
    reads perfectly well against a dark edge. As lettering on white, the lighter members
    fall under the 4.5:1 contrast a small serif face needs. This scales the color towards
    black until its relative luminance (WCAG 2.1) is at most `target`, which is 4.5:1
    against white, and leaves the hue where it was, so a label still matches its marker.
    """
    rgb = np.array([int(hexcol[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])

    def lum(c):
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return float(np.dot([0.2126, 0.7152, 0.0722], lin))

    if lum(rgb) <= target:
        return hexcol
    lo, hi = 0.0, 1.0
    for _ in range(40):  # bisect the scale factor; luminance is monotone in it
        mid = 0.5 * (lo + hi)
        if lum(rgb * mid) > target:
            hi = mid
        else:
            lo = mid
    return "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in rgb * lo)


# the lettering twin of each rule color: same hue, dark enough to read at 8 pt on white
RULE_TEXT = {r: _dark(c) for r, c in RULE_COLOUR.items()}
# a truncated Greys, so the densest catalog cell stays lighter than any plotted marker
# (Labbe et al. 2025 Fig. 2 and Hviding et al. 2025 Fig. 6 both keep the density map light)
DENS_CMAP = LinearSegmentedColormap.from_list(
    "greys_light", plt.get_cmap("Greys")(np.linspace(0.0, 0.58, 256))
)

# round 3 removed RNAME, whose seven values were the internal rule keys rather than author
# names, so the key of fig_recall_burden printed "labbe23" where it meant "Labbé+23". The
# names now come from RULE_LABEL_SHORT below, which every figure that names a rule shares.
LIST_BARRO, LIST_DEGRAAFF, LIST_HVIDING = "Barro+26", "de Graaff+26", "Hviding+25"

FONT_REPORT = []


def min_font(fig):
    """Smallest point size actually drawn on this figure, measured from the artists."""
    sizes = [
        t.get_fontsize()
        for t in fig.findobj(Text)
        if t.get_visible() and str(t.get_text()).strip()
    ]
    return min(sizes) if sizes else float("nan")


def save(fig, name, width_mm):
    """Write the PDF and PNG at exactly the canvas size, and record the font floor.

    No bbox_inches: main.tex rescales the file to \\columnwidth or \\textwidth, so the
    canvas must already be that width or every point size on it moves.
    """
    w_in, h_in = fig.get_size_inches()
    got = w_in / MM
    assert abs(got - width_mm) < 0.05, "%s is %.2f mm wide, wanted %.2f" % (
        name,
        got,
        width_mm,
    )
    mf = min_font(fig)
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches=None)
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches=None)
    plt.close(fig)
    FONT_REPORT.append((name, got, h_in / MM, mf))
    print(
        "wrote %-26s %6.1f x %6.1f mm  smallest font %.1f pt"
        % (name, got, h_in / MM, mf)
    )


def panel(ax, letter, text=None, y=1.012, x=0.0):
    """Bold panel letter above the axes. Descriptions belong in the caption."""
    s = "(%s)" % letter if text is None else "(%s) %s" % (letter, text)
    ax.text(
        x,
        y,
        s,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=BASE,
        fontweight="bold",
        color=K,
    )


def J(f):
    return json.load(open(os.path.join(V, f)))


# ------------------------------------------------------------------ data used by several figures
feat = pd.read_parquet(os.path.join(V, "features.parquet"))
sup = feat[feat.in_support.astype(bool)].set_index("source_id")
oof = pd.read_parquet(os.path.join(V, "oof_scores.parquet")).set_index("source_id")
sup["score"] = oof.loc[sup.index, "score_mean"].values
sup["sel_burden"] = oof.loc[sup.index, "sel_burden"].values
sup["t_burden"] = oof.loc[sup.index, "t_burden"].values
# The seven rules impose their F277W-F444W thresholds on ordinary AB colors, while the
# model feature c_f277w_f444w is a difference of inverse-hyperbolic-sine magnitudes, which
# keeps an undetected band finite. Every color bin and every threshold line below is drawn
# on the AB color, computed from the catalog fluxes in labels.parquet exactly as
# build_numbers.py does (paper/reviews/round1_facts.md section 9.5). A source with
# a non-positive flux in either band has no AB color and is dropped from those panels.
_lab = pd.read_parquet(
    os.path.join(V, "labels.parquet"), columns=["source_id", "f_f277w", "f_f444w"]
).set_index("source_id")
with np.errstate(divide="ignore", invalid="ignore"):
    _lab["c_ab_f277w_f444w"] = -2.5 * np.log10(
        _lab.f_f277w.to_numpy(float) / _lab.f_f444w.to_numpy(float)
    )
sup["c_ab_f277w_f444w"] = _lab.loc[sup.index, "c_ab_f277w_f444w"].values
pos = sup[sup.y == 1]
neg = sup[sup.y == 0]
cand = pd.read_csv(os.path.join(V, "candidates.csv"))
cand["c_ab_f277w_f444w"] = _lab.reindex(cand.source_id.values)[
    "c_ab_f277w_f444w"
].values
eb = cand[cand.tier == "equal_burden"]
fu = cand[cand.followup_tier]
assert np.isfinite(pos.c_ab_f277w_f444w).all(), "a positive has no finite AB color"
ev = J("evaluation.json")
rules = [
    "labbe23",
    "kokorev24",
    "kocevski24",
    "perezgonzalez24",
    "barro23",
    "greene24",
    "akins24",
]
RTAG = {
    "labbe23": "Labbe",
    "kokorev24": "Kokorev",
    "kocevski24": "Kocevski",
    "perezgonzalez24": "PerezGonzalez",
    "barro23": "Barro",
    "greene24": "Greene",
    "akins24": "Akins",
}
# the display names used where a rule is named on a figure, with the journal year of the
# bibliography entry; the same seven names Table 1 lists
RULE_LABEL_SHORT = {
    "labbe23": "Labbé+23",
    "kokorev24": "Kokorev+24",
    "kocevski24": "Kocevski+24",
    "perezgonzalez24": "Pérez-González+24",
    "barro23": "Barro+23",
    "greene24": "Greene+24",
    "akins24": "Akins+24",
}

# ================================================================== Figure 1: pipeline schematic
# Written as figures/fig_pipeline.tex (TikZ standalone) and compiled with pdflatex from
# inside the figures directory. Nothing to draw here.

# ================================================================== Figure 2: recall versus burden
# 84 mm, because main.tex includes it at \columnwidth. Two crowding problems are solved by
# moving text off the data rather than by shrinking it: the seven rule points sit inside
# half a decade of the log axis, so no 8 pt label fits beside them and each is written in a
# key in the empty region under the curve with a hairline leader; and the six-entry key of
# the series sits under the axes in two columns, where it can cover neither the shuffled
# band nor the literature points.
fig, ax = plt.subplots(figsize=(COL2, 98 * MM))
fig.subplots_adjust(left=0.07, right=0.985, top=0.985, bottom=0.30)
# Round 9: the five held-out ensembles do not share a score scale, so the curve is
# built region by region. At each budget fraction f every region keeps the top f times
# the union's row count there, by that region's own out-of-fold ranking, and the
# recoveries are summed; the x coordinate is the total number of rows kept.
_KTOT = float(sum(CURVE["K_union"].values()))
_fx = np.array(CURVE["fractions"]) * _KTOT
ax.plot(_fx, CURVE["primary"], color=C_MODEL, lw=1.0, label="learned ranking (within-region ranks, one budget fraction)", zorder=3)
ax.plot(_fx, CURVE["imitation"], color=K, lw=0.8, ls=(0, (3.0, 1.6)), label="imitation of the rules, same construction", zorder=3)
un = ev["union"]
pe = ev["primary_equal_burden"]
p5 = ev["primary_0p5pct"]
ax.scatter(
    [un["selected_total"]],
    [un["recall_147"]],
    marker="s",
    s=20,
    color=C_UNION,
    zorder=5,
    label="union of seven rules",
)
ax.scatter(
    [CURVE["union"]["rows"]],
    [CURVE["union"]["primary"] if "primary" in CURVE["union"] else N9["matchedRecallSupport"]],
    marker="D",
    s=18,
    color=C_MODEL,
    zorder=6,
    label="learned, region-matched burden",
)
ax.scatter(
    [pe["selected_total"]],
    [pe["recall_147"]],
    marker="x",
    s=22,
    color=C_MODEL,
    lw=0.9,
    zorder=6,
    label="learned, deployed thresholds",
)
ax.scatter(
    [CURVE["union8"]["rows"]],
    [CURVE["union8"]["recall"]],
    marker="s",
    s=20,
    facecolor="white",
    edgecolor=C_UNION,
    lw=0.8,
    zorder=5,
    label="union of eight (with Barro+24b)",
)
ax.scatter(
    [CURVE["union8"]["rows"]],
    [CURVE["union8"]["primary"]],
    marker="D",
    s=18,
    facecolor="white",
    edgecolor=C_MODEL,
    lw=0.9,
    zorder=6,
    label="learned at the union of eight's row counts",
)
ax.scatter(
    [p5["selected_total"]],
    [p5["recall_147"]],
    marker="+",
    s=30,
    color=C_MODEL,
    lw=0.9,
    zorder=6,
    label="learned, 0.5% threshold",
)
bi = ev["baselines"]["rules_reproduction"]
# round 2 of round 1: the imitation's realised catalog burden is measured
# (baseline_burden.json), so the triangle sits at the burden it actually spends rather than
# at the union's.
ax.scatter(
    [N["blImitationSelected"]],
    [bi["recall_at_burden"]],
    marker="^",
    s=20,
    facecolor="white",
    edgecolor=K,
    lw=0.7,
    zorder=6,
    label="imitation, own realized burden",
)
ax.fill_between(
    [800, 1700],
    N["shuffledMin"],
    N["shuffledMax"],
    color=C_MISS,
    alpha=0.16,
    lw=0,
    label="shuffled labels (%d)" % N["shuffledN"],
    zorder=7.5,
)
ax.axhline(N["nPosSupport"], color=C_UNION, lw=0.5, ls=":", zorder=2)
ax.text(
    118,
    N["nPosSupport"] + 2.0,
    "all %d in-support LRDs" % N["nPosSupport"],
    fontsize=BASE,
    color=GREY_TXT,
    va="bottom",
)
# the seven rules, keyed into the empty band under the curve
rule_xy = {
    r: (N["rule" + RTAG[r] + "Selected"], int(pos["sel_" + r].astype(bool).sum()))
    for r in rules
}
# Round 3: each rule takes its palette color here, and the same color names it in the key,
# edges its shaded region in fig_colour_planes and fills its bar on the fig_miss_anatomy
# ladder. The marker is filled in that color with a dark edge (style rule 11), so a point
# and its name are matched by color and not only by a leader line. The key also carries the
# published author name in place of the internal rule key.
for r in rules:
    x, y = rule_xy[r]
    ax.scatter(
        [x],
        [y],
        marker="o",
        s=11,
        facecolor=RULE_COLOUR[r],
        edgecolor=K,
        lw=0.35,
        zorder=4,
    )
KEY_X = 2100.0
key_order = sorted(rules, key=lambda r: -rule_xy[r][1])
key_y = [74, 65, 56, 47, 38, 29, 20]
for r, ky in zip(key_order, key_y):
    x, y = rule_xy[r]
    ax.annotate(
        RULE_LABEL_SHORT[r],
        xy=(x, y),
        xytext=(KEY_X, ky),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=BASE,
        color=RULE_TEXT[r],
        zorder=7,
        arrowprops=dict(
            arrowstyle="-", lw=0.35, color="#b8b8b8", shrinkA=2.0, shrinkB=2.5
        ),
    )
ax.set_xscale("log")
ax.set_xlim(100, 40000)
ax.set_ylim(0, 162)
ax.set_xlabel("catalog rows selected, summed over the five regions")
ax.set_ylabel("spectroscopic LRDs recovered")
h, lb = ax.get_legend_handles_labels()
fig.legend(
    h,
    lb,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    ncol=3,
    fontsize=BASE,
    handletextpad=0.4,
    columnspacing=1.0,
    labelspacing=0.35,
)
save(fig, "fig_recall_burden", 178.0)

# ================================================================== Figure 3: where the gain is
fig, axes = plt.subplots(1, 2, figsize=(COL2, 62 * MM), layout="constrained")
fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.05, hspace=0.0)
P = pos.copy()
cbins = [-5, 0.5, 1.0, 1.5, 2.0, 10]
clab = ["< 0.5", "0.5 to 1.0", "1.0 to 1.5", "1.5 to 2.0", "> 2.0"]
mbins = [0, 24, 25, 26, 27, 40]
mlab = ["< 24", "24 to 25", "25 to 26", "26 to 27", "> 27"]
GAIN_COUNTS = {}
for k, (ax, col, bins, labels, xl) in enumerate(
    (
        (axes[0], "c_ab_f277w_f444w", cbins, clab, "F277W $-$ F444W (AB mag)"),
        (axes[1], "mag_f444w", mbins, mlab, "F444W (AB mag)"),
    )
):
    b = pd.cut(P[col], bins, labels=labels)
    g = P.groupby(b, observed=True)
    nn = g.size().values
    u = g.picked.apply(lambda s: s.astype(bool).sum()).values
    m = g.sel_burden.sum().values
    GAIN_COUNTS["ab"[k]] = (list(labels), list(nn), list(u), list(m))
    x = np.arange(len(labels))
    # round 3, style rule 11: a plotted body carries a dark edge, so bars of similar tone
    # separate from each other and from the page
    hu = ax.bar(
        x - 0.19,
        u / nn,
        0.36,
        color=C_UNION,
        edgecolor=K,
        lw=0.3,
        label="union of seven rules",
    )
    hm = ax.bar(
        x + 0.19,
        m / nn,
        0.36,
        color=C_MODEL,
        edgecolor=K,
        lw=0.3,
        label="learned selection",
    )
    for i in range(len(labels)):
        ax.text(
            x[i] - 0.19,
            u[i] / nn[i] + 0.022,
            "%d" % u[i],
            ha="center",
            fontsize=SMALL,
            color=C_UNION,
        )
        ax.text(
            x[i] + 0.19,
            m[i] / nn[i] + 0.022,
            "%d" % m[i],
            ha="center",
            fontsize=SMALL,
            color=C_MODEL,
        )
        ax.text(
            x[i],
            -0.155,
            "n = %d" % nn[i],
            ha="center",
            fontsize=SMALL,
            color=GREY_TXT,
            transform=ax.get_xaxis_transform(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=BASE)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel(xl, labelpad=15)
    ax.set_ylabel("fraction recovered")
    panel(ax, "ab"[k])
fig.legend(
    handles=[hu, hm],
    labels=["union of seven rules", "learned selection"],
    loc="outside upper center",
    ncol=2,
    fontsize=BASE,
    handletextpad=0.5,
    columnspacing=2.0,
)
save(fig, "fig_gain_by_colour_mag", 178.0)

# ================================================================== Figure 4: color-magnitude and color-color
# Round 3 rebuilds this figure to the literature's grammar (reviews/FIGURE_STYLE_LITERATURE.md
# rules 3, 4, 5, 11, 20): the catalog is a log-density greyscale with a labeled color bar,
# as Labbe et al. (2025) Fig. 2 and Hviding et al. (2025) Fig. 6 draw theirs; each published
# rule's selection region is a translucent band named at the axis edge in the rule's own
# color, as Barro et al. (2026) Fig. 5 names his color bins; and every plotted point carries
# a dark edge. The geometry is placed by hand in figure fractions, as fig_confusion's is, so
# that the two panels are provably the same size once a color bar is attached to one of them.
#
# Nothing plotted changes. The thresholds are the same three values Table 1 lists, the same
# rows are drawn, and the same two color axes are used: panel (a) is the ordinary AB color,
# in which every published threshold is written, and panel (b) is the pair of asinh colors
# the model actually sees. That is why the shaded regions appear on panel (a) only: drawing a
# threshold published in AB magnitudes onto an asinh axis would misstate it.
CP_W, CP_H = 178.0, 88.0
CP_L, CP_PW, CP_MID = (
    13.0,
    68.0,
    14.0,
)  # left label strip, panel width, gap + b's y label
CP_CGAP, CP_CW, CP_R = 2.0, 3.0, 10.0  # color-bar gap, bar width, right tick strip
assert abs(CP_L + CP_PW + CP_MID + CP_PW + CP_CGAP + CP_CW + CP_R - CP_W) < 1e-9, (
    "color-plane geometry does not add up to 178 mm"
)
CP_PH, CP_BOT = 60.0, 23.0  # panel height, and the height of the strip below the panels
fig = plt.figure(figsize=(COL2, CP_H * MM))


def _cpx(mm_):
    return mm_ / CP_W


def _cpy(mm_):
    return mm_ / CP_H


axa = fig.add_axes([_cpx(CP_L), _cpy(CP_BOT), _cpx(CP_PW), _cpy(CP_PH)])
axb = fig.add_axes(
    [_cpx(CP_L + CP_PW + CP_MID), _cpy(CP_BOT), _cpx(CP_PW), _cpy(CP_PH)]
)
cbax = fig.add_axes(
    [
        _cpx(CP_L + CP_PW + CP_MID + CP_PW + CP_CGAP),
        _cpy(CP_BOT),
        _cpx(CP_CW),
        _cpy(CP_PH),
    ]
)

rand = sup.sample(60000, random_state=1)
hit = pos[pos.picked.astype(bool)]
mis = pos[~pos.picked.astype(bool)]

# the catalog, as one log-density greyscale shared by both panels so that a shade means
# the same number of rows in each
hba = axa.hexbin(
    rand.mag_f444w,
    rand.c_ab_f277w_f444w,
    gridsize=60,
    cmap=DENS_CMAP,
    mincnt=1,
    linewidths=0.1,
    extent=(21, 29, -1.5, 3.5),
    zorder=1,
)
hbb = axb.hexbin(
    rand.c_f277w_f444w,
    rand.c_f200w_f356w,
    gridsize=60,
    cmap=DENS_CMAP,
    mincnt=1,
    linewidths=0.1,
    extent=(-1.5, 3.5, -1.5, 3.5),
    zorder=1,
)
_vmax = float(max(hba.get_array().max(), hbb.get_array().max()))
_dnorm = LogNorm(vmin=1.0, vmax=_vmax)
hba.set_norm(_dnorm)
hbb.set_norm(_dnorm)
cb = fig.colorbar(hba, cax=cbax)
cb.set_label("catalog rows per hexagon", fontsize=BASE, labelpad=3)
cb.ax.tick_params(labelsize=BASE, width=0.6, length=2.4, direction="in")
cb.outline.set_linewidth(0.6)

# ------------------------------------------------------------------ panel (a): the rules
# The seven rules impose three distinct F277W-F444W thresholds, so their selection regions
# nest. Each is washed at the same low alpha, which makes the shade darken step by step with
# the number of rules that accept, and the rules that own each boundary are named in the key,
# every name in that rule's palette color.
THR_RULES = [
    (1.5, ["barro23", "akins24"]),
    (1.0, ["labbe23", "perezgonzalez24", "greene24"]),
    (0.7, ["kokorev24"]),
]
CP_YTOP = 3.5
for thr, owners in THR_RULES:
    axa.axhspan(thr, CP_YTOP, facecolor="#4b6b86", alpha=0.055, lw=0, zorder=2)
    axa.axhline(thr, color=K, lw=0.5, ls=(0, (4, 2)), zorder=6)
# Round 4, F45: the names used to ride on the lines they name, where the marker scatter
# ran across them. They are one key in the empty upper-left corner of the panel instead,
# thresholds ascending, no background box, above the scatter.
key_rows = []
for thr, owners in sorted(THR_RULES):
    parts = [
        TextArea(
            "$>$ %g" % thr, textprops=dict(color=K, fontsize=BASE, fontweight="bold")
        )
    ]
    for i, r in enumerate(owners):
        parts.append(
            TextArea(
                RULE_LABEL_SHORT[r] + ("," if i < len(owners) - 1 else ""),
                textprops=dict(color=RULE_TEXT[r], fontsize=BASE),
            )
        )
    key_rows.append(HPacker(children=parts, pad=0, sep=3.0, align="baseline"))
axa.add_artist(
    AnnotationBbox(
        VPacker(children=key_rows, pad=0, sep=2.0, align="left"),
        (0.015, 0.985),
        xycoords="axes fraction",
        box_alignment=(0.0, 1.0),
        frameon=False,
        pad=0.0,
        zorder=8,
        annotation_clip=False,
    )
)
# Kocevski+24 selects on continuum slope and sets no F277W-F444W threshold, which the panel
# says rather than hides
axa.text(
    21.16,
    -1.34,
    "Kocevski+24 selects on continuum slope: no color threshold",
    fontsize=SMALL,
    color=RULE_TEXT["kocevski24"],
    ha="left",
    va="bottom",
    zorder=8,
)

h_cand = axa.scatter(
    eb.mag_f444w,
    eb.c_ab_f277w_f444w,
    s=4,
    color=C_CAND,
    alpha=0.85,
    lw=0,
    zorder=3,
    label="candidates, equal-burden tier (%d)" % len(eb),
)
h_hit = axa.scatter(
    hit.mag_f444w,
    hit.c_ab_f277w_f444w,
    s=11,
    marker="o",
    facecolor=C_MODEL,
    edgecolor=K,
    lw=0.3,
    zorder=4,
    label="spectroscopic LRDs selected by a rule (%d)" % len(hit),
)
h_mis = axa.scatter(
    mis.mag_f444w,
    mis.c_ab_f277w_f444w,
    s=17,
    marker="D",
    facecolor=C_MISS,
    edgecolor=K,
    lw=0.45,
    zorder=5,
    label="spectroscopic LRDs missed by every rule (%d)" % len(mis),
)
axa.set_xlim(21, 29)
axa.set_ylim(-1.5, CP_YTOP)
axa.set_xlabel("F444W (AB mag)")
axa.set_ylabel("F277W $-$ F444W (AB mag)")
panel(axa, "a")

# ------------------------------------------------------------------ panel (b): the model's colors
axb.scatter(
    eb.c_f277w_f444w,
    eb.c_f200w_f356w,
    s=4,
    color=C_CAND,
    alpha=0.85,
    lw=0,
    zorder=3,
)
axb.scatter(
    hit.c_f277w_f444w,
    hit.c_f200w_f356w,
    s=11,
    marker="o",
    facecolor=C_MODEL,
    edgecolor=K,
    lw=0.3,
    zorder=4,
)
axb.scatter(
    mis.c_f277w_f444w,
    mis.c_f200w_f356w,
    s=17,
    marker="D",
    facecolor=C_MISS,
    edgecolor=K,
    lw=0.45,
    zorder=5,
)
axb.set_xlim(-1.5, 3.5)
axb.set_ylim(-1.5, 3.5)
axb.set_xlabel("F277W $-$ F444W (asinh mag)")
axb.set_ylabel("F200W $-$ F356W (asinh mag)")
panel(axb, "b")

fig.legend(
    handles=[h_cand, h_hit, h_mis],
    loc="lower center",
    bbox_to_anchor=(0.5, _cpy(1.0)),
    ncol=3,
    fontsize=BASE,
    handletextpad=0.4,
    columnspacing=1.6,
)
save(fig, "fig_colour_planes", 178.0)

# ================================================================== Figure 5: per region
fig, ax = plt.subplots(figsize=(COL2, 66 * MM))
fig.subplots_adjust(left=0.07, right=0.975, top=0.985, bottom=0.20)
x = np.arange(5)
u = [N["unionRegion" + r.replace("-", "")] for r in REG]
m = [N9["matchedRegion" + r.replace("-", "")] for r in REG]
t = [N["posSup" + r.replace("-", "")] for r in REG]
mm_ = [N9["matchedRegionRuleMissed" + r.replace("-", "")] for r in REG]
b_u = ax.bar(x - 0.19, u, 0.36, color=C_UNION, edgecolor=K, lw=0.3)
b_m = ax.bar(x + 0.19, m, 0.36, color=C_MODEL, edgecolor=K, lw=0.3)
h_t = ax.scatter(x, t, marker="_", s=170, color=K, lw=1.0, zorder=5)
# the counts sit at the outer edge of each bar, clear of the total marker's dash
for i in range(5):
    ax.text(
        x[i] - 0.31, u[i] + 1.1, str(u[i]), ha="center", fontsize=SMALL, color=C_UNION
    )
    ax.text(
        x[i] + 0.31, m[i] + 1.1, str(m[i]), ha="center", fontsize=SMALL, color=C_MODEL
    )
ax.set_xticks(x)
ax.set_xticklabels(
    [
        "%s\n%d of %d rule-missed" % (r, mm_[i], N["missedSup" + r.replace("-", "")])
        for i, r in enumerate(REG)
    ],
    fontsize=BASE,
    linespacing=1.35,
)
ax.set_xlim(-0.6, 4.6)
ax.set_ylim(0, 88)
ax.xaxis.set_minor_locator(NullLocator())
ax.tick_params(axis="x", length=0)
ax.set_ylabel("LRDs recovered at the union's row count in the region")
ax.legend(
    handles=[h_t, b_u, b_m],
    labels=[
        "spectroscopic LRDs in region",
        "union of seven rules",
        "learned selection (region held out, region-matched burden)",
    ],
    loc="upper left",
    fontsize=BASE,
    handletextpad=0.5,
    borderaxespad=0.7,
)
save(fig, "fig_regions", 178.0)

# ================================================================== Figure 6: feature importance
imp = J("feature_importance_gain.json")
top = sorted(imp.items(), key=lambda kv: -kv[1])[:14]
# short forms: at 8 pt a 33-character label would eat half of an 84 mm figure. The features
# themselves are defined in Appendix A; these are the same quantities under shorter names.
PRETTY = {
    "c_f200w_f356w": "F200W $-$ F356W",
    "c_f277w_f444w": "F277W $-$ F444W",
    "slope_red": "long-wavelength slope",
    "m_f444w": "F444W magnitude",
    "asnr_f444w": "F444W S/N",
    "c_f200w_f277w": "F200W $-$ F277W",
    "log_rh": "log $r_{h}$",
    "c_f356w_f444w": "F356W $-$ F444W",
    "log_rh_over_rstar": "log $r_{h}/r_{\\star}$",
    "m_f356w": "F356W magnitude",
    "m_f150w": "F150W magnitude",
    "v_curv": "V curvature",
    "c_f277w_f356w": "F277W $-$ F356W",
    "m_f200w": "F200W magnitude",
    "asnr_f356w": "F356W S/N",
    "slope_blue": "short-wavelength slope",
    "asnr_f090w": "F090W S/N",
}
fig, ax = plt.subplots(figsize=(COL1, 74 * MM))
fig.subplots_adjust(left=0.335, right=0.975, top=0.985, bottom=0.135)
names = [PRETTY.get(k, k) for k, _ in top][::-1]
vals = [100 * v for _, v in top][::-1]
cols = [
    C_MODEL if ("F277W $-$ F444W" in nm or "F200W $-$ F356W" in nm) else C_ALT
    for nm in names
]
ax.barh(names, vals, color=cols, height=0.66, edgecolor=K, lw=0.3)
for i, v in enumerate(vals):
    ax.text(v + 0.6, i, "%.1f" % v, va="center", fontsize=SMALL, color=GREY_TXT)
ax.set_xlabel("share of split gain, mean over %d bags (percent)" % N["nBags"])
ax.set_xlim(0, max(vals) * 1.20)
ax.set_ylim(-0.7, len(names) - 0.3)
ax.yaxis.set_minor_locator(NullLocator())
ax.tick_params(axis="y", length=0)
save(fig, "fig_importance", 84.0)

# ================================================================== Figure 7: confusion matrices
# Three 2 x 2 tile matrices of identical size, shaded by row fraction on a light blue ramp.
# The axes are placed by hand in figure fractions so that the tiles are provably equal: the
# three sets of row labels have different widths, and any automatic layout would trade tile
# size against label width.
CONF_H = 56.0
fig = plt.figure(figsize=(COL2, CONF_H * MM))
RAMP = LinearSegmentedColormap.from_list("ramp", ["#ffffff", C_MODEL])
AX_W, AX_H = 38.0 / 178.0, 34.0 / CONF_H
AX_Y = 15.0 / CONF_H
AX_X = [26.0 / 178.0, 67.0 / 178.0, 136.0 / 178.0]


def tiles(ax, letter, head, cells, rowlab, collab, rowaxis=None, colaxis=None):
    """cells[i][j] is a count; the shading and the printed percentage are per row."""
    for i in range(2):
        tot = cells[i][0] + cells[i][1]
        for j in range(2):
            frac = cells[i][j] / tot
            fc = RAMP(0.10 + 0.80 * frac)
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=fc,
                    edgecolor=K,
                    linewidth=0.5,
                    zorder=1,
                )
            )
            lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
            tc = "white" if lum < 0.55 else K
            ax.text(
                j,
                i - 0.13,
                "{:,}".format(cells[i][j]),
                ha="center",
                va="center",
                fontsize=10.0,
                color=tc,
                zorder=2,
            )
            pc = "%.1f%%" % (100 * frac) if tot < 1000 else "%.2f%%" % (100 * frac)
            ax.text(
                j,
                i + 0.22,
                pc,
                ha="center",
                va="center",
                fontsize=SMALL,
                color=tc,
                zorder=2,
            )
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(collab, fontsize=BASE)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(rowlab, fontsize=BASE, linespacing=1.2)
    if colaxis:
        ax.set_xlabel(colaxis, fontsize=BASE, labelpad=3)
    if rowaxis:
        ax.set_ylabel(rowaxis, fontsize=BASE, labelpad=3)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    panel(ax, letter, head, y=1.05)


ax = fig.add_axes([AX_X[0], AX_Y, AX_W, AX_H])
tiles(
    ax,
    "a",
    "union of seven rules",
    [
        [N["unionRecallEnd"], N["nPos"] - N["unionRecallEnd"]],
        [N["unionAnchors"], N["nNegSupport"] - N["unionAnchors"]],
    ],
    [
        "spectroscopic\nLRDs (%d)" % N["nPos"],
        "comparison\nobjects (%s)" % "{:,}".format(N["nNegSupport"]),
    ],
    ["selected", "not selected"],
)
ax = fig.add_axes([AX_X[1], AX_Y, AX_W, AX_H])
tiles(
    ax,
    "b",
    "learned selection, region-matched",
    [
        [N9["matchedRecallEnd"], N["nPos"] - N9["matchedRecallEnd"]],
        [N9["matchedAnchors"], N["nNegSupport"] - N9["matchedAnchors"]],
    ],
    ["", ""],
    ["selected", "not selected"],
)
ax = fig.add_axes([AX_X[2], AX_Y, AX_W, AX_H])
tiles(
    ax,
    "c",
    "the %d LRDs, paired" % N["nPos"],
    [
        [N9["matchedPairedBoth"], N9["matchedPairedUnionOnlyEnd"]],
        [N9["matchedPairedModelOnly"], N9["matchedPairedNeitherEnd"]],
    ],
    ["yes", "no"],
    ["yes", "no"],
    rowaxis="selected by the rules",
    colaxis="selected by the learned selection",
)
save(fig, "fig_confusion", 178.0)

# ================================================================== Figure 8: score distributions
# Round 9: the five ensembles' scores are not on one scale, so the distributions are drawn
# one region per panel, each with its own fold's equal-burden threshold.
fig, axes = plt.subplots(5, 1, figsize=(COL1, 124 * MM), sharex=True)
fig.subplots_adjust(left=0.155, right=0.975, top=0.965, bottom=0.115, hspace=0.42)
bins = np.linspace(0, 1, 41)
for _ax, _r in zip(axes, REG):
    _s = sup[sup.region == _r]
    _n = neg[neg.region == _r]
    _p = pos[pos.region == _r]
    _ax.hist(_s.score, bins=bins, color=C_CAT, lw=0, label="support rows")
    _ax.hist(_n.score, bins=bins, color=C_NEG, lw=0, label="comparison objects")
    _ax.hist(_p.score, bins=bins, color=C_MODEL, lw=0, label="spectroscopic LRDs")
    _tb = float(np.unique(_s.t_burden)[0])
    _ax.axvline(_tb, color=C_MISS, lw=0.8, zorder=4)
    _ax.set_yscale("log")
    _ax.set_ylim(0.7, 4e5)
    _ax.set_xlim(0, 1)
    _ax.set_yticks([1, 1e2, 1e4])
    _ax.set_title("%s: %s rows, %d comparison objects, %d LRDs" % (_r, "{:,}".format(len(_s)), len(_n), len(_p)), fontsize=SMALL, loc="left", pad=2.0)
    _ax.text(_tb - 0.012, 1.0e4, "threshold %.3f" % _tb, fontsize=SMALL, color=C_MISS, ha="right", va="top")
axes[-1].set_xlabel("out-of-fold ranking score (held-out region)")
axes[2].set_ylabel("rows")
_h, _l = axes[0].get_legend_handles_labels()
fig.legend(_h, _l, loc="lower center", bbox_to_anchor=(0.56, 0.0), fontsize=SMALL, ncol=3, handlelength=1.2, columnspacing=0.8, handletextpad=0.4, frameon=False)
save(fig, "fig_scores", 84.0)

# ================================================================== Figure 9: image gallery
calib = json.load(
    open(os.path.join(ROOT, "inputs", "r1b_calibration.json"))
)
BANDS = ["F150W-CLEAR", "F277W-CLEAR", "F444W-CLEAR"]
RGB_SCL = calib["rgb"]["rgb_scl"]
STRETCH = float(calib["rgb"]["stretch"])
Q = float(calib["rgb"]["Q"])
GRAY_TOP = float(calib["gray_f444w"]["T"])
ASINH_A = float(calib["gray_f444w"]["a"])
# the six single-band panels round 3 adds, in wavelength order, with the composite last
BANDS6 = [
    "F115W-CLEAR",
    "F150W-CLEAR",
    "F200W-CLEAR",
    "F277W-CLEAR",
    "F356W-CLEAR",
    "F444W-CLEAR",
]
BAND_SHORT = {b: b.split("-")[0] for b in BANDS6}


def load_bands(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with afits.open(path, memmap=False) as h:
            ims = {}
            for hdu in h:
                name = (hdu.header.get("FILTER") or hdu.name or "").upper()
                for b in BANDS:
                    if b in name:
                        ims[b] = np.nan_to_num(np.asarray(hdu.data, float), nan=0.0)
    assert len(ims) == 3, path
    return {b: im - np.median(im) for b, im in ims.items()}


def load_bands6(path):
    """The six-band cutout, median-subtracted band by band; a band with no coverage is None.

    The file is the one fetch_stamps_6band.py wrote for this object: same service, same
    coordinates, same 5 arcsec radius as the three-band file beside it.
    """
    ims = {b: None for b in BANDS6}
    if not os.path.exists(path):
        return ims
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with afits.open(path, memmap=False) as h:
            for hdu in h:
                if hdu.data is None:
                    continue
                name = (hdu.header.get("FILTER") or hdu.name or "").upper()
                for b in BANDS6:
                    if b in name:
                        ims[b] = np.nan_to_num(
                            np.asarray(hdu.data, float), nan=0.0
                        ) - np.median(np.nan_to_num(np.asarray(hdu.data, float), 0.0))
    return ims


def render(ims):
    rgb = make_lupton_rgb(
        ims["F444W-CLEAR"] * RGB_SCL["F444W-CLEAR"],
        ims["F277W-CLEAR"] * RGB_SCL["F277W-CLEAR"],
        ims["F150W-CLEAR"] * RGB_SCL["F150W-CLEAR"],
        minimum=0.0,
        stretch=STRETCH,
        Q=Q,
    )
    x = np.clip(ims["F444W-CLEAR"] / GRAY_TOP, 0, None)
    g = np.clip(np.arcsinh(x / ASINH_A) / np.arcsinh(1.0 / ASINH_A), 0, 1)
    return rgb, g


def gray_panel(im):
    """The one asinh rule every single-band panel uses, in every band and every object.

    It is the frozen R1b gray rule (inputs/r1b_calibration.json, key
    gray_f444w) that the F444W panel has always used, applied unchanged to the other five
    bands: the image is divided by the same normalisation T, clipped below at zero, and put
    through arcsinh(x/a) / arcsinh(1/a) with the same softening a. Because T and a never
    change from panel to panel, the strip is photometric: a band in which an object is
    faint prints faint, which is the point of showing the blue bands at all. No new
    calibration constant is introduced.
    """
    x = np.clip(im / GRAY_TOP, 0, None)
    return np.clip(np.arcsinh(x / ASINH_A) / np.arcsinh(1.0 / ASINH_A), 0, 1)


man = pd.read_csv(os.path.join(FIG, "stamps", "manifest.csv"))
_rec = man[man.group == "recovered"]
_red = _rec[_rec.c_f277w_f444w > 1.0]
_blue = _rec[_rec.c_f277w_f444w < 0.5]
assert len(_red) + len(_blue) == len(_rec), (
    "the recovered block no longer splits in two"
)
# the three sections, in the order they have always stood in, with the recovered block's two
# color halves now separate sub-blocks so that each starts on a row of its own
SECTIONS = [
    dict(
        key="recovered",
        title="Spectroscopic LRDs missed by every rule and recovered by the learned selection",
        # Round 9: the three reddest recovered misses are the objects of fig_miss_anatomy,
        # so only the three bluest are drawn here
        parts=[
            ("F277W$-$F444W below 0.5", _blue),
        ],
    ),
    dict(
        key="missed",
        title="Spectroscopic LRDs the learned selection still misses at equal burden",
        parts=[(None, man[man.group == "missed"])],
    ),
    dict(
        key="candidate",
        title="Highest-ranked follow-up-tier candidates: no rule, no list, no usable archival spectrum",
        parts=[(None, man[man.group == "candidate"])],
    ),
]
GCOL = {"recovered": C_MODEL, "missed": C_MISS, "candidate": C_CAND}

# ------------------------------------------------------------------ geometry, in millimetres
# Round 3 presents each object as a strip of six single-band panels and the color composite,
# the way Akins et al. (2025) Fig. 4 and Labbe et al. (2025) Fig. 4 present theirs. Seven
# panels per object is three and a half times the panel count of round 2, so two objects fit
# across 178 mm instead of three, and each panel is 11.4 mm rather than 25.1 mm. That is the
# price of the strip; the alternative, one object per row, would make the figure 360 mm tall
# and unplaceable.
G_W = 178.0
G_NCOL = 2  # objects across
G_NPAN = 7  # six bands and the composite
G_MARGIN = 3.5
G_PAIRGAP = 6.0
G_INGAP = 0.5
G_STAMP = (
    G_W - 2 * G_MARGIN - (G_NCOL - 1) * G_PAIRGAP - G_NCOL * (G_NPAN - 1) * G_INGAP
) / (G_NCOL * G_NPAN)
G_STRIP = G_NPAN * G_STAMP + (G_NPAN - 1) * G_INGAP  # width of one object's strip
G_ABOVE = 3.4  # the "field id" line
G_BELOW = 3.4  # the measurement line
G_ROWGAP = 1.3
G_ROW = G_ABOVE + G_STAMP + G_BELOW + G_ROWGAP
G_HEAD = 5.2  # a section title line
G_SUB = 4.2  # a sub-block qualifier line
G_TOP, G_BOT = 1.0, 1.0

# The cutout service is asked for size=5, which is a radius: it returns 200 x 200 pixels at
# 0.05 arcsec per pixel, a 10 arcsec field, as the CD1_1 = -1.388889e-05 deg of every stamp
# header says. Round 1 and round 2 read that as 5 arcsec across and so used half the true
# pixel scale, which made the crop 2.0 arcsec on paper when it is 4.0, and the bar labeled
# 0.5 arcsec 1.0 arcsec long. The scale below is the header's. build_figures_extra.py
# already uses it, so the two sets of stamps now agree.
# Round 3 narrows the crop from 80 to 40 pixels: a strip of seven panels on a two-column
# page gives each panel about 11 mm, and at 4.0 arcsec the object filled too little of it.
# The field is the central 2.0 arcsec of the same stamp; the pixel scale, the calibration
# and the objects are unchanged, and the same crop is used for the composite.
crop = slice(
    80, 120
)  # central 2.0 arcsec of the 10 arcsec stamp (200 px, 0.05 arcsec/px)
PX = 0.05  # arcsec per pixel, from CD1_1 in the stamp headers
BAR_ARCSEC = 0.5
BAR_PX = BAR_ARCSEC / PX  # 10 px inside the 40 px crop
NPX = crop.stop - crop.start


def draw_gallery(name, sections):
    """Draw one gallery canvas from a list of sections and save it at 178 mm.

    Round 4 splits the gallery in two. Nine objects of known class and six candidates on
    one canvas made a figure 201.1 mm tall, and main.tex had to cap its height, which
    printed the whole thing at 0.919 and its lettering below the 8 pt floor. The sections,
    the geometry, the crop, the stretch and the order are exactly what they were; only the
    partition into canvases is new, so a reader sees the same strips at their intended size.
    The full canvas is still written, under its old name, as the reference version.
    """
    n_rows, n_subs = 0, 0
    for s in sections:
        for sub, d in s["parts"]:
            n_rows += int(np.ceil(len(d) / G_NCOL))
            n_subs += sub is not None
    G_H = G_TOP + G_BOT + len(sections) * G_HEAD + n_subs * G_SUB + n_rows * G_ROW

    fig = plt.figure(figsize=(COL2, G_H * MM))

    def fx(mm_):
        return mm_ / G_W

    def fy(mm_from_top):
        return 1.0 - mm_from_top / G_H

    def stamp_axes(x_mm, y_mm, edge):
        """One square panel of a strip, at a place fixed in millimetres."""
        a = fig.add_axes([fx(x_mm), fy(y_mm + G_STAMP), fx(G_STAMP), G_STAMP / G_H])
        a.set_xticks([])
        a.set_yticks([])
        a.xaxis.set_minor_locator(NullLocator())
        a.yaxis.set_minor_locator(NullLocator())
        for s in a.spines.values():
            s.set_visible(True)
            s.set_color(edge)
            s.set_linewidth(0.6)
        return a

    def band_tag(ax, text, color="white"):
        """The filter name inside the panel corner, white on a dark box (Akins Fig. 4)."""
        ax.text(
            0.055,
            0.945,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=SMALL,
            color=color,
            zorder=6,
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.55, pad=0.9),
        )

    cursor = G_TOP
    GALLERY_ROWS = []
    for sec in sections:
        g = sec["key"]
        fig.text(
            fx(G_MARGIN),
            fy(cursor + G_HEAD - 1.4),
            sec["title"],
            fontsize=BASE,
            fontweight="bold",
            color=K,
            va="baseline",
            ha="left",
        )
        cursor += G_HEAD
        for sub, d in sec["parts"]:
            d = d.reset_index(drop=True)
            if sub is not None:
                fig.text(
                    fx(G_MARGIN),
                    fy(cursor + G_SUB - 1.3),
                    sub,
                    ha="left",
                    va="baseline",
                    fontsize=BASE,
                    color=GCOL[g],
                )
                cursor += G_SUB
            for k in range(len(d)):
                r, c = divmod(k, G_NCOL)
                m = d.iloc[k]
                stem = "%s_%s_%d" % (g, m.field, int(m.id))
                ims3 = load_bands(os.path.join(FIG, "stamps", stem + ".fits"))
                rgb, _gray = render(
                    ims3
                )  # the composite, frozen R1b calibration, unchanged
                ims6 = load_bands6(os.path.join(FIG, "stamps", stem + "_6band.fits"))
                row_top = cursor + r * G_ROW
                x0 = G_MARGIN + c * (G_STRIP + G_PAIRGAP)
                y_st = row_top + G_ABOVE
                for j, b in enumerate(BANDS6):
                    ax = stamp_axes(x0 + j * (G_STAMP + G_INGAP), y_st, GCOL[g])
                    im = ims6[b]
                    if (
                        im is None
                    ):  # the band has no coverage here: say so, do not fake it
                        ax.set_facecolor("#f2f2f2")
                        band_tag(ax, BAND_SHORT[b], color="white")
                        ax.text(
                            0.5,
                            0.42,
                            "no\ncoverage",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=SMALL,
                            color="#8a8a8a",
                            linespacing=1.15,
                        )
                        continue
                    ax.imshow(
                        gray_panel(im)[crop, crop],
                        origin="lower",
                        cmap="gray",
                        vmin=0,
                        vmax=1,
                        interpolation="nearest",
                    )
                    band_tag(ax, BAND_SHORT[b])
                # the composite closes the strip, and carries the one scale bar of the object
                axr = stamp_axes(x0 + 6 * (G_STAMP + G_INGAP), y_st, GCOL[g])
                axr.imshow(rgb[crop, crop], origin="lower", interpolation="nearest")
                band_tag(axr, "RGB")
                # the bar is inset far enough that its centered label clears the left border
                _bx = 0.15 * NPX
                axr.plot(
                    [_bx, _bx + BAR_PX],
                    [0.085 * NPX, 0.085 * NPX],
                    color="white",
                    lw=1.1,
                    solid_capstyle="butt",
                    zorder=6,
                )
                axr.text(
                    _bx + BAR_PX / 2,
                    0.115 * NPX,
                    '0.5"',
                    color="white",
                    fontsize=SMALL,
                    ha="center",
                    va="bottom",
                    zorder=6,
                )
                if g == "candidate":
                    rk = int(
                        cand[(cand.field == m.field) & (cand.id == int(m.id))][
                            "rank"
                        ].iloc[0]
                    )
                    tail = "rank %d" % rk
                else:
                    tail = "score %.2f" % m.score
                xc = fx(x0 + G_STRIP / 2)
                fig.text(
                    xc,
                    fy(row_top + G_ABOVE - 1.0),
                    "%s %d" % (m.field, int(m.id)),
                    ha="center",
                    va="baseline",
                    fontsize=SMALL,
                    color=K,
                )
                fig.text(
                    xc,
                    fy(y_st + G_STAMP + 2.7),
                    "F444W %.1f, F277W$-$F444W %.2f, %s"
                    % (m.mag_f444w, m.c_f277w_f444w, tail),
                    ha="center",
                    va="baseline",
                    fontsize=SMALL,
                    color=K,
                )
                GALLERY_ROWS.append((g, str(m.field), int(m.id), stem + "_6band.fits"))
            cursor += int(np.ceil(len(d) / G_NCOL)) * G_ROW
    save(fig, name, 178.0)
    print(
        "  %s: %d objects, %d panels, stamp %.2f mm, six-band files %d"
        % (
            name,
            len(GALLERY_ROWS),
            len(GALLERY_ROWS) * G_NPAN,
            G_STAMP,
            sum(
                os.path.exists(os.path.join(FIG, "stamps", r[3])) for r in GALLERY_ROWS
            ),
        )
    )


KNOWN = [s for s in SECTIONS if s["key"] in ("recovered", "missed")]
CAND = [s for s in SECTIONS if s["key"] == "candidate"]
assert len(KNOWN) == 2 and len(CAND) == 1, "the gallery sections changed"
# Round 4, Reviewer C housekeeping: the combined fig_gallery canvas is no longer written.
# main.tex includes fig_gallery_known and fig_gallery_cand only, so the third canvas was a
# stale artefact that no caption referred to.
draw_gallery("fig_gallery_known", KNOWN)
draw_gallery("fig_gallery_cand", CAND)

# ================================================================== Figure 10: candidates
fig, axes = plt.subplots(1, 3, figsize=(COL2, 62 * MM), layout="constrained")
fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.06, hspace=0.0)
ax = axes[0]
bins = np.arange(22, 29.5, 0.5)
ax.hist(
    pos.mag_f444w,
    bins=bins,
    density=True,
    histtype="step",
    color=C_MODEL,
    lw=1.0,
    label="spectroscopic LRDs (%d)" % len(pos),
)
ax.hist(
    eb.mag_f444w,
    bins=bins,
    density=True,
    histtype="step",
    color=C_MISS,
    lw=1.0,
    label="equal-burden tier (%d)" % len(eb),
)
ax.hist(
    fu.mag_f444w,
    bins=bins,
    density=True,
    histtype="step",
    color=C_UNION,
    lw=0.9,
    ls=(0, (3.5, 1.8)),
    label="follow-up tier (%d)" % len(fu),
)
ax.set_xlabel("F444W (AB mag)")
ax.set_ylabel("density")
ax.set_xlim(22, 29)
ax.set_ylim(0, 0.80)  # headroom so the key cannot reach the histograms
ax.legend(fontsize=BASE, loc="upper left", borderaxespad=0.5)
panel(ax, "a")
ax = axes[1]
bins = np.arange(0, 12.5, 0.5)
ax.hist(
    pos.z_phot.clip(0, 12),
    bins=bins,
    density=True,
    histtype="step",
    color=C_MODEL,
    lw=1.0,
)
ax.hist(
    eb.z_phot.clip(0, 12),
    bins=bins,
    density=True,
    histtype="step",
    color=C_MISS,
    lw=1.0,
)
ax.hist(
    fu.z_phot.clip(0, 12),
    bins=bins,
    density=True,
    histtype="step",
    color=C_UNION,
    lw=0.9,
    ls=(0, (3.5, 1.8)),
)
ax.axvline(3, color=K, lw=0.5, ls=":")
ax.set_xlim(0, 12)
ax.set_ylim(0, 0.52)
ax.set_xlabel("catalog photometric redshift")
ax.set_ylabel("density")
panel(ax, "b")
ax = axes[2]
sets = [
    ("all\n%d" % N["candTotal"], cand),
    ("equal-burden\n%d" % N["candEqualBurden"], eb),
    ("follow-up\n%d" % N["followupN"], fu),
]
x = np.arange(3)
hs = []
CAND_STATUS = {}
for i, (lab, d) in enumerate(sets):
    c_ = d.status.value_counts()
    u_ = c_.get("untested", 0)
    h_ = c_.get("secure z>3 (V untested)", 0)
    l_ = c_.get("secure low-z", 0)
    tot = len(d)
    CAND_STATUS[lab.split("\n")[0]] = (int(u_), int(h_), int(l_), int(tot))
    b0 = ax.bar(i, u_ / tot, 0.62, color=C_CAT, edgecolor=K, lw=0.3)
    b1 = ax.bar(i, h_ / tot, 0.62, bottom=u_ / tot, color=C_MODEL, edgecolor=K, lw=0.3)
    b2 = ax.bar(
        i, l_ / tot, 0.62, bottom=(u_ + h_) / tot, color=C_MISS, edgecolor=K, lw=0.3
    )
    # Round 9: the counts are printed, so the axis can run from 0 to 1 without hiding them
    ax.text(i, 0.5 * u_ / tot, "%d" % u_, ha="center", va="center", fontsize=SMALL, color=K)
    ax.text(i, 1.012, "%d / %d" % (h_, l_), ha="center", va="bottom", fontsize=SMALL, color=K)
    if i == 0:
        hs = [b0, b1, b2]
ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in sets], fontsize=BASE, linespacing=1.35)
ax.set_xlim(-0.6, 2.6)
ax.set_ylim(0.0, 1.10)
ax.xaxis.set_minor_locator(NullLocator())
ax.tick_params(axis="x", length=0)
ax.set_ylabel("fraction")
ax.legend(
    handles=hs,
    labels=["no usable spectrum", "secure $z > 3$", "secure $z \\leq 3$"],
    loc="lower right",
    bbox_to_anchor=(1.0, 1.005),
    ncol=3,
    frameon=False,
    fontsize=BASE,
    borderaxespad=0.0,
    handlelength=1.0,
    handletextpad=0.45,
    columnspacing=1.1,
)
# the key sits on the row directly above the axes, so the panel label moves up a row
panel(ax, "c", "archive status", y=1.13)
save(fig, "fig_candidates", 178.0)

# ================================================================== Figure 11: robustness dot plot
# A forest plot: one row per run, filled marker at the burden matched to the union region by
# region, open marker at the run's own threshold, joined by a hairline. Every number that
# used to be printed on the rows is in Table 3 and in Section 4.4; a figure shows the shape.
ROB_H = 96.0
fig = plt.figure(figsize=(COL2, ROB_H * MM))
ROB_LEFT = 48.0  # millimetres kept for the row labels and the group headings
ax = fig.add_axes(
    [
        ROB_LEFT / 178.0,
        13.5 / ROB_H,
        (178.0 - ROB_LEFT - 3.5) / 178.0,
        1.0 - (13.5 + 6.5) / ROB_H,
    ]
)
items = [
    # label, matched recall, color, marker, recall at the run's own threshold
    (
        "learned selection (primary)",
        N["primaryMatchedRecall"],
        C_MODEL,
        "o",
        N["modelRecallSupport"],
    ),
    ("8-bag replica", N["blReplicaMatchedRecall"], C_ALT, "o", N["blReplicaRecall"]),
    (
        "anchor weight 0",
        N["blAnchorZeroMatchedRecall"],
        C_ALT,
        "o",
        N["blAnchorZeroRecall"],
    ),
    (
        "anchor weight 0.5",
        N["blAnchorHalfMatchedRecall"],
        C_ALT,
        "o",
        N["blAnchorHalfRecall"],
    ),
    (
        "positives against comparison objects only",
        N["blPNMatchedRecall"],
        C_ALT,
        "o",
        N["blPNRecall"],
    ),
    (
        "with photometric redshift",
        N["blWithZphotMatchedRecall"],
        C_ALT,
        "o",
        N["blWithZphotRecall"],
    ),
    (
        "without F277W$-$F444W",
        N["blAblateColourMatchedRecall"],
        C_ALT,
        "o",
        N["blAblateColourRecall"],
    ),
    (
        "without size features",
        N["blAblateSizeMatchedRecall"],
        C_ALT,
        "o",
        N["blAblateSizeRecall"],
    ),
    (
        "magnitudes and S/N only",
        N["blMagOnlyMatchedRecall"],
        C_ALT,
        "o",
        N["blMagOnlyRecall"],
    ),
    ("sizes only", N["blMorphOnlyMatchedRecall"], C_ALT, "o", N["blMorphOnlyRecall"]),
    (
        "imitation of the rules (no labels)",
        N["blImitationMatchedRecall"],
        K,
        "^",
        N["blImitationRecall"],
    ),
    (
        "trained without %s LRDs" % LIST_BARRO,
        N["locoBarroMatchedRecall"],
        C_LOCO,
        "s",
        N["locoBarroRecall"],
    ),
    (
        "trained without %s LRDs" % LIST_DEGRAAFF,
        N["locoDeGraaffMatchedRecall"],
        C_LOCO,
        "s",
        N["locoDeGraaffRecall"],
    ),
    (
        "trained without %s LRDs" % LIST_HVIDING,
        N["locoHvidingAMatchedRecall"],
        C_LOCO,
        "s",
        N["locoHvidingARecall"],
    ),
    ("shuffled labels (%d repeats)" % N["shuffledN"], None, C_MISS, None, None),
]
# groups in the order the rows already stand in
GROUPS = [
    ("primary and ablations", 0, 9),
    ("baseline", 10, 10),
    ("leave one list out", 11, 13),
    ("shuffled-label control", 14, 14),
]
slot = 0.0
yof = {}
head_at = []
for gname, i0, i1 in GROUPS:
    head_at.append((slot, gname))
    slot += 1.0
    for i in range(i0, i1 + 1):
        yof[i] = slot
        slot += 1.0
    slot += 0.35
TOP = slot
for i, (lab, rec, col, mk, dep) in enumerate(items):
    yy = TOP - yof[i]
    if rec is None:
        ax.plot(
            [N["shuffledMatchedMin"], N["shuffledMatchedMax"]],
            [yy, yy],
            color=col,
            lw=2.6,
            solid_capstyle="butt",
            zorder=5,
        )
        ax.plot(
            [N["shuffledMin"], N["shuffledMax"]],
            [yy - 0.3, yy - 0.3],
            color=col,
            lw=0.8,
            alpha=0.55,
            solid_capstyle="butt",
            zorder=4,
        )
    else:
        ax.plot(
            [min(rec, dep), max(rec, dep)], [yy, yy], color="#c4c4c4", lw=0.6, zorder=3
        )
        ax.scatter(
            [dep],
            [yy],
            facecolor="white",
            edgecolor=col,
            lw=0.7,
            s=17,
            marker=mk,
            zorder=4,
        )
        ax.scatter([rec], [yy], color=col, s=19, marker=mk, zorder=5)
ax.set_yticks([TOP - yof[i] for i in range(len(items))])
ax.set_yticklabels([it[0] for it in items], fontsize=BASE)
ax.set_ylim(TOP - slot + 0.4, TOP + 0.55)
ax.set_xlim(-3, 143)  # "sizes only" recovers 4 and 3; the axis must reach them
ax.set_xticks([0, 20, 40, 60, 80, 100, 120, 140])
ax.yaxis.set_minor_locator(NullLocator())
ax.tick_params(axis="y", length=0)
ax.set_xlabel("spectroscopic LRDs recovered, of %d (out of fold)" % N["nPosSupport"])
# the two reference lines, named at the top so no row carries a number string
for xv, txt, ha, dx in (
    (
        N["unionRecallSupport"],
        "union of rules %d" % N["unionRecallSupport"],
        "right",
        -1.5,
    ),
    (N["primaryMatchedRecall"], "primary %d" % N["primaryMatchedRecall"], "left", 1.5),
):
    ax.axvline(xv, color="#9a9a9a", lw=0.6, ls=(0, (4, 2)), zorder=1)
    ax.text(
        xv + dx,
        TOP + 0.62,
        txt,
        fontsize=BASE,
        color=GREY_TXT,
        ha=ha,
        va="bottom",
        clip_on=False,
    )
# group headings on their own blank slot, at the far left of the row-label column, with a
# hairline above each group after the first
for gi, (s, gname) in enumerate(head_at):
    ax.text(
        -(ROB_LEFT - 2.0) / (178.0 - ROB_LEFT - 3.5),
        TOP - s,
        gname,
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=BASE,
        fontweight="bold",
        color=GREY_TXT,
        clip_on=False,
    )
    if gi:
        ax.axhline(TOP - s + 0.62, color="#dcdcdc", lw=0.5, zorder=0)
key = [
    Line2D(
        [],
        [],
        color=C_ALT,
        marker="o",
        ls="none",
        ms=4.0,
        label="matched to the union's burden, region by region",
    ),
    Line2D(
        [],
        [],
        markerfacecolor="white",
        markeredgecolor=C_ALT,
        marker="o",
        ls="none",
        ms=4.0,
        mew=0.7,
        label="at the run's own training-region threshold",
    ),
]
ax.legend(
    handles=key, loc="lower right", fontsize=BASE, borderaxespad=0.6, handletextpad=0.4
)
save(fig, "fig_robustness", 178.0)

# ================================================================== Figure 1, compiled
# fig_pipeline is a TikZ standalone that reads its numbers from ../numbers.tex, so it has to
# be recompiled whenever numbers.tex changes or the PDF goes stale against the text. Running
# it here means one command rebuilds all eleven figures.
if shutil.which("pdflatex"):
    _r = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig_pipeline.tex"],
        cwd=FIG,
        capture_output=True,
        text=True,
        check=False,  # the return code is inspected below, with the log attached
    )
    if _r.returncode:
        raise SystemExit(
            "fig_pipeline.tex failed to compile:\n"
            + _r.stdout[-3000:]
            + _r.stderr[-2000:]
        )
    for _ext in (".aux", ".log"):
        _f = os.path.join(FIG, "fig_pipeline" + _ext)
        if os.path.exists(_f):
            os.remove(_f)
    from pypdf import PdfReader as _PR

    _b = _PR(os.path.join(FIG, "fig_pipeline.pdf")).pages[0].mediabox
    _w, _h = float(_b.width) / 72 * 25.4, float(_b.height) / 72 * 25.4
    assert abs(_w - 178.0) < 0.1, "fig_pipeline.pdf is %.2f mm wide, wanted 178.00" % _w
    print(
        "wrote %-26s %6.1f x %6.1f mm  smallest font 8.0 pt" % ("fig_pipeline", _w, _h)
    )
    FONT_REPORT.insert(0, ("fig_pipeline", _w, _h, 8.0))
else:
    print("pdflatex not found: figures/fig_pipeline.pdf left as it stands")

# rasterise it so the contact sheet shows all eleven figures side by side
try:
    import pypdfium2 as pdfium

    _pp = os.path.join(FIG, "fig_pipeline.pdf")
    if os.path.exists(_pp):
        _pg = pdfium.PdfDocument(_pp)[0]
        _pg.render(scale=300 / 72).to_pil().save(os.path.join(FIG, "fig_pipeline.png"))
        print("wrote fig_pipeline.png from fig_pipeline.pdf")
except Exception as e:  # noqa: BLE001 - the PDF is the deliverable, the PNG only reviews it
    print("fig_pipeline.png not rasterised:", e)

# ================================================================== font report
with open(
    os.path.join(FIG, "font_report.txt"), "w", encoding="utf-8", newline="\n"
) as f:
    f.write("figure                       width_mm  height_mm  min_font_pt\n")
    for nm, w, h, mf in FONT_REPORT:
        f.write("%-28s %8.1f %10.1f %12.1f\n" % (nm, w, h, mf))
    f.write(
        "\nfig_pipeline is a TikZ standalone: its one type size is set in"
        " figures/fig_pipeline.tex and measured back from the compiled PDF.\n"
        "Every figure is emitted at the exact width main.tex includes it at. Under"
        " aastex701 (twocolumn), measured from the class, \\columnwidth is 85.15 mm and"
        " \\textwidth 180.34 mm, so a 178 mm canvas is scaled up by 1.3 percent and an"
        " 84 mm canvas by 1.4 percent: 8.0 pt prints at 8.1 pt.\n"
        "main.tex includes fig_gallery_known and fig_gallery_cand at \\textwidth with"
        " no height limit, so both print at the width their canvas is emitted for and the"
        " sizes above are the sizes on the page.\n"
    )
print("wrote font_report.txt")

# ================================================================== counts the captions need
with open(
    os.path.join(FIG, "caption_facts.txt"), "w", encoding="utf-8", newline="\n"
) as f:
    f.write("fig_candidates panel (c), counts no longer printed on the bars\n")
    f.write("tier: untested, secure z>3, secure z<=3, total\n")
    for k, v in CAND_STATUS.items():
        f.write("  %-14s %5d %5d %5d %6d\n" % (k, v[0], v[1], v[2], v[3]))
    f.write("\nfig_confusion panel (c) overlap line, moved to the caption\n")
    f.write("  rows selected by both: %d\n" % N["setBoth"])
    f.write("  Jaccard overlap of the two selections: %.2f\n" % N["setJaccard"])
    f.write("\nfig_gain_by_colour_mag bin counts (bin, n, union, learned)\n")
    for p, (labels, nn, u, m) in GAIN_COUNTS.items():
        f.write("  panel (%s)\n" % p)
        for i in range(len(labels)):
            f.write("    %-12s %5d %5d %5d\n" % (labels[i], nn[i], u[i], m[i]))
print("wrote caption_facts.txt")

# ================================================================== contact sheet for review
ORDER = [
    "fig_pipeline",
    "fig_recall_burden",
    "fig_gain_by_colour_mag",
    "fig_colour_planes",
    "fig_regions",
    "fig_importance",
    "fig_confusion",
    "fig_scores",
    "fig_gallery_known",  # the two gallery canvases the paper uses
    "fig_gallery_cand",
    "fig_candidates",
    "fig_robustness",
]
ncol = 3
nrow = int(np.ceil(len(ORDER) / ncol))
cs, cs_axes = plt.subplots(nrow, ncol, figsize=(13.5, 4.6 * nrow))
for a in cs_axes.ravel():
    a.axis("off")
for i, nm in enumerate(ORDER):
    a = cs_axes.ravel()[i]
    p = os.path.join(FIG, nm + ".png")
    if os.path.exists(p):
        a.imshow(plt.imread(p))
    a.set_title(nm + ".png", fontsize=11, color=K, pad=6)
cs.tight_layout()
cs.savefig(
    os.path.join(FIG, "contact_sheet.png"),
    dpi=110,
    bbox_inches="tight",
    pad_inches=0.15,
)
plt.close(cs)
print("wrote contact_sheet")
print("all figures written")
