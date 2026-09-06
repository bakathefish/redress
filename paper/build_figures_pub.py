"""The two new figures of the publication rewrite.

fig_zdist          the spectroscopic redshift distribution of all nPos spectroscopic LRDs,
                   stacked by which of the two selections recovers each one, end to end.
fig_colour_colour  the canonical color-color view of the seven published rules, drawn on
                   ordinary AB colors because that is the space in which the rules write
                   their thresholds.

Run from the repository root:
    python paper/build_figures_pub.py

Writes paper/figures/fig_zdist.{pdf,png} and fig_colour_colour.{pdf,png}, copies
the four files to paper/figures/, appends the figure-local facts to
paper/figures/caption_facts.txt and paper/figures/caption_facts.txt, and
appends the measured font floor to paper/figures/font_report.txt.

House style is build_figures.py's and is reproduced verbatim below: serif through STIX,
8 pt everywhere at output size with 7.5 pt allowed only for threshold and in-image labels,
thin rules, ticks inward on all four sides, no gridlines, no titles, panel letters outside
above the top left. Every canvas is emitted at exactly the width main.tex includes it at,
with no bounding-box cropping, so a point specified here is a point on the page.
"""

import json
import os
import shutil
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    LinearSegmentedColormap,
    LogNorm,
    to_rgba,  # noqa: E402
)
from matplotlib.offsetbox import (  # noqa: E402
    AnnotationBbox,
    HPacker,
    TextArea,
    VPacker,
)
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

# Optional visual self-check: render the PNG, audit it programmatically for missing glyphs,
# clipped text and overlapping tick labels, then inspect the PNG against the checklist and
# fix what shows. The helper module is not part of this repository; point FIGURE_QA_SCRIPTS
# at a directory holding visual_qa.py to enable it. render_preview is given the PNG this
# script already wrote at the exact canvas size, rather than a figure, because the house
# rule is that the canvas is the final size and is never cropped.
_SKILL = os.environ.get("FIGURE_QA_SCRIPTS", "")
if _SKILL not in sys.path:
    sys.path.insert(0, _SKILL)
try:
    from visual_qa import audit_layout, print_report, render_preview  # noqa: E402

    HAVE_QA = True
except ImportError:  # pragma: no cover - the helper is not part of this repository
    HAVE_QA = False
    print("WARNING: scipilot-figure-skill not found at %s; the programmatic" % _SKILL)
    print("         layout audit did not run. Install it or run the audit by hand.")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "recovery")
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
PUB = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(PUB, exist_ok=True)
N = json.load(open(os.path.join(HERE, "numbers.json")))
N9 = json.load(open(os.path.join(ROOT, "recovery", "round9_numbers.json")))
POUT = pd.read_csv(os.path.join(ROOT, "recovery", "positive_outcomes.csv")).set_index("source_id")

# ==========================================================================================
# House style, copied from paper/build_figures.py lines 78 to 290.
# Copied rather than imported: importing build_figures.py runs all eleven of its figures.
# ==========================================================================================

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
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "text.usetex": False,
        "font.size": BASE,
        "axes.labelsize": BASE,
        "axes.titlesize": BASE,
        "legend.fontsize": BASE,
        "xtick.labelsize": BASE,
        "ytick.labelsize": BASE,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.6,
        "grid.linewidth": 0.4,
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
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": None,  # never crop: the canvas IS the final size
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # a hairline hatch; its color follows the patch edge color in matplotlib 3.10,
        # so the white hatch of the rules-only class is produced by drawing that bar with a
        # white edge and laying a separate dark outline over it (see fig_zdist below)
        "hatch.linewidth": 0.5,
    }
)

# base is black and gray; exactly two accents, used the same way on every figure
K = "#000000"
C_UNION = "#666666"  # the seven published rules
C_MODEL = "#1a5a8a"  # accent 1, the learned selection
C_MISS = "#c0532a"  # accent 2, missed by every rule / shuffled labels
C_CAND = "#e0a06a"  # tint of accent 2, unlisted candidates
C_NEG = "#a0a0a0"  # spectroscopic non-LRD anchors
GREY_TXT = "#7f7f7f"  # secondary lettering (threshold and literature labels)

# ------------------------------------------------------------------ the seven rules
# Paul Tol's "muted" qualitative scheme (Tol 2021, SRON/EPS/TN/09-002 issue 3.2), in Table 1
# order, exactly as build_figures.py assigns it; a rule's color is carried redundantly by
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
    """The same hue at a luminance a reader can take as 8 pt text on white."""
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
DENS_CMAP = LinearSegmentedColormap.from_list(
    "greys_light", plt.get_cmap("Greys")(np.linspace(0.0, 0.58, 256))
)
# the display names used where a rule is named on a figure, with the journal year of the
# bibliography entry; the same seven names Table 1 lists
RULE_LABEL_SHORT = {
    "labbe23": "Labbé+25",
    "kokorev24": "Kokorev+24",
    "kocevski24": "Kocevski+25",
    "perezgonzalez24": "Pérez-González+24",
    "barro23": "Barro+24b",
    "greene24": "Greene+24",
    "akins24": "Akins+25",
}

FONT_REPORT = []
FACTS = []
QA_REPORT = []
GREY_REPORT = []
BLOCK_SIZES = []


def white_hatch(*patches):
    """Give a hatched patch a white hatch and keep its dark outline.

    A patch's hatch takes the patch's edge color in matplotlib 3.10, and setting the edge
    color overwrites the hatch color, so a single artist cannot be given a white hatch and
    a dark edge through the public interface. The private attribute is set after the fact
    instead, which keeps the hatched class in one artist rather than two stacked ones. The
    assertion makes a future matplotlib that drops the attribute fail loudly here rather
    than silently print a black hatch.
    """
    for p in patches:
        assert hasattr(p, "_hatch_color"), (
            "this matplotlib no longer carries Patch._hatch_color; "
            "the white hatch needs another route"
        )
        p._hatch_color = to_rgba("white")
    return patches[0] if len(patches) == 1 else patches


def marker_boxes(collections, renderer, pad_px=1.0):
    """Display-space boxes of every plotted marker in the given axes.

    Only the scatter markers count. A hexbin is a PolyCollection whose get_offsets() returns
    one entry per hexagon, so counting it would treat the whole catalog density map as
    plotted data and push every label into the few cells that hold no catalog row at all.
    The density map is background (style rule 3) and a label may sit over it, as Labbe et
    al. (2025) Fig. 2 and Hviding et al. (2025) Fig. 6 both do.
    """
    boxes = []
    for sc in collections:
        if not isinstance(sc, PathCollection):
            continue
        off = sc.get_offsets()
        if len(off) == 0:
            continue
        pts = sc.get_offset_transform().transform(off)
        # sizes are in points squared; half the side of the marker in device pixels
        dpi = sc.get_figure().dpi
        sizes = sc.get_sizes()
        if len(sizes) == 0:
            continue
        r = 0.5 * np.sqrt(float(np.max(sizes))) * dpi / 72.0 + pad_px
        for x, y in pts:
            boxes.append((x - r, y - r, x + r, y + r))
    return boxes


def label_collisions(fig, texts, collections, pad_px=1.0):
    """Labels whose rendered box touches a plotted marker, or another label.

    The scipilot skill's audit catches clipped text and overlapping tick labels; this is the
    third test the corrections asked for, that no in-panel label touches a plotted point.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    mboxes = marker_boxes(collections, renderer, pad_px)
    tboxes = []
    for t in texts:
        bb = t.get_window_extent(renderer)
        tboxes.append(
            (str(t.get_text()).replace("\n", " / "), (bb.x0, bb.y0, bb.x1, bb.y1))
        )
    hits = []

    def touch(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    for label, box in tboxes:
        n = sum(1 for m in mboxes if touch(box, m))
        if n:
            hits.append("label %r touches %d plotted markers" % (label[:44], n))
    for i in range(len(tboxes)):
        for j in range(i + 1, len(tboxes)):
            if touch(tboxes[i][1], tboxes[j][1]):
                hits.append(
                    "labels %r and %r overlap" % (tboxes[i][0][:30], tboxes[j][0][:30])
                )
    return hits


def in_panel_texts(ax):
    """Every visible label drawn inside the axes: in-panel notes and the key's rows.

    Tick labels and axis labels sit outside the axes rectangle, because the ticks point
    inward, so a center-inside test separates them cleanly from the labels this figure
    places by hand.
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ab = ax.get_window_extent(renderer)
    out = []
    for t in ax.findobj(Text):
        if not t.get_visible() or not str(t.get_text()).strip():
            continue
        bb = t.get_window_extent(renderer)
        cx, cy = 0.5 * (bb.x0 + bb.x1), 0.5 * (bb.y0 + bb.y1)
        if ab.x0 <= cx <= ab.x1 and ab.y0 <= cy <= ab.y1:
            out.append(t)
    return out


def patch_boxes(patches, renderer, pad_px=0.0):
    """Display-space boxes of drawn patches, for the bar-versus-key test."""
    boxes = []
    for p in patches:
        bb = p.get_window_extent(renderer)
        if bb.width <= 0 or bb.height <= 0:
            continue
        boxes.append((bb.x0 - pad_px, bb.y0 - pad_px, bb.x1 + pad_px, bb.y1 + pad_px))
    return boxes


def report_collisions(name, hits):
    """Print the collision test and fail the build if a label sits on the data."""
    if hits:
        print("  label collision test on %s: %d problems" % (name, len(hits)))
        for h in hits:
            print("    [FAIL] " + h)
    else:
        print(
            "  label collision test on %s: [PASS] no label touches a marker or"
            " another label" % name
        )
    QA_REPORT.append((name + " label collisions", "PASS" if not hits else "FAIL"))
    assert not hits, "%s: %d label collisions, see above" % (name, len(hits))


def grayscale_check(name, classes, min_luma_gap=20.0):
    """Report greyscale separation, per the scientific-visualization skill.

    `classes` is a list of (label, color, encoding, note). Rec. 709 luma is what a greyscale
    print reduces an sRGB color to. Two classes are separable in greyscale if their luma
    differs by at least `min_luma_gap` out of 255, or if they carry different redundant
    encodings (a different marker shape, hatch or dash pattern). A pair that shares an
    encoding and differs by less than the gap is reported as a weak pair: in a greyscale
    print a reader has nothing left to tell them apart with.
    """

    def luma(hexcol):
        r, g, b = (int(hexcol[i : i + 2], 16) for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    rows = [(lbl, luma(col), enc, note) for lbl, col, enc, note in classes]
    lines = ["%s greyscale separation (Rec. 709 luma out of 255)" % name]
    for lbl, lm, enc, note in rows:
        lines.append("    %-30s luma %5.1f   encoding %s" % (lbl, lm, note))
    weak = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            d = abs(rows[i][1] - rows[j][1])
            if rows[i][2] == rows[j][2] and d < min_luma_gap:
                weak.append((d, rows[i][0], rows[j][0]))
    if weak:
        for d, a, b in sorted(weak):
            lines.append(
                "    WEAK PAIR: %s and %s share an encoding and differ by %.1f luma"
                " levels, under the %.0f a greyscale print needs"
                % (a, b, d, min_luma_gap)
            )
    else:
        lines.append(
            "    PASS: every pair is separated by at least %.0f luma levels or by a"
            " different shape, hatch or dash" % min_luma_gap
        )
    GREY_REPORT.append("\n".join(lines))
    for ln in lines:
        print("  " + ln)
    QA_REPORT.append((name + " greyscale", "PASS" if not weak else "WEAK"))
    return weak


def min_font(fig):
    """Smallest point size actually drawn on this figure, measured from the artists."""
    sizes = [
        t.get_fontsize()
        for t in fig.findobj(Text)
        if t.get_visible() and str(t.get_text()).strip()
    ]
    return min(sizes) if sizes else float("nan")


def save(fig, name, width_mm):
    """Write the PDF and PNG at exactly the canvas size, and record the font floor."""
    w_in, h_in = fig.get_size_inches()
    got = w_in / MM
    assert abs(got - width_mm) < 0.05, "%s is %.2f mm wide, wanted %.2f" % (
        name,
        got,
        width_mm,
    )
    mf = min_font(fig)
    assert mf >= SMALL - 1e-9, "%s draws %.2f pt lettering, floor is %.1f pt" % (
        name,
        mf,
        SMALL,
    )
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches=None)
    png = os.path.join(FIG, name + ".png")
    fig.savefig(png, bbox_inches=None)
    # the skill's programmatic layer, run on the live figure before it is closed
    if HAVE_QA:
        print("  scipilot visual_qa.audit_layout on %s" % name)
        verdict = print_report(audit_layout(fig))
        render_preview(png)  # a bitmap path passes straight through, uncropped
    else:
        verdict = "NOT RUN"
    QA_REPORT.append((name, verdict))
    assert verdict in ("PASS", "INFO", "NOT RUN"), (
        "%s failed the scipilot layout audit with verdict %s" % (name, verdict)
    )
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


# ==========================================================================================
# The tables of record, and the derivations copied from build_figures.py lines 288 to 520
# and build_numbers.py lines 1560 to 1650.
# ==========================================================================================
RULES = [
    "labbe23",
    "kokorev24",
    "kocevski24",
    "perezgonzalez24",
    "barro23",
    "greene24",
    "akins24",
]

feat = pd.read_parquet(os.path.join(V, "features.parquet"))
oof = pd.read_parquet(os.path.join(V, "oof_scores.parquet")).set_index("source_id")

# The seven rules impose their thresholds on ordinary AB colors, while the model features
# c_* are differences of inverse-hyperbolic-sine magnitudes, which keep an undetected band
# finite. Every threshold line, shaded region and plotted point below is drawn on the AB
# color, computed from the catalog fluxes in labels.parquet exactly as build_numbers.py
# and build_figures.py do. A source with a non-positive flux in either band has no AB
# color in that pair and is not plotted in a panel that uses it.
_lab = pd.read_parquet(
    os.path.join(V, "labels.parquet"),
    columns=["source_id", "f_f115w", "f_f150w", "f_f200w", "f_f277w", "f_f444w"],
).set_index("source_id")


def _ab(blue, red):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -2.5 * np.log10(
            _lab["f_" + blue].to_numpy(float) / _lab["f_" + red].to_numpy(float)
        )


_lab["ab_f115w_f200w"] = _ab("f115w", "f200w")
_lab["ab_f150w_f200w"] = _ab("f150w", "f200w")
_lab["ab_f277w_f444w"] = _ab("f277w", "f444w")
AB_COLS = ["ab_f115w_f200w", "ab_f150w_f200w", "ab_f277w_f444w"]

sup = feat[feat.in_support.astype(bool)].set_index("source_id")
for c in AB_COLS:
    sup[c] = _lab.loc[sup.index, c].values
sup["sel_burden"] = oof.loc[sup.index, "sel_burden"].to_numpy(bool)

posAll = feat[feat.y == 1].set_index("source_id")
for c in AB_COLS:
    posAll[c] = _lab.loc[posAll.index, c].values
assert len(posAll) == N["nPos"], "the positive count moved from numbers.json"
POS_INSUP = posAll.in_support.astype(bool).to_numpy()
assert int(POS_INSUP.sum()) == N["nPosSupport"], "the in-support positive count moved"

# The union of the seven rules is the `picked` flag; the assertion proves it is exactly
# "at least one of the seven selects", which is how both figures read it.
POS_UNION = posAll.picked.astype(bool).to_numpy()
assert (
    posAll[["sel_" + r for r in RULES]].astype(bool).any(axis=1).to_numpy() == POS_UNION
).all(), "picked is not the disjunction of the seven rule flags"

# The learned selection at the equal-burden operating point, out of fold. The model scores
# only the seven-band support, so the nPosOutside positives outside it are not recovered by
# the learned selection, end to end (Section 2.3 of main.tex).
# Round 9: the primary operating point is the region-matched one (Section 5.1)
POS_MODEL = np.zeros(len(posAll), dtype=bool)
POS_MODEL[POS_INSUP] = POUT.loc[posAll.index[POS_INSUP], "sel_matched"].to_numpy(bool)

# The spectroscopic redshift of each positive: the redshift column of the published list the
# object is in, else the DAWN JWST Archive v4.4 secure redshift, else the catalog
# photometric redshift. build_numbers.py asserts that in the run of record no positive falls
# through to the photometric redshift, so the distribution is spectroscopic throughout.
_extra = pd.read_parquet(
    os.path.join(V, "published_catalogues_extra.parquet"),
    columns=["source_id", "z_spec", "z_spec_source"],
).set_index("source_id")
ZSPEC = _extra.loc[posAll.index, "z_spec"].to_numpy(float)
assert np.isfinite(ZSPEC).all(), "a positive has no spectroscopic redshift"
assert int((_extra.loc[posAll.index, "z_spec_source"] == "zphot").sum()) == 0, (
    "a positive falls through to a photometric redshift"
)

cand = pd.read_csv(os.path.join(V, "candidates.csv"))
_ec = _lab.reindex(cand.source_id.values)
for c in AB_COLS:
    cand[c] = _ec[c].values
EB = cand[cand.tier == "equal_burden"]
assert len(EB) == N["candEqualBurden"], "the equal-burden candidate count moved"

# ==========================================================================================
# Figure A: fig_zdist, the redshift distribution of the spectroscopic LRDs by outcome
# Rules 1, 2, 11, 12, 19, 22 of reviews/FIGURE_STYLE_LITERATURE.md.
# ==========================================================================================
Z_LO, Z_HI, Z_W = 2.0, 10.0, 0.5
ZDIST_EXTENDED = not (ZSPEC.min() >= Z_LO and ZSPEC.max() < Z_HI)
if ZDIST_EXTENDED:  # extend the range rather than drop a positive off the end
    Z_LO = min(Z_LO, np.floor(ZSPEC.min() / Z_W) * Z_W)
    Z_HI = max(Z_HI, np.ceil((ZSPEC.max() + 1e-9) / Z_W) * Z_W)
ZEDGES = np.arange(Z_LO, Z_HI + 0.5 * Z_W, Z_W)

# the four outcome classes at the equal-burden operating point, out of fold, end to end
ZCLASS = [
    ("both", POS_UNION & POS_MODEL, "matchedPairedBoth", C_UNION, None),
    ("union only", POS_UNION & ~POS_MODEL, "matchedPairedUnionOnlyEnd", C_UNION, "////"),
    ("model only", POS_MODEL & ~POS_UNION, "matchedPairedModelOnly", C_MODEL, None),
    ("neither", ~POS_UNION & ~POS_MODEL, "matchedPairedNeitherEnd", C_MISS, None),
]
ZLEGEND = [
    "rules and learned selection",
    "rules only",
    "learned selection only",
    "neither",
]

print("\nfig_zdist assertions")
_tot = 0
for name, mask, key, _c, _h in ZCLASS:
    got = int(mask.sum())
    assert got == N9[key], "%s is %d, round9 %s is %d" % (name, got, key, N9[key])
    print("  class %-11s %3d == round9 %-20s %3d" % (name, got, key, N9[key]))
    _tot += got
assert _tot == N["nPos"], "the four outcome classes do not partition the positives"
print("  sum %d == nPos %d" % (_tot, N["nPos"]))

ZCOARSE = [
    (-np.inf, 3.0, "posZspecBinBelowThree", "z < 3"),
    (3.0, 4.0, "posZspecBinThreeFour", "3 to 4"),
    (4.0, 5.0, "posZspecBinFourFive", "4 to 5"),
    (5.0, 7.0, "posZspecBinFiveSeven", "5 to 7"),
    (7.0, 9.0, "posZspecBinSevenNine", "7 to 9"),
    (9.0, np.inf, "posZspecBinAboveNine", "z > 9"),
]
for lo, hi, key, lbl in ZCOARSE:
    got = int(((ZSPEC >= lo) & (ZSPEC < hi)).sum())
    assert got == N[key], "coarse bin %s is %d, numbers.json %s is %d" % (
        lbl,
        got,
        key,
        N[key],
    )
    print("  coarse %-8s %3d == numbers.json %-24s %3d" % (lbl, got, key, N[key]))

fig, ax = plt.subplots(figsize=(COL1, 55 * MM))
fig.subplots_adjust(left=0.135, right=0.975, top=0.975, bottom=0.185)
centres = 0.5 * (ZEDGES[:-1] + ZEDGES[1:])
bottom = np.zeros(len(centres))
ZHIST = {}
for (name, mask, _k, color, hatch), lbl in zip(ZCLASS, ZLEGEND):
    h, _ = np.histogram(ZSPEC[mask], bins=ZEDGES)
    ZHIST[name] = h
    # The rules-only class is the same gray as the both class, separated by a white hatch.
    # Every segment is one artist with a dark 0.3 pt outline (style rule 11); the hatched
    # one gets its white hatch from white_hatch() rather than from a second stacked patch,
    # so its key swatch is a single box the size of the other three.
    bars = ax.bar(
        centres,
        h,
        width=Z_W,
        bottom=bottom,
        facecolor=color,
        edgecolor=K,
        linewidth=0.3,
        hatch=hatch,
        zorder=3,
        align="center",
    )
    if hatch:
        white_hatch(*bars.patches)
    bottom = bottom + h
assert int(bottom.sum()) == N["nPos"], "the stacked histogram lost a positive"
ZTOP = float(bottom.max())

# the lower bound two of the seven rules share, as a thin dashed gray rule
ax.axvline(4.0, color=GREY_TXT, lw=0.5, ls=(0, (3.0, 2.0)), zorder=2)
ax.text(
    4.0 + 0.09,
    ZTOP * 1.42,
    "z = 4",
    fontsize=BASE,
    color=GREY_TXT,
    ha="left",
    va="top",
)

ax.set_xlim(Z_LO, Z_HI)
ax.set_ylim(0, ZTOP * 1.46)
ax.set_xlabel("spectroscopic redshift")
ax.set_ylabel("spectroscopic LRDs")

# Key inside the panel, upper right, no frame, the four classes in the order they stack.
# Four swatches of one patch each, all the same size; the rules-only one carries the white
# hatch through white_hatch(), the same route the bars take.
_h_both = Patch(facecolor=C_UNION, edgecolor=K, linewidth=0.3)
_h_uonly = white_hatch(
    Patch(facecolor=C_UNION, edgecolor=K, linewidth=0.3, hatch="////")
)
_h_monly = Patch(facecolor=C_MODEL, edgecolor=K, linewidth=0.3)
_h_neither = Patch(facecolor=C_MISS, edgecolor=K, linewidth=0.3)
ax.legend(
    handles=[_h_both, _h_uonly, _h_monly, _h_neither],
    labels=ZLEGEND,
    loc="upper right",
    frameon=False,
    fontsize=BASE,
    handlelength=1.4,
    handletextpad=0.45,
    labelspacing=0.30,
    borderaxespad=0.5,
)

# --- the two figure-skill self-checks on this figure -------------------------------------
# the key must not cover a bar: the bars are patches rather than scatter markers, so the
# same box test runs against the drawn rectangles
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_barboxes = patch_boxes(
    [p for c in ax.containers for p in c.patches if p.get_height() > 0], _rend
)
_ztexts = in_panel_texts(ax)


def _touch(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


_zhits = []
for _t in _ztexts:
    _bb = _t.get_window_extent(_rend)
    _box = (_bb.x0, _bb.y0, _bb.x1, _bb.y1)
    _n = sum(1 for _m in _barboxes if _touch(_box, _m))
    if _n:
        _zhits.append("label %r covers %d bar segments" % (str(_t.get_text())[:40], _n))
report_collisions("fig_zdist", _zhits)
grayscale_check(
    "fig_zdist",
    [
        ("rules and learned selection", C_UNION, "plain", "plain fill"),
        ("rules only", C_UNION, "hatch////", "white //// hatch"),
        ("learned selection only", C_MODEL, "plain", "plain fill"),
        ("neither", C_MISS, "plain", "plain fill"),
    ],
)
save(fig, "fig_zdist", 84.0)

FACTS.append(
    "fig_zdist, built by paper/build_figures_pub.py\n"
    "  all %d spectroscopic LRDs, spectroscopic redshift, bins of %.1f from z = %.1f to %.1f\n"
    "  redshift range of the positives: %.2f to %.2f (median %.2f); "
    "the plotted range was %s\n"
    "  outcome classes at the equal-burden operating point, out of fold, end to end:\n"
    % (
        N["nPos"],
        Z_W,
        Z_LO,
        Z_HI,
        float(ZSPEC.min()),
        float(ZSPEC.max()),
        float(np.median(ZSPEC)),
        "extended past z = 2 to 10 to hold every positive"
        if ZDIST_EXTENDED
        else "not extended: every positive falls inside z = 2 to 10",
    )
    + "".join(
        "    %-41s %3d  (numbers.json %s)\n" % (lbl, int(mask.sum()), key)
        for (_nm, mask, key, _c, _h), lbl in zip(
            ZCLASS,
            (
                "recovered by rules and learned selection",
                "recovered by the rules only",
                "recovered by the learned selection only",
                "recovered by neither",
            ),
        )
    )
    + "    %-41s %3d  (numbers.json nPos)\n" % ("total", N["nPos"])
    + "  coarse redshift bins, all four classes together:\n"
    + "".join(
        "    %-8s %3d  (numbers.json %s)\n"
        % (lbl, int(((ZSPEC >= lo) & (ZSPEC < hi)).sum()), key)
        for lo, hi, key, lbl in ZCOARSE
    )
    + "  tallest stacked bin: %d spectroscopic LRDs\n" % int(ZTOP)
)

# ==========================================================================================
# Figure B: fig_colour_colour, the canonical color-color view of the seven rules
# Rules 1 to 5, 11, 12, 16, 20, 22, 23 of reviews/FIGURE_STYLE_LITERATURE.md.
#
# Thresholds are Table 1 of main.tex (lines 222 to 238), each checked against the constant
# in its module under src/ember/cuts/:
#   greene24          VSHAPE_BLUE_LO -0.5, VSHAPE_BLUE_HI 1.0, COLOR_F277W_F444W_MIN 1.0
#   kokorev24 red2    ("f150w","f200w","lt",0.8) and ("f277w","f444w","gt",0.7)
#   labbe23 red2      ("f150w","f200w","lt",0.8) and ("f277w","f444w","gt",1.0)
#   perezgonzalez24   COLOR_F150W_F200W_MAX 0.5, COLOR_F277W_F444W_MIN 1.0
#   barro23           COLOR_F277W_F444W_MIN 1.5
#   akins24           COLOR_F277W_F444W_MIN 1.5
# Table 1 and the modules agree on every one of these values.
# ==========================================================================================
CC_W, CC_H = 178.0, 80.0
CC_L, CC_PW, CC_MID = (
    13.0,
    71.0,
    8.0,
)  # left label strip, panel width, gap between panels
CC_CGAP, CC_CW, CC_R = 2.0, 3.0, 10.0  # color-bar gap, bar width, right tick strip
assert abs(CC_L + CC_PW + CC_MID + CC_PW + CC_CGAP + CC_CW + CC_R - CC_W) < 1e-9, (
    "color-color geometry does not add up to 178 mm"
)
CC_PH, CC_BOT = 64.0, 10.5  # panel height, and the strip below the panels

XA_LO, XA_HI = -3.0, 5.0  # F115W - F200W, panel (a)
XB_LO, XB_HI = -4.0, 3.5  # F150W - F200W, panel (b)
Y_LO, Y_HI = -1.5, 4.2  # F277W - F444W, shared

fig = plt.figure(figsize=(COL2, CC_H * MM))


def _ccx(mm_):
    return mm_ / CC_W


def _ccy(mm_):
    return mm_ / CC_H


axa = fig.add_axes([_ccx(CC_L), _ccy(CC_BOT), _ccx(CC_PW), _ccy(CC_PH)])
axb = fig.add_axes(
    [_ccx(CC_L + CC_PW + CC_MID), _ccy(CC_BOT), _ccx(CC_PW), _ccy(CC_PH)]
)
cbax = fig.add_axes(
    [
        _ccx(CC_L + CC_PW + CC_MID + CC_PW + CC_CGAP),
        _ccy(CC_BOT),
        _ccx(CC_CW),
        _ccy(CC_PH),
    ]
)

POS_SUP = posAll[POS_INSUP]
PS_UNION = POS_UNION[POS_INSUP]
YC = "ab_f277w_f444w"
PANELS = [
    ("a", axa, "ab_f115w_f200w", XA_LO, XA_HI, "F115W $-$ F200W (AB mag)"),
    ("b", axb, "ab_f150w_f200w", XB_LO, XB_HI, "F150W $-$ F200W (AB mag)"),
]

# ------------------------------------------------------------------ the catalog, rule 3
# All in-support catalog rows, as one log-density greyscale shared by both panels, so a
# shade means the same number of rows in each. Rows without both AB colors of a panel have
# no position in it and are dropped from that panel only.
HB = {}
CC_FACTS = {}
for letter, ax, xc, xlo, xhi, _xl in PANELS:
    m = np.isfinite(sup[xc].to_numpy()) & np.isfinite(sup[YC].to_numpy())
    inframe = (
        m
        & (sup[xc].to_numpy() >= xlo)
        & (sup[xc].to_numpy() <= xhi)
        & (sup[YC].to_numpy() >= Y_LO)
        & (sup[YC].to_numpy() <= Y_HI)
    )
    HB[letter] = ax.hexbin(
        sup[xc].to_numpy()[m],
        sup[YC].to_numpy()[m],
        gridsize=68,
        cmap=DENS_CMAP,
        mincnt=1,
        linewidths=0.1,
        extent=(xlo, xhi, Y_LO, Y_HI),
        zorder=1,
    )
    CC_FACTS[letter] = {
        "cat_both": int(m.sum()),
        "cat_inframe": int(inframe.sum()),
    }
_vmax = float(max(HB["a"].get_array().max(), HB["b"].get_array().max()))
_dnorm = LogNorm(vmin=1.0, vmax=_vmax)
HB["a"].set_norm(_dnorm)
HB["b"].set_norm(_dnorm)
cb = fig.colorbar(HB["b"], cax=cbax)
cb.set_label("catalog rows per bin", fontsize=BASE, labelpad=3)
cb.ax.tick_params(labelsize=BASE, width=0.6, length=2.4, direction="in")
cb.outline.set_linewidth(0.6)

# ------------------------------------------------------------------ panel (a): the rules
# The greene24 window, and the four F277W-F444W thresholds the other rules impose. The
# window is shaded and edged in greene24's color on its three real boundaries; the top is
# the panel edge, because the rule sets no upper limit.
G_XLO, G_XHI, G_Y = -0.5, 1.0, 1.0
axa.add_patch(
    Rectangle(
        (G_XLO, G_Y),
        G_XHI - G_XLO,
        Y_HI - G_Y,
        facecolor=RULE_COLOUR["greene24"],
        alpha=0.07,
        linewidth=0,
        zorder=2,
    )
)
for _seg in (
    ([G_XLO, G_XHI], [G_Y, G_Y]),
    ([G_XLO, G_XLO], [G_Y, Y_HI]),
    ([G_XHI, G_XHI], [G_Y, Y_HI]),
):
    axa.plot(
        _seg[0],
        _seg[1],
        color=RULE_COLOUR["greene24"],
        lw=0.7,
        ls=RULE_DASH["greene24"],
        zorder=6,
        solid_capstyle="butt",
    )
# horizontal inside the window near its top, on two lines so it fits between the window's
# two vertical edges rather than spilling past them
LBL_GREENE = axa.text(
    0.5 * (G_XLO + G_XHI) - 0.00,
    3.70,
    RULE_LABEL_SHORT["greene24"] + "\nwindow",
    ha="center",
    va="center",
    linespacing=1.15,
    fontsize=SMALL,
    color=RULE_TEXT["greene24"],
    zorder=8,
)

# the horizontal F277W-F444W thresholds, in Table 1's values; barro23 and akins24 share
# 1.5, so both lines are drawn at that exact value and one label names both
# ---- label machinery shared by both panels ----------------------------------------------
# A label is built now and positioned later, after the markers and the key are on the canvas,
# because a position can only be judged against what it would sit on. Each label carries an
# ordered list of candidate anchors; place_block takes the first that touches neither a
# plotted marker nor a label already placed.
LH_F = (
    SMALL / 72.0 * 25.4
) / CC_PH  # one 7.5 pt line as a fraction of the panel height
PENDING_A, PENDING_B = [], []


def _row(parts, sep=2.5):
    return HPacker(
        children=[
            TextArea(txt, textprops=dict(color=col, fontsize=SMALL))
            for txt, col in parts
        ],
        pad=0,
        sep=sep,
        align="baseline",
    )


def make_block(ax, rows, sep=2.0, align="left"):
    ab = AnnotationBbox(
        VPacker(children=rows, pad=0, sep=sep, align=align),
        (0.0, 0.0),
        xycoords="axes fraction",
        box_alignment=(0.0, 0.0),
        frameon=False,
        pad=0.0,
        zorder=8,
        annotation_clip=False,
    )
    ax.add_artist(ab)
    return ab


def _fya(y):  # data y to axes fraction, shared by both panels
    return (y - Y_LO) / (Y_HI - Y_LO)


def _fxa(x):  # data x to axes fraction on panel (a)
    return (x - XA_LO) / (XA_HI - XA_LO)


def _fxb(x):  # data x to axes fraction on panel (b)
    return (x - XB_LO) / (XB_HI - XB_LO)


def place_blocks(pending, ax, occupied):
    """Anchor each pending label at its first collision-free candidate.

    A block's size does not depend on where it is anchored, and AnnotationBbox offsets it
    from the anchor by box_alignment times its own width and height, so the box of every
    candidate follows from one measurement. That makes a dense ladder of anchors cheap:
    the figure is drawn once here, not once per candidate.
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    mboxes = marker_boxes(ax.collections, rend, pad_px=1.0)

    def touch(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    for name, ab, owner, cands in pending:
        bb = ab.get_window_extent(rend)
        w, h = bb.width, bb.height
        _o = ax.transAxes.transform((0.0, 0.0))
        _u = ax.transAxes.transform((1.0, 1.0)) - _o
        _xs = ax.get_xlim()
        _ys = ax.get_ylim()
        BLOCK_SIZES.append(
            (name, w / _u[0] * (_xs[1] - _xs[0]), h / _u[1] * (_ys[1] - _ys[0]))
        )
        best = None
        for i, (fx, fy, ba) in enumerate(cands):
            ax_, ay_ = ax.transAxes.transform((fx, fy))
            x0 = ax_ - ba[0] * w
            y0 = ay_ - ba[1] * h
            box = (x0, y0, x0 + w, y0 + h)
            n = sum(1 for m in mboxes if touch(box, m))
            n += sum(1 for p in occupied if touch(box, p))
            if best is None or n < best[0]:
                best = (n, i, fx, fy, ba, box)
            if n == 0:
                break
        n, i, fx, fy, ba, box = best
        ab.xy = (fx, fy)
        ab.xybox = (fx, fy)
        ab._box_alignment = ba
        occupied.append(box)
        if n == 0:
            print(
                "    %-46s candidate %2d of %2d, clean, block %.2f x %.2f data units"
                % (name, i + 1, len(cands), BLOCK_SIZES[-1][1], BLOCK_SIZES[-1][2])
            )
        else:
            print(
                "    %-46s NO CLEAN ANCHOR in %d candidates, best has %d touches"
                % (name, len(cands), n)
            )
    fig.canvas.draw()


THR = [
    (0.7, ["kokorev24"], "below"),
    (1.0, ["perezgonzalez24"], "above"),
    (1.5, ["barro23", "akins24"], "above"),
]
for thr, owners, side in THR:
    for r in owners:
        axa.axhline(
            thr,
            color=RULE_COLOUR[r],
            lw=0.7,
            ls=RULE_DASH[r],
            zorder=6,
        )
# The three thresholds are named in one key rather than each label riding the right end of
# its own line. Round 2 tried the latter, which is what style rule 5 asks for, and the
# collision test refused it: this plane carries data at every x between 0.4 and 1.9 mag, so
# a label long enough to name a rule and its value touches a plotted point wherever it is
# put on those lines, and moving it far enough away to be clean detaches it from the line it
# names. fig_colour_planes met the same problem in its round 4 and resolved it the same way,
# a key of thresholds ascending with no background box; this follows that decision. Each row
# is the value in bold black and the rules that impose it, each in its own color, which is
# also the color of its dashed line.
_key_rows = []
for _thr, _owners, _side in sorted(THR):
    _parts = [("$>$ %.1f" % _thr, K)]
    for _i, _r in enumerate(_owners):
        _parts.append(
            (
                RULE_LABEL_SHORT[_r] + ("," if _i < len(_owners) - 1 else ""),
                RULE_TEXT[_r],
            )
        )
    _key_rows.append(_row(_parts))
PENDING_A.append(
    (
        "panel (a) threshold key",
        make_block(axa, _key_rows, sep=2.0),
        axa,
        # free corners first, then a sweep along the bottom, which is the one band of this
        # panel that carries neither a marker nor another label
        [(0.012, 0.015, (0.0, 0.0)), (0.988, 0.015, (1.0, 0.0))]
        + [(0.012 + 0.05 * k, 0.015, (0.0, 0.0)) for k in range(16)]
        + [(0.012, 0.985, (0.0, 1.0)), (0.988, 0.985, (1.0, 1.0))],
    )
)

# kocevski24 sets no F277W-F444W threshold and therefore draws no line here. The in-panel
# sentence that used to say so was removed in round 2: the caption of this figure already
# carries it, and it collided with the key.

# ------------------------------------------------------------------ panel (b): the windows
# Three nested red2-style windows, each shaded at the same low alpha so the shade darkens
# with the number of rules that accept, each edged by its own dashed coloured boundary.
WIN = [
    ("labbe23", 0.8, 1.0, "Labbé+25 red2"),
    ("kokorev24", 0.8, 0.7, "Kokorev+24 red2"),
    ("perezgonzalez24", 0.5, 1.0, "Pérez-González+24"),
]
for r, xmax, ymin, lbl in WIN:
    axb.add_patch(
        Rectangle(
            (XB_LO, ymin),
            xmax - XB_LO,
            Y_HI - ymin,
            facecolor=RULE_COLOUR[r],
            alpha=0.07,
            linewidth=0,
            zorder=2,
        )
    )
    axb.plot(
        [XB_LO, xmax],
        [ymin, ymin],
        color=RULE_COLOUR[r],
        lw=0.7,
        ls=RULE_DASH[r],
        zorder=6,
        solid_capstyle="butt",
    )
    axb.plot(
        [xmax, xmax],
        [ymin, Y_HI],
        color=RULE_COLOUR[r],
        lw=0.7,
        ls=RULE_DASH[r],
        zorder=6,
        solid_capstyle="butt",
    )


# Each window is named in one key rather than each label riding the boundary it names.
# Round 2 put two labels at the top of the panel just inside the vertical edges and two
# more at the left end of the horizontal lines, and the review asked for the other shape:
# one stacked block in the upper-left, which is the emptiest part of this panel and lies
# inside all three windows, one row per window, entirely in that window's color, which is
# also the color of its dashed boundary. Each row gives the pair of bounds in the order of
# the panel's own axes, F150W - F200W first and F277W - F444W second; the caption states
# them in words. The values are read from WIN above, which carries the module constants, so
# no threshold is retyped here. place_blocks picks the first anchor that touches neither a
# plotted marker nor another label, and label_collisions() fails the build if none does.
_win_rows = []
for _r, _xmax, _ymin, _lbl in WIN:
    # Round 9c: one line per window. Two lines per window made the block tall enough to
    # touch a marker at every upper-left anchor, and it fell to the bottom of the panel.
    _win_rows.append(
        _row(
            [
                (_lbl + "  ", RULE_TEXT[_r]),
                ("$<%.1f$, $>%.1f$" % (_xmax, _ymin), RULE_TEXT[_r]),
            ]
        )
    )
PENDING_B.append(
    (
        "panel (b) window key",
        make_block(axb, _win_rows, sep=2.6),
        axb,
        # the upper-left corner first, then straight down the left margin, then a second
        # column a little to the right, then the far corners as a last resort
        [(0.012, 0.985 - 0.035 * k, (0.0, 1.0)) for k in range(12)]
        + [(0.10, 0.985 - 0.035 * k, (0.0, 1.0)) for k in range(12)]
        + [(0.012, 0.015, (0.0, 0.0)), (0.988, 0.985, (1.0, 1.0))],
    )
)

# ------------------------------------------------------------------ the plotted points
print("\nfig_colour_colour assertions")
for letter, ax, xc, xlo, xhi, xl in PANELS:
    px = POS_SUP[xc].to_numpy(float)
    py = POS_SUP[YC].to_numpy(float)
    ok = np.isfinite(px) & np.isfinite(py)
    hit = ok & PS_UNION
    mis = ok & ~PS_UNION
    ex = EB[xc].to_numpy(float)
    ey = EB[YC].to_numpy(float)
    eok = np.isfinite(ex) & np.isfinite(ey)
    ax.scatter(
        ex[eok],
        ey[eok],
        s=4,
        color=C_CAND,
        alpha=0.85,
        lw=0,
        zorder=3,
        label="equal-burden candidate",
    )
    ax.scatter(
        px[hit],
        py[hit],
        s=11,
        marker="o",
        facecolor=C_MODEL,
        edgecolor=K,
        lw=0.3,
        zorder=4,
        label="rule-selected LRD",
    )
    ax.scatter(
        px[mis],
        py[mis],
        s=17,
        marker="D",
        facecolor=C_MISS,
        edgecolor=K,
        lw=0.45,
        zorder=5,
        label="rule-missed LRD",
    )
    # every spectroscopic LRD that has both colors must be inside the frame
    assert (
        (px[ok] > xlo) & (px[ok] < xhi) & (py[ok] > Y_LO) & (py[ok] < Y_HI)
    ).all(), "panel (%s) clips a spectroscopic LRD" % letter
    assert int(hit.sum()) + int(mis.sum()) == int(ok.sum()), (
        "panel (%s): the two spectroscopic-LRD classes do not add to the plotted total"
        % letter
    )
    CC_FACTS[letter].update(
        plotted=int(ok.sum()),
        missing=int((~ok).sum()),
        hit=int(hit.sum()),
        mis=int(mis.sum()),
        eb=int(eok.sum()),
        eb_missing=int((~eok).sum()),
        xlo=xlo,
        xhi=xhi,
        xmin=float(px[ok].min()),
        xmax=float(px[ok].max()),
    )
    print(
        "  panel (%s) spectroscopic LRDs plotted %3d = %3d selected by a rule + %3d by none;"
        " %d of %d have no AB color in this pair"
        % (
            letter,
            int(ok.sum()),
            int(hit.sum()),
            int(mis.sum()),
            int((~ok).sum()),
            N["nPosSupport"],
        )
    )
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xlabel(xl)
    panel(ax, letter)

# The labels are anchored now, with the markers and (for panel (a)) the key already on the
# canvas, so a candidate can be judged against everything it might sit on.
axa.set_ylabel("F277W $-$ F444W (AB mag)")
axb.tick_params(labelleft=False)  # rule 16: tick labels on the outer panel only

# Key inside panel (a), upper right, no frame (rule 12). The two counts differ between the
# panels, because a source needs both colors of a panel to appear in it, so the key carries
# no count and the facts file carries all of them. "rule-selected" and "rule-missed" are the
# paper's own words for the two classes.
_hh, _ll = axa.get_legend_handles_labels()
axa.legend(
    _hh,
    _ll,
    loc="upper right",
    frameon=False,
    fontsize=BASE,
    scatterpoints=1,
    handletextpad=0.45,
    labelspacing=0.24,
    borderaxespad=0.5,
)
print("  label placement, first clean candidate wins")
_occupied_a, _occupied_b = [], []
_lg = axa.get_legend()
fig.canvas.draw()
_r0 = fig.canvas.get_renderer()
_lgb = _lg.get_window_extent(_r0)
_occupied_a.append((_lgb.x0, _lgb.y0, _lgb.x1, _lgb.y1))
_gb = LBL_GREENE.get_window_extent(_r0)
_occupied_a.append((_gb.x0, _gb.y0, _gb.x1, _gb.y1))
place_blocks(PENDING_A, axa, _occupied_a)
place_blocks(PENDING_B, axb, _occupied_b)

print(
    "  y range of the spectroscopic LRDs %.4f to %.4f, panel y limits %.1f to %.1f"
    % (
        float(POS_SUP[YC][np.isfinite(POS_SUP[YC])].min()),
        float(POS_SUP[YC][np.isfinite(POS_SUP[YC])].max()),
        Y_LO,
        Y_HI,
    )
)

# --- the two figure-skill self-checks on this figure -------------------------------------
_cchits = []
for _letter, _ax in (("a", axa), ("b", axb)):
    _cchits += [
        "panel (%s): %s" % (_letter, h)
        for h in label_collisions(fig, in_panel_texts(_ax), _ax.collections, pad_px=1.0)
    ]
report_collisions("fig_colour_colour", _cchits)
grayscale_check(
    "fig_colour_colour markers",
    [
        ("rule-selected LRD", C_MODEL, "o11", "filled circle, 11 pt2, dark edge"),
        ("rule-missed LRD", C_MISS, "D17", "filled diamond, 17 pt2, dark edge"),
        ("equal-burden candidate", C_CAND, "o4", "small unedged dot, 4 pt2"),
    ],
)
grayscale_check(
    "fig_colour_colour rule lines",
    [
        (
            RULE_LABEL_SHORT[r],
            RULE_COLOUR[r],
            str(RULE_DASH[r][1]),
            "dash pattern %s" % (RULE_DASH[r][1],),
        )
        for r in ("labbe23", "kokorev24", "perezgonzalez24", "barro23", "greene24")
    ],
)
save(fig, "fig_colour_colour", 178.0)

_ymin = float(POS_SUP[YC][np.isfinite(POS_SUP[YC])].min())
_ymax = float(POS_SUP[YC][np.isfinite(POS_SUP[YC])].max())
FACTS.append(
    "fig_colour_colour, built by paper/build_figures_pub.py\n"
    "  ordinary AB colors throughout; a source with a non-positive catalog flux in\n"
    "  either band of a pair has no AB color and is not plotted in the panel that uses it\n"
    "  axis limits: panel (a) x from %.1f to %.1f, panel (b) x from %.1f to %.1f,\n"
    "    both panels y from %.1f to %.1f; the spectroscopic LRDs span x %.2f to %.2f in (a),\n"
    "    x %.2f to %.2f in (b) and y %.2f to %.2f in both, so every one is inside the frame\n"
    "  panel (a), F115W - F200W against F277W - F444W:\n"
    "    spectroscopic LRDs plotted            %3d  of the %d in support\n"
    "    of which at least one rule selects %3d\n"
    "    of which no rule selects           %3d\n"
    "    missing an AB color, not plotted  %3d\n"
    "    equal-burden candidates plotted   %3d  of %d (%d have no AB color in this pair)\n"
    "    catalog support rows with both colors %6d, of which %6d inside the frame\n"
    "  panel (b), F150W - F200W against F277W - F444W:\n"
    "    spectroscopic LRDs plotted            %3d  of the %d in support\n"
    "    of which at least one rule selects %3d\n"
    "    of which no rule selects           %3d\n"
    "    missing an AB color, not plotted  %3d\n"
    "    equal-burden candidates plotted   %3d  of %d (%d have no AB color in this pair)\n"
    "    catalog support rows with both colors %6d, of which %6d inside the frame\n"
    "  thresholds drawn, all from Table 1 of main.tex and checked against src/ember/cuts:\n"
    "    panel (a) greene24 window   F115W - F200W in (-0.5, 1.0) and F277W - F444W > 1.0\n"
    "    panel (a) kokorev24 red2    F277W - F444W > 0.7\n"
    "    panel (a) perezgonzalez24   F277W - F444W > 1.0\n"
    "    panel (a) barro23           F277W - F444W > 1.5\n"
    "    panel (a) akins24           F277W - F444W > 1.5\n"
    "    panel (b) kokorev24 red2    F150W - F200W < 0.8 and F277W - F444W > 0.7\n"
    "    panel (b) labbe23 red2      F150W - F200W < 0.8 and F277W - F444W > 1.0\n"
    "    panel (b) perezgonzalez24   F150W - F200W < 0.5 and F277W - F444W > 1.0\n"
    "    kocevski24 sets no F277W - F444W threshold and draws no line\n"
    % (
        XA_LO,
        XA_HI,
        XB_LO,
        XB_HI,
        Y_LO,
        Y_HI,
        CC_FACTS["a"]["xmin"],
        CC_FACTS["a"]["xmax"],
        CC_FACTS["b"]["xmin"],
        CC_FACTS["b"]["xmax"],
        _ymin,
        _ymax,
        CC_FACTS["a"]["plotted"],
        N["nPosSupport"],
        CC_FACTS["a"]["hit"],
        CC_FACTS["a"]["mis"],
        CC_FACTS["a"]["missing"],
        CC_FACTS["a"]["eb"],
        N["candEqualBurden"],
        CC_FACTS["a"]["eb_missing"],
        CC_FACTS["a"]["cat_both"],
        CC_FACTS["a"]["cat_inframe"],
        CC_FACTS["b"]["plotted"],
        N["nPosSupport"],
        CC_FACTS["b"]["hit"],
        CC_FACTS["b"]["mis"],
        CC_FACTS["b"]["missing"],
        CC_FACTS["b"]["eb"],
        N["candEqualBurden"],
        CC_FACTS["b"]["eb_missing"],
        CC_FACTS["b"]["cat_both"],
        CC_FACTS["b"]["cat_inframe"],
    )
)

# ==========================================================================================
# Outputs: copies into paper/figures, the facts, the font report
# ==========================================================================================
COPIED = []
for nm in ("fig_zdist", "fig_colour_colour"):
    for ext in (".pdf", ".png"):
        src = os.path.join(FIG, nm + ext)
        dst = os.path.join(PUB, nm + ext)
        shutil.copy2(src, dst)
        COPIED.append(dst)
print("\ncopied %d files to paper/figures" % len(COPIED))

_block = (
    "\n"
    "==========================================================================\n"
    "Appended by paper/build_figures_pub.py (publication rewrite figures)\n"
    "==========================================================================\n"
    + "\n".join(FACTS)
)
for d in (FIG, PUB):
    with open(
        os.path.join(d, "caption_facts.txt"), "a", encoding="utf-8", newline="\n"
    ) as f:
        f.write(_block)
print("appended the facts to caption_facts.txt in figures/ and paper/figures/")

with open(
    os.path.join(FIG, "font_report.txt"), "a", encoding="utf-8", newline="\n"
) as f:
    f.write(
        "\nAppended by paper/build_figures_pub.py"
        " (publication rewrite figures)\n"
    )
    f.write("figure                       width_mm  height_mm  min_font_pt\n")
    for nm, w, h, mf in FONT_REPORT:
        f.write("%-28s %8.1f %10.1f %12.1f\n" % (nm, w, h, mf))
    f.write(
        "Both canvases are emitted at the exact width main.tex includes them at, so the"
        " point sizes above are the point sizes on the page. The 7.5 pt entries are the"
        " threshold and region labels inside the color-color panels, which the font floor"
        " allows at that size.\n"
    )
    f.write("\nfigure-skill self-checks\n")
    for nm, verdict in QA_REPORT:
        f.write("  %-44s %s\n" % (nm, verdict))
    for blk in GREY_REPORT:
        f.write("  " + blk.replace("\n", "\n  ") + "\n")
print("appended the font floor to font_report.txt")

print("\nfigure-skill self-checks")
for nm, verdict in QA_REPORT:
    print("  %-40s %s" % (nm, verdict))
print("done")
