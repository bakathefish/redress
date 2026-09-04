"""The three explanatory figures of the recovery paper: what the object class looks like,
why a real one fails every published rule, and where on the sky the data are.

Nothing here is hand picked. Each object is chosen by a fixed rule stated in the caption and
printed by this script when it runs; each number drawn or printed is read from the tables of
record in recovery and from the archival spectra the spectral test itself uses.

Run from the repository root: python paper/build_figures_extra.py
Writes paper/figures/fig_lrd_anatomy, fig_miss_anatomy, fig_fields (.pdf and .png),
paper/figures/stamps/anatomy_*.fits, paper/figures/spectra/*.spec.fits,
and paper/figures/manifest_extra.csv.

House style is build_figures.py's, with the round-2 font floor: no lettering below 8 pt at
print size, two-column (178 mm) width for every multi-panel figure, vector PDF plus a 300 dpi
PNG. The stamp calibration is the frozen R1b calibration build_figures.py's gallery uses; the
stamp-rendering function is copied from it unchanged so both figures render identically.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from astropy.io import fits as afits  # noqa: E402
from astropy.visualization import make_lupton_rgb  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402
from sedpy import observate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "recovery")
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
STAMPS = os.path.join(FIG, "stamps")
SPECD = os.path.join(FIG, "spectra")
for _d in (FIG, STAMPS, SPECD):
    os.makedirs(_d, exist_ok=True)

# the seven re-implemented selections, as published modules (staged public copy)
REDRESS = os.path.join(os.path.dirname(ROOT), "redress", "selections")
sys.path.insert(0, REDRESS)
from redress.cuts import _shared as sh  # noqa: E402
from redress.cuts import (  # noqa: E402
    akins24,
    barro23,
    greene24,
    kocevski24,
    kokorev24,
    labbe23,
    perezgonzalez24,
)

CUTS = {
    "labbe23": labbe23.select,
    "kokorev24": kokorev24.select,
    "kocevski24": kocevski24.select,
    "perezgonzalez24": perezgonzalez24.select,
    "barro23": barro23.select,
    "greene24": greene24.select,
    "akins24": akins24.select,
}
# Table 1 order and display names of the paper
RULE_ORDER = [
    "labbe23",
    "kokorev24",
    "kocevski24",
    "perezgonzalez24",
    "barro23",
    "greene24",
    "akins24",
]
RULE_LABEL = {
    "labbe23": "Labbé+23",
    "kokorev24": "Kokorev+24",
    "kocevski24": "Kocevski+24",
    "perezgonzalez24": "Pérez-González+24",
    "barro23": "Barro+23",
    "greene24": "Greene+24",
    "akins24": "Akins+24",
}

# ------------------------------------------------------------------ house style
# Copied from build_figures.py. The one change is the round-2 font floor: every size that
# was below 8 pt is raised to 8 pt, because MNRAS body text is 9 pt and figure lettering
# below 8 pt is unreadable at print size.
plt.rcParams.update(
    {
        # serif body matching the LaTeX paper, without a LaTeX dependency at runtime
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "text.usetex": False,
        # 8 pt floor: nothing on any figure here falls below 8 pt at print size
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
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
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.7,
        "ytick.minor.size": 1.7,
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
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.32,
        "legend.borderpad": 0.2,
        "legend.borderaxespad": 0.4,
        "legend.columnspacing": 1.0,
        # vector PDF with real glyphs, 300 dpi raster companion
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# base is black and grey; exactly two accents, used the same way on every figure
K = "#000000"
C_UNION = "#666666"
C_MODEL = "#1a5a8a"  # accent 1, the learned selection
C_MISS = "#c0532a"  # accent 2, missed by every rule
C_CAND = "#e0a06a"  # tint of accent 2, unlisted candidates
C_NEG = "#a0a0a0"  # spectroscopic non-LRD anchors
C_CAT = "#dcdcdc"  # the catalogue as a whole
GREY_TXT = "#7f7f7f"

# ------------------------------------------------------------------ the seven rules
# The same categorical palette build_figures.py defines, copied here for the same reason the
# style block is copied: importing build_figures would run every figure in it. Paul Tol's
# "muted" qualitative scheme, safe under deuteranopia, protanopia and tritanopia, with a
# darkened twin for lettering so a rule name reads at 8 pt on white.
RULE_COLOUR = {
    "labbe23": "#332288",  # indigo
    "kokorev24": "#88CCEE",  # cyan
    "kocevski24": "#44AA99",  # teal
    "perezgonzalez24": "#117733",  # green
    "barro23": "#999933",  # olive
    "greene24": "#CC6677",  # rose
    "akins24": "#AA4499",  # purple
}


def _dark(hexcol, target=0.16):
    """The same hue at a relative luminance of at most `target`, i.e. 4.5:1 against white."""
    rgb = np.array([int(hexcol[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])

    def lum(c):
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return float(np.dot([0.2126, 0.7152, 0.0722], lin))

    if lum(rgb) <= target:
        return hexcol
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if lum(rgb * mid) > target:
            hi = mid
        else:
            lo = mid
    return "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in rgb * lo)


RULE_TEXT = {r: _dark(c) for r, c in RULE_COLOUR.items()}

# One tint per sky region for fig_fields, from Paul Tol's "light" qualitative scheme, which
# is the set he designs for filled areas rather than for lines. None of the five is the
# orange of the candidate points or the blue of the LRD points, so the markers stay legible
# on top of the density. The outline of each field is the same tint darkened.
REGION_TINT = {
    "CEERS": "#77AADD",  # light blue
    "GOODS-N": "#44BB99",  # mint
    "GOODS-S": "#BBCC33",  # olive
    "COSMOS": "#FFAABB",  # pink
    "UDS": "#EEDD88",  # light yellow
}

MM = 1.0 / 25.4
COL1 = 84.0 * MM  # single column, as main.tex includes it
COL2 = 178.0 * MM  # two columns, as main.tex includes it
FS = 8.0  # the font floor; every explicit fontsize in this file is exactly this
FS_TAG = 7.5  # the one exception the brief allows: labels drawn inside a stamp


FONT_REPORT = []


def save(fig, name, width_mm=178.0):
    """Write the PDF and PNG at exactly the canvas size, and record the font floor.

    Round 3 stopped cropping to a tight bounding box. main.tex includes each of these three
    at \\textwidth, so LaTeX scales whatever it is given to 178 mm; a tight box gave an
    arbitrary width and therefore an arbitrary scale factor. fig_fields had drifted to
    197.9 mm, which LaTeX was shrinking by ten per cent, taking its 8 pt lettering to 7.2 pt
    on the page. The canvas is now the final size and the scale factor is 1.
    """
    w_in, h_in = fig.get_size_inches()
    got = w_in / MM
    assert abs(got - width_mm) < 0.05, "%s is %.2f mm wide, wanted %.2f" % (
        name,
        got,
        width_mm,
    )
    sizes = [
        t.get_fontsize()
        for t in fig.findobj(Text)
        if t.get_visible() and str(t.get_text()).strip()
    ]
    mf = min(sizes) if sizes else float("nan")
    # An uncropped canvas will silently cut any lettering that runs past its edge, which is
    # exactly what a tight bounding box used to hide by growing the figure. Every visible
    # string is measured against the canvas and any overflow is named, in millimetres.
    fig.canvas.draw()
    px_w, px_h = fig.canvas.get_width_height()
    over = []
    for t in fig.findobj(Text):
        s = str(t.get_text()).strip()
        if not (t.get_visible() and s):
            continue
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        # a string wholly outside the canvas is an out-of-range tick label that matplotlib
        # keeps as an artist and never draws; only one that straddles an edge is being cut
        if bb.x1 <= 0 or bb.x0 >= px_w or bb.y1 <= 0 or bb.y0 >= px_h:
            continue
        d = max(-bb.x0, bb.x1 - px_w, -bb.y0, bb.y1 - px_h)
        if d > 0.5:  # half a pixel of rounding is not an overflow
            over.append((d / fig.dpi * 25.4, s[:44]))
    if over:
        over.sort(reverse=True)
        print("  %s: %d strings run off the canvas" % (name, len(over)))
        for d_mm, s in over[:6]:
            print("     %5.2f mm  %r" % (d_mm, s))
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches=None)
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches=None)
    plt.close(fig)
    FONT_REPORT.append((name, got, h_in / MM, mf))
    print(
        "wrote %-20s %6.1f x %6.1f mm  smallest font %.1f pt"
        % (name, got, h_in / MM, mf)
    )


def panel(ax, letter, pad=3.0):
    """Panel letter above the axes, a fixed distance up in points so that panels of
    different heights (an equal-aspect stamp beside a plot) carry it at the same offset.
    Descriptions belong in the caption, not in the panel."""
    ax.annotate(
        f"({letter})",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, pad),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=FS,
        fontweight="bold",
        color=K,
    )


# ------------------------------------------------------------------ stamps and spectra
CUTOUT = (
    "https://grizli-cutout.herokuapp.com/thumb?ra={ra:.6f}&dec={dec:.6f}"
    "&size=5&filters=f150w-clear,f277w-clear,f444w-clear&output=fits"
)
# the six-band request round 3 adds: same service, same radius, same output, longer filter
# list. fetch_stamps_6band.py writes these for every object in the two manifests; the call
# here only covers the case of running this script on a clean checkout.
BANDS6 = [
    "F115W-CLEAR",
    "F150W-CLEAR",
    "F200W-CLEAR",
    "F277W-CLEAR",
    "F356W-CLEAR",
    "F444W-CLEAR",
]
BAND_SHORT = {b: b.split("-")[0] for b in BANDS6}
CUTOUT6 = (
    "https://grizli-cutout.herokuapp.com/thumb?ra={ra:.6f}&dec={dec:.6f}"
    "&size=5&filters=" + ",".join(b.lower() for b in BANDS6) + "&output=fits"
)
SPEC_BASE = "https://s3.amazonaws.com/msaexp-nirspec/extractions"


def valid(p, minsize=50000):
    try:
        with open(p, "rb") as fh:
            return fh.read(6) == b"SIMPLE" and os.path.getsize(p) > minsize
    except OSError:
        return False


def retrieve(url, dest, minsize=50000):
    """Four attempts, exactly as fetch_stamps.py does; returns whether the file is usable."""
    if valid(dest, minsize):
        return True
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url, dest)
            if valid(dest, minsize):
                return True
        except Exception as e:  # noqa: BLE001
            print("retry", os.path.basename(dest), type(e).__name__, e, flush=True)
        time.sleep(3 * (attempt + 1))
    print("FAILED", url)
    return False


# frozen R1b stamp calibration, the same one build_figures.py's gallery uses
calib = json.load(
    open(os.path.join(ROOT, "inputs", "r1b_calibration.json"))
)
BANDS_RGB = ["F150W-CLEAR", "F277W-CLEAR", "F444W-CLEAR"]
RGB_SCL = calib["rgb"]["rgb_scl"]
STRETCH = float(calib["rgb"]["stretch"])
Q = float(calib["rgb"]["Q"])
GRAY_TOP = float(calib["gray_f444w"]["T"])
ASINH_A = float(calib["gray_f444w"]["a"])


def load_bands(path):
    """Copied unchanged from build_figures.py."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with afits.open(path, memmap=False) as h:
            ims = {}
            for hdu in h:
                name = (hdu.header.get("FILTER") or hdu.name or "").upper()
                for b in BANDS_RGB:
                    if b in name:
                        ims[b] = np.nan_to_num(np.asarray(hdu.data, float), nan=0.0)
    assert len(ims) == 3, path
    return {b: im - np.median(im) for b, im in ims.items()}


def render(ims):
    """Copied unchanged from build_figures.py."""
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


# The cutout service returns 200 x 200 pixels at 0.05 arcsec per pixel, a 10 arcsec field.
# The gallery in build_figures.py crops pixels 80:120; the same crop is used here so both
# figures show the same angular field, which is 40 x 0.05 = 2.0 arcsec. Round 3 narrowed it
# from 80 pixels: seven panels in a strip leave each about 11 mm on a two-column page, and
# at 4.0 arcsec the object filled too little of that. Same objects, same calibration.
PIXSCALE = 0.05
CROP = slice(80, 120)
CROP_ARCSEC = (CROP.stop - CROP.start) * PIXSCALE
BAR_ARCSEC = 0.5
BAR_PIX = BAR_ARCSEC / PIXSCALE


def show_stamp(ax, im, rgb=False, edge=K, bar=True):
    if rgb:
        ax.imshow(im[CROP, CROP], origin="lower", interpolation="nearest")
    else:
        ax.imshow(
            im[CROP, CROP],
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_anchor(
        "N"
    )  # square panels hang from the top of their cell, aligned with the plots
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(edge)
        s.set_linewidth(0.8)
    if bar:
        scale_bar(ax)


def scale_bar(ax, fontsize=FS):
    """The one 0.5 arcsec bar an object carries, white, lower left of the composite."""
    n = CROP.stop - CROP.start
    x0 = 0.15 * n  # inset far enough that the centred label clears the left border
    ax.plot(
        [x0, x0 + BAR_PIX],
        [0.085 * n, 0.085 * n],
        color="white",
        lw=1.1,
        solid_capstyle="butt",
        zorder=6,
    )
    ax.text(
        x0 + BAR_PIX / 2,
        0.115 * n,
        f'{BAR_ARCSEC:g}"',
        color="white",
        fontsize=fontsize,
        ha="center",
        va="bottom",
        zorder=6,
    )


def load_bands6(path):
    """The six-band cutout, median-subtracted band by band; a band with no coverage is None."""
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
                        d = np.nan_to_num(np.asarray(hdu.data, float), nan=0.0)
                        ims[b] = d - np.median(d)
    return ims


def gray_panel(im):
    """The one asinh rule every single-band panel uses, in every band and every object.

    It is the frozen R1b grey rule (inputs/r1b_calibration.json, key
    gray_f444w) that the F444W panel has always used, applied unchanged to the other five
    bands: divide by the same normalisation T, clip below at zero, then
    arcsinh(x/a) / arcsinh(1/a) with the same softening a. T and a never change from panel
    to panel, so a band in which the object is faint prints faint. No new calibration
    constant is introduced.
    """
    x = np.clip(im / GRAY_TOP, 0, None)
    return np.clip(np.arcsinh(x / ASINH_A) / np.arcsinh(1.0 / ASINH_A), 0, 1)


def draw_strip(fig, spec, ims6, rgb, edge, letter=None):
    """Six single-band panels and the composite, in the footprint two stamps used to fill.

    The arrangement is a 2 x 4 block: F115W to F277W across the top, F356W, F444W and the
    composite across the bottom. Akins et al. (2025) Fig. 4 runs the same sequence in one
    line above a wide SED; two lines are what fits beside a panel of plots at 178 mm.
    """
    sub = spec.subgridspec(2, 4, wspace=0.05, hspace=0.05)
    order = list(BANDS6) + ["RGB"]
    first = None
    for k, b in enumerate(order):
        ax = fig.add_subplot(sub[divmod(k, 4)])
        if first is None:
            first = ax
        if b == "RGB":
            ax.imshow(rgb[CROP, CROP], origin="lower", interpolation="nearest")
            scale_bar(ax, fontsize=FS_TAG)
            tag = "RGB"
        elif ims6[b] is None:  # no coverage in this band: say so, do not fake it
            ax.set_facecolor("#f2f2f2")
            ax.text(
                0.5,
                0.40,
                "no\ncoverage",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=FS_TAG,
                color="#8a8a8a",
                linespacing=1.15,
            )
            tag = BAND_SHORT[b]
        else:
            ax.imshow(
                gray_panel(ims6[b])[CROP, CROP],
                origin="lower",
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            tag = BAND_SHORT[b]
        ax.text(
            0.055,
            0.945,
            tag,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FS_TAG,
            color="white",
            zorder=6,
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.55, pad=0.9),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(edge)
            s.set_linewidth(0.6)
    if letter:
        panel(first, letter)
    return first


# ------------------------------------------------------------------ tables of record
BANDS = ["f090w", "f115w", "f150w", "f200w", "f277w", "f356w", "f444w"]
MAN = json.load(open(os.path.join(V, "feature_manifest.json")))
SOFT = np.array([MAN["asinh_softening_uJy_per_band"][b] for b in BANDS])
LAM_UM = {  # NIRCam pivot wavelengths, microns; the same table v4_features.py uses
    "f090w": 0.901,
    "f115w": 1.154,
    "f150w": 1.501,
    "f200w": 1.990,
    "f277w": 2.786,
    "f356w": 3.563,
    "f444w": 4.421,
}


def spectrum_band_ratios(w_um, f_uJy, phot_uJy, bands):
    """Photometry over synthetic photometry, band by band, for a spectrum in microjansky.

    The synthetic flux is the photon-counting average of f_nu through the NIRCam total
    throughput curve that sedpy ships (Johnson, sedpy 0.4.1, data/filters/jwst_*.par),
    the same definition the AB system uses:

        f_nu_syn = int f_nu(l) T(l) dl / l  /  int T(l) dl / l.

    A band whose tabulated curve is not wholly inside the spectrum's wavelength range is
    dropped and named, so no ratio rests on an extrapolation. This serves the display
    scaling of the anatomy figure's spectrum panel only; nothing analysed uses it.
    """
    out = {}
    for b, filt in zip(bands, observate.load_filters(["jwst_" + b for b in bands])):
        lam = filt.wavelength / 1e4  # sedpy tabulates in angstrom
        thr = filt.transmission
        if lam.min() < w_um.min() or lam.max() > w_um.max():
            print("    %s: outside the spectrum's range, dropped" % b.upper())
            continue
        fs = np.interp(lam, w_um, f_uJy)
        syn = np.trapezoid(fs * thr / lam, lam) / np.trapezoid(thr / lam, lam)
        out[b] = float(phot_uJy[b]) / float(syn)
    return out


PIVOT_A = {  # NIRCam pivot wavelengths, angstrom; the table the rule modules are given
    "f090w": 9022.92,
    "f115w": 11543.01,
    "f150w": 15007.45,
    "f200w": 19886.48,
    "f277w": 27623.47,
    "f356w": 35682.28,
    "f410m": 40820.73,
    "f444w": 44037.14,
    "f435w": 4318.83,
    "f606w": 5920.82,
    "f814w": 8056.88,
    "f105w": 10543.52,
    "f125w": 12470.52,
    "f140w": 13924.16,
    "f160w": 15396.62,
}
KMAG = 2.5 / np.log(10.0)
AB_ZP = 23.9  # microjansky zeropoint; asinh magnitude + AB_ZP is the AB magnitude when detected
UV_COLS = [0, 1, 2, 3]  # slope_blue: F090W, F115W, F150W, F200W
OPT_COLS = [4, 5, 6]  # slope_red: F277W, F356W, F444W
# an 8 pt slope label is about 0.28 decades wide in these panels, so a band within this
# factor in wavelength of the label's anchor is a band the label could sit on
LABEL_HALFWIDTH = 10**0.14

lab = pd.read_parquet(
    os.path.join(V, "labels.parquet"),
    columns=(
        ["source_id", "field", "id", "ra", "dec", "y", "bands_complete", "z_phot"]
        + ["r_h_arcsec", "r_star_v1_arcsec", "mag_f444w", "snr_f444w"]
        + [f"sel_{k}" for k in RULE_ORDER]
        + [f"f_{b}" for b in BANDS]
        + [f"e_{b}" for b in BANDS]
    ),
)
feat = pd.read_parquet(
    os.path.join(V, "features.parquet"),
    columns=(
        ["source_id", "field", "id", "region", "ra", "dec", "y", "in_support"]
        + ["c_f277w_f444w", "slope_blue", "slope_red", "z_phot", "mag_f444w"]
        + [f"m_{b}" for b in BANDS]
        + [f"sel_{k}" for k in RULE_ORDER]
    ),
)
oof = pd.read_parquet(
    os.path.join(V, "oof_scores.parquet"),
    columns=["source_id", "field", "id", "region", "score_mean", "t_burden", "y"],
)
status = pd.read_csv(os.path.join(V, "anchor_test_status.csv"))
cand = pd.read_csv(
    os.path.join(V, "candidates.csv"),
    usecols=[
        "rank",
        "tier",
        "field",
        "id",
        "ra",
        "dec",
        "ranking_score",
        "followup_tier",
    ],
)
comp = pd.read_csv(
    os.path.join(V, "compactness.csv"),
    usecols=["source_id", "compactness_f444w", "labbe_compactness", "measurable"],
)

sup = feat[feat.in_support.astype(bool)].copy()
sup = sup.merge(
    oof[["source_id", "score_mean", "t_burden"]], on="source_id", how="left"
)
sup["rank_all"] = sup.score_mean.rank(ascending=False, method="min").astype(int)
N_SUPPORT = len(sup)
print(f"support rows {N_SUPPORT:,}")


def ab_colour(row, blue, red):
    """Ordinary AB colour from the catalogue fluxes, the convention every threshold uses."""
    fb, fr = float(row[f"f_{blue}"]), float(row[f"f_{red}"])
    if not (np.isfinite(fb) and np.isfinite(fr) and fb > 0 and fr > 0):
        return np.nan
    return -2.5 * np.log10(fb) + 2.5 * np.log10(fr)


def asinh_sed(row):
    """The seven-band spectral energy distribution as the features define it.

    Returns the asinh magnitude placed on the AB scale (it equals the AB magnitude wherever
    the band is detected), its uncertainty, and the two weighted slopes refitted here. The
    refit is asserted against the stored slope_blue and slope_red features.
    """
    f = np.array([float(row[f"f_{b}"]) for b in BANDS])
    e = np.array([float(row[f"e_{b}"]) for b in BANDS])
    with np.errstate(all="ignore"):
        m = -KMAG * (np.arcsinh(f / (2.0 * SOFT)) + np.log(SOFT))
        sig = KMAG * e / np.sqrt(f * f + (2.0 * SOFT) ** 2)
        w = 1.0 / np.clip(sig, 0.01, None) ** 2

    def fit(cols):
        x = np.log10(np.array([LAM_UM[BANDS[c]] for c in cols]))
        y, ww = m[cols], w[cols]
        ok = np.isfinite(y) & np.isfinite(ww)
        ww = np.where(ok, ww, 0.0)
        y = np.where(ok, y, 0.0)
        sw = ww.sum()
        xm = (ww * x).sum() / sw
        ym = (ww * y).sum() / sw
        num = (ww * (x - xm) * (y - ym)).sum()
        den = (ww * (x - xm) ** 2).sum()
        s = num / den if (ok.sum() >= 2 and den > 0) else 0.0
        return float(s), float(ym - s * xm)  # slope, intercept in the asinh magnitude

    s_uv, b_uv = fit(UV_COLS)
    s_opt, b_opt = fit(OPT_COLS)
    return m + AB_ZP, sig, (s_uv, b_uv + AB_ZP), (s_opt, b_opt + AB_ZP)


def draw_sed(ax, row, letter=None):
    """Panel (c): the seven-band SED with the two slopes the features measure."""
    m, sig, (s_uv, b_uv), (s_opt, b_opt) = asinh_sed(row)
    lam = np.array([LAM_UM[b] for b in BANDS])
    err = np.clip(sig, 0, 1.5)
    # Round 3, style rule 9: the two windows the V shape is measured in are shaded behind
    # the points, the way Barro et al. (2026) Fig. 3 shades the ranges his optical colour is
    # computed over. The bands are the ones the features already use; nothing is refitted.
    for cols in (UV_COLS, OPT_COLS):
        ax.axvspan(
            LAM_UM[BANDS[cols[0]]],
            LAM_UM[BANDS[cols[-1]]],
            facecolor="#9a9a9a",
            alpha=0.13,
            lw=0,
            zorder=0,
        )
    ax.errorbar(
        lam,
        m,
        yerr=err,
        fmt="o",
        ms=3.0,
        mfc=C_MODEL,
        mec=C_MODEL,
        ecolor=C_NEG,
        elinewidth=0.7,
        capsize=1.4,
        lw=0,
        zorder=3,
    )
    anchors = []
    for cols, xcols, s, b, col, name in (
        (UV_COLS, UV_COLS[-2:], s_uv, b_uv, C_UNION, "UV slope"),
        (OPT_COLS, OPT_COLS[:1] + OPT_COLS[-1:], s_opt, b_opt, C_MISS, "optical slope"),
    ):
        xs = np.array([LAM_UM[BANDS[c]] for c in (cols[0], cols[-1])])
        ax.plot(xs, s * np.log10(xs) + b, color=col, lw=1.2, zorder=2)
        # One placement rule in every panel: the ultraviolet label sits below its line's
        # midpoint and the optical label above its line's midpoint, each on an opaque white
        # patch. The anchor is pushed past the furthest point and error bar of the bands
        # that slope is fitted to, so no label can land on a marker in any panel; on this
        # inverted magnitude axis "below" means the larger magnitude.
        # the label sits on the stretch of its own line spanned by xcols, and clears only
        # the points that actually fall under it (within a factor LABEL_HALFWIDTH in
        # wavelength), capped so it never detaches from the line it names
        xm = float(np.sqrt(LAM_UM[BANDS[xcols[0]]] * LAM_UM[BANDS[xcols[-1]]]))
        below = name == "UV slope"
        ymid = s * np.log10(xm) + b
        span = float(np.nanmax(m) - np.nanmin(m))
        cap = 0.20 * span
        under = [
            c
            for c in cols
            if xm / LABEL_HALFWIDTH <= LAM_UM[BANDS[c]] <= xm * LABEL_HALFWIDTH
        ]
        if below:
            reach = max([m[c] + err[c] for c in under], default=ymid)
            y = min(max(ymid, float(reach)), ymid + cap)
        else:
            reach = min([m[c] - err[c] for c in under], default=ymid)
            y = max(min(ymid, float(reach)), ymid - cap)
        anchors.append(y)
        ax.annotate(
            name,
            xy=(xm, y),
            xytext=(0, -11 if below else 10),
            textcoords="offset points",
            ha="center",
            va="top" if below else "bottom",
            fontsize=FS,
            color=col,
            zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.2),
        )
    ax.set_xscale("log")
    ax.set_xlim(0.75, 5.4)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1", "2", "3", "4"])
    ax.xaxis.set_minor_locator(NullLocator())
    lo = min(float(np.nanmin(m)), anchors[1])
    hi = max(float(np.nanmax(m)), anchors[0])
    # the limits leave room for the two label patches, which hang past the anchors above.
    # Round 3 raised the upper headroom from 0.44 to 0.54 of the span: in the two shallower
    # panels of fig_miss_anatomy the optical-slope label was reaching the top spine.
    ax.set_ylim(hi + 0.32 * (hi - lo) + 0.3, lo - 0.54 * (hi - lo) - 0.4)
    ax.set_xlabel(r"observed wavelength ($\mu$m)")
    ax.set_ylabel("AB magnitude")
    if letter:
        panel(ax, letter)
    return s_uv, s_opt


# ------------------------------------------------------------------ the rule ladder
def contract(df):
    """The harmonized-photometry table the seven rule modules require."""
    p = pd.DataFrame(index=range(len(df)))
    for b in BANDS:
        f = df[f"f_{b}"].to_numpy(float)
        e = df[f"e_{b}"].to_numpy(float)
        p[f"f_{b}_ujy"] = f
        p[f"e_{b}_ujy"] = e
        p[f"cov_{b}"] = np.isfinite(f) & np.isfinite(e)
    p["compactness_f444w"] = df["compactness_f444w"].to_numpy(float)
    p["flux_radius_f444w_arcsec"] = df["r_h_arcsec"].to_numpy(float)
    return p


def run_rules(df):
    """Every rule's per-criterion masks for the rows of df, from the modules themselves."""
    p = contract(df)
    ratio = df["labbe_compactness"].to_numpy(float)
    z = df["z_phot"].to_numpy(float)
    rstar = df["r_star_v1_arcsec"].to_numpy(float)
    out = {}
    for k in RULE_ORDER:
        fn = CUTS[k]
        if k in ("labbe23", "kokorev24"):
            out[k] = fn(p, ratio)
        elif k == "kocevski24":
            out[k] = fn(p, z_phot=z, r_h_stars_arcsec=rstar, pivots_angstrom=PIVOT_A)
        else:
            out[k] = fn(p)
    return out


def _thr_from_branch(branch, blue, red):
    for b, r, _op, thr in branch:
        if b == blue and r == red:
            return float(thr)
    return None


# each rule's own threshold on F277W-F444W, read out of the module, never retyped
THR_277_444 = {
    "labbe23": _thr_from_branch(labbe23.RED2, "f277w", "f444w"),
    "kokorev24": _thr_from_branch(kokorev24.RED2, "f277w", "f444w"),
    "kocevski24": None,  # a continuum-slope selection: it sets no F277W-F444W threshold
    "perezgonzalez24": float(perezgonzalez24.COLOR_F277W_F444W_MIN),
    "barro23": float(barro23.COLOR_F277W_F444W_MIN),
    "greene24": float(greene24.COLOR_F277W_F444W_MIN),
    "akins24": float(akins24.COLOR_F277W_F444W_MIN),
}


def colour_terms(row, phot=None, i=0):
    """Every colour term of every rule, evaluated on this row from the modules' constants.

    Ordinary AB colours are used, which is the convention the published thresholds are
    written in, with one exception: Kokorev+24 defines its colours through a per-colour
    detection rule with a 2 sigma upper-limit substitution, so its terms are evaluated
    with that module's own colour function rather than with plain AB colours.
    """

    def c(blue, red):
        return ab_colour(row, blue, red)

    def ck(blue, red):
        return float(kokorev24._color_with_upper_limits(phot, blue, red)[0][i])

    def term(blue, red, op, thr, cf=None):
        v = (cf or c)(blue, red)
        lo = "$-$".join([blue.upper(), red.upper()])
        ok = (
            bool(sh.gt(np.array([v]), thr)[0])
            if op == "gt"
            else bool(sh.lt(np.array([v]), thr)[0])
        )
        return (f"{lo} {'>' if op == 'gt' else '<'} {thr:g}", ok)

    def branches(mod, cf=None):
        """(red1 | red2): report the branch that comes closest, red2 breaking a tie.

        The branch masks rebuilt here are checked against the module's own red1 and red2,
        so a decomposition that disagrees with the rule it claims to explain is caught.
        """
        b1 = [term(*t, cf=cf) for t in mod.RED1]
        b2 = [term(*t, cf=cf) for t in mod.RED2]
        n1 = sum(1 for _, ok in b1 if not ok)
        n2 = sum(1 for _, ok in b2 if not ok)
        pick, tag = (b2, "red2") if n2 <= n1 else (b1, "red1")
        agree = (all(ok for _, ok in b1), all(ok for _, ok in b2))
        return [(f"{t} ({tag})", ok) for t, ok in pick], agree

    out, agree = {}, {}
    out["labbe23"], agree["labbe23"] = branches(labbe23)
    out["kokorev24"], agree["kokorev24"] = branches(kokorev24, cf=ck)
    out["kocevski24"] = []
    out["perezgonzalez24"] = [
        term("f277w", "f444w", "gt", perezgonzalez24.COLOR_F277W_F444W_MIN),
        term("f150w", "f200w", "lt", perezgonzalez24.COLOR_F150W_F200W_MAX),
        (
            "F115W$-$F150W $\\geq$ -0.5",
            bool(
                sh.ge(
                    np.array([c("f115w", "f150w")]),
                    perezgonzalez24.BD_COLOR_F115W_F150W_MIN,
                )[0]
            ),
        ),
    ]
    out["barro23"] = [term("f277w", "f444w", "gt", barro23.COLOR_F277W_F444W_MIN)]
    v = c("f115w", "f200w")
    out["greene24"] = [
        term("f277w", "f444w", "gt", greene24.COLOR_F277W_F444W_MIN),
        (
            f"F115W$-$F200W in ({greene24.VSHAPE_BLUE_LO:g}, {greene24.VSHAPE_BLUE_HI:g})",
            bool(
                sh.between(
                    np.array([v]), greene24.VSHAPE_BLUE_LO, greene24.VSHAPE_BLUE_HI
                )[0]
            ),
        ),
    ]
    out["akins24"] = [term("f277w", "f444w", "gt", akins24.COLOR_F277W_F444W_MIN)]
    return out, agree


# the criterion mask names that need a measurement this build cannot make locally: the
# 0.4/0.2 arcsec aperture ratio and the Akins C444 concentration are measured on the field
# mosaics, which are not part of the released tables for rows outside the candidate list.
RATIO_MASKS = {"compact", "compact_valid", "compact_artifact"}
AUDIT_MASKS = {
    "selected",
    "main",
    "bd_measurable",
    "used_upper_limit_red1",
    "used_upper_limit_red2",
    "used_upper_limit_bd",
    "lineboost_356_applies",
    "lineboost_410_applies",
}
CRIT_NAME = {
    "detection": "F444W detection gate",
    "mag_gate": "F444W magnitude gate",
    "red_color": "F277W$-$F444W colour",
    "red1": "red1 colours",
    "red2": "red2 colours",
    "vshape_blue": "F115W$-$F200W window",
    "vshape_red": "F277W$-$F444W colour",
    "blue_color": "F150W$-$F200W colour",
    "bd_retention": "F115W$-$F150W colour",
    "bd_removal": "F115W$-$F200W colour",
    "beta_uv_window": "UV slope window",
    "beta_opt_red": "optical slope sign",
    "size": "half-light radius",
    "lineboost_356": "F277W$-$F444W line boost",
    "lineboost_410": "F410M line boost",
}


def rule_ladder(row_frame, i):
    """For one object: per rule, the F277W-F444W threshold and why the rule rejects it."""
    masks = run_rules(row_frame)
    row = row_frame.iloc[i]
    terms, agree = colour_terms(row, phot=contract(row_frame), i=i)
    # the rebuilt branches must reproduce the modules' own red1 and red2 for this row
    for k, (r1, r2) in agree.items():
        got = (bool(masks[k]["red1"][i]), bool(masks[k]["red2"][i]))
        if got != (r1, r2):
            print(
                f"   WARNING {k}: rebuilt branches {(r1, r2)} disagree with the module "
                f"{got}; the criterion name falls back to the module's mask"
            )
            terms[k] = []
    obs = ab_colour(row, "f277w", "f444w")
    rows = []
    for k in RULE_ORDER:
        m = masks[k]
        failed = [
            kk
            for kk, vv in m.items()
            if isinstance(vv, np.ndarray)
            and vv.dtype == bool
            and kk not in AUDIT_MASKS
            and not bool(vv[i])
        ]
        unmeasured = sorted(set(failed) & RATIO_MASKS)
        failed = [f for f in failed if f not in RATIO_MASKS]
        thr = THR_277_444[k]
        # does the object fail THIS rule on the F277W-F444W axis the ladder draws?
        on_axis = thr is not None and not bool(sh.gt(np.array([obs]), thr)[0])
        # the criterion actually named: the colour term that fails, else the mask name
        bad_terms = [t for t, ok in terms[k] if not ok]
        if on_axis:
            why = f"F277W$-$F444W $\\leq$ {thr:g}"
        elif bad_terms:
            why = (
                bad_terms[0]
                if len(bad_terms) == 1
                else f"{bad_terms[0]} (+{len(bad_terms) - 1})"
            )
        elif failed:
            why = CRIT_NAME.get(failed[0], failed[0])
        else:
            why = "aperture ratio not measured" if unmeasured else "selected"
        rows.append(
            dict(
                rule=k,
                thr=thr,
                obs=obs,
                on_axis=on_axis,
                why=why,
                failed=failed,
                unmeasured=unmeasured,
                selected=bool(m["selected"][i]),
                stored=bool(row[f"sel_{k}"]),
            )
        )
    return rows


def draw_ladder(ax, rows, letter=None):
    """Panel (d): one bar per rule on a shared F277W-F444W axis."""
    obs = rows[0]["obs"]
    ymax = len(rows) - 0.4
    ax.axvline(obs, color=C_MODEL, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for j, r in enumerate(rows):
        y = len(rows) - 1 - j
        col = C_MISS if r["on_axis"] else C_NEG
        if r["thr"] is not None:
            ax.plot(
                [min(r["thr"], obs), max(r["thr"], obs)],
                [y, y],
                color=col,
                lw=2.6,
                solid_capstyle="butt",
                alpha=0.55 if not r["on_axis"] else 0.9,
                zorder=2,
            )
            # round 3: the threshold tick and the rule's name carry the rule's palette
            # colour, the same colour that names it in fig_recall_burden and edges its
            # shaded region in fig_colour_planes. The bar keeps its meaning colour, so the
            # reader still reads off why the rule rejects the object.
            ax.plot(
                [r["thr"], r["thr"]],
                [y - 0.32, y + 0.32],
                color=RULE_COLOUR[r["rule"]],
                lw=1.6,
                solid_capstyle="butt",
                zorder=4,
            )
        ax.plot(
            [obs],
            [y],
            "o",
            ms=4.0,
            mfc=C_MODEL if not r["on_axis"] else C_MISS,
            mec=K,
            mew=0.5,
            zorder=5,
        )
        ax.text(
            2.34,
            y,
            r["why"],
            fontsize=FS,
            color=C_MISS if r["on_axis"] else "#555555",
            ha="left",
            va="center",
        )
    # the rule names live inside this panel, so they can never run into the panel at its
    # left; the colour axis occupies 0.35 to 2.2 and the strip right of it carries the
    # criterion the rule actually rejects the object on
    for j, r in enumerate(rows):
        ax.text(
            -1.99,
            len(rows) - 1 - j,
            RULE_LABEL[r["rule"]],
            fontsize=FS,
            color=RULE_TEXT[r["rule"]],
            ha="left",
            va="center",
        )
    ax.set_yticks([])
    ax.set_ylim(-0.6, ymax)
    # Round 3 widened this axis from (-1.78, 5.55) to (-2.05, 5.95) and widened its column,
    # because the figure is now saved uncropped: the widest criterion string is 34.6 mm at
    # 8 pt and the widest rule name 22.8 mm, both measured, and under the old geometry the
    # criterion strip was 28 mm and the tight bounding box was quietly growing the canvas to
    # 175.6 mm to hide the overflow. Nothing plotted moved; only the empty margins either
    # side of the colour axis, whose bounds stay 0.35 to 2.2.
    ax.set_xlim(-2.05, 5.95)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("F277W$-$F444W (AB)", x=0.413, ha="center", labelpad=2.0)
    ax.tick_params(axis="x", top=False)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_locator(NullLocator())
    for s in ("right", "top", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_bounds(0.35, 2.2)
    if letter:
        panel(ax, letter)


# ================================================================== Figure N1: anatomy
print("\n--- figure N1: what a Little Red Dot looks like")
RULES_ALL = feat[[f"sel_{k}" for k in RULE_ORDER]].sum(axis=1)
pool = feat.assign(n_rule=RULES_ALL).merge(
    status[["source_id", "elig", "testable", "vshaped_spec"]],
    on="source_id",
    how="left",
)
pool = pool.merge(oof[["source_id", "score_mean"]], on="source_id", how="left")
pool = pool[
    (pool.y == 1)
    & (pool.n_rule == len(RULE_ORDER))
    & pool.elig.fillna(False).astype(bool)
    & pool.testable.fillna(False).astype(bool)
    & pool.vshaped_spec.fillna(False).astype(bool)
    & pool.in_support.astype(bool)
]
pool = pool.sort_values("score_mean", ascending=False)
print(
    f"N1 pool (all seven rules, tested V-shaped prism, in support): {len(pool)} objects"
)
print(
    pool[
        ["field", "id", "score_mean", "c_f277w_f444w", "mag_f444w", "z_phot"]
    ].to_string(index=False)
)
n1 = pool.iloc[0]
n1_lab = lab[lab.source_id == n1.source_id].iloc[0]
n1_sup = sup[sup.source_id == n1.source_id].iloc[0]
print(
    f"N1 object: {n1.field} {int(n1.id)}  score {n1.score_mean:.6f} "
    f"(runner-up {pool.iloc[1].score_mean:.6f})  rank {int(n1_sup.rank_all)} of {N_SUPPORT:,}"
)

n1_stamp = os.path.join(STAMPS, f"anatomy_{n1.field}_{int(n1.id)}.fits")
have_stamp = retrieve(CUTOUT.format(ra=float(n1.ra), dec=float(n1.dec)), n1_stamp)
assert have_stamp, "N1 stamp could not be fetched"

# The archival prism spectrum the spectral test would use: the DJA extractions index is the
# same index census3.py draws from. Fixed rule: among PRISM-CLEAR extractions matched within
# 0.5 arcsec, the highest grade, ties broken on the smallest separation.
idx = pd.read_csv(
    os.path.join(ROOT, "inputs", "dja_index.csv.gz"),
    low_memory=False,
    usecols=["file", "root", "ra", "dec", "grating", "filter", "grade", "z_best"],
)
sep = (
    np.hypot(
        (idx.ra - float(n1.ra)) * np.cos(np.radians(float(n1.dec))),
        idx.dec - float(n1.dec),
    )
    * 3600.0
)
near = idx[(sep < 0.5) & (idx.grating.str.upper() == "PRISM")].copy()
near["sep"] = sep[near.index]
near = near.sort_values(["grade", "sep"], ascending=[False, True])
spec_ok, spec_file, z_spec, spec_grade, spec_sep = False, None, np.nan, np.nan, np.nan
if len(near):
    s = near.iloc[0]
    spec_file, z_spec = str(s.file), float(s.z_best)
    spec_grade, spec_sep = float(s.grade), float(s.sep)
    print(
        f"prism spectrum: {spec_file} grade {spec_grade:g} z_best {z_spec:.4f} "
        f"sep {spec_sep:.3f} arcsec ({len(near)} prism matches within 0.5 arcsec)"
    )
    spec_ok = retrieve(
        f"{SPEC_BASE}/{s.root}/{s.file}", os.path.join(SPECD, s.file), minsize=10000
    )
else:
    print("no prism extraction within 0.5 arcsec: panel (d) will be dropped")

# the six-band cutout of the same object, at the same coordinates, from the same service;
# fetch_stamps_6band.py normally has it on disk already
n1_stamp6 = os.path.join(STAMPS, f"anatomy_{n1.field}_{int(n1.id)}_6band.fits")
have6 = retrieve(CUTOUT6.format(ra=float(n1.ra), dec=float(n1.dec)), n1_stamp6)
if not have6:
    print("six-band stamp unavailable: the strip will show 'no coverage' panels")

# The rest-frame features a prism spectrum of a Little Red Dot is read for. Each is drawn
# only where it falls inside the plotted range, and names that would collide at print size
# are merged into one label, as Greene et al. (2024) Fig. 2 merges its blended pair.
FEATURES = [
    (3645.0, "Balmer break"),
    (4862.7, r"H$\beta$"),
    (5008.2, "[O III]"),
    (6564.6, r"H$\alpha$"),
]
FEATURE_MIN_SEP_DEX = 0.022  # about 4 mm at this panel's width: closer than this, merge

fig = plt.figure(figsize=(COL2, 4.40))
gs = fig.add_gridspec(
    2,
    2,
    width_ratios=[1.0, 1.02],
    height_ratios=[1.0, 0.78],
    wspace=0.26,
    hspace=0.46,
    left=0.070,
    right=0.988,
    top=0.945,
    bottom=0.088,
)
ims = load_bands(n1_stamp)
rgb, gray = render(ims)
ims6 = load_bands6(n1_stamp6)
# (a) the strip: six single-band panels and the composite, in the footprint the colour and
# F444W stamps used to fill between them
draw_strip(fig, gs[0, 0], ims6, rgb, edge=C_MODEL, letter="a")
axc = fig.add_subplot(gs[0, 1])
n1_uv, n1_opt = draw_sed(axc, n1_lab, "b")
axd = fig.add_subplot(gs[1, :])
if spec_ok:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = afits.open(os.path.join(SPECD, spec_file))[1].data
    w = np.asarray(sp["wave"], float)
    fl = np.asarray(sp["flux"], float)
    er = np.asarray(sp["err"], float)
    good = np.isfinite(w) & np.isfinite(fl) & np.isfinite(er) & (er > 0)
    # the broad-band photometry the SED panel plots, over the spectrum in the same units:
    # labels.parquet stores f_ and e_ in microjansky, which is the spectrum's own axis
    f_ph = np.array([float(n1_lab[f"f_{b}"]) for b in BANDS])
    e_ph = np.array([float(n1_lab[f"e_{b}"]) for b in BANDS])
    l_ph = np.array([LAM_UM[b] for b in BANDS])
    okp = np.isfinite(f_ph) & np.isfinite(e_ph) & (f_ph > 0)
    # The archival extraction is a slit spectrum and the photometry is a total flux, so the
    # two sit at different levels; the extraction carries no slit-loss correction. For the
    # display the spectrum is multiplied by one constant, the median over the six broad
    # bands of the strip of the ratio of the catalogue flux to the spectrum's synthetic
    # flux in that band. One constant cannot remove a wavelength-dependent loss, and the
    # residual per-band spread is printed below so the caption can be honest about it.
    _sb = [
        b
        for b, ok in zip(BANDS, okp)
        if ok and b != "f090w"  # the six bands the strip shows
    ]
    _ratios = spectrum_band_ratios(
        w[good], fl[good], {b: float(n1_lab[f"f_{b}"]) for b in _sb}, _sb
    )
    SPEC_SCALE = float(np.median(np.array(list(_ratios.values()))))
    print(
        "N1 spectrum scaling: bands "
        + ", ".join("%s %.2f" % (b.upper(), r) for b, r in _ratios.items())
        + "; median %.4f, drawn as %.1f" % (SPEC_SCALE, SPEC_SCALE)
    )
    fl = fl * SPEC_SCALE
    er = er * SPEC_SCALE
    # a logarithmic flux axis, so the faint rest-ultraviolet continuum, the break and the
    # H-alpha line are all legible together; on a linear axis the line flattens the rest
    axd.fill_between(
        w[good],
        np.clip((fl - er)[good], 1e-3, None),
        (fl + er)[good],
        color=C_NEG,
        alpha=0.35,
        lw=0,
        zorder=1,
    )
    (h_spec,) = axd.plot(
        w[good], fl[good], color=K, lw=0.7, zorder=2, label="prism spectrum"
    )
    axd.set_yscale("log")
    top = max(float(np.nanmax(fl[good])), float(np.nanmax(f_ph[okp]))) * 3.0
    axd.set_ylim(0.02, top)
    axd.set_xlim(0.75, 5.4)
    LO, HI = axd.get_xlim()
    # (c) the rest-frame axis on top, so a reader reads the features off directly
    secx = axd.secondary_xaxis(
        "top",
        functions=(lambda o: o / (1.0 + z_spec), lambda r: r * (1.0 + z_spec)),
    )
    secx.set_xlabel(r"rest wavelength ($\mu$m), $z = %.3f$" % z_spec, labelpad=2.5)
    secx.tick_params(labelsize=FS, direction="in", width=0.6)
    drawn = []  # (observed micron, merged name) for the labels actually placed
    for rest, name in FEATURES:
        obs = rest * (1.0 + z_spec) / 1e4
        if not (LO < obs < HI):
            continue
        axd.axvline(obs, color=C_MISS, lw=0.7, ls=(0, (3, 2)), zorder=3)
        if drawn and abs(np.log10(obs) - np.log10(drawn[-1][0])) < FEATURE_MIN_SEP_DEX:
            drawn[-1] = (obs, drawn[-1][1] + " + " + name)
        else:
            drawn.append((obs, name))
    for obs, name in drawn:
        axd.text(
            obs,
            0.965,
            " " + name,
            transform=axd.get_xaxis_transform(),
            color=C_MISS,
            fontsize=FS,
            rotation=90,
            ha="left",
            va="top",
            zorder=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.8),
        )
    h_phot = axd.errorbar(
        l_ph[okp],
        f_ph[okp],
        yerr=e_ph[okp],
        fmt="o",
        ms=4.4,
        mfc=C_MODEL,
        mec=K,
        mew=0.5,
        ecolor=K,
        elinewidth=0.7,
        capsize=1.6,
        lw=0,
        zorder=6,
        label="broad-band photometry",
    )
    axd.set_xlabel(r"observed wavelength ($\mu$m)")
    axd.set_ylabel(r"$f_\nu$ ($\mu$Jy)")
    axd.legend(
        handles=[h_spec, h_phot],
        loc="upper left",
        fontsize=FS,
        borderaxespad=0.5,
        handletextpad=0.5,
    )
    panel(axd, "c", pad=13.0)
    print(
        "N1 spectrum panel: features drawn "
        + ", ".join("%s at %.3f um" % (n, o) for o, n in drawn)
        + "; %d of 7 bands overlaid" % int(okp.sum())
    )
else:
    fig.delaxes(axd)
save(fig, "fig_lrd_anatomy")

# ================================================================== Figure N2: a miss
print("\n--- figure N2: anatomy of a miss")
gal = pd.read_csv(os.path.join(STAMPS, "manifest.csv"))
red3 = gal[(gal.group == "recovered") & (gal.c_f277w_f444w > 1.0)].sort_values(
    "score", ascending=False
)
print("three red recovered rule-missed LRDs (fixed rule of fetch_stamps.py):")
print(
    red3[["field", "id", "mag_f444w", "c_f277w_f444w", "score"]].to_string(index=False)
)

rows3 = lab.merge(red3[["field", "id"]], on=["field", "id"]).merge(
    comp, on="source_id", how="left"
)
rows3 = (
    rows3.set_index(["field", "id"]).loc[list(zip(red3.field, red3.id))].reset_index()
)
ladders = [rule_ladder(rows3, i) for i in range(len(rows3))]

N2_LEFT = 0.040
# Round 3 reallocated the columns from measured string widths: the ladder needs 81 mm to
# hold a 22.8 mm rule name, the 0.35 to 2.2 colour axis and a 34.6 mm criterion, and used
# to have 65 mm and overflow the canvas.
fig = plt.figure(figsize=(COL2, 5.85))
gs = fig.add_gridspec(
    3,
    4,
    width_ratios=[1.06, 1.06, 1.30, 4.38],
    wspace=0.24,
    hspace=0.60,
    left=N2_LEFT,
    right=0.996,
    top=0.885,
    bottom=0.070,
)
n2_report = []
for i in range(len(rows3)):
    r = rows3.iloc[i]
    s = sup[sup.source_id == r.source_id].iloc[0]
    stem = f"recovered_{r.field}_{int(r.id)}"
    ims = load_bands(os.path.join(STAMPS, stem + ".fits"))
    rgb, gray = render(ims)
    ims6 = load_bands6(os.path.join(STAMPS, stem + "_6band.fits"))
    # the strip takes exactly the footprint the colour and F444W stamps used to fill, so the
    # figure keeps its height while showing six bands instead of one
    draw_strip(fig, gs[i, 0:2], ims6, rgb, edge=C_MISS, letter="a" if i == 0 else None)
    a2 = fig.add_subplot(gs[i, 2])
    uv, opt = draw_sed(a2, r)
    a3 = fig.add_subplot(gs[i, 3])
    draw_ladder(a3, ladders[i])
    if i == 0:
        for ax, let in ((a2, "b"), (a3, "c")):
            panel(ax, let)
    # the object's identity above its row, clear of the panel letters on the first row
    fig.text(
        N2_LEFT,
        a2.get_position().y1 + (0.027 if i == 0 else 0.007),
        f"{r.field} {int(r.id)}\nranking score {s.score_mean:.2f}, "
        f"rank {int(s.rank_all)} of {N_SUPPORT:,}",
        fontsize=FS,
        color=K,
        ha="left",
        va="bottom",
        linespacing=1.3,
    )
    n2_report.append(
        dict(
            field=r.field,
            id=int(r.id),
            score=float(s.score_mean),
            rank=int(s.rank_all),
            ab_c277_444=ab_colour(r, "f277w", "f444w"),
            slope_uv=uv,
            slope_opt=opt,
            ladder=ladders[i],
        )
    )
save(fig, "fig_miss_anatomy")

for d in n2_report:
    print(f"\n{d['field']} {d['id']}  AB F277W-F444W {d['ab_c277_444']:.3f}")
    for r in d["ladder"]:
        t = "none" if r["thr"] is None else f"{r['thr']:g}"
        print(
            f"   {RULE_LABEL[r['rule']]:<18} thr {t:<5} fails-on-axis {str(r['on_axis']):<5}"
            f" why '{r['why']}'  module-selected {r['selected']} record {r['stored']}"
            + (f"  unmeasured {r['unmeasured']}" if r["unmeasured"] else "")
        )

# ================================================================== Figure N3: the fields
print("\n--- figure N3: nine fields, five regions")
REGIONS = ["CEERS", "GOODS-N", "GOODS-S", "COSMOS", "UDS"]
FIELDS = MAN["regions"]
cand_eb = cand[cand.tier.astype(str).str.contains("equal", case=False, na=False)].copy()
if not len(
    cand_eb
):  # the tier column may name the tier differently; fall back on the flag
    cand_eb = cand[~cand.followup_tier.astype(bool)].copy()
cand_eb["region"] = cand_eb.field.map(FIELDS)
print("candidate tiers on record:", sorted(cand.tier.astype(str).unique()))
print(f"equal-burden candidates: {len(cand_eb)}")

macros = {}
for line in open(os.path.join(HERE, "numbers.tex")):
    if line.startswith("\\newcommand{\\"):
        nm = line.split("}")[0][len("\\newcommand{\\") :]
        val = line[line.index("}{") + 2 : line.rindex("}")]
        macros[nm] = val.replace("{,}", ",")

# Round 3: the canvas is exactly 178 mm and is saved uncropped, so \textwidth scales it by
# 1 and the 8 pt lettering is 8 pt on the page. Round 2's 2.1 mm undersize was a correction
# for the tight bounding box, which no longer applies.
fig = plt.figure(figsize=(COL2, 3.62))
gs = fig.add_gridspec(
    1, 5, wspace=0.50, left=0.078, right=0.990, top=0.775, bottom=0.325
)
counts = []
MAXFIELDS = max(sum(1 for v in FIELDS.values() if v == r) for r in REGIONS)
rng = np.random.default_rng(0)
for j, reg in enumerate(REGIONS):
    ax = fig.add_subplot(gs[0, j])
    g = sup[sup.region == reg]
    gl = g[g.y == 1]
    gn = g[g.y == 0]
    gc = cand_eb[cand_eb.region == reg]
    show = g if len(g) <= 50000 else g.iloc[rng.choice(len(g), 50000, replace=False)]
    # round 3: each region's density is drawn in its own tint, and the outline of every
    # field in it is traced, the way a survey-footprint figure draws its pointings. The
    # outline is the boundary of the sky cells that actually hold catalogue rows, on a
    # 44 x 44 grid over the region, so it states coverage and does not invent a hull.
    tint = REGION_TINT[reg]
    ax.hexbin(
        show.ra,
        show.dec,
        gridsize=42,
        cmap=LinearSegmentedColormap.from_list("t" + reg, ["#ffffff", tint]),
        bins="log",
        linewidths=0.0,
        mincnt=1,
        vmax=None,
        zorder=1,
        alpha=0.95,
    )
    ra_lo, ra_hi = float(g.ra.min()), float(g.ra.max())
    de_lo, de_hi = float(g.dec.min()), float(g.dec.max())
    for f in sorted(k for k, v in FIELDS.items() if v == reg):
        fr = g[g.field == f]
        if len(fr) < 10:
            continue
        Hf, xe, ye = np.histogram2d(
            fr.ra.to_numpy(float),
            fr.dec.to_numpy(float),
            bins=(44, 44),
            range=[[ra_lo, ra_hi], [de_lo, de_hi]],
        )
        ax.contour(
            0.5 * (xe[:-1] + xe[1:]),
            0.5 * (ye[:-1] + ye[1:]),
            (Hf.T > 0).astype(float),
            levels=[0.5],
            colors=[_dark(tint, target=0.30)],
            linewidths=0.6,
            zorder=2,
        )
    an = gn if len(gn) <= 500 else gn.iloc[rng.choice(len(gn), 500, replace=False)]
    ax.plot(an.ra, an.dec, "o", ms=1.6, mfc="none", mec=C_NEG, mew=0.35, lw=0, zorder=3)
    ax.plot(gc.ra, gc.dec, ".", ms=1.8, color=C_CAND, lw=0, zorder=4)
    ax.plot(gl.ra, gl.dec, "o", ms=2.6, mfc=C_MODEL, mec=K, mew=0.3, lw=0, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_xaxis()
    ax.locator_params(axis="x", nbins=2)
    ax.locator_params(axis="y", nbins=4)
    ax.tick_params(labelsize=FS)
    ax.set_xlabel("RA (deg)", labelpad=2.0)
    if j == 0:
        ax.set_ylabel("Dec (deg)", labelpad=2.0)
    # region name, then its field names one per line; ngdeep has no band-complete row and
    # so contributes nothing to support, which the panel says rather than hides
    fl = sorted(k for k, v in FIELDS.items() if v == reg)
    fl = [f + (" (none)" if int((sup.field == f).sum()) == 0 else "") for f in fl]
    # offsets in points, so they do not depend on how tall the equal-aspect panel ends up
    ax.set_title(
        reg, fontsize=FS, pad=4.0 + 10.6 * MAXFIELDS
    )  # one height for all five
    ax.annotate(
        "\n".join(fl),
        xy=(0.5, 1.0),
        xycoords="axes fraction",
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=FS,
        color="#555555",
        linespacing=1.3,
    )
    ax.annotate(
        f"{len(g):,} rows\n{len(gl)} LRDs\n{len(gn):,} anchors\n{len(gc)} candidates",
        xy=(0.5, 0.0),
        xycoords="axes fraction",
        xytext=(0, -32),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=FS,
        linespacing=1.32,
    )
    counts.append(
        dict(region=reg, rows=len(g), lrds=len(gl), anchors=len(gn), candidates=len(gc))
    )
handles = [
    Line2D(
        [],
        [],
        marker="h",
        ls="",
        mfc="#b0b0b0",
        mec="none",
        ms=5,
        label="catalogue rows in support",
    ),
    Line2D(
        [],
        [],
        marker="o",
        ls="",
        mfc="none",
        mec=C_NEG,
        mew=0.6,
        ms=4,
        label="spectroscopic non-LRD anchors",
    ),
    Line2D(
        [],
        [],
        marker=".",
        ls="",
        color=C_CAND,
        ms=7,
        label="candidates at the rules' burden",
    ),
    Line2D(
        [],
        [],
        marker="o",
        ls="",
        mfc=C_MODEL,
        mec=C_MODEL,
        ms=4,
        label="spectroscopic LRDs",
    ),
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=4,
    fontsize=FS,
    bbox_to_anchor=(0.5, 0.002),
    handletextpad=0.4,
    columnspacing=1.3,
)
save(fig, "fig_fields")

MK = {
    "CEERS": "CEERS",
    "GOODS-N": "GOODSN",
    "GOODS-S": "GOODSS",
    "COSMOS": "COSMOS",
    "UDS": "UDS",
}
print("\ncounts printed on N3, against numbers.tex:")
print(
    f"{'region':<9}{'rows':>10}{'macro':>10}{'LRDs':>6}{'macro':>7}{'anchors':>9}{'macro':>8}{'cands':>7}"
)
disagree = []
for c in counts:
    k = MK[c["region"]]
    mr, mp, mn = (
        macros.get("rows" + k),
        macros.get("posSup" + k),
        macros.get("negSup" + k),
    )
    print(
        f"{c['region']:<9}{c['rows']:>10,}{mr:>10}{c['lrds']:>6}{mp:>7}"
        f"{c['anchors']:>9,}{mn:>8}{c['candidates']:>7}"
    )
    for got, want, nm in (
        (f"{c['rows']:,}", mr, "rows"),
        (str(c["lrds"]), mp, "posSup"),
        (f"{c['anchors']:,}", mn, "negSup"),
    ):
        if got != want:
            disagree.append((c["region"], nm, got, want))
tot = sum(c["candidates"] for c in counts)
print(
    f"{'total':<9}{sum(c['rows'] for c in counts):>10,}{macros.get('supportRows'):>10}"
    f"{sum(c['lrds'] for c in counts):>6}{macros.get('nPosSupport'):>7}"
    f"{sum(c['anchors'] for c in counts):>9,}{macros.get('nNegSupport'):>8}{tot:>7}"
)
print("candidate total macro candEqualBurden =", macros.get("candEqualBurden"))
print("DISAGREEMENTS:", disagree if disagree else "none")

# ------------------------------------------------------------------ manifest
rowsm = [
    dict(
        figure="fig_lrd_anatomy",
        panel="a,b,c,d",
        group="all-seven-rules spectroscopic LRD",
        field=n1.field,
        id=int(n1.id),
        ra=float(n1.ra),
        dec=float(n1.dec),
        z=float(z_spec) if spec_ok else float(n1.z_phot),
        z_kind="spectroscopic (DJA grade 3)" if spec_ok else "photometric",
        score=float(n1.score_mean),
        rank=int(n1_sup.rank_all),
        stamp=os.path.basename(n1_stamp),
        spectrum=spec_file if spec_ok else "",
    )
]
for d, (_, r) in zip(n2_report, rows3.iterrows()):
    rowsm.append(
        dict(
            figure="fig_miss_anatomy",
            panel="a,b,c,d",
            group="recovered rule-missed LRD, F277W-F444W above 1.0",
            field=d["field"],
            id=d["id"],
            ra=float(r.ra),
            dec=float(r.dec),
            z=float(r.z_phot),
            z_kind="photometric",
            score=d["score"],
            rank=d["rank"],
            stamp=f"recovered_{d['field']}_{d['id']}.fits",
            spectrum="",
        )
    )
pd.DataFrame(rowsm).to_csv(os.path.join(FIG, "manifest_extra.csv"), index=False)
print("\nwrote figures/manifest_extra.csv with", len(rowsm), "objects")
print(
    f"stamp field {CROP_ARCSEC:.1f} arcsec at {PIXSCALE} arcsec per pixel; "
    f"scale bar {BAR_ARCSEC} arcsec = {BAR_PIX:g} pixels"
)
if spec_ok:
    print(
        f"prism spectrum drawn at {SPEC_SCALE:.4f} times the archive extraction "
        f"({SPEC_SCALE:.1f} in the caption), display only"
    )
print(f"smallest font on every figure: {FS} pt; widths {COL2} in (178 mm)")

# ================================================================== the second schematic
# fig_architecture.tex is a TikZ standalone like fig_pipeline.tex, but nothing recompiled it
# when its colours changed. Round 3 compiles it here, so one command rebuilds the three
# explanatory figures and the schematic that shares their palette.
if shutil.which("pdflatex"):
    _r = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "fig_architecture.tex",
        ],
        cwd=FIG,
        capture_output=True,
        text=True,
        check=False,  # the return code is inspected below, with the log attached
    )
    if _r.returncode:
        raise SystemExit(
            "fig_architecture.tex failed to compile:\n"
            + _r.stdout[-3000:]
            + _r.stderr[-2000:]
        )
    for _ext in (".aux", ".log"):
        _f = os.path.join(FIG, "fig_architecture" + _ext)
        if os.path.exists(_f):
            os.remove(_f)
    from pypdf import PdfReader as _PR

    _b = _PR(os.path.join(FIG, "fig_architecture.pdf")).pages[0].mediabox
    print(
        "wrote fig_architecture.pdf  %.1f x %.1f mm"
        % (float(_b.width) / 72 * 25.4, float(_b.height) / 72 * 25.4)
    )
    try:
        import pypdfium2 as pdfium

        _pg = pdfium.PdfDocument(os.path.join(FIG, "fig_architecture.pdf"))[0]
        _pg.render(scale=300 / 72).to_pil().save(
            os.path.join(FIG, "fig_architecture.png")
        )
        print("wrote fig_architecture.png from fig_architecture.pdf")
    except Exception as e:  # noqa: BLE001 - the PDF is the deliverable, the PNG only reviews it
        print("fig_architecture.png not rasterised:", e)
else:
    print("pdflatex not found: figures/fig_architecture.pdf left as it stands")

# ================================================================== font report
# build_figures.py writes font_report.txt for its eleven; this is the companion for the
# three explanatory figures and the schematic compiled above, kept in a separate file so
# neither script can overwrite the other's.
with open(
    os.path.join(FIG, "font_report_extra.txt"), "w", encoding="utf-8", newline="\n"
) as f:
    f.write("figure                 width_mm  height_mm  min_font_pt\n")
    for nm, w, h, mf in FONT_REPORT:
        f.write("%-22s %8.1f %10.1f %12.1f\n" % (nm, w, h, mf))
    f.write(
        "\nEvery figure here is emitted at exactly 178 mm, the width main.tex includes it"
        " at, and saved uncropped, so the LaTeX scale factor is 1 and a point specified in"
        " this file is a point on the page.\n"
        "fig_architecture is a TikZ standalone: its one type size is set in"
        " figures/fig_architecture.tex.\n"
    )
print("wrote font_report_extra.txt")

# ================================================================== second contact sheet
# build_figures.py writes contact_sheet.png for its eleven. This is the other four plus the
# second schematic: the three explanatory figures and the two TikZ standalones, so a
# reviewer can check the whole set of fifteen in two images.
ORDER2 = [
    "fig_lrd_anatomy",
    "fig_miss_anatomy",
    "fig_fields",
    "fig_pipeline",
    "fig_architecture",
]
_ncol = 2
_nrow = int(np.ceil(len(ORDER2) / _ncol))
cs, cs_axes = plt.subplots(_nrow, _ncol, figsize=(15.0, 5.4 * _nrow))
_flat = np.asarray(cs_axes).ravel()
for a in _flat:
    a.axis("off")
for i, nm in enumerate(ORDER2):
    p = os.path.join(FIG, nm + ".png")
    if os.path.exists(p):
        _flat[i].imshow(plt.imread(p))
    _flat[i].set_title(nm + ".png", fontsize=11, color=K, pad=6)
cs.tight_layout()
cs.savefig(
    os.path.join(FIG, "contact_sheet_extra.png"),
    dpi=110,
    bbox_inches="tight",
    pad_inches=0.15,
)
plt.close(cs)
print("wrote contact_sheet_extra.png with", len(ORDER2), "figures")
