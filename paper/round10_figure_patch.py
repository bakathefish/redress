"""Round 10 (2026-09-07) figure fixes from the r7 referee reports.

Exact-string edits to the three figure builders:
  * rule labels carry the publication year of the citation (Labbe+25, Kocevski+25,
    Akins+25, Barro+24b for the barro23 rule) and the eighth selection reads Barro+24a;
  * fig_regions: the y range is trimmed to the data;
  * fig_gain_by_colour_mag: Wilson 95% intervals on every bar;
  * fig_miss_anatomy: the rank is the within-region rank, which is the only rank the
    five fold models make comparable;
  * fig_fields: the panel labels say "comparison objects", as the paper does.
Run from the build repository root: python paper/round10_figure_patch.py
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PR = os.path.join(ROOT, "paper")

LABELS_OLD = """    "labbe23": "Labbé+23",
    "kokorev24": "Kokorev+24",
    "kocevski24": "Kocevski+24",
    "perezgonzalez24": "Pérez-González+24",
    "barro23": "Barro+23",
    "greene24": "Greene+24",
    "akins24": "Akins+24",
"""
LABELS_NEW = """    "labbe23": "Labbé+25",
    "kokorev24": "Kokorev+24",
    "kocevski24": "Kocevski+25",
    "perezgonzalez24": "Pérez-González+24",
    "barro23": "Barro+24b",
    "greene24": "Greene+24",
    "akins24": "Akins+25",
"""

WILSON = '''

def _wilson(k, n, z=1.959964):
    """Wilson 95% interval for k of n, as (lower, upper) fractions."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return c - h, c + h

'''

EDITS = {
    "build_figures.py": [
        (LABELS_OLD, LABELS_NEW),
        (
            'label="union of eight (with Barro+24b)",',
            'label="union of eight (with Barro+24a)",',
        ),
        (
            '    "Kocevski+24 selects on continuum slope: no color threshold",',
            '    "Kocevski+25 selects on continuum slope: no color threshold",',
        ),
        ("ax.set_ylim(0, 88)\n", "ax.set_ylim(0, 55)\n"),
        # Wilson intervals on the gain bars
        (
            'mlab = ["< 24", "24 to 25", "25 to 26", "26 to 27", "> 27"]\nGAIN_COUNTS = {}\n',
            'mlab = ["< 24", "24 to 25", "25 to 26", "26 to 27", "> 27"]\nGAIN_COUNTS = {}'
            + WILSON,
        ),
        (
            """    for i in range(len(labels)):
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
""",
            """    # Round 10: Wilson 95% intervals on every bar, so the small bins carry their
    # uncertainty on the figure and not only in the caption.
    ulo, uhi = _wilson(u, nn)
    mlo, mhi = _wilson(m, nn)
    for xx, frac, lo, hi in ((x - 0.19, u / nn, ulo, uhi), (x + 0.19, m / nn, mlo, mhi)):
        ax.errorbar(
            xx,
            frac,
            yerr=[frac - lo, hi - frac],
            fmt="none",
            ecolor=K,
            elinewidth=0.6,
            capsize=1.6,
            zorder=4,
        )
    for i in range(len(labels)):
        ax.text(
            x[i] - 0.19,
            uhi[i] + 0.022,
            "%d" % u[i],
            ha="center",
            fontsize=SMALL,
            color=C_UNION,
        )
        ax.text(
            x[i] + 0.19,
            mhi[i] + 0.022,
            "%d" % m[i],
            ha="center",
            fontsize=SMALL,
            color=C_MODEL,
        )
""",
        ),
        ("    ax.set_ylim(0, 1.16)\n", "    ax.set_ylim(0, 1.24)\n"),
    ],
    "build_figures_pub.py": [
        (LABELS_OLD, LABELS_NEW),
        (
            '    ("labbe23", 0.8, 1.0, "Labbé+23 red2"),',
            '    ("labbe23", 0.8, 1.0, "Labbé+25 red2"),',
        ),
    ],
    "build_figures_extra.py": [
        (LABELS_OLD, LABELS_NEW),
        (
            'sup["rank_all"] = sup.score_mean.rank(ascending=False, method="min").astype(int)\n',
            'sup["rank_all"] = sup.score_mean.rank(ascending=False, method="min").astype(int)\n'
            "# Round 10: the five fold models' scores are not on one scale, so the rank the\n"
            "# figure prints is the rank within the object's own region.\n"
            'sup["rank_region"] = (\n'
            '    sup.groupby("region").score_mean.rank(ascending=False, method="min").astype(int)\n'
            ")\n"
            'N_REGION = sup.groupby("region").size().to_dict()\n',
        ),
        (
            """        f"{r.field} {int(r.id)}\\nranking score {s.score_mean:.2f}, "
        f"rank {int(s.rank_all)} of {N_SUPPORT:,}",""",
            """        f"{r.field} {int(r.id)}\\nranking score {s.score_mean:.2f}, "
        f"rank {int(s.rank_region)} of {N_REGION[s.region]:,} in {s.region}",""",
        ),
        (
            """            rank=int(s.rank_all),
            ab_c277_444=ab_colour(r, "f277w", "f444w"),""",
            """            rank=int(s.rank_all),
            rank_region=int(s.rank_region),
            region=str(s.region),
            ab_c277_444=ab_colour(r, "f277w", "f444w"),""",
        ),
        (
            """        f"{len(g):,} rows\\n{len(gl)} LRDs\\n{len(gn):,} anchors\\n{len(gc)} candidates",""",
            """        f"{len(g):,} rows\\n{len(gl)} LRDs\\n{len(gn):,} comparison\\nobjects, {len(gc)} candidates",""",
        ),
    ],
}

for name, edits in EDITS.items():
    p = os.path.join(PR, name)
    t = open(p, encoding="utf-8").read()
    for old, new in edits:
        n = t.count(old)
        assert n == 1, (name, n, old[:70])
        t = t.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(t)
    print(name, "patched", len(edits), "edits")
