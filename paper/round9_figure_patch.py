"""Round 9 figure and table patches (2026-09-06): idempotent string edits to the figure
scripts, the two TikZ figures and the table builder, for the astra-r5 referee items.

  fig_recall_burden  regional-budget curve (within-region ranks at one budget fraction)
  fig_confusion      comparison-object labels; learned panels at the region-matched burden
  fig_scores         faceted by region, axis range reduced
  fig_regions        region-matched counts; rule-missed labels without the plus sign
  fig_candidates     full status axis with counts printed
  fig_gallery_known  the three objects Figure 10 already shows are dropped
  fig_robustness     comparison-object wording
  fig_zdist          outcome classes at the region-matched burden
  fig_fields         comparison-object wording; ngdeep marked "no support"
  fig_pipeline.tex   "The fix" becomes "The learned ranking"; matched counts; frozen-input wording
  fig_architecture.tex  comparison-object wording; final-fit threshold label
  build_tables.py    spectroscopic redshift and within-region rank columns; eighth-rule column

Run from the repository root: python paper/round9_figure_patch.py
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PR = os.path.join(ROOT, "paper")


def patch(path, pairs, must=True):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        if old in s:
            s = s.replace(old, new, 1)
        elif new in s:
            continue  # already applied
        elif must:
            raise SystemExit("anchor not found in %s:\n%s" % (path, old[:200]))
    open(path, "w", encoding="utf-8").write(s)
    print("patched", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------- build_figures.py
BF = os.path.join(PR, "build_figures.py")
patch(
    BF,
    [
        (
            'N = json.load(open(os.path.join(HERE, "numbers.json")))\n',
            'N = json.load(open(os.path.join(HERE, "numbers.json")))\n'
            "# round 9 (2026-09-06): the region-matched comparison is the primary result, and its\n"
            "# numbers, the regional-budget curve and the per-object outcomes come from these records\n"
            'N9 = json.load(open(os.path.join(ROOT, "recovery", "round9_numbers.json")))\n'
            'CURVE = json.load(open(os.path.join(ROOT, "recovery", "regional_budget_curve.json")))\n'
            'POUT = pd.read_csv(os.path.join(ROOT, "recovery", "positive_outcomes.csv")).set_index("source_id")\n',
        ),
        # ---- Figure 2 (recall against burden): the regional-budget curve
        (
            'sc = sup.score.values\nys = (sup.y == 1).values\norder = np.argsort(-sc)\ncum = np.cumsum(ys[order])\nn = np.arange(1, len(cum) + 1)\nax.plot(n, cum, color=C_MODEL, lw=1.0, label="learned ranking, out of fold", zorder=3)\n',
            "# Round 9: the five held-out ensembles do not share a score scale, so the curve is\n"
            "# built region by region. At each budget fraction f every region keeps the top f times\n"
            "# the union's row count there, by that region's own out-of-fold ranking, and the\n"
            "# recoveries are summed; the x coordinate is the total number of rows kept.\n"
            '_KTOT = float(sum(CURVE["K_union"].values()))\n'
            '_fx = np.array(CURVE["fractions"]) * _KTOT\n'
            'ax.plot(_fx, CURVE["primary"], color=C_MODEL, lw=1.0, label="learned ranking, within-region ranks at one budget fraction", zorder=3)\n'
            'ax.plot(_fx, CURVE["imitation"], color=K, lw=0.8, ls=(0, (3.0, 1.6)), label="imitation of the rules, same construction", zorder=3)\n',
        ),
        (
            'ax.scatter(\n    [pe["selected_total"]],\n    [pe["recall_147"]],\n    marker="D",\n    s=18,\n    color=C_MODEL,\n    zorder=6,\n    label="learned, equal burden",\n)\n',
            'ax.scatter(\n    [CURVE["union"]["rows"]],\n    [CURVE["union"]["primary"] if "primary" in CURVE["union"] else N9["matchedRecallSupport"]],\n    marker="D",\n    s=18,\n    color=C_MODEL,\n    zorder=6,\n    label="learned, region-matched burden",\n)\n'
            'ax.scatter(\n    [pe["selected_total"]],\n    [pe["recall_147"]],\n    marker="x",\n    s=22,\n    color=C_MODEL,\n    lw=0.9,\n    zorder=6,\n    label="learned, deployed thresholds",\n)\n'
            'ax.scatter(\n    [CURVE["union8"]["rows"]],\n    [CURVE["union8"]["recall"]],\n    marker="s",\n    s=20,\n    facecolor="white",\n    edgecolor=C_UNION,\n    lw=0.8,\n    zorder=5,\n    label="union of eight (with Barro+24b)",\n)\n'
            'ax.scatter(\n    [CURVE["union8"]["rows"]],\n    [CURVE["union8"]["primary"]],\n    marker="D",\n    s=18,\n    facecolor="white",\n    edgecolor=C_MODEL,\n    lw=0.9,\n    zorder=6,\n    label="learned at the union of eight\'s regional counts",\n)\n',
        ),
        (
            'ax.scatter(\n    [p5["selected_total"]],\n    [p5["recall_147"]],\n    marker="D",\n    s=18,\n    facecolor="white",\n    edgecolor=C_MODEL,\n    lw=0.7,\n    zorder=6,\n    label="learned, 0.5% threshold",\n)\n',
            'ax.scatter(\n    [p5["selected_total"]],\n    [p5["recall_147"]],\n    marker="x",\n    s=22,\n    facecolor="none",\n    edgecolor=C_MODEL,\n    lw=0.9,\n    zorder=6,\n    label="learned, 0.5% threshold",\n)\n',
        ),
        (
            'ax.scatter(\n    [N["blImitationSelected"]],\n    [bi["recall_at_burden"]],\n    marker="^",\n    s=20,\n    facecolor="white",\n    edgecolor=K,\n    lw=0.7,\n    zorder=6,\n    label="imitation of the rules",\n)\n',
            'ax.scatter(\n    [N["blImitationSelected"]],\n    [bi["recall_at_burden"]],\n    marker="^",\n    s=20,\n    facecolor="white",\n    edgecolor=K,\n    lw=0.7,\n    zorder=6,\n    label="imitation at its own realized burden",\n)\n',
        ),
        (
            'ax.set_xlim(100, 30000)\nax.set_ylim(0, 162)\nax.set_xlabel("catalog rows selected (burden)")',
            'ax.set_xlim(100, 40000)\nax.set_ylim(0, 162)\nax.set_xlabel("catalog rows selected, summed over the five regions")',
        ),
        # ---- Figure 7 (confusion): labels and the matched operating point
        (
            '        "non-LRD\\nanchors (%s)" % "{:,}".format(N["nNegSupport"]),',
            '        "comparison\\nobjects (%s)" % "{:,}".format(N["nNegSupport"]),',
        ),
        (
            '    "learned selection",\n    [\n        [N["modelRecallEnd"], N["nPos"] - N["modelRecallEnd"]],\n        [N["modelAnchors"], N["nNegSupport"] - N["modelAnchors"]],\n    ],',
            '    "learned selection, region-matched",\n    [\n        [N9["matchedRecallEnd"], N["nPos"] - N9["matchedRecallEnd"]],\n        [N9["matchedAnchors"], N["nNegSupport"] - N9["matchedAnchors"]],\n    ],',
        ),
        (
            '    [\n        [N["pairedBoth"], N["pairedUnionOnlyEnd"]],\n        [N["pairedModelOnly"], N["pairedNeitherEnd"]],\n    ],',
            '    [\n        [N9["matchedPairedBoth"], N9["matchedPairedUnionOnlyEnd"]],\n        [N9["matchedPairedModelOnly"], N9["matchedPairedNeitherEnd"]],\n    ],',
        ),
        # ---- Figure 8 (score distributions): one panel per region
        (
            'fig, ax = plt.subplots(figsize=(COL1, 64 * MM))\nfig.subplots_adjust(left=0.155, right=0.975, top=0.985, bottom=0.135)\nbins = np.linspace(0, 1, 51)\nax.hist(\n    sup.score,\n    bins=bins,\n    color=C_CAT,\n    lw=0,\n    label="all {:,} support rows".format(len(sup)),\n)\nax.hist(\n    neg.score,\n    bins=bins,\n    color=C_NEG,\n    lw=0,\n    label="{:,} non-LRD anchors".format(len(neg)),\n)\nax.hist(pos.score, bins=bins, color=C_MODEL, lw=0, label="%d spectroscopic LRDs" % len(pos))\ntb = np.unique(sup.t_burden)\nax.axvspan(tb.min(), tb.max(), color=C_MISS, alpha=0.15, lw=0)\nax.text(\n    (tb.min() + tb.max()) / 2,\n    3.0e3,\n    "equal-burden\\nthresholds\\n(five folds)",\n    ha="center",\n    va="center",\n    fontsize=BASE,\n    color=C_MISS,\n    linespacing=1.25,\n)\nax.set_yscale("log")\nax.set_ylim(0.7, 3e8)\nax.set_xlim(0, 1)\nax.set_xlabel("out-of-fold ranking score")\nax.set_ylabel("rows")\nax.legend(loc="upper left", fontsize=BASE, borderaxespad=0.6)\nsave(fig, "fig_scores", 84.0)\n',
            "# Round 9: the five ensembles' scores are not on one scale, so the distributions are drawn\n"
            "# one region per panel, each with its own fold's equal-burden threshold.\n"
            'fig, axes = plt.subplots(5, 1, figsize=(COL1, 118 * MM), sharex=True)\nfig.subplots_adjust(left=0.155, right=0.975, top=0.975, bottom=0.075, hspace=0.12)\nbins = np.linspace(0, 1, 41)\nfor _ax, _r in zip(axes, REG):\n    _s = sup[sup.region == _r]\n    _n = neg[neg.region == _r]\n    _p = pos[pos.region == _r]\n    _ax.hist(_s.score, bins=bins, color=C_CAT, lw=0, label="support rows")\n    _ax.hist(_n.score, bins=bins, color=C_NEG, lw=0, label="comparison objects")\n    _ax.hist(_p.score, bins=bins, color=C_MODEL, lw=0, label="spectroscopic LRDs")\n    _tb = float(np.unique(_s.t_burden)[0])\n    _ax.axvline(_tb, color=C_MISS, lw=0.8, zorder=4)\n    _ax.set_yscale("log")\n    _ax.set_ylim(0.7, 4e5)\n    _ax.set_xlim(0, 1)\n    _ax.set_yticks([1, 1e2, 1e4])\n    _ax.text(0.985, 0.92, "%s: %s rows, %d comparison, %d LRDs" % (_r, "{:,}".format(len(_s)), len(_n), len(_p)), transform=_ax.transAxes, ha="right", va="top", fontsize=SMALL, color=K)\n    _ax.text(_tb + 0.012, 1.4, "threshold %.3f" % _tb, fontsize=SMALL, color=C_MISS, ha="left", va="bottom")\naxes[-1].set_xlabel("out-of-fold ranking score (held-out region)")\naxes[2].set_ylabel("rows")\naxes[0].legend(loc="upper left", fontsize=SMALL, borderaxespad=0.4, handlelength=1.2)\nsave(fig, "fig_scores", 84.0)\n',
        ),
        # ---- Figure 5 (per region): region-matched counts
        (
            'm = [N["modelRegion" + r.replace("-", "")] for r in REG]\nt = [N["posSup" + r.replace("-", "")] for r in REG]\nmm_ = [N["modelRegionMissed" + r.replace("-", "")] for r in REG]',
            'm = [N9["matchedRegion" + r.replace("-", "")] for r in REG]\nt = [N["posSup" + r.replace("-", "")] for r in REG]\nmm_ = [N9["matchedRegionRuleMissed" + r.replace("-", "")] for r in REG]',
        ),
        (
            '        "%s\\n+%d of %d" % (r, mm_[i], N["missedSup" + r.replace("-", "")])',
            '        "%s\\n%d of %d rule-missed" % (r, mm_[i], N["missedSup" + r.replace("-", "")])',
        ),
        (
            'ax.set_ylabel("LRDs recovered, equal burden")',
            'ax.set_ylabel("LRDs recovered at the union\'s row count in the region")',
        ),
        (
            '        "learned selection (region held out)",\n    ],\n    loc="upper left",',
            '        "learned selection (region held out, region-matched burden)",\n    ],\n    loc="upper left",',
        ),
        # ---- Figure 10 (candidates), panel (c): full axis with counts
        (
            "    b0 = ax.bar(i, u_ / tot, 0.62, color=C_CAT, edgecolor=K, lw=0.3)\n    b1 = ax.bar(i, h_ / tot, 0.62, bottom=u_ / tot, color=C_MODEL, edgecolor=K, lw=0.3)\n    b2 = ax.bar(\n        i, l_ / tot, 0.62, bottom=(u_ + h_) / tot, color=C_MISS, edgecolor=K, lw=0.3\n    )\n    if i == 0:\n        hs = [b0, b1, b2]\n",
            '    b0 = ax.bar(i, u_ / tot, 0.62, color=C_CAT, edgecolor=K, lw=0.3)\n    b1 = ax.bar(i, h_ / tot, 0.62, bottom=u_ / tot, color=C_MODEL, edgecolor=K, lw=0.3)\n    b2 = ax.bar(\n        i, l_ / tot, 0.62, bottom=(u_ + h_) / tot, color=C_MISS, edgecolor=K, lw=0.3\n    )\n    # Round 9: the counts are printed, so the axis can run from 0 to 1 without hiding them\n    ax.text(i, 0.5 * u_ / tot, "%d" % u_, ha="center", va="center", fontsize=SMALL, color=K)\n    ax.text(i, 1.012, "%d / %d" % (h_, l_), ha="center", va="bottom", fontsize=SMALL, color=K)\n    if i == 0:\n        hs = [b0, b1, b2]\n',
        ),
        (
            "ax.set_xlim(-0.6, 2.6)\nax.set_ylim(0.88, 1.005)\n",
            "ax.set_xlim(-0.6, 2.6)\nax.set_ylim(0.0, 1.10)\n",
        ),
        (
            '    labels=["untested", "secure $z > 3$", "secure $z \\\\leq 3$"],',
            '    labels=["no usable spectrum", "secure $z > 3$", "secure $z \\\\leq 3$"],',
        ),
        # ---- the gallery: drop the three objects Figure 10 already shows
        (
            '        parts=[\n            ("F277W$-$F444W above 1.0", _red),\n            ("F277W$-$F444W below 0.5", _blue),\n        ],',
            '        # Round 9: the three reddest recovered misses are the objects of fig_miss_anatomy,\n        # so only the three bluest are drawn here\n        parts=[\n            ("F277W$-$F444W below 0.5", _blue),\n        ],',
        ),
        # ---- robustness: wording
        (
            '        "positives vs anchors only (no PU)",',
            '        "positives against comparison objects only",',
        ),
    ],
)

# ---------------------------------------------------------------- build_figures_pub.py (fig_zdist)
BP = os.path.join(PR, "build_figures_pub.py")
patch(
    BP,
    [
        (
            'N = json.load(open(os.path.join(HERE, "numbers.json")))\n',
            'N = json.load(open(os.path.join(HERE, "numbers.json")))\n'
            'N9 = json.load(open(os.path.join(ROOT, "recovery", "round9_numbers.json")))\n'
            'POUT = pd.read_csv(os.path.join(ROOT, "recovery", "positive_outcomes.csv")).set_index("source_id")\n',
        ),
        (
            'POS_MODEL = np.zeros(len(posAll), dtype=bool)\nPOS_MODEL[POS_INSUP] = sup.loc[posAll.index[POS_INSUP], "sel_burden"].to_numpy(bool)\n',
            "# Round 9: the primary operating point is the region-matched one (Section 5.1)\n"
            'POS_MODEL = np.zeros(len(posAll), dtype=bool)\nPOS_MODEL[POS_INSUP] = POUT.loc[posAll.index[POS_INSUP], "sel_matched"].to_numpy(bool)\n',
        ),
        (
            '    ("both", POS_UNION & POS_MODEL, "pairedBoth", C_UNION, None),\n    ("union only", POS_UNION & ~POS_MODEL, "pairedUnionOnlyEnd", C_UNION, "////"),\n    ("model only", POS_MODEL & ~POS_UNION, "pairedModelOnly", C_MODEL, None),\n    ("neither", ~POS_UNION & ~POS_MODEL, "pairedNeitherEnd", C_MISS, None),\n',
            '    ("both", POS_UNION & POS_MODEL, "matchedPairedBoth", C_UNION, None),\n    ("union only", POS_UNION & ~POS_MODEL, "matchedPairedUnionOnlyEnd", C_UNION, "////"),\n    ("model only", POS_MODEL & ~POS_UNION, "matchedPairedModelOnly", C_MODEL, None),\n    ("neither", ~POS_UNION & ~POS_MODEL, "matchedPairedNeitherEnd", C_MISS, None),\n',
        ),
        (
            '    assert got == N[key], "%s is %d, numbers.json %s is %d" % (name, got, key, N[key])\n    print("  class %-11s %3d == numbers.json %-20s %3d" % (name, got, key, N[key]))',
            '    assert got == N9[key], "%s is %d, round9 %s is %d" % (name, got, key, N9[key])\n    print("  class %-11s %3d == round9 %-20s %3d" % (name, got, key, N9[key]))',
        ),
    ],
)

# ---------------------------------------------------------------- build_figures_extra.py (fig_fields)
BE = os.path.join(PR, "build_figures_extra.py")
patch(
    BE,
    [
        (
            '        label="spectroscopic non-LRD anchors",',
            '        label="comparison objects (not established as LRDs)",',
        ),
        (
            '    fl = [f + (" (none)" if int((sup.field == f).sum()) == 0 else "") for f in fl]',
            '    fl = [f + (" (no support)" if int((sup.field == f).sum()) == 0 else "") for f in fl]',
        ),
    ],
)

# ---------------------------------------------------------------- fig_pipeline.tex
FP = os.path.join(PR, "figures", "fig_pipeline.tex")
patch(
    FP,
    [
        (
            "\\node[tier] at (0, 16.5) {The fix};",
            "\\node[tier] at (0, 16.5) {The learned ranking};",
        ),
        (
            "  (t2c) at (70.0, 16.5) {threshold at the rules' burden (\\modelSelected{} against\n  \\unionSelected{} rows)};",
            "  (t2c) at (70.0, 16.5) {cut at the rules' own row count in each held-out region (\\unionSelected{} rows in all)};",
        ),
        (
            "\\resultbar{16.5}{\\modelRecallEnd}{learned}{accent}{accent}%\n  {\\modelRecallEnd{} recovered}{\\modelMissed{} missed}%\n  {\\modelRuleMissed{} of the \\nRuleMissed{} rule-missed recovered}",
            "\\resultbar{16.5}{\\matchedRecallEnd}{learned}{accent}{accent}%\n  {\\matchedRecallEnd{} recovered}{\\matchedMissedEnd{} missed}%\n  {\\matchedRuleMissed{} of the \\nRuleMissed{} rule-missed recovered}",
        ),
        (
            "  \\candEqualBurden{} at the rules' burden; \\followupN{} follow-up tier; none confirmed};",
            "  \\candEqualBurden{} at the rules' burden; \\followupN{} follow-up tier; none confirmed in the frozen inputs};",
        ),
    ],
)

# ---------------------------------------------------------------- fig_architecture.tex
FA = os.path.join(PR, "figures", "fig_architecture.tex")
patch(
    FA,
    [
        (
            "  {\\bxt{Negative anchors}{40mm}{%\n   spectroscopic non-LRDs\\\\",
            "  {\\bxt{Comparison objects}{40mm}{%\n   spectroscopic, not established as LRDs\\\\",
        ),
        (
            "\\node[alab, text=picked, anchor=west, inner sep=0.3mm] at (140.3,12.4) {\\tBurden{}};",
            "\\node[alab, text=picked, anchor=west, inner sep=0.3mm] at (140.3,12.4) {\\tBurden{} (final fit)};",
        ),
    ],
)

# ---------------------------------------------------------------- build_tables.py
BT = os.path.join(PR, "build_tables.py")
patch(
    BT,
    [
        (
            'JOIN_COLS = ["c_f277w_f444w", "z_phot"]',
            'JOIN_COLS = ["c_f277w_f444w", "z_phot", "source_id"]\n# Round 9: each object table also prints the spectroscopic redshift (published_catalogues_extra)\n# and the within-region rank of the still-missed objects (recovery)\nZSPEC = pd.read_parquet(os.path.join(V, "published_catalogues_extra.parquet"), columns=["source_id", "z_spec"]).set_index("source_id").z_spec\nRRANK = {(r["field"], int(r["id"])): r["rank_pct_region"] for r in json.load(open(os.path.join(ROOT, "recovery", "still_missed_region_rank.json")))}\nB24 = pd.read_parquet(os.path.join(V, "barro24b_flags.parquet")).set_index("source_id").sel_barro24b',
        ),
        (
            '                num("%.2f", r["c_f277w_f444w"]),\n                num("%.2f", r["z_phot"]),\n                lists,\n                num("%.3f", r["score"]),\n            ]',
            '                num("%.2f", r["c_f277w_f444w"]),\n                num("%.2f", ZSPEC.loc[int(r["source_id"])]),\n                num("%.2f", r["z_phot"]),\n                lists,\n                num("%.3f", r["score"]),\n            ]',
        ),
        (
            '                num("%.2f", r["c_f277w_f444w"]),\n                num("%.2f", r["z_phot"]),\n                yesno(r["picked"]),\n                num("%.3f", r["score"]),\n                # score_rank_pct is already 100 x the fraction of the catalogue scoring at\n                # least as high (recovery/v4_evaluate.py), so it is printed as it stands.\n                num("%.2f", r["score_rank_pct"]),\n            ]',
            '                num("%.2f", r["c_f277w_f444w"]),\n                num("%.2f", ZSPEC.loc[int(r["source_id"])]),\n                num("%.2f", r["z_phot"]),\n                yesno(r["picked"]),\n                num("%.3f", r["score"]),\n                # Round 9: the percentile rank is taken within the object\'s own region, since the\n                # five held-out ensembles do not share a score scale.\n                num("%.2f", RRANK[(str(r["field"]), int(r["id"]))]),\n            ]',
        ),
        (
            '                num("%.3f", r["ranking_score"]),\n                STATUS_TEX[r["status"]],\n            ]',
            '                num("%.3f", r["ranking_score"]),\n                yesno(B24.get(int(r["source_id"]), False)),\n                STATUS_TEX[r["status"]],\n            ]',
        ),
    ],
)
print("done")
