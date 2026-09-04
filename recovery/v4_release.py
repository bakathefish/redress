"""v4 build step 11: stage the recovery classifier release into the public clone.

What goes public (owner's order: pipeline, tables, classifier code and weights, candidate
catalog, model card; no internal notes):
  recovery/README.md            the model card (numbers read from the JSON of record)
  recovery/RUN_MANIFEST.md      execution order, inputs, environment, artifact -> producer
  recovery/requirements.txt     the pinned environment the run used
  recovery/*.py                 the scripts as run (labels, features, training, evaluation,
                                final fit, candidates, archive refit, compactness, purity)
  recovery/models/bag_*.txt     the 48 LightGBM boosters of the final ensemble
  recovery/*.json, *.csv, *.md  every table of record, including the variant runs
Nothing is committed here; the script stages files and prints the prohibited-term scan.
Usage: python v4_release.py [suffix]
"""

import importlib
import json
import os
import re
import shutil
import sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "recovery"
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
REL = os.path.join("release", "recovery")
os.makedirs(os.path.join(REL, "models"), exist_ok=True)


def load(name, required=False):
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        return json.load(open(p))
    if required:
        raise FileNotFoundError(p)
    return {}


fin = load(f"final{SUF}.json", True)
ev = load(f"evaluation{SUF}.json", True)
den = load("denominators.json", True)
cc = load(f"candidate_counts{SUF}.json", True)
ar = load(f"archive_refit_counts{SUF}.json")
man = load(f"feature_manifest{SUF}.json", True)
cm = load(f"compactness_counts{SUF}.json")
ap = load(f"archive_purity{SUF}.json")
ms = load(f"model_selection{SUF}.json")

# the scripts that ran, in execution order (see RUN_MANIFEST.md)
scripts = [
    "build_labels.py",
    "v4_features.py",
    "v4lib.py",
    "v4_train_lgbm.py",
    "vm_extract_shape.py",
    "v4_features_v34.py",
    "v4_train_variant.py",
    "v4_mlp.py",
    "v4_tabpfn.py",
    "v4_tabicl.py",
    "v4_evaluate.py",
    "v4_final.py",
    "v4_mlp_final.py",
    "v4_blend.py",
    "v4_candidates.py",
    "v4_refit_archive.py",
    "vm_compactness.py",
    "v4_merge_compactness.py",
    "v4_archive_purity.py",
    "v4_release.py",
]
for s_ in scripts:
    if os.path.exists(os.path.join(OUT, s_)):
        shutil.copy(os.path.join(OUT, s_), REL)
tables = {
    f"feature_manifest{SUF}.json": "feature_manifest.json",
    "denominators.json": "denominators.json",
    f"final{SUF}.json": "final.json",
    f"evaluation{SUF}.json": "evaluation.json",
    f"evaluation{SUF}.md": "evaluation.md",
    f"train_results{SUF}.json": "train_results.json",
    f"candidates{SUF}.csv": "candidates.csv",
    f"candidates_ledger{SUF}.csv": "candidates_ledger.csv",
    f"candidate_counts{SUF}.json": "candidate_counts.json",
    f"archive_refits{SUF}.csv": "archive_refits.csv",
    f"archive_refit_counts{SUF}.json": "archive_refit_counts.json",
    f"compactness{SUF}.csv": "compactness.csv",
    f"compactness_counts{SUF}.json": "compactness_counts.json",
    f"archive_purity{SUF}.json": "archive_purity.json",
    f"candidate_archive_records{SUF}.csv": "candidate_archive_records.csv",
    f"model_selection{SUF}.json": "model_selection.json",
    "evaluation_v34.md": "evaluation_v34.md",
    "evaluation_v34.json": "evaluation_v34.json",
    "train_results_v34.json": "train_results_v34.json",
    "feature_manifest_v34.json": "feature_manifest_v34.json",
    "mlp_results.json": "mlp_results.json",
    "label_audit.csv": "label_audit.csv",
}
for src, dst in tables.items():
    if os.path.exists(os.path.join(OUT, src)):
        shutil.copy(os.path.join(OUT, src), os.path.join(REL, dst))
for f in os.listdir(fin["models_dir"]):
    shutil.copy(os.path.join(fin["models_dir"], f), os.path.join(REL, "models", f))
# the released copy of final.json names the weights by their public path
fin_rel = dict(fin)
fin_rel["models_dir"] = "models"
fin_rel["models_dir_at_run_time"] = fin["models_dir"]
json.dump(fin_rel, open(os.path.join(REL, "final.json"), "w"), indent=1)

# pinned environment of the run
pins = []
for mod, pipname in (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("sklearn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
    ("joblib", "joblib"),
    ("torch", "torch"),
    ("astropy", "astropy"),
    ("photutils", "photutils"),
    ("tabpfn", "tabpfn"),
):
    try:
        v = importlib.import_module(mod).__version__
        pins.append(f"{pipname}=={v.split('+')[0]}")
    except Exception:  # noqa: BLE001
        pins.append(f"# {pipname}: not importable on the release machine")
open(os.path.join(REL, "requirements.txt"), "w", encoding="utf-8").write(
    "# Environment of the v4 run (Python %s). torch was the CPU build; tabpfn was used with the\n"
    "# open Prior-Labs/TabPFN-v2-clf checkpoint and did not finish within the time budget.\n"
    % sys.version.split()[0]
    + "\n".join(pins)
    + "\n"
)

u, p, h = ev["union"], ev["primary_equal_burden"], ev["primary_0p5pct"]
shuf = ev["sanity"].get("shuffled_labels_recall_at_burden", [])
bl = ev.get("baselines", {})
n_sup = den["support_rows"] if "support_rows" in den else ev.get("support_rows", "?")
rr = bl.get("rules_reproduction", {})
REGIONS = list(p["per_region_recall"].keys())


def wins(a, b):
    """regions in which selection a recovers more positives than selection b"""
    return sum(
        int(a["per_region_recall"][R].split("/")[0])
        > int(b["per_region_recall"][R].split("/")[0])
        for R in REGIONS
    )


def _sel_text(ms):
    if not ms:
        return "No challenger record for this run."
    t30, t34, m, b = (
        ms["trees_30_features"],
        ms["trees_34_features"],
        ms["mlp_30_features"],
        ms["blend_trees30_mlp"],
    )
    e30, e34 = ms["external_check_trees_30"], ms["external_check_trees_34"]
    w34 = wins(t34, t30)
    return (
        "Challengers were held to a pre-stated rule: promoted only with at least three more "
        "positives (or two more rule-missed) at the union-targeted burden, in at least four of "
        "five regions, without raising the targeted-negative selection rate by more than 0.3 "
        "points. Out of fold (recovered of 147, rule-missed of 41, targeted non-LRD anchors "
        f"selected of 4,979): trees on the 30-feature set of the plan of record {t30['recall_147']}, "
        f"{t30['rule_missed_41']}, {t30['neg_selected']}; trees on a 34-feature set (adding F444W and "
        f"F200W aperture concentrations, log r90/r20 and axis ratio) {t34['recall_147']}, "
        f"{t34['rule_missed_41']}, {t34['neg_selected']}, regions won {w34} of 5, so one more positive and "
        "no more rule-missed: fails the rule; nnPU multilayer perceptron (30 features, 15 networks) "
        f"{m['recall_147']}, {m['rule_missed_41']}, {m['neg_selected']}, regions won "
        f"{ms['mlp_promotion_vs_trees30']['regions_won']} of 5: fails the rule; rank-average blend of "
        f"trees and MLP {b['recall_147']}, {b['rule_missed_41']}, {b['neg_selected']}, regions won "
        f"{ms['blend_promotion_vs_trees30']['regions_won']} of 5: fails the rule. TabPFN v2 and TabICL "
        "did not complete within the time budget. The 30-feature model stays deployed because no "
        "challenger met the rule. Two further comparisons were made after that and are exploratory, "
        "not part of the rule: the 34-feature set's lower targeted-negative rate (which counts only "
        "z > 3 targets), and the photo-z and archival-redshift make-up of the two candidate lists "
        f"(34-feature equal-burden tier: catalog photo-z < 3 for {e34['zphot_lt_3_pct']}% against "
        f"{e30['zphot_lt_3_pct']}%; secure archival z <= 3 for {e34['secure_z_le_3']} of "
        f"{e34['equal_burden_candidates']} against {e30['secure_z_le_3']} of {e30['equal_burden_candidates']}; "
        f"secure z > 3 for {e34['secure_z_gt_3']} of {e34['equal_burden_candidates']} against "
        f"{e30['secure_z_gt_3']} of {e30['equal_burden_candidates']}). During the build the 34-feature "
        "model was briefly chosen on the first of these and reverted on the second; that sequence is "
        "recorded in `model_selection.json`, and because the archive comparison was consulted in it, "
        "the archive paragraph above is post hoc and selection-influenced, not an independent test. "
        "The 34-feature out-of-fold tables are in `evaluation_v34.md`."
    )


sel_text = _sel_text(ms)
eb = ap.get("model_candidates_equal_burden", {})
fu = ap.get("model_candidates_followup", {}) or {}
un = ap.get("union_novel_picks", {})
st_eb = ar.get("status_counts_equal_burden", cc.get("status_counts_equal_burden", {}))
st_line = ", ".join(f"{k} {v}" for k, v in st_eb.items())
per_region = ", ".join(
    f"{R} {p['per_region_recall'][R]} vs {u['per_region_recall'][R].split('/')[0]}"
    for R in REGIONS
)
features = ", ".join(man["features"])
card = f"""# recovery classifier (v4): a learned selection for little red dots

**What this is.** A ranking of the {n_sup} catalog rows inside the model's seven-band support
(all nine JWST/NIRCam fields of the catalog) by how much they resemble the
spectroscopically confirmed little red dots (LRDs) the community has published, trained on
catalog quantities alone (seven NIRCam bands, catalog sizes) with positive-unlabelled
learning, and a candidate list of the highest-ranked rows that no published selection rule
picks and no published list contains.

**Why.** The seven published photometric selections, re-implemented and applied to the same
catalog, together recover {den["union_selects_of_151"]} of the {den["positives_151"]} spectroscopic LRDs in these fields
({100 * den["union_selects_of_151"] / den["positives_151"]:.1f}%); {den["rule_missed_44"]} are missed by all seven.

**Result (out of fold).** Five macroregions (CEERS, GOODS-N, GOODS-S, COSMOS, UDS). In the
out-of-fold evaluation every in-support row is scored by an ensemble trained without its
macroregion, and every threshold was fixed on the other four regions. The thresholds are
fold-specific and trained to target the union's burden; pooled over the five folds the model
selects {p["selected_total"]} rows against the union's {u["selected_total"]}:

| selection | rows selected | spectroscopic LRDs recovered (of {den["positives_151"]}) | of the {den["rule_missed_44"]} missed by all rules | strict 32 | strict misses (7) | targeted spectroscopic non-LRD anchors selected (of {u["neg_total"]}) |
|---|---|---|---|---|---|---|
| union of seven published rules | {u["selected_total"]} | {u["recall_151_end_to_end"]} ({u["recall_151_pct"]}%) | 0 | {u["strict_in_support"]} | 0 | {u["neg_selected"]} ({u["neg_selection_rate_pct"]}%) |
| this model, thresholds targeting the union burden | {p["selected_total"]} | {p["recall_151_end_to_end"]} ({p["recall_151_pct"]}%) | {p["rule_missed_41"]} | {p["strict_in_support"]} | {p["strict_missed"]} | {p["neg_selected"]} ({p["neg_selection_rate_pct"]}%) |
| this model, thresholds targeting 0.5% of training-region support | {h["selected_total"]} | {h["recall_151_end_to_end"]} ({h["recall_151_pct"]}%) | {h["rule_missed_41"]} | {h["strict_in_support"]} | {h["strict_missed"]} | {h["neg_selected"]} ({h["neg_selection_rate_pct"]}%) |

Wilson 95% interval on the union-burden recall of the {den["positives_in_support_147"]} in-support positives: {p["recall_147_wilson"]}.
Per region (model vs union): {per_region}.

**What the number is not.** The targeted-negative selection rate is a selection rate on
targeted spectroscopic objects, not a precision or a purity of the catalog. A shuffled-label
control ({len(shuf)} repeats) still recovers {min(shuf) if shuf else "?"} to {max(shuf) if shuf else "?"} of 147 at the union-targeted burden,
because every labelled object was a spectroscopic target; the model must be read against
that floor and against the union, not against chance. A learned approximation to the union
alone recovers {rr.get("recall_at_burden", "?")} and {rr.get("rule_missed_recovered", "?")} of the misses: it reproduces a substantial part of the observed
recovery difference, without establishing its cause.

**Candidates.** {cc["candidates_total"]} rows score above the final 0.5% threshold ({fin["n_above_05"]} of {n_sup} support
rows), are selected by no rule in any catalog copy, sit in no published list within
0.5 arcsec, and are deduplicated to one highest-ranked row per 0.5-arcsec sky group;
{cc["candidates_equal_burden_tier"]} of them are in the equal-burden tier (above the final threshold that selects the
union's burden fraction). Status column: untested = no usable archival spectrum within
0.5 arcsec; secure low-z = a grade >= 3 archival redshift at z <= 3; secure z>3 (V untested) =
a grade >= 3 archival redshift above 3 from a grating spectrum the frozen V test cannot use.
A secure redshift is a redshift classification, not an LRD spectral-shape verdict. The
statuses archive consistent / inconclusive / disfavoured / untestable would record the frozen
refit of an eligible archival PRISM spectrum (grade >= 3, z > 3); no candidate has one:
{ar.get("eligible_spectra", 0)} eligible spectra, {len(ar.get("object_verdicts", {}))} refit verdicts. Equal-burden tier: {st_line}.
No candidate has been spectroscopically established as an LRD. `ranking_score` is a rank
statistic, not a probability.

**Compactness and the follow-up tier.** The frozen aperture protocol of the paper was run on
the F444W mosaics for {cm.get("measured", "?")} of the {cm.get("candidates", "?")} candidates; in the equal-burden tier
{cm.get("compact_labbe_equal_burden", "?")} of {cm.get("measured_equal_burden", "?")} measured are compact by the Labbe criterion (f0.4/f0.2 < 1.7).
The follow-up tier ({cm.get("followup_tier", "?")} sources) is a post-hoc cut stated as such: equal-burden tier, no
archival verdict against the source, inside the training positives' feature range, compact,
catalog photo-z >= 3. It is a prioritisation for follow-up, not a validation; `followup_tier`
is a column, not a claim.

**What the archive says (post hoc and selection-influenced).** Counting sources with a
secure (grade >= 3) archival redshift within 0.5 arcsec, with the size of each set: model
candidates in the equal-burden tier {eb.get("secure_z_gt_3", "?")} of {eb.get("n", "?")} at z > 3 and {eb.get("secure_z_le_3", "?")} of {eb.get("n", "?")} at z <= 3;
follow-up tier {fu.get("secure_z_gt_3", "?")} of {fu.get("n", "?")} and {fu.get("secure_z_le_3", "?")} of {fu.get("n", "?")}; the seven rules' own novel picks (in no list,
unlabelled) {un.get("secure_z_gt_3", "?")} of {un.get("n", "?")} and {un.get("secure_z_le_3", "?")} of {un.get("n", "?")}. Catalog photo-z below 3: {eb.get("zphot_lt_3_pct", "?")}% of the equal-burden
tier ({eb.get("zphot_lt_3", "?")} of {eb.get("n", "?")}), {un.get("zphot_lt_3_pct", "?")}% of the rules' novel picks, {ap.get("known_positives_zphot_lt_3", "?")} of the known LRDs.
These are rates on whatever the archive happened to observe; the observed subsets were
targeted, not drawn at random, so no purity can be inferred from them. The same quantities
were inspected for a variant model during model selection (next paragraph), so this
comparison is not an independent test of the deployed model; an independent archival test
needs observations that were not consulted here.

**Model selection.** {sel_text}

**Model.** {fin["n_bags"]} bagged LightGBM classifiers (7 leaves, depth 3, {fin["chosen_trees"]} trees, learning rate 0.025),
each bag = all positives + 20 unlabelled rows per positive (half uniform, half matched in
region, F444W bin and S/N quartile) + the targeted spectroscopic non-LRD anchors at total
weight {fin["chosen_anchor"]} of the positives; settings chosen by an inner one-standard-error rule.
Features ({man["n_features"]}): {features}.
Weights: `models/bag_*.txt` (sha256 of the concatenation {fin["models_sha256"][:16]}...).

**Provenance and reproduction.** Positives: Hviding et al. 2025 (RUBIES, table A1), Barro
et al. 2025, de Graaff et al. 2026, plus the 32 strict compact V-shaped sources of the
census; anchors: census sources with a secure z > 3 PRISM spectrum, not V-shaped, in no LRD
list. Photometry and sizes: DJA grizli v7 catalogs. Archive: DJA msaexp v4.4. The scripts in
this directory are copies of the ones that ran; `RUN_MANIFEST.md` gives the execution order,
the inputs that are not in this directory, the environment (`requirements.txt`) and the script
that produced each artifact. `evaluation.md` carries every out-of-fold table. All dates in this
directory are IST (UTC+5:30); the run spanned 2026-09-03 21:00 to 2026-09-04 00:30 UTC.
"""
open(os.path.join(REL, "README.md"), "w", encoding="utf-8").write(card)

manifest = f"""# Run manifest: recovery classifier v4

Every script here ran from `<build repository>/recovery/` with the working directory two
levels above (each script sets it from its own location) and read and wrote `recovery/`.
To rerun, place this directory's scripts under `<repo>/recovery/` next to the inputs listed
below. Times are IST (UTC+5:30) on 2026-09-04; in UTC the run spanned 2026-09-03 21:00 to
2026-09-04 00:30. Environment: Python {sys.version.split()[0]} with `requirements.txt`.
Compute: a Windows 11 laptop for the label table, features, the primary training run, the
final fit, the candidate chain and this release; Azure VM the burst VM (Standard_D16as_v5,
Ubuntu, 16 vCPU) for the 34-feature variant, the nnPU MLP and the TabPFN attempt; Azure VM
the staging VM for the DJA shape-column extraction and the frozen compactness run (it holds
the DJA catalog staging and the grizli v7 F444W mosaics).

## Inputs not in this directory

- the DJA grizli v7 photometric catalogs (`*fix_phot_apcorr*`) for the nine fields, staged on
  the staging VM and joined by (field, id);
- the DJA msaexp v4.4 emission-line table (`dja_v4.4_zenodo_dja_msaexp_emission_lines_v4.4.csv.gz`);
- the grizli v7 F444W science and weight mosaics (`https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/`);
- the published LRD lists (Hviding et al. 2025 table A1 and B1, Barro et al. 2025, de Graaff
  et al. 2026) and the census tables of this repository (`results/catalog/`);
- `labels.parquet`, the label table built by `build_labels.py` from the above (not released;
  the per-source audit of it is `label_audit.csv`).

## Execution order and artifacts

| step | script (arguments) | where | produces |
|---|---|---|---|
| 1 | `build_labels.py` | laptop | `labels.parquet` (not released) |
| 2 | `v4_features.py` | laptop | `features.parquet` (not released), `feature_manifest.json`, `denominators.json`, `label_audit.csv` |
| 3 | `v4_train_lgbm.py` (uses `v4lib.py`) | laptop | `oof_scores.parquet` (not released), `train_results.json`, `baseline_scores.parquet` (not released) |
| 4 | `vm_extract_shape.py` | the staging VM | `v4_shape_columns.csv.gz` (not released) |
| 5 | `v4_features_v34.py` | laptop | `features_v34.parquet` (not released), `feature_manifest_v34.json` |
| 6 | `v4_train_variant.py _v34` | the burst VM | `oof_scores_v34.parquet` (not released), `train_results_v34.json` |
| 7 | `v4_mlp.py` | the burst VM | `mlp_oof_scores.parquet` (not released), `mlp_results.json` |
| 8 | `v4_tabpfn.py`, `v4_tabicl.py` | the burst VM, laptop | stopped before any fold finished; no artifact |
| 9 | `v4_evaluate.py` and `v4_evaluate.py _v34` | laptop | `evaluation.json`, `evaluation.md`, `evaluation_v34.json`, `evaluation_v34.md` |
| 10 | `v4_final.py` | laptop | `final.json`, `models/bag_00..47.txt`, `scores.parquet` (not released; every in-support row's ranking score, reproducible from the weights) |
| 11 | `v4_mlp_final.py`, `v4_blend.py` | not run for the release | the MLP and the blend were not promoted |
| 12 | `v4_candidates.py` | laptop | `candidates.csv`, `candidates_ledger.csv`, `candidate_archive_records.csv`, `candidate_counts.json` |
| 13 | `v4_refit_archive.py` | laptop | `archive_refits.csv`, `archive_refit_counts.json` (zero eligible spectra) |
| 14 | `vm_compactness.py candidates.csv compactness.csv` | the staging VM | `compactness.csv` |
| 15 | `v4_merge_compactness.py` | laptop | `candidates.csv` (compactness columns, `followup_tier`), `compactness_counts.json` |
| 16 | `v4_archive_purity.py` | laptop | `archive_purity.json` |
| 17 | `v4_release.py` | laptop | this directory, `README.md`, this manifest, `requirements.txt`, the released `final.json` (`models_dir` set to `models`) |

`model_selection.json` was written by hand-run Python from the evaluation files during the
build and records the challenger comparison and the decision sequence.
"""
open(os.path.join(REL, "RUN_MANIFEST.md"), "w", encoding="utf-8").write(manifest)

# prohibited vocabulary scan over the release directory text files
bad = re.compile(
    # scan-allow: this line IS the vocabulary guard, so it has to name the words it bans
    r"\bp_lrd\b|\bconfirmed LRD candidates?\b|\bdiscover(?:y|ed|ies)\b|(?<!not a )(?<!never )\bprecision\b(?! within| on the targeted)",  # scan-allow: vocabulary guard
    re.I,
)
hits = []
for root, _, files in os.walk(REL):
    for f in files:
        if f.endswith((".md", ".py", ".csv", ".json", ".txt")) and not f.startswith(
            "bag_"
        ):
            txt = (
                open(os.path.join(root, f), encoding="utf-8", errors="ignore")
                .read()
                .replace("\n", " ")
            )
            for m in bad.finditer(txt):
                hits.append(
                    (f, txt[max(0, m.start() - 40) : m.end() + 40].replace("\n", " "))
                )
print("release staged in", REL)
print("prohibited-term hits:", len(hits))
for hh in hits[:20]:
    print("  ", hh)
