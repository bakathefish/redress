# REDRESS: a learned selection for JWST Little Red Dots

This repository holds the code, the released tables, the trained weights and the model card
for a learned selection of little red dots (LRDs) in JWST/NIRCam imaging: a ranker trained on
catalog quantities alone to resemble the spectroscopically confirmed LRDs the community has
published, together with the seven published photometric selections it is measured against,
the scripts that built the paper's numbers, figures and tables, and the frozen spectral-refit
engine used for the archival test. Author: Rudra. The paper is in preparation; nothing here
depends on it, and every number in the model card below is read from the tables in
`recovery/`.

## Model card

**What this is.** A ranking of the 466419 catalog rows inside the model's seven-band support
(all nine JWST/NIRCam fields of the catalog) by how much they resemble the
spectroscopically confirmed little red dots (LRDs) the community has published, trained on
catalog quantities alone (seven NIRCam bands, catalog sizes) with positive-unlabelled
learning, and a candidate list of the highest-ranked rows that no published selection rule
picks and no published list contains.

**Why.** The seven published photometric selections, re-implemented and applied to the same
catalog, together recover 107 of the 151 spectroscopic LRDs in these fields
(70.9%); 44 are missed by all seven.

**Result (out of fold).** Five macroregions (CEERS, GOODS-N, GOODS-S, COSMOS, UDS). In the
out-of-fold evaluation every in-support row is scored by an ensemble trained without its
macroregion, and every threshold was fixed on the other four regions. The thresholds are
fold-specific and trained to target the union's burden; pooled over the five folds the model
selects 1115 rows against the union's 1133:

| selection | rows selected | spectroscopic LRDs recovered (of 151) | of the 44 missed by all rules | strict 32 | strict misses (7) | spectroscopic comparison objects not established as LRDs (anchors) selected (of 4979) |
|---|---|---|---|---|---|---|
| union of seven published rules | 1133 | 107 (70.9%) | 0 | 25 | 0 | 46 (0.92%) |
| this model, thresholds targeting the union burden | 1115 | 123 (81.5%) | 20 | 30 | 5 | 70 (1.41%) |
| this model, thresholds targeting 0.5% of training-region support | 2659 | 133 (88.1%) | 29 | 30 | 5 | 225 (4.52%) |

Wilson 95% interval on the union-burden recall of the 147 in-support positives: [0.769, 0.888].
Per region (model vs union): CEERS 29/33 vs 23, GOODS-N 11/17 vs 10, GOODS-S 15/18 vs 13, COSMOS 23/28 vs 22, UDS 45/51 vs 38.

**What the number is not.** The targeted-negative selection rate is a selection rate on
targeted spectroscopic objects, not a precision or a purity of the catalog. A shuffled-label
control (5 repeats) still recovers 38 to 52 of 147 at the union-targeted burden,
because every labelled object was a spectroscopic target; the model must be read against
that floor and against the union, not against chance. A learned approximation to the union
alone recovers 119 and 17 of the misses: it reproduces a substantial part of the observed
recovery difference, without establishing its cause.

**Candidates.** 1207 rows score above the final 0.5% threshold (2333 of 466419 support
rows), are selected by no rule in any catalog copy, sit in no published list within
0.5 arcsec, and are deduplicated to one highest-ranked row per 0.5-arcsec sky group;
362 of them are in the equal-burden tier (above the final threshold that selects the
union's burden fraction). Status column: untested = no usable archival spectrum within
0.5 arcsec; secure low-z = a grade >= 3 archival redshift at z <= 3; secure z>3 (V untested) =
a grade >= 3 archival redshift above 3 from a grating spectrum the frozen V test cannot use.
A secure redshift is a redshift classification, not an LRD spectral-shape verdict. The
statuses archive consistent / inconclusive / disfavoured / untestable would record the frozen
refit of an eligible archival PRISM spectrum (grade >= 3, z > 3); no candidate has one:
0 eligible spectra, 0 refit verdicts. Equal-burden tier: untested 337, secure z>3 (V untested) 17, secure low-z 8.
No candidate had been spectroscopically established as an LRD when the lists were frozen.
`ranking_score` is a rank statistic, not a probability.

**Catalog versions.** The three reference lists and the archive index were frozen with
version 0.1 of the de Graaff et al. (2026) release (Zenodo record 17665942, 20 November 2025,
116 unique sources). A later version (Zenodo record 21977747, 17 August 2026, 181 spectra of
146 unique sources) holds 3 of the 1207 candidates within 0.5 arcsec: ranks 78, 154 and 176,
at spectroscopic redshifts 4.59, 3.40 and 2.04, all in the equal-burden tier and two in the
follow-up tier, all marked `untested` because the archive index was frozen with the lists.
`recovery/v4_match_degraaff_v2.py` records the match in the columns `in_degraaff26_v2` and
`zspec_degraaff26_v2` of `candidates.csv`; the labels, the model and every count above are
left as frozen. Under the paper's definition the three are spectroscopic LRDs that no rule
selects and the ranking placed in its top 362; three objects chosen by another group's
targeting are not a yield measurement.

**Compactness and the follow-up tier.** The frozen aperture protocol of the paper was run on
the F444W mosaics for 1196 of the 1207 candidates; in the equal-burden tier
189 of 359 measured are compact by the Labbe criterion (f0.4/f0.2 < 1.7).
The follow-up tier (126 sources) is a post-hoc cut stated as such: equal-burden tier, no
archival verdict against the source, inside the training positives' feature range, compact,
catalog photo-z >= 3. It is a prioritisation for follow-up, not a validation; `followup_tier`
is a column, not a claim.

**What the archive says (post hoc and selection-influenced).** Counting sources with a
secure (grade >= 3) archival redshift within 0.5 arcsec, with the size of each set: model
candidates in the equal-burden tier 17 of 362 at z > 3 and 8 of 362 at z <= 3;
follow-up tier 8 of 126 and 0 of 126; the seven rules' own novel picks (in no list,
unlabelled) 13 of 688 and 19 of 688. Catalog photo-z below 3: 21.5% of the equal-burden
tier (78 of 362), 40.8% of the rules' novel picks, 15 of 147 of the known LRDs.
These are rates on whatever the archive happened to observe; the observed subsets were
targeted, not drawn at random, so no purity can be inferred from them. The same quantities
were inspected for a variant model during model selection (next paragraph), so this
comparison is not an independent test of the deployed model; an independent archival test
needs observations that were not consulted here.

**Model selection.** Challengers were held to a pre-stated rule: promoted only with at least three more positives (or two more rule-missed) at the union-targeted burden, in at least four of five regions, without raising the targeted-negative selection rate by more than 0.3 points. Out of fold (recovered of 147, rule-missed of 41, comparison objects not established as LRDs (anchors) selected of 4,979): trees on the 30-feature set of the plan of record 123, 20, 70; trees on a 34-feature set (adding F444W and F200W aperture concentrations, log r90/r20 and axis ratio) 124, 20, 48, regions won 3 of 5, so one more positive and no more rule-missed: fails the rule; nnPU multilayer perceptron (30 features, 15 networks) 123, 25, 65, regions won 2 of 5: fails the rule; rank-average blend of trees and MLP 125, 24, 61, regions won 3 of 5: fails the rule. TabPFN v2 and TabICL did not complete within the time budget. The 30-feature model stays deployed because no challenger met the rule. Two further comparisons were made after that and are exploratory, not part of the rule: the 34-feature set's lower targeted-negative rate (which counts only z > 3 targets), and the photo-z and archival-redshift make-up of the two candidate lists (34-feature equal-burden tier: catalog photo-z < 3 for 34.9% against 21.5%; secure archival z <= 3 for 21 of 390 against 8 of 362; secure z > 3 for 15 of 390 against 17 of 362). During the build the 34-feature model was briefly chosen on the first of these and reverted on the second; that sequence is recorded in `recovery/model_selection.json`, and because the archive comparison was consulted in it, the archive paragraph above is post hoc and selection-influenced, not an independent test. The 34-feature out-of-fold tables are in `recovery/evaluation_v34.md`.

**Model.** 48 bagged LightGBM classifiers (7 leaves, depth 3, 350 trees, learning rate 0.025),
each bag = all positives + 20 unlabelled rows per positive (half uniform, half matched in
region, F444W bin and S/N quartile) + the spectroscopic comparison objects not established as LRDs (anchors) at total
weight 0.1 of the positives; settings chosen by an inner one-standard-error rule.
Features (30): m_f090w, m_f115w, m_f150w, m_f200w, m_f277w, m_f356w, m_f444w, asnr_f090w, asnr_f115w, asnr_f150w, asnr_f200w, asnr_f277w, asnr_f356w, asnr_f444w, c_f090w_f115w, c_f115w_f150w, c_f150w_f200w, c_f200w_f277w, c_f277w_f356w, c_f356w_f444w, c_f115w_f200w, c_f200w_f356w, c_f277w_f444w, slope_blue, slope_red, v_curv, log_rh, log_rh_over_rstar, n_snr_ge1, n_snr_ge3.
Weights: `recovery/models/bag_*.txt` (sha256 of the concatenation dd03f0d8467e37c0...).

**Provenance and reproduction.** Positives: Hviding et al. 2025 (RUBIES, table A1), Barro
et al. 2025, de Graaff et al. 2026, plus the 32 strict compact V-shaped sources of the
census; anchors: census sources with a secure z > 3 PRISM spectrum, not V-shaped, in no LRD
list. Photometry and sizes: DJA grizli v7 catalogs. Archive: DJA msaexp v4.4. The scripts in
`recovery/` are copies of the ones that ran; `RUN_MANIFEST.md` gives the execution order,
the inputs that are not in this repository, the environment (`requirements.txt`) and the script
that produced each artifact. `recovery/evaluation.md` carries every out-of-fold table. All dates in
this repository are IST (UTC+5:30); the run spanned 2026-09-03 21:00 to 2026-09-04 00:30 UTC.

## How to run

Python 3.13.3 with the pins in `requirements.txt`:

```
python -m venv .venv
.venv/Scripts/activate          # on Linux or macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

`RUN_MANIFEST.md` gives the execution order in full, including which inputs are not released
and which machine each step ran on. In short, from the repository root:

```
python recovery/build_labels.py            # 1  label table
python recovery/v4_features.py             # 2  30-feature table, folds, denominators
python recovery/v4_train_lgbm.py           # 3  out-of-fold scores (uses v4lib.py)
python recovery/vm_extract_shape.py        # 4  catalog shape columns (staging VM)
python recovery/v4_features_v34.py         # 5  34-feature variant table
python recovery/v4_train_variant.py _v34   # 6  34-feature out-of-fold scores
python recovery/v4_mlp.py                  # 7  nnPU multilayer perceptron challenger
python recovery/v4_tabpfn.py               # 8  TabPFN and TabICL challengers (not completed)
python recovery/v4_tabicl.py
python recovery/v4_evaluate.py             # 9  evaluation tables (also: v4_evaluate.py _v34)
python recovery/v4_final.py                # 10 final fit, models/bag_00..47.txt
python recovery/v4_mlp_final.py            # 11 not promoted
python recovery/v4_blend.py                # 11 not promoted
python recovery/v4_candidates.py           # 12 candidate list, ledger, archive records
python recovery/v4_refit_archive.py        # 13 archival refit (zero eligible spectra)
python recovery/vm_compactness.py candidates.csv compactness.csv   # 14 (staging VM)
python recovery/v4_merge_compactness.py    # 15 compactness columns, follow-up tier
python recovery/v4_archive_purity.py       # 16 archival rates
python recovery/v4_release.py              # 17 stages the released directory
```

The paper chain runs after that, from the repository root:

```
python paper/build_numbers.py        # every number the paper quotes -> numbers.json, numbers.tex
python paper/fetch_stamps.py         # three-band cutouts for the gallery figures
python paper/fetch_stamps_6band.py   # the other four bands for the same objects
python paper/build_figures.py        # twelve of the paper's sixteen figures
python paper/build_figures_extra.py  # the other four
python paper/build_tables.py         # the three appendix tables
```

The manuscript itself is not released, so nothing here typesets it. What is released is
the chain that produces every number, figure and table it quotes, so any one of them can
be audited by rerunning the script that made it and diffing the output against the tables
in `recovery/`.

The unit tests of the selection modules need `pytest` and `hypothesis` in addition to
`requirements.txt`, and run from the repository root:

```
python -m pytest selections -q
```

## Layout

```
recovery/       the released run: every script that ran, the 48 LightGBM weight files
                under models/, and every table of record (.md, .json, .csv, .parquet)
paper/          the scripts that turn the tables of record into the paper's numbers,
                figures and appendix tables, plus numbers.json and numbers.tex
selections/     the seven published photometric selections re-implemented, the Barro et al.
                (2024b) comparator added in revision, the shared fail-closed comparators,
                the compactness, slope-fitting and fold modules (package redress), the frozen
                spectral-refit engine (spectra/), and the unit tests that pin all of them
tools/          the cleanliness scan run over this tree before publication
CITATION.cff    how to cite this repository
```

Some things are named in the code but are not files in this repository. Module docstrings
cite the author's frozen protocol documents (the data contracts, the compactness
measurement protocol, the build plan) to explain why a definition or a threshold is what
it is; those documents are the author's working records and are not released. The run
manifest lists the data files that are likewise not released, and where they belong if
you want to rerun the chain.

## Added in the revision of 2026-09-06

Four things a reader of the first release will not find in it. `recovery/labelled_rows.csv`
carries every labelled or ambiguous row with its label, list memberships, the seven rule
flags of record, its region and sky group and its out-of-fold score and outcomes, so every
per-rule and paired count can be recomputed from released files. `recovery/rule_flags.parquet`
carries the seven flags of record for all 630869 catalog rows. `recovery/candidates.csv`
carries, next to the in-sample ranking score, the out-of-fold score of the fold model that
never saw the candidate's region and that fold's thresholds (`oof_*` columns); a reader who
prefers the validated ranking can define the tiers on those. And the Barro et al. (2024b)
two-color selection with its aperture compactness is coded in
`selections/redress/cuts/barro24b.py`, measured in `recovery/v4_barro24b_apertures.py` on DJA
F444W cutouts, and reported as a separate eighth comparator (`redress.cuts.COMPARATORS`); it
is not one of the seven of record and no union or burden above includes it. The counts of all
of this are in `recovery/referee_compute_2026_09_06.json`, and `RUN_MANIFEST.md` lists the
files with their sha256.

**Second revision of the same day (steps 24 to 26 of the manifest).** `recovery/label_provenance.csv`
is now the one table of record for every label: one row per labelled or ambiguous source
with its identifiers, sky group, region, every list membership, the spectral eligibility,
evaluability and verdict at the source and at the row level, the final class, the
ambiguity reason and the spectroscopic redshift with its source. `recovery/test_status_reconciliation.csv`
lists every row where the two earlier test-status files (`anchor_test_status.csv`,
`positive_test_status.csv`) differ and records that none of the differences moves a label
or a printed count; those two files are kept for the record and superseded.
`recovery/rule_audit_positives.csv` gives each rule's outcome on each of the 151 positives
by criterion, `recovery/positive_match_alternatives.csv` the alternative catalog rows of the
multi-row positives, and `recovery/positive_apertures.csv` the aperture compactness of every
positive measured with the candidates' protocol. `recovery/candidates.csv` gains the eighth
selection's flag (`sel_barro24b`), a five-way `spectrum_status`, the secure archival
redshift where one exists (`z_spec_secure`) and a post-freeze list flag
(`listed_after_freeze`). The paper's primary equal-cost comparison is now the
region-matched one: each held-out region cut at the union's own row count there
(`recovery/positive_outcomes.csv` carries that outcome per positive, and
`recovery/regional_budget_curve.json` the curve). The counts are in
`recovery/round9_compute_2026_09_06.json`.

## Errata and record notes

Facts about the released files that a reader reproducing from them needs and that the model
card above does not carry.

**`recovery/build_labels.py`'s docstring and comment are wrong about Hviding table B1.** The
docstring says a positive is a source in "Hviding 2025 A1 or B1, Barro 2025 or de Graaff
2026". The code does not use B1 that way: `in_spec_list` is `in_hviding25_A1 | in_barro25 |
in_degraaff26`, so B1 never makes a positive. The comment beside the negative mask says B1
members "are excluded from the negatives"; that is also not what the code does. The negative
mask is `neg = notv & ~in_any_list`, and table B1 is not in `in_any_list`, so a B1 member that
the spectral test does not establish as V-shaped carries `y = 0` and is an anchor in training
and in every anchor rate: 19 sources, 16 of them with complete seven-band photometry, 2 of them
selected by the union of the rules. The `ambiguous` flag is set on the same rows, but that flag
only removes a row from the unlabelled pool; it does not mask `y`. `label_counts.json` counts
these 19 in `negatives` (5397) and, with the 48 photometric-list sources that are ambiguous and
unlabelled, in `ambiguous_excluded` (67). Every count in this repository is read from the columns
the code writes, so no number is affected; the labelled-row table `recovery/labelled_rows.csv`
carries `y`, `ambiguous` and `in_hviding25_B1` per row so the treatment can be checked.

**`recovery/archive_purity.json` does not report purity.** Every rate in it is a selection
rate on targeted spectroscopic objects: the share of a candidate set that the archive
happens to have observed and given a secure redshift. Those observed subsets were targeted,
not drawn at random, so nothing in the file bounds a purity. The file name is historical and
was kept so the released table matches the run that wrote it; the file's own `note` field
says the same thing.

**One `kocevski24` flag sits outside the module's redshift table.** The record carries
`sel_kocevski24` true for primer-uds-north 3592, whose released `z_phot` is 1.3043; the
module assigns a band set only for `z_phot` > 2 and cannot select a row with no bin. The row
is unlabelled, is not a positive, and moves no printed number, but rerunning the module on
the released quantities does not reproduce this one flag. The cause is not established:
the row is a singleton sky group, and the redshift the selection run used for it is not in
any released table.

**Two test-flag files that do not agree row for row.** `positive_test_status.csv` carries
`spec_tested` and `spec_vshaped` for the 151 positives; `anchor_test_status.csv` carries
`elig`, `testable` and `vshaped_spec` for the positives and the anchors. They record the
spectral V-shape test at different stages and disagree on 490 anchor rows and on one
positive, primer-cosmos-east 38011. Every count for the positives is read from
`positive_test_status.csv` and every count for the anchors from `anchor_test_status.csv`.

**The seven-band rerun of the rule modules is not the record.** Running the released rule
modules on a seven-band table (F090W to F444W only) reproduces the stored flags on every row
except those where kocevski24's criteria touch F410M (vacuous without the band, so the rerun
is permissive) or the ACS bands below z = 4.75 (fails closed without them, so the rerun is
restrictive), plus one kokorev24 aperture-ratio boundary case (1.695458 against the 1.7
cut). The record was built from the full DJA band set: NIRCam, MIRI, ACS and WFC3 where
present. No positive changes its union verdict under the rerun; the union burden would read
1145 rather than 1133 and the union's anchors 57 rather than 46. The record stands.

**There is no combined gallery.** The paper's gallery is two figures, `fig_gallery_known`
and `fig_gallery_cand`, and `paper/build_figures.py` no longer writes the older combined
`fig_gallery` canvas. A `fig_gallery` file left over from an earlier run is stale.

**Two of the sixteen figures are TikZ standalones.** `fig_pipeline` and `fig_architecture`
are compiled with `pdflatex` from `paper/figures/fig_pipeline.tex` and
`paper/figures/fig_architecture.tex`, which are part of the manuscript and are not released.
The other fourteen figures are drawn from the tables in `recovery/` alone.

## Citation

Paper in preparation, 2026. Please cite the repository until it appears; `CITATION.cff` carries
the author and title fields and will carry the arXiv identifier once the manuscript is posted.
