# Run manifest: the recovery classifier

Every script in `recovery/` sets its working directory to the repository root from its own
location, and reads and writes `recovery/`. The inputs that are not released are expected
under `inputs/` at the repository root; the paths in the scripts point there. Times are IST
(UTC+5:30) on 2026-09-04; in UTC the run spanned 2026-09-03 21:00 to 2026-09-04 00:30.
Environment: Python 3.13.3 with `requirements.txt`.
Compute: a Windows 11 laptop for the label table, features, the primary training run, the
final fit, the candidate chain and this release; a burst VM (Standard_D16as_v5,
Ubuntu, 16 vCPU) for the 34-feature variant, the nnPU MLP and the TabPFN attempt; a staging
VM for the DJA shape-column extraction and the frozen compactness run (it holds
the DJA catalog staging and the grizli v7 F444W mosaics).

## Inputs not in this repository

Place these under `inputs/` (the `prior_art_catalogues` ones inside
`inputs/prior_art_catalogues/`):

- the DJA grizli v7 photometric catalogs (`*fix_phot_apcorr*`) for the nine fields, staged on
  the staging VM and joined by (field, id);
- the DJA msaexp v4.4 emission-line table (`dja_v4.4_zenodo_dja_msaexp_emission_lines_v4.4.csv.gz`);
- the grizli v7 F444W science and weight mosaics (`https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/`);
- the published LRD lists (Hviding et al. 2025 table A1 and B1, Barro et al. 2025, de Graaff
  et al. 2026) and the census tables of the earlier work (`inputs/sources_v3.parquet`,
  `inputs/training.parquet`);
- `labels.parquet`, the label table built by `build_labels.py` from the above (not released;
  the per-source audit of it is `recovery/label_audit.csv`).

The paper chain reads three further unreleased files, expected in the same place:
`inputs/r1b_calibration.json`, the frozen stamp calibration the gallery figures use;
`inputs/dja_index.csv.gz`, the archive index the spectrum panels are drawn from; and
`inputs/round3_eprints/`, the coordinate excerpts read out of two published papers' arXiv
sources. Two of the sixteen figures are TikZ standalones compiled from
`paper/figures/fig_pipeline.tex` and `paper/figures/fig_architecture.tex`, which belong to
the manuscript and are not released either.

Three intermediate tables are likewise not released, because they are large and fully
reproducible from the scripts and the inputs above: `features.parquet`, `oof_scores.parquet`
and `scores.parquet`. The paper scripts in `paper/` read them, so the paper chain can only be
rerun after steps 1 to 3 have been rerun.

## Execution order and artifacts

| step | script (arguments) | where | produces |
|---|---|---|---|
| 1 | `build_labels.py` | laptop | `labels.parquet` (not released), `label_counts.json` |
| 2 | `v4_features.py` | laptop | `features.parquet` (not released), `feature_manifest.json`, `denominators.json`, `label_audit.csv` |
| 3 | `v4_train_lgbm.py` (uses `v4lib.py`) | laptop | `oof_scores.parquet` (not released), `train_results.json`, `baseline_scores.parquet` (not released) |
| 4 | `vm_extract_shape.py` | staging VM | `v4_shape_columns.csv.gz` (not released) |
| 5 | `v4_features_v34.py` | laptop | `features_v34.parquet` (not released), `feature_manifest_v34.json` |
| 6 | `v4_train_variant.py _v34` | burst VM | `oof_scores_v34.parquet` (not released), `train_results_v34.json` |
| 7 | `v4_mlp.py` | burst VM | `mlp_oof_scores.parquet` (not released), `mlp_results.json` |
| 8 | `v4_tabpfn.py`, `v4_tabicl.py` | burst VM, laptop | stopped before any fold finished; no artifact |
| 9 | `v4_evaluate.py` and `v4_evaluate.py _v34` | laptop | `evaluation.json`, `evaluation.md`, `evaluation_v34.json`, `evaluation_v34.md` |
| 10 | `v4_final.py` | laptop | `final.json`, `models/bag_00..47.txt`, `scores.parquet` (not released; every in-support row's ranking score, reproducible from the weights) |
| 11 | `v4_mlp_final.py`, `v4_blend.py` | not run for the release | the MLP and the blend were not promoted |
| 12 | `v4_candidates.py` | laptop | `candidates.csv`, `candidates_ledger.csv`, `candidate_archive_records.csv`, `candidate_counts.json` |
| 13 | `v4_refit_archive.py` | laptop | `archive_refits.csv`, `archive_refit_counts.json` (zero eligible spectra) |
| 14 | `vm_compactness.py candidates.csv compactness.csv` | staging VM | `compactness.csv` |
| 15 | `v4_merge_compactness.py` | laptop | `candidates.csv` (compactness columns, `followup_tier`), `compactness_counts.json` |
| 16 | `v4_archive_purity.py` | laptop | `archive_purity.json` |
| 17 | `v4_release.py` | laptop | the released directory, its model card, its manifest, its `requirements.txt`, the released `final.json` (`models_dir` set to `models`) |
| 18 | `v4_baseline_burden.py` | laptop | `baseline_burden.json`, `baseline_burden.md`, `baseline_oof_full.parquet` |
| 19 | `v4_round3_analysis.py` | laptop | `published_catalogues_extra.parquet`, `anchor_test_status.csv`, `round3_analysis_counts.json` |
| 20 | `v4_match_degraaff_v2.py` | laptop | `candidates.csv` (columns `in_degraaff26_v2`, `zspec_degraaff26_v2`; every other column unchanged) |
| 21 | `paper/referee_compute_2026_09_06.py` (parts a to d) | laptop | `labelled_rows.csv`, `rule_flags.parquet`, `candidates.csv` (five `oof_*` columns added; every other column unchanged), `referee_compute_2026_09_06.json` |
| 22 | `v4_barro24b_apertures.py` | laptop, DJA cutout service | `barro24b_colors.parquet`, `barro24b_apertures.csv` |
| 23 | `paper/referee_compute_2026_09_06.py` (part e, same run as 21 once step 22 exists) | laptop | `barro24b_flags.parquet`, the `b24*` and `union8*` keys of `referee_compute_2026_09_06.json` |

`model_selection.json` was written by hand-run Python from the evaluation files during the
build and records the challenger comparison and the decision sequence.

## Records of the review rounds

Steps 18 to 20 ran after step 17, during the paper's review rounds, and their records were
added to `recovery/` after the release was first staged. sha256 of the released bytes, in
`sha256sum` order so the list can be checked as it stands:

```
c54de83d2c2f72d3637c175b03f08b77e04a821fcb4448254d1049e01e0e34ab  v4_baseline_burden.py
aa2b9c8c0ba06be6c15958893a6d8eac271812329e76ef215618f7391eac1b7b  baseline_burden.json
d292c4000a38d2141d2cc7dbed52dae8eba61323e7914f3fd55e513a2c42159a  baseline_burden.md
eacff2c3eadde0fd9a9f1ffecfee632969ddd8e8daba42947be0489faef34091  baseline_oof_full.parquet
da15623a77eceaa5abe2ae9f303f3190f61d60a55374df9b0c76ea3e97483365  v4_round3_analysis.py
14cd5cc196e5d19b1d8bfe47bc5c17c7f66f114a9b964064c89087d1e7929bac  published_catalogues_extra.parquet
c5e3fbacc4721c97fce18390f9bd4861af488e50be42a967a1788a815373cf98  anchor_test_status.csv
76e9c30d45f1a9540dceb6f57e5aabe94445ca5914d31c77dec6ed24efc0f513  round3_analysis_counts.json
```

Three further records were added at the same time:

```
a58a0f7ae0cf4d62da9c40a35d139aa00099a6251cf59b3f14b0c45d446cb337  positive_test_status.csv
132f5f73395d84f9334504ced692eda87161196d5edd989fdf4017713dbe251d  label_counts.json
7ae9a817491b95c1da69e3902dc025f6c0af062410a111af7ca1ae3e7463dd60  feature_importance_gain.json
```

`positive_test_status.csv` (151 rows and 5 columns) carries the `spec_tested` and `spec_vshaped` flags of the 151 positives, taken from the build's feature table, which is not released; 118 are tested and 41 established as V-shaped. `label_counts.json` is written by step 1. `feature_importance_gain.json` is the deployed model's split-gain share by feature (30 entries summing to 1); the command that wrote it is not in the release, so it is a record only. `final.json`'s `models_dir_at_run_time` was rewritten from the build-tree path to `recovery/models`, the release convention used for every other pointer.

Row counts: `baseline_oof_full.parquet` 466419 rows and 56 columns,
`published_catalogues_extra.parquet` 630869 rows and 28 columns, `anchor_test_status.csv`
5548 rows and 7 columns; the three JSON and Markdown files are not tables.

`baseline_oof_full.parquet` carries every in-support row's out-of-fold score under the
deployed model and under each baseline the paper compares it against, and
`baseline_burden.json` and `baseline_burden.md` are the burden-matched tables read from it.
`published_catalogues_extra.parquet` adds each positive's spectroscopic redshift and a
per-catalogue decomposition of the published census; `anchor_test_status.csv` splits the
anchors into the tested and the untested.


## Records of the revision of 2026-09-06

Steps 21 to 23 ran on 2026-09-06 (IST), after the referee rounds of 2026-09-05, from the
tables of record and the unreleased `inputs/` tables of steps 1 to 3. Nothing was retrained
and no released number of steps 1 to 20 changed. sha256 of the released bytes:

```
75c48a0329e4fe48a97449da9d55feaf4656d00dca1b746ce5004b9af002c3e0  referee_compute_2026_09_06.py
aaf6a58e29c73cbc0fc225c477d756b54cf22fb25426acca02ffa7e642bc8939  referee_compute_2026_09_06.json
c4e0e5648c154ca9119dde6ae123d7130ad479e321cb08b2ae6f54f404b58d9b  labelled_rows.csv
c29c278fc7788e1eec714d21fa303df56032468ea0a8442b059f2c4ed0f5aa6c  rule_flags.parquet
da1b44e91a9d73ad0f85bf21624028cfadbda6c9482b0c119934478a708ba8cd  candidates.csv
0ffcf099fffb8f626e642a5c7f19e77b89a2a1d7adc92353715a9372d560aa83  v4_barro24b_apertures.py
d5cd8dee0968725d994249dd8ff7446f2a9ea1d472d6be353551940a29204238  barro24b_colors.parquet
3797d11afb1cab41d3273677066632bc8f5cbff5e74b8c322eba4a00e75e1420  barro24b_apertures.csv
525cbb28511b6ad02444549feb383d7e31d3bff8f5eb9dbe75d2265d13b71272  barro24b_flags.parquet
5f0829ecb16d5e28eb229b51b06c232cf76a15ce84efc5933da780ae255a61fc  barro24b.py
b56e1e2e0e73833ee8fe40d4856c13a3866664a8198b9e67d340b55f809c03ac  subsets.json
eea545dfa05ccc463553a72ddcabd0089678aa83a8be7f2863089ba06e3c5453  barro_membership.json
```

`labelled_rows.csv` (5596 rows and 38 columns) carries every labelled or
ambiguous row: `y`, `y_strict`, `ambiguous`, the list memberships, the seven rule flags of
record, the sky group, the region, `in_support`, and the out-of-fold score with its three
outcomes for the rows in support. `rule_flags.parquet` (630869 rows and 11 columns) carries the
seven rule flags of record and `picked` for every catalog row, so the union of record is
regenerable without the full-band inputs (see the seven-band note in the README).
`candidates.csv` now carries `oof_score`, `oof_t_burden`, `oof_t_05`, `oof_above_burden` and
`oof_above_05` (90 columns in all): the out-of-fold score of the fold model that never
saw the candidate's region and that fold's two thresholds. `referee_compute_2026_09_06.json`
holds every count the script prints, keyed by the paper's macro names.

The Barro et al. (2024b) comparator was added at referee request and is kept apart from the
seven of `redress.cuts.CUTS` (it is `redress.cuts.COMPARATORS["barro24b"]`). Step 22 applies its
color and magnitude criteria to every catalog row from the catalog fluxes (fail-closed on a
non-positive flux) and writes `barro24b_colors.parquet` (630869 rows), then measures its
aperture ratio F444W(r=0.5")/F444W(r=0.2") on an F444W cutout from the DJA grizli cutout
service for each of the 1021 color survivors, forced at the catalog position with
exact-overlap apertures and no recentering; `barro24b_apertures.csv` records the two fluxes,
the ratio, the mosaic version (`grizliv`) and, for the 11 rows
it could not measure, the reason. `barro24b_flags.parquet` is the resulting per-row flag.
Deviations from the published procedure are stated in the module docstring
(`selections/redress/cuts/barro24b.py`).

The paper tooling was brought level with the manuscript in the same commit. The scripts below
were added or replaced; `numbers.json` and `numbers.tex` are their output. Order of
application when regenerating the numbers: `build_numbers.py`, `round5_subsets.py`,
`round5_barro_membership.py`, `round5_macros_patch.py`, `referee_macros_2026_09_05.py`,
`referee_compute_2026_09_06.py`; `check_pub.py` is the manuscript's literal-number check and
reads `numbers.tex`.

```
52c237e832b17dfda50b57e87e212a62301f732eba506cf6e57f3a2dc68cddaf  round5_subsets.py
c24c622a7ac6286dd8497caa7f5e1db57df76579f061de3e7fc0123d368213ef  round5_barro_membership.py
c64c12583d51f30f18948020a5217fb8298b93a91438f65c20cfc4d7cdd0e385  round5_macros_patch.py
d0cb262ef320943aa241f695e673a41ae7a62d67021b5bf167f1e773c0a2985b  referee_macros_2026_09_05.py
6a814ab6290e959f1f604a7fb1f1a10a75747332998e58a28d0b7c2e5586e7b1  build_figures_pub.py
85b91a2afe8e392fb05b18e73344c1bbaa74f2ba63dbb806e9e67d62af908bb0  check_pub.py
a3fca795c1b76ecc01fefc80e92f6599d5b737ec1bf04b45781eee855ffe35d6  numbers.tex
```

## The paper chain

`paper/build_numbers.py` reads the tables in `recovery/` (and the two unreleased parquet
tables) and writes `paper/numbers.json` and `paper/numbers.tex`; the round-5 and referee
scripts listed above append their macros to the same two files; `paper/fetch_stamps.py` and
`paper/fetch_stamps_6band.py` download the cutouts for the image figures; `paper/build_figures.py`,
`paper/build_figures_extra.py`, `paper/build_figures_pub.py` and `paper/build_tables.py` draw the
figures and write the appendix table bodies.

The manuscript source is not released and nothing here typesets it. What these
scripts give a reader is every number, figure and table the manuscript quotes, each one
traceable to the table in `recovery/` it was read from, so any of them can be checked by
rerunning the script and diffing its output.

## The selection modules

`selections/redress/cuts/` re-implements the seven published photometric selections used as
the comparison baseline throughout, plus the Barro et al. (2024b) comparator added in revision
and registered apart from them, `selections/redress/compactness.py` is the frozen
aperture protocol of step 14, `selections/redress/folds.py` builds the sky groups and folds,
and `selections/spectra/` is the frozen spectral-refit engine `v4_refit_archive.py` calls in
step 13. `selections/tests/` pins all of them; run `python -m pytest selections -q` from the
repository root.
