# Realised catalogue burden of every v4 variant

Round 2, reviewer E findings 1 and 2. Every variant in the run of record was scored only on the 5126 labelled rows, so no variant had a catalogue-wide burden. This table re-runs each of them with the record's exact code path and scores all 466419 in-support rows of each held-out region.

Machine: DESKTOP-2N9PO1E, Windows-11-10.0.26200-SP0, 16 logical cores, Python 3.13.3, LightGBM 4.7.0, NumPy 2.4.1, LightGBM num_threads 4 and joblib n_jobs 4 exactly as in the record. Wall time 266 s. Every variant reproduced the record's labelled-row numbers exactly (recall, rule-missed, strict, anchors, pooled case-control PR-AUC and the per-region recalls all identical to recovery/train_results.json).

The inner nested selection was not re-run; its chosen anchor, tree count and training-region union burden were read from the record, and the 20 draws it took on the shared rng_sub stream were replayed call for call so every variant's 50000-row threshold subsample is the same draw the record used. The training-region union burdens recomputed from the data equal the record's to the last bit.

The catalogue-wide out-of-fold scores this table is built from are released in `recovery/baseline_oof_full.parquet`.

## Realised burden at the record's equal-burden threshold

Selected rows is the realised catalogue burden: in-support rows the variant selects at the threshold the record set inside the training regions. Recall, rule-missed and anchors are the record's labelled-row numbers, reproduced here.

| variant | selected rows | burden vs union | recall of 147 | rule-missed of 41 | anchors of 4979 | pooled AP | reproduces |
|---|---:|---:|---:|---:|---:|---:|:--:|
| union of the seven rules | 1133 | 1.00x | 106 | 0 | 46 | - | - |
| primary, 48 bags (deployed) | 1115 | 0.98x | 123 | 20 | 70 | 0.8214 | - |
| primary_8bag_replica | 1167 | 1.03x | 124 | 21 | 79 | 0.8217 | yes |
| magnitude_only | 1202 | 1.06x | 98 | 14 | 116 | 0.4414 | yes |
| morphology_only | 1769 | 1.56x | 3 | 0 | 17 | 0.1562 | yes |
| ablate_c_f277w_f444w | 1162 | 1.03x | 123 | 21 | 90 | 0.8013 | yes |
| ablate_size | 1110 | 0.98x | 114 | 16 | 118 | 0.6074 | yes |
| anchor_0 | 1126 | 0.99x | 124 | 22 | 81 | 0.8318 | yes |
| anchor_0.50 | 1148 | 1.01x | 124 | 21 | 62 | 0.8588 | yes |
| pn_lightgbm | 1091 | 0.96x | 123 | 20 | 43 | 0.8648 | yes |
| rules_reproduction | 1293 | 1.14x | 119 | 17 | 56 | 0.7497 | yes |
| leave_out_hviding25_A1 | 1111 | 0.98x | 124 | 21 | 63 | 0.8241 | yes |
| leave_out_barro25 | 1125 | 0.99x | 84 | 9 | 49 | 0.4807 | yes |
| leave_out_degraaff26 | 1105 | 0.98x | 116 | 18 | 65 | 0.7144 | yes |
| shuffled_labels_0 | 1154 | 1.02x | 41 | 7 | 141 | 0.1364 | yes |
| shuffled_labels_1 | 1370 | 1.21x | 38 | 12 | 162 | 0.1301 | yes |
| shuffled_labels_2 | 1140 | 1.01x | 39 | 9 | 171 | 0.1147 | yes |
| shuffled_labels_3 | 1030 | 0.91x | 39 | 9 | 91 | 0.1027 | yes |
| shuffled_labels_4 | 1133 | 1.00x | 52 | 14 | 124 | 0.1976 | yes |
| with_zphot | 1155 | 1.02x | 125 | 21 | 70 | 0.8194 | yes |

## Realised burden per region

| variant | CEERS | GOODS-N | GOODS-S | COSMOS | UDS | pooled |
|---|---:|---:|---:|---:|---:|---:|
| union of the seven rules | 134 | 129 | 127 | 254 | 489 | 1133 |
| primary, 48 bags (deployed) | 180 | 146 | 138 | 278 | 373 | 1115 |
| primary_8bag_replica | 199 | 148 | 149 | 248 | 423 | 1167 |
| magnitude_only | 163 | 125 | 102 | 259 | 553 | 1202 |
| morphology_only | 0 | 313 | 778 | 578 | 100 | 1769 |
| ablate_c_f277w_f444w | 194 | 145 | 150 | 245 | 428 | 1162 |
| ablate_size | 201 | 123 | 130 | 268 | 388 | 1110 |
| anchor_0 | 184 | 148 | 157 | 319 | 318 | 1126 |
| anchor_0.50 | 177 | 154 | 133 | 313 | 371 | 1148 |
| pn_lightgbm | 141 | 144 | 120 | 279 | 407 | 1091 |
| rules_reproduction | 98 | 123 | 117 | 286 | 669 | 1293 |
| leave_out_hviding25_A1 | 181 | 142 | 133 | 297 | 358 | 1111 |
| leave_out_barro25 | 144 | 123 | 114 | 428 | 316 | 1125 |
| leave_out_degraaff26 | 189 | 140 | 107 | 299 | 370 | 1105 |
| shuffled_labels_0 | 249 | 149 | 254 | 233 | 269 | 1154 |
| shuffled_labels_1 | 291 | 241 | 186 | 203 | 449 | 1370 |
| shuffled_labels_2 | 267 | 211 | 166 | 181 | 315 | 1140 |
| shuffled_labels_3 | 219 | 176 | 142 | 223 | 270 | 1030 |
| shuffled_labels_4 | 262 | 166 | 115 | 222 | 368 | 1133 |
| with_zphot | 178 | 152 | 138 | 311 | 376 | 1155 |

## Matched burden: top N_union rows in each held-out region

N_union per region is the union's own in-support selection count: CEERS 134, GOODS-N 129, GOODS-S 127, COSMOS 254, UDS 489, pooled 1133. Every variant is cut to exactly those counts, so the comparison against the union is burden matched by construction.

| variant | matched recall of 147 | matched rule-missed of 41 | matched anchors | tie-free |
|---|---:|---:|---:|:--:|
| union of the seven rules | 106 | 0 | 46 | - |
| primary, 48 bags | 124 | 21 | 56 | yes |
| primary_8bag_replica | 123 | 20 | 57 | yes |
| magnitude_only | 100 | 14 | 108 | yes |
| morphology_only | 4 | 0 | 7 | no |
| ablate_c_f277w_f444w | 122 | 20 | 60 | yes |
| ablate_size | 112 | 15 | 114 | yes |
| anchor_0 | 120 | 19 | 70 | yes |
| anchor_0.50 | 125 | 22 | 51 | yes |
| pn_lightgbm | 123 | 20 | 41 | yes |
| rules_reproduction | 116 | 13 | 52 | yes |
| leave_out_hviding25_A1 | 124 | 21 | 62 | yes |
| leave_out_barro25 | 85 | 9 | 57 | yes |
| leave_out_degraaff26 | 119 | 20 | 57 | yes |
| shuffled_labels_0 | 41 | 10 | 124 | yes |
| shuffled_labels_1 | 35 | 10 | 122 | yes |
| shuffled_labels_2 | 39 | 9 | 151 | yes |
| shuffled_labels_3 | 43 | 9 | 86 | yes |
| shuffled_labels_4 | 51 | 12 | 108 | yes |
| with_zphot | 124 | 21 | 58 | yes |

## Paired tests at matched burden

| comparison | subset | both | A only | B only | neither | exact two-sided p |
|---|---:|---:|---:|---:|---:|---:|
| imitation (A) vs union (B), matched | 147 positives | 103 | 13 | 3 | 28 | 0.0212708 |
| imitation (A) vs union (B), matched | 41 rule-missed | 0 | 13 | 0 | 28 | 0.000244141 |
| primary (A) vs imitation (B), matched | 147 positives | 110 | 14 | 6 | 17 | 0.115318 |
| primary (A) vs imitation (B), matched | 41 rule-missed | 10 | 11 | 3 | 17 | 0.057373 |
| primary (A) vs union (B), matched | 147 positives | 103 | 21 | 3 | 20 | 0.000277162 |
| imitation (A) vs union (B), deployed thresholds | 147 positives | 102 | 17 | 4 | 24 | 0.00719738 |
| primary (A) vs imitation (B), deployed thresholds | 147 positives | 111 | 12 | 8 | 16 | 0.503445 |

