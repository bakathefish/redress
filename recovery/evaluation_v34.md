# v4 recovery classifier: out-of-fold evaluation_v34

Positives: 151 spectroscopic LRDs, 147 inside the seven-band support (4 abstentions counted as misses end to end). Rule-missed: 44 (41 in support). Confirmed non-LRDs in support: 4979. Support rows: 466419. Union burden: 1133 selections (0.243%).

| selection | selected | recall of 147 | end-to-end recall | rule-missed | strict 32 | strict missed | targeted-negative selection rate |
|---|---|---|---|---|---|---|---|
| union of seven rules | 1133 | 106 (72.1%) | 107 of 151 (70.9%) | 0 of 41 | 25 of 32 | 0 of 7 | 46 of 4979 (0.92%) |
| primary at equal burden | 1098 | 124 (84.4%) | 124 of 151 (82.1%) | 20 of 41 | 30 of 32 | 5 of 7 | 48 of 4979 (0.96%) |
| primary at inner 1% negative rate | 1179 | 125 (85.0%) | 125 of 151 (82.8%) | 21 of 41 | 30 of 32 | 5 of 7 | 52 of 4979 (1.04%) |
| primary at 0.5% of the catalog | 2560 | 135 (91.8%) | 135 of 151 (89.4%) | 29 of 41 | 30 of 32 | 5 of 7 | 185 of 4979 (3.72%) |
| mlp at equal burden | 1161 | 123 (83.7%) | 123 of 151 (81.5%) | 25 of 41 | 28 of 32 | 5 of 7 | 65 of 4979 (1.31%) |
| rank-average blend (trees + MLP) at equal burden | 1085 | 125 (85.0%) | 125 of 151 (82.8%) | 24 of 41 | 29 of 32 | 5 of 7 | 55 of 4979 (1.1%) |

Case-control PR-AUC (positives vs targeted negatives, pooled OOF): primary 0.8619, MLP 0.8352

## Per region at equal burden (recall / rule-missed recovered)
| region | union | primary | primary rule-missed | selected vs union | settings (anchor, trees) |
|---|---|---|---|---|---|
| CEERS | 23/33 | 28/33 | 5/10 | 160 vs 134 | (0.5, 350) |
| GOODS-N | 10/17 | 12/17 | 2/7 | 139 vs 129 | (0.5, 450) |
| GOODS-S | 13/18 | 16/18 | 3/5 | 141 vs 127 | (0.25, 300) |
| COSMOS | 22/28 | 24/28 | 2/6 | 280 vs 254 | (0.5, 250) |
| UDS | 38/51 | 44/51 | 8/13 | 378 vs 489 | (0.5, 400) |

## Paired against the union (positives)
both 104, model only 20, union only 2, neither 21; selected sets: both 616, model only 482, union only 517, Jaccard 0.381

## By catalog and by F444W magnitude (primary at equal burden)
catalogs: hviding25_A1 33/36, barro25 87/104, degraaff26 77/83
magnitude: 0-24: 7/10 (union 7), 24-25: 25/27 (union 23), 25-26: 39/45 (union 32), 26-27: 47/54 (union 39), 27-40: 6/11 (union 5)

## Baselines and sanity checks (equal burden, 8 bags, settings copied from the primary fold)
| variant | recall of 147 | rule-missed | strict missed | negatives selected | PR-AUC | per region |
|---|---|---|---|---|---|---|
| shuffled_labels_1 | 22 | 3 | 0 | 149 | 0.0922 | 7 2 7 5 1 |
| shuffled_labels_2 | 26 | 8 | 1 | 160 | 0.1217 | 11 1 9 4 1 |

Shuffled-label recall at equal burden (2 repeats): [22, 26]; chance level if the labels carried nothing: about 0.36. The floor sits far above chance because every labelled object, positive or negative, was a spectroscopic target: a shuffled model still learns what a targeted object looks like (brightness, colour selection of the programs), so the honest comparison for the primary is against this floor and against the union, not against chance.
Field adversary: accuracy 0.7729 (chance 0.2); top features [['asnr_f115w', 0.114], ['m_f115w', 0.087], ['m_f277w', 0.082], ['asnr_f277w', 0.07], ['asnr_f090w', 0.065], ['log_rh', 0.062]].
Ensemble dispersion (std over 48 bags): median all 0.0009, positives 0.0032, selected 0.0081.

mlp promotion rule: {"gain_positives": -1, "gain_rule_missed": 5, "regions_won": 2, "neg_rate_change_points": 0.35, "promoted": false}

## Rule-missed positives the primary model recovers at equal burden
| field | id | region | F444W | A1 | Barro | deGraaff | strict | score |
|---|---|---|---|---|---|---|---|---|
| gdn | 6046 | GOODS-N | 26.32 | False | True | True | True | 0.989 |
| ceers-full | 73652 | CEERS | 26.16 | False | True | True | True | 0.987 |
| gdn | 24803 | GOODS-N | 25.38 | False | True | False | True | 0.986 |
| primer-uds-north | 74752 | UDS | 25.63 | False | True | False | False | 0.976 |
| primer-uds-north | 25768 | UDS | 24.84 | False | True | False | False | 0.971 |
| primer-cosmos-east | 37681 | COSMOS | 26.25 | False | True | True | True | 0.955 |
| ceers-full | 54373 | CEERS | 25.08 | False | True | False | False | 0.955 |
| primer-uds-north | 36569 | UDS | 27.21 | False | True | False | False | 0.952 |
| gds | 24555 | GOODS-S | 26.78 | False | True | False | False | 0.949 |
| primer-uds-south | 15006 | UDS | 26.04 | False | False | True | False | 0.943 |
| primer-uds-south | 10036 | UDS | 25.87 | False | True | False | False | 0.935 |
| primer-uds-south | 23438 | UDS | 24.98 | True | True | True | True | 0.935 |
| ceers-full | 18848 | CEERS | 25.75 | False | True | False | False | 0.927 |
| primer-uds-north | 46223 | UDS | 26.37 | False | True | False | False | 0.913 |
| ceers-full | 20143 | CEERS | 25.54 | False | True | False | False | 0.895 |
| gds-sw | 12551 | GOODS-S | 23.82 | False | False | True | False | 0.886 |
| primer-cosmos-east | 79141 | COSMOS | 26.86 | False | True | False | False | 0.879 |
| gds | 11107 | GOODS-S | 26.39 | False | True | False | False | 0.851 |
| ceers-full | 58886 | CEERS | 25.90 | False | True | False | False | 0.847 |
| primer-uds-south | 65705 | UDS | 26.36 | False | True | False | False | 0.803 |

## Positives still missed at equal burden (score percentile among all support rows)
| field | id | region | F444W | union picks it | strict | score | top % |
|---|---|---|---|---|---|---|---|
| gds | 49351 | GOODS-S | 25.65 | False | False | 0.791 | 0.22 |
| primer-uds-south | 50431 | UDS | 23.51 | True | False | 0.695 | 0.28 |
| primer-uds-south | 33451 | UDS | 26.93 | True | False | 0.683 | 0.29 |
| ceers-full | 55163 | CEERS | 25.93 | False | False | 0.667 | 0.3 |
| ceers-full | 19938 | CEERS | 26.09 | False | False | 0.655 | 0.31 |
| primer-cosmos-west | 89952 | COSMOS | 27.46 | False | False | 0.646 | 0.32 |
| primer-cosmos-west | 65299 | COSMOS | 25.20 | False | False | 0.627 | 0.33 |
| primer-cosmos-east | 83388 | COSMOS | 26.71 | False | False | 0.607 | 0.34 |
| gdn | 77875 | GOODS-N | 27.18 | False | False | 0.556 | 0.4 |
| primer-uds-north | 8768 | UDS | 25.33 | False | False | 0.525 | 0.43 |
| primer-uds-north | 67741 | UDS | 24.68 | False | False | 0.399 | 0.6 |
| primer-uds-south | 37427 | UDS | 25.76 | False | False | 0.383 | 0.64 |
| ceers-full | 7491 | CEERS | 27.85 | False | False | 0.304 | 1.01 |
| ceers-full | 72397 | CEERS | 28.61 | False | False | 0.299 | 1.05 |
| primer-cosmos-west | 56264 | COSMOS | 28.02 | False | True | 0.280 | 1.23 |
| ceers-full | 51045 | CEERS | 26.42 | False | False | 0.249 | 1.6 |
| primer-uds-north | 49209 | UDS | 26.66 | False | False | 0.242 | 1.7 |
| gdn | 67601 | GOODS-N | 22.02 | False | False | 0.209 | 2.31 |
| gdn | 27348 | GOODS-N | 26.47 | False | False | 0.186 | 2.93 |
| primer-uds-south | 38479 | UDS | 26.79 | False | False | 0.138 | 4.64 |
| gdn | 24477 | GOODS-N | 25.01 | False | True | 0.084 | 7.82 |
| gdn | 54665 | GOODS-N | 24.70 | False | False | 0.079 | 8.34 |
| gds-sw | 12550 | GOODS-S | 23.97 | False | False | 0.013 | 51.26 |

Note: the MLP challenger and the blend in this file use the nnPU MLP trained on the 30-feature set (mlp_oof_scores.parquet copied to the _v34 name); the MLP was not retrained on the 34 features. This feature set was not deployed; the model-selection record is model_selection.json.
