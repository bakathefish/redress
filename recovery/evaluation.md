# v4 recovery classifier: out-of-fold evaluation

Positives: 151 spectroscopic LRDs, 147 inside the seven-band support (4 abstentions counted as misses end to end). Rule-missed: 44 (41 in support). Confirmed non-LRDs in support: 4979. Support rows: 466419. Union burden: 1133 selections (0.243%).

| selection | selected | recall of 147 | end-to-end recall | rule-missed | strict 32 | strict missed | targeted-negative selection rate |
|---|---|---|---|---|---|---|---|
| union of seven rules | 1133 | 106 (72.1%) | 107 of 151 (70.9%) | 0 of 41 | 25 of 32 | 0 of 7 | 46 of 4979 (0.92%) |
| primary at equal burden | 1115 | 123 (83.7%) | 123 of 151 (81.5%) | 20 of 41 | 30 of 32 | 5 of 7 | 70 of 4979 (1.41%) |
| primary at inner 1% negative rate | 1074 | 123 (83.7%) | 123 of 151 (81.5%) | 20 of 41 | 30 of 32 | 5 of 7 | 61 of 4979 (1.23%) |
| primary at 0.5% of the catalog | 2659 | 133 (90.5%) | 133 of 151 (88.1%) | 29 of 41 | 30 of 32 | 5 of 7 | 225 of 4979 (4.52%) |
| mlp at equal burden | 1161 | 123 (83.7%) | 123 of 151 (81.5%) | 25 of 41 | 28 of 32 | 5 of 7 | 65 of 4979 (1.31%) |
| rank-average blend (trees + MLP) at equal burden | 1092 | 125 (85.0%) | 125 of 151 (82.8%) | 24 of 41 | 29 of 32 | 5 of 7 | 61 of 4979 (1.23%) |

Case-control PR-AUC (positives vs targeted negatives, pooled OOF): primary 0.8214, MLP 0.8352

## Per region at equal burden (recall / rule-missed recovered)
| region | union | primary | primary rule-missed | selected vs union | settings (anchor, trees) |
|---|---|---|---|---|---|
| CEERS | 23/33 | 29/33 | 6/10 | 180 vs 134 | (0.0, 400) |
| GOODS-N | 10/17 | 11/17 | 2/7 | 146 vs 129 | (0.5, 150) |
| GOODS-S | 13/18 | 15/18 | 2/5 | 138 vs 127 | (0.5, 250) |
| COSMOS | 22/28 | 23/28 | 2/6 | 278 vs 254 | (0.25, 350) |
| UDS | 38/51 | 45/51 | 8/13 | 373 vs 489 | (0.25, 400) |

## Paired against the union (positives)
both 103, model only 20, union only 3, neither 21; selected sets: both 645, model only 470, union only 488, Jaccard 0.402

## By catalog and by F444W magnitude (primary at equal burden)
catalogs: hviding25_A1 33/36, barro25 87/104, degraaff26 74/83
magnitude: 0-24: 6/10 (union 7), 24-25: 24/27 (union 23), 25-26: 40/45 (union 32), 26-27: 47/54 (union 39), 27-40: 6/11 (union 5)

## Baselines and sanity checks (equal burden, 8 bags, settings copied from the primary fold)
| variant | recall of 147 | rule-missed | strict missed | negatives selected | PR-AUC | per region |
|---|---|---|---|---|---|---|
| primary_8bag_replica | 124 | 21 | 5 | 79 | 0.8217 | 29 11 16 23 45 |
| magnitude_only | 98 | 14 | 6 | 116 | 0.4414 | 23 11 11 18 35 |
| morphology_only | 3 | 0 | 0 | 17 | 0.1562 | 0 1 0 0 2 |
| ablate_c_f277w_f444w | 123 | 21 | 5 | 90 | 0.8013 | 28 11 16 22 46 |
| ablate_size | 114 | 16 | 5 | 118 | 0.6074 | 26 10 14 22 42 |
| anchor_0 | 124 | 22 | 5 | 81 | 0.8318 | 29 11 15 25 44 |
| anchor_0.50 | 124 | 21 | 5 | 62 | 0.8588 | 29 11 15 24 45 |
| pn_lightgbm | 123 | 20 | 5 | 43 | 0.8648 | 29 11 14 23 46 |
| rules_reproduction | 119 | 17 | 5 | 56 | 0.7497 | 22 13 14 24 46 |
| leave_out_hviding25_A1 | 124 | 21 | 5 | 63 | 0.8241 | 29 11 15 24 45 | left-out catalog: 33 of 36 (primary 33 of 36)
| leave_out_barro25 | 84 | 9 | 3 | 49 | 0.4807 | 15 9 9 16 35 | left-out catalog: 54 of 104 (primary 87 of 104)
| leave_out_degraaff26 | 116 | 18 | 5 | 65 | 0.7144 | 26 11 15 25 39 | left-out catalog: 70 of 83 (primary 74 of 83)
| shuffled_labels_0 | 41 | 7 | 0 | 141 | 0.1364 | 19 5 7 6 4 |
| shuffled_labels_1 | 38 | 12 | 1 | 162 | 0.1301 | 18 1 8 4 7 |
| shuffled_labels_2 | 39 | 9 | 0 | 171 | 0.1147 | 12 5 7 3 12 |
| shuffled_labels_3 | 39 | 9 | 2 | 91 | 0.1027 | 11 1 7 12 8 |
| shuffled_labels_4 | 52 | 14 | 3 | 124 | 0.1976 | 23 5 5 4 15 |
| with_zphot | 125 | 21 | 5 | 70 | 0.8194 | 29 11 15 25 45 |

Shuffled-label recall at equal burden (5 repeats): [41, 38, 39, 39, 52]; chance level if the labels carried nothing: about 0.36. The floor sits far above chance because every labelled object, positive or negative, was a spectroscopic target: a shuffled model still learns what a targeted object looks like (brightness, colour selection of the programs), so the honest comparison for the primary is against this floor and against the union, not against chance.
Field adversary: accuracy 0.7761 (chance 0.2); top features [['asnr_f115w', 0.117], ['m_f115w', 0.082], ['m_f277w', 0.079], ['log_rh', 0.074], ['log_rh_over_rstar', 0.069], ['asnr_f277w', 0.069]].
Ensemble dispersion (std over 48 bags): median all 0.0011, positives 0.0030, selected 0.0093.

mlp promotion rule: {"gain_positives": 0, "gain_rule_missed": 5, "regions_won": 2, "neg_rate_change_points": -0.1, "promoted": false}

## Rule-missed positives the primary model recovers at equal burden
| field | id | region | F444W | A1 | Barro | deGraaff | strict | score |
|---|---|---|---|---|---|---|---|---|
| ceers-full | 73652 | CEERS | 26.16 | False | True | True | True | 0.990 |
| primer-cosmos-east | 37681 | COSMOS | 26.25 | False | True | True | True | 0.982 |
| ceers-full | 54373 | CEERS | 25.08 | False | True | False | False | 0.976 |
| primer-uds-south | 23438 | UDS | 24.98 | True | True | True | True | 0.973 |
| primer-uds-north | 25768 | UDS | 24.84 | False | True | False | False | 0.972 |
| primer-uds-north | 36569 | UDS | 27.21 | False | True | False | False | 0.970 |
| primer-uds-north | 74752 | UDS | 25.63 | False | True | False | False | 0.963 |
| ceers-full | 19938 | CEERS | 26.09 | False | True | False | False | 0.958 |
| gds | 24555 | GOODS-S | 26.78 | False | True | False | False | 0.954 |
| ceers-full | 18848 | CEERS | 25.75 | False | True | False | False | 0.944 |
| gdn | 6046 | GOODS-N | 26.32 | False | True | True | True | 0.937 |
| ceers-full | 58886 | CEERS | 25.90 | False | True | False | False | 0.937 |
| gdn | 24803 | GOODS-N | 25.38 | False | True | False | True | 0.934 |
| primer-uds-south | 15006 | UDS | 26.04 | False | False | True | False | 0.926 |
| ceers-full | 20143 | CEERS | 25.54 | False | True | False | False | 0.925 |
| primer-uds-north | 46223 | UDS | 26.37 | False | True | False | False | 0.910 |
| primer-uds-south | 10036 | UDS | 25.87 | False | True | False | False | 0.894 |
| primer-cosmos-east | 79141 | COSMOS | 26.86 | False | True | False | False | 0.887 |
| primer-uds-south | 37427 | UDS | 25.76 | False | True | False | False | 0.846 |
| gds | 49351 | GOODS-S | 25.65 | False | True | False | False | 0.771 |

## Positives still missed at equal burden (score percentile among all support rows)
| field | id | region | F444W | union picks it | strict | score | top % |
|---|---|---|---|---|---|---|---|
| ceers-full | 55163 | CEERS | 25.93 | False | False | 0.869 | 0.18 |
| primer-uds-north | 67741 | UDS | 24.68 | False | False | 0.753 | 0.29 |
| primer-uds-south | 65705 | UDS | 26.36 | False | False | 0.745 | 0.3 |
| primer-cosmos-west | 89952 | COSMOS | 27.46 | False | False | 0.741 | 0.3 |
| primer-cosmos-west | 45444 | COSMOS | 24.81 | True | False | 0.733 | 0.31 |
| gds | 11107 | GOODS-S | 26.39 | False | False | 0.717 | 0.33 |
| primer-cosmos-east | 83388 | COSMOS | 26.71 | False | False | 0.674 | 0.39 |
| primer-uds-north | 8768 | UDS | 25.33 | False | False | 0.635 | 0.44 |
| gds-sw | 12551 | GOODS-S | 23.82 | False | False | 0.623 | 0.46 |
| gdn | 77875 | GOODS-N | 27.18 | False | False | 0.573 | 0.54 |
| ceers-full | 7491 | CEERS | 27.85 | False | False | 0.533 | 0.6 |
| primer-uds-south | 7111 | UDS | 25.23 | True | False | 0.484 | 0.7 |
| gdn | 4064 | GOODS-N | 22.08 | True | False | 0.482 | 0.71 |
| primer-cosmos-west | 65299 | COSMOS | 25.20 | False | False | 0.385 | 1.06 |
| primer-uds-north | 49209 | UDS | 26.66 | False | False | 0.357 | 1.26 |
| gdn | 27348 | GOODS-N | 26.47 | False | False | 0.339 | 1.42 |
| gdn | 54665 | GOODS-N | 24.70 | False | False | 0.332 | 1.49 |
| ceers-full | 51045 | CEERS | 26.42 | False | False | 0.309 | 1.77 |
| ceers-full | 72397 | CEERS | 28.61 | False | False | 0.300 | 1.88 |
| primer-cosmos-west | 56264 | COSMOS | 28.02 | False | True | 0.230 | 3.08 |
| gdn | 67601 | GOODS-N | 22.02 | False | False | 0.192 | 4.09 |
| primer-uds-south | 38479 | UDS | 26.79 | False | False | 0.163 | 5.2 |
| gdn | 24477 | GOODS-N | 25.01 | False | True | 0.076 | 11.62 |
| gds-sw | 12550 | GOODS-S | 23.97 | False | False | 0.011 | 65.32 |