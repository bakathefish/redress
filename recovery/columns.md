# Column dictionary

Units: fluxes in microjansky, magnitudes AB unless the column name says asinh feature,
coordinates J2000 degrees, sizes arcsec. Boolean columns are True/False. The join key of
every table is `source_id`. The feature columns are the 30 model inputs listed in
`feature_manifest.json`, with the asinh softening per band recorded there.

## candidates.csv (1207 rows, 116 columns)

| column | meaning |
|---|---|
| `rank` | rank in the candidate catalog by the deployed model's score |
| `tier` | equal-burden or high-recall tier of the deployed thresholds |
| `status` | frozen archival status (see the paper, Section 6.2) |
| `ranking_score` | in-sample score of the deployed model fitted on all five regions |
| `score_std` | standard deviation of the 48 bag outputs |
| `score_percentile` | percentile of ranking_score over the support |
| `field` | DJA grizli v7 catalog name (ceers-full, gdn, gds, gds-sw, ngdeep, primer-cosmos-east, primer-cosmos-west, primer-uds-north, primer-uds-south) |
| `id` | catalog id within the field |
| `source_id` | integer key of the catalog row across the nine fields (the join key of every released table) |
| `ra` | catalog right ascension, J2000, degrees |
| `dec` | catalog declination, J2000, degrees |
| `sky_group` | friends-of-friends group id at 0.5 arcsec; a group is one object where objects are counted |
| `sky_group_size` | number of catalog rows in the sky group |
| `mag_f444w` | catalog total AB magnitude in F444W |
| `snr_f444w` | catalog signal-to-noise in F444W |
| `r_h_arcsec` | catalog half-light radius (flux_radius) in arcsec |
| `z_phot` | catalog photometric redshift (EAZY, DJA) |
| `ood_flag` | one or more features outside the range spanned by the positives |
| `n_features_outside_positive_range` | count of such features |
| `disagreement_flag` | bag disagreement above the release cut |
| `nearest_list` | nearest input-list object |
| `nearest_list_sep_arcsec` | its separation in arcsec |
| `n_archive_records` | archival spectra within 0.5 arcsec in the frozen DJA index |
| `n_eligible_records` | of those, prism spectra eligible for the V-shape test |
| `n_secure_lowz_records` | records with a secure (grade 3) redshift at z <= 3 |
| `n_secure_highz_records` | records with a secure redshift above 3 |
| `archive_best_grade` | best redshift grade among the records |
| `archive_z_best_grade3` | the secure redshift where one exists |
| `archive_gratings` | gratings of the records |
| `archive_files` | spectrum files of the records |
| `in_census` | row in the spectroscopic census of the earlier work |
| `spec_tested` | row-level flag: the V-shape test was evaluated on a spectrum attached to this row |
| `spec_vshaped` | row-level flag: that spectrum passed the V-shape test |
| `sel_labbe23` | selected by the labbe23 module (Labbe et al. 2025), flag of record |
| `sel_kokorev24` | selected by the kokorev24 module (Kokorev et al. 2024), flag of record |
| `sel_kocevski24` | selected by the kocevski24 module (Kocevski et al. 2025), flag of record |
| `sel_perezgonzalez24` | selected by the perezgonzalez24 module (Perez-Gonzalez et al. 2024), flag of record |
| `sel_barro23` | selected by the barro23 module (Barro et al. 2024b), flag of record |
| `sel_greene24` | selected by the greene24 module (Greene et al. 2024), flag of record |
| `sel_akins24` | selected by the akins24 module (Akins et al. 2025), flag of record |
| `m_f090w` | inverse-hyperbolic-sine magnitude feature of F090W (no zero point; add 23.9 to compare with AB at high S/N) |
| `m_f115w` | inverse-hyperbolic-sine magnitude feature of F115W (no zero point; add 23.9 to compare with AB at high S/N) |
| `m_f150w` | inverse-hyperbolic-sine magnitude feature of F150W (no zero point; add 23.9 to compare with AB at high S/N) |
| `m_f200w` | inverse-hyperbolic-sine magnitude feature of F200W (no zero point; add 23.9 to compare with AB at high S/N) |
| `m_f277w` | inverse-hyperbolic-sine magnitude feature of F277W (no zero point; add 23.9 to compare with AB at high S/N) |
| `m_f356w` | inverse-hyperbolic-sine magnitude feature of F356W (no zero point; add 23.9 to compare with AB at high S/N) |
| `m_f444w` | inverse-hyperbolic-sine magnitude feature of F444W (no zero point; add 23.9 to compare with AB at high S/N) |
| `asnr_f090w` | asinh signal-to-noise feature of F090W |
| `asnr_f115w` | asinh signal-to-noise feature of F115W |
| `asnr_f150w` | asinh signal-to-noise feature of F150W |
| `asnr_f200w` | asinh signal-to-noise feature of F200W |
| `asnr_f277w` | asinh signal-to-noise feature of F277W |
| `asnr_f356w` | asinh signal-to-noise feature of F356W |
| `asnr_f444w` | asinh signal-to-noise feature of F444W |
| `c_f090w_f115w` | feature colour F090W minus F115W on the asinh scale |
| `c_f115w_f150w` | feature colour F115W minus F150W on the asinh scale |
| `c_f150w_f200w` | feature colour F150W minus F200W on the asinh scale |
| `c_f200w_f277w` | feature colour F200W minus F277W on the asinh scale |
| `c_f277w_f356w` | feature colour F277W minus F356W on the asinh scale |
| `c_f356w_f444w` | feature colour F356W minus F444W on the asinh scale |
| `c_f115w_f200w` | feature colour F115W minus F200W on the asinh scale |
| `c_f200w_f356w` | feature colour F200W minus F356W on the asinh scale |
| `c_f277w_f444w` | feature colour F277W minus F444W on the asinh scale |
| `slope_blue` | weighted observed-frame slope over the blue bands (feature) |
| `slope_red` | weighted observed-frame slope over the red bands (feature) |
| `v_curv` | V-curvature feature |
| `log_rh` | log10 of the catalog half-light radius in arcsec |
| `log_rh_over_rstar` | log10 of r_h over the per-field stellar radius |
| `n_snr_ge1` | number of bands with S/N >= 1 |
| `n_snr_ge3` | number of bands with S/N >= 3 |
| `archive_verdict` | verdict of the frozen archival refit (none produced) |
| `compactness_f444w` | Akins et al. (2025) ratio: flux in 0.2 arcsec over 0.5 arcsec diameter apertures on the F444W mosaic |
| `labbe_compactness` | Labbe et al. (2025) ratio: flux in 0.4 arcsec over 0.2 arcsec diameter apertures |
| `flux_d020` | F444W aperture flux, 0.2 arcsec diameter, microjansky |
| `flux_d040` | F444W aperture flux, 0.4 arcsec diameter, microjansky |
| `flux_d050` | F444W aperture flux, 0.5 arcsec diameter, microjansky |
| `compactness_measurable` | the apertures lie inside the valid footprint |
| `compactness_reason` | why not, when not measurable |
| `compactness_mosaic` | grizli mosaic version the apertures were measured on |
| `compactness_pixscale` | mosaic pixel scale, arcsec |
| `compact_akins` | 0.5 < compactness_f444w <= 0.7 |
| `compact_labbe` | labbe_compactness < 1.7 |
| `followup_tier` | member of the follow-up tier defined in Section 6.3 |
| `in_degraaff26_v2` | member of the de Graaff et al. (2026) v2 list (post-freeze status field) |
| `zspec_degraaff26_v2` | its spectroscopic redshift |
| `oof_score` | out-of-fold ranking score of the primary 48-bag model (fold model that never saw the row's region) |
| `oof_t_05` | the row's own fold 0.5 percent threshold |
| `oof_t_burden` | the row's own fold equal-burden threshold |
| `oof_above_05` | oof_score at or above oof_t_05 |
| `oof_above_burden` | oof_score at or above oof_t_burden |
| `sel_barro24b` | selected by the eighth module, barro24b (Barro et al. 2024a), coded after the freeze |
| `spectrum_status` | five-way spectrum status of Section 6.2 |
| `z_spec_secure` | secure archival redshift where one exists |
| `listed_after_freeze` | listed as a spectroscopic LRD by a later list version (post-freeze status field) |
| `f_f090w_ujy` | catalog total flux in F090W, microjansky (DJA <band>_tot_1) |
| `f_f115w_ujy` | catalog total flux in F115W, microjansky (DJA <band>_tot_1) |
| `f_f150w_ujy` | catalog total flux in F150W, microjansky (DJA <band>_tot_1) |
| `f_f200w_ujy` | catalog total flux in F200W, microjansky (DJA <band>_tot_1) |
| `f_f277w_ujy` | catalog total flux in F277W, microjansky (DJA <band>_tot_1) |
| `f_f356w_ujy` | catalog total flux in F356W, microjansky (DJA <band>_tot_1) |
| `f_f444w_ujy` | catalog total flux in F444W, microjansky (DJA <band>_tot_1) |
| `e_f090w_ujy` | its uncertainty, microjansky (DJA <band>_etot_1) |
| `e_f115w_ujy` | its uncertainty, microjansky (DJA <band>_etot_1) |
| `e_f150w_ujy` | its uncertainty, microjansky (DJA <band>_etot_1) |
| `e_f200w_ujy` | its uncertainty, microjansky (DJA <band>_etot_1) |
| `e_f277w_ujy` | its uncertainty, microjansky (DJA <band>_etot_1) |
| `e_f356w_ujy` | its uncertainty, microjansky (DJA <band>_etot_1) |
| `e_f444w_ujy` | its uncertainty, microjansky (DJA <band>_etot_1) |
| `mag_f090w_ab` | AB magnitude in F090W from the total flux, 23.9 - 2.5 log10(flux) |
| `mag_f115w_ab` | AB magnitude in F115W from the total flux, 23.9 - 2.5 log10(flux) |
| `mag_f150w_ab` | AB magnitude in F150W from the total flux, 23.9 - 2.5 log10(flux) |
| `mag_f200w_ab` | AB magnitude in F200W from the total flux, 23.9 - 2.5 log10(flux) |
| `mag_f277w_ab` | AB magnitude in F277W from the total flux, 23.9 - 2.5 log10(flux) |
| `mag_f356w_ab` | AB magnitude in F356W from the total flux, 23.9 - 2.5 log10(flux) |
| `mag_f444w_ab` | AB magnitude in F444W from the total flux, 23.9 - 2.5 log10(flux) |
| `bd_screen` | brown-dwarf screen: F115W-F200W feature colour below -0.5 (a heuristic photometric screen, not a stellar classification) |

## labelled_rows.csv (5596 rows, 91 columns)

| column | meaning |
|---|---|
| `source_id` | integer key of the catalog row across the nine fields (the join key of every released table) |
| `field` | DJA grizli v7 catalog name (ceers-full, gdn, gds, gds-sw, ngdeep, primer-cosmos-east, primer-cosmos-west, primer-uds-north, primer-uds-south) |
| `id` | catalog id within the field |
| `ra` | catalog right ascension, J2000, degrees |
| `dec` | catalog declination, J2000, degrees |
| `y` | training label: 1 spectroscopic LRD, 0 comparison object, empty for ambiguous or unlabeled |
| `y_strict` | 1 for the 32 sources the spectral test with the compactness criterion admits, else 0 or empty |
| `ambiguous` | flagged ambiguous: removed from the unlabeled pool (and, for the 48 of final class ambiguous, from both classes) |
| `bands_complete` | all seven bands carry a finite flux and uncertainty |
| `in_hviding25_A1` | member of Hviding et al. (2025) table A1 |
| `in_hviding25_B1` | member of Hviding et al. (2025) table B1 (broad-line galaxies) |
| `in_barro25` | member of the Barro et al. (2026) spectroscopic list |
| `in_degraaff26` | member of the de Graaff et al. (2026) v0.1 list |
| `in_perger25` | member of the Perger et al. (2025) compilation |
| `in_kocevski24` | member of the published Kocevski et al. (2025) catalog |
| `spec_tested` | row-level flag: the V-shape test was evaluated on a spectrum attached to this row |
| `spec_vshaped` | row-level flag: that spectrum passed the V-shape test |
| `mag_f444w` | catalog total AB magnitude in F444W |
| `snr_f444w` | catalog signal-to-noise in F444W |
| `r_h_arcsec` | catalog half-light radius (flux_radius) in arcsec |
| `r_star_v1_arcsec` | per-field stellar half-light radius used to normalise r_h |
| `z_phot` | catalog photometric redshift (EAZY, DJA) |
| `sel_labbe23` | selected by the labbe23 module (Labbe et al. 2025), flag of record |
| `sel_kokorev24` | selected by the kokorev24 module (Kokorev et al. 2024), flag of record |
| `sel_kocevski24` | selected by the kocevski24 module (Kocevski et al. 2025), flag of record |
| `sel_perezgonzalez24` | selected by the perezgonzalez24 module (Perez-Gonzalez et al. 2024), flag of record |
| `sel_barro23` | selected by the barro23 module (Barro et al. 2024b), flag of record |
| `sel_greene24` | selected by the greene24 module (Greene et al. 2024), flag of record |
| `sel_akins24` | selected by the akins24 module (Akins et al. 2025), flag of record |
| `picked` | selected by the union of the seven rules (flags of record, full band set) |
| `region` | one of the five held-out sky regions (CEERS, GOODS-N, GOODS-S, COSMOS, UDS) |
| `sky_group` | friends-of-friends group id at 0.5 arcsec; a group is one object where objects are counted |
| `sky_group_size` | number of catalog rows in the sky group |
| `in_support` | row lies in the seven-band support (all seven wide NIRCam bands covered) |
| `oof_score` | out-of-fold ranking score of the primary 48-bag model (fold model that never saw the row's region) |
| `oof_sel_burden` | above the fold's equal-burden threshold |
| `oof_sel_neg1` | above the fold's 1 percent comparison-set threshold |
| `oof_sel_05` | above the fold's 0.5 percent training-row threshold |
| `m_f090w` | inverse-hyperbolic-sine magnitude feature of F090W (no zero point; add 23.9 to compare with AB at high S/N) |
| `m_f115w` | inverse-hyperbolic-sine magnitude feature of F115W (no zero point; add 23.9 to compare with AB at high S/N) |
| `m_f150w` | inverse-hyperbolic-sine magnitude feature of F150W (no zero point; add 23.9 to compare with AB at high S/N) |
| `m_f200w` | inverse-hyperbolic-sine magnitude feature of F200W (no zero point; add 23.9 to compare with AB at high S/N) |
| `m_f277w` | inverse-hyperbolic-sine magnitude feature of F277W (no zero point; add 23.9 to compare with AB at high S/N) |
| `m_f356w` | inverse-hyperbolic-sine magnitude feature of F356W (no zero point; add 23.9 to compare with AB at high S/N) |
| `m_f444w` | inverse-hyperbolic-sine magnitude feature of F444W (no zero point; add 23.9 to compare with AB at high S/N) |
| `asnr_f090w` | asinh signal-to-noise feature of F090W |
| `asnr_f115w` | asinh signal-to-noise feature of F115W |
| `asnr_f150w` | asinh signal-to-noise feature of F150W |
| `asnr_f200w` | asinh signal-to-noise feature of F200W |
| `asnr_f277w` | asinh signal-to-noise feature of F277W |
| `asnr_f356w` | asinh signal-to-noise feature of F356W |
| `asnr_f444w` | asinh signal-to-noise feature of F444W |
| `c_f090w_f115w` | feature colour F090W minus F115W on the asinh scale |
| `c_f115w_f150w` | feature colour F115W minus F150W on the asinh scale |
| `c_f150w_f200w` | feature colour F150W minus F200W on the asinh scale |
| `c_f200w_f277w` | feature colour F200W minus F277W on the asinh scale |
| `c_f277w_f356w` | feature colour F277W minus F356W on the asinh scale |
| `c_f356w_f444w` | feature colour F356W minus F444W on the asinh scale |
| `c_f115w_f200w` | feature colour F115W minus F200W on the asinh scale |
| `c_f200w_f356w` | feature colour F200W minus F356W on the asinh scale |
| `c_f277w_f444w` | feature colour F277W minus F444W on the asinh scale |
| `slope_blue` | weighted observed-frame slope over the blue bands (feature) |
| `slope_red` | weighted observed-frame slope over the red bands (feature) |
| `v_curv` | V-curvature feature |
| `log_rh` | log10 of the catalog half-light radius in arcsec |
| `log_rh_over_rstar` | log10 of r_h over the per-field stellar radius |
| `n_snr_ge1` | number of bands with S/N >= 1 |
| `n_snr_ge3` | number of bands with S/N >= 3 |
| `f_f090w_ujy` | catalog total flux in F090W, microjansky (DJA <band>_tot_1) |
| `f_f115w_ujy` | catalog total flux in F115W, microjansky (DJA <band>_tot_1) |
| `f_f150w_ujy` | catalog total flux in F150W, microjansky (DJA <band>_tot_1) |
| `f_f200w_ujy` | catalog total flux in F200W, microjansky (DJA <band>_tot_1) |
| `f_f277w_ujy` | catalog total flux in F277W, microjansky (DJA <band>_tot_1) |
| `f_f356w_ujy` | catalog total flux in F356W, microjansky (DJA <band>_tot_1) |
| `f_f444w_ujy` | catalog total flux in F444W, microjansky (DJA <band>_tot_1) |
| `e_f090w_ujy` | its uncertainty, microjansky (DJA <band>_etot_1) |
| `e_f115w_ujy` | its uncertainty, microjansky (DJA <band>_etot_1) |
| `e_f150w_ujy` | its uncertainty, microjansky (DJA <band>_etot_1) |
| `e_f200w_ujy` | its uncertainty, microjansky (DJA <band>_etot_1) |
| `e_f277w_ujy` | its uncertainty, microjansky (DJA <band>_etot_1) |
| `e_f356w_ujy` | its uncertainty, microjansky (DJA <band>_etot_1) |
| `e_f444w_ujy` | its uncertainty, microjansky (DJA <band>_etot_1) |
| `c_f115w_f200w_ab` | AB colour F115W minus F200W from the catalog total fluxes (empty where either flux is not positive) |
| `c_f150w_f200w_ab` | AB colour F150W minus F200W from the catalog total fluxes (empty where either flux is not positive) |
| `c_f200w_f356w_ab` | AB colour F200W minus F356W from the catalog total fluxes (empty where either flux is not positive) |
| `c_f200w_f444w_ab` | AB colour F200W minus F444W from the catalog total fluxes (empty where either flux is not positive) |
| `c_f277w_f444w_ab` | AB colour F277W minus F444W from the catalog total fluxes (empty where either flux is not positive) |
| `sel_region_matched` | in the top K of the out-of-fold ranking within its region, K the union of seven's in-support row count there (the primary comparison) |
| `sel_region_matched_eight` | the same with K the union of eight's row count in the region |
| `sel_barro24b` | selected by the eighth module, barro24b (Barro et al. 2024a), coded after the freeze |
| `hand_added_barro26` | one of the seven objects Barro et al. (2026) added by hand; in_barro25 and not hand_added_barro26 is the Barro et al. (2024a) selection's own output |

## label_provenance.csv (5596 rows, 33 columns)

| column | meaning |
|---|---|
| `source_id` | integer key of the catalog row across the nine fields (the join key of every released table) |
| `field` | DJA grizli v7 catalog name (ceers-full, gdn, gds, gds-sw, ngdeep, primer-cosmos-east, primer-cosmos-west, primer-uds-north, primer-uds-south) |
| `id` | catalog id within the field |
| `ra` | catalog right ascension, J2000, degrees |
| `dec` | catalog declination, J2000, degrees |
| `region` | one of the five held-out sky regions (CEERS, GOODS-N, GOODS-S, COSMOS, UDS) |
| `sky_group` | friends-of-friends group id at 0.5 arcsec; a group is one object where objects are counted |
| `sky_group_size` | number of catalog rows in the sky group |
| `in_support` | row lies in the seven-band support (all seven wide NIRCam bands covered) |
| `bands_complete` | all seven bands carry a finite flux and uncertainty |
| `in_hviding25_A1` | member of Hviding et al. (2025) table A1 |
| `in_hviding25_B1` | member of Hviding et al. (2025) table B1 (broad-line galaxies) |
| `in_barro25` | member of the Barro et al. (2026) spectroscopic list |
| `in_degraaff26` | member of the de Graaff et al. (2026) v0.1 list |
| `in_perger25` | member of the Perger et al. (2025) compilation |
| `in_kocevski24` | member of the published Kocevski et al. (2025) catalog |
| `spectrum_eligible` | the source has a spectrum eligible for the V-shape test |
| `test_evaluable_source` | the test could be evaluated on at least one eligible spectrum of the source |
| `verdict_source` | source-level verdict: True if any eligible spectrum passes |
| `test_evaluable_row` | row-level: evaluated on the spectrum attached to this row |
| `verdict_row` | row-level verdict |
| `label` | training label as in labelled_rows.csv |
| `label_strict` | strict label |
| `ambiguous` | flagged ambiguous: removed from the unlabeled pool (and, for the 48 of final class ambiguous, from both classes) |
| `final_class` | spectroscopic LRD, comparison object, or ambiguous (used in neither class) |
| `verdict_rule` | which rule fixed the class |
| `ambiguity_reason` | why the source is ambiguous (empty for the seven B1 broad-line galaxies without an eligible spectrum) |
| `z_spec` | spectroscopic redshift used in the paper |
| `z_spec_source` | list or archive |
| `z_list` | redshift quoted by the input list |
| `z_archive` | archival redshift |
| `archive_grade` | its grade |
| `hand_added_barro26` | one of the seven objects Barro et al. (2026) added by hand; in_barro25 and not hand_added_barro26 is the Barro et al. (2024a) selection's own output |

## oof_scores_primary.parquet (466419 rows, 21 columns)

| column | meaning |
|---|---|
| `source_id` | integer key of the catalog row across the nine fields (the join key of every released table) |
| `field` | DJA grizli v7 catalog name (ceers-full, gdn, gds, gds-sw, ngdeep, primer-cosmos-east, primer-cosmos-west, primer-uds-north, primer-uds-south) |
| `region` | one of the five held-out sky regions (CEERS, GOODS-N, GOODS-S, COSMOS, UDS) |
| `id` | catalog id within the field |
| `ra` | catalog right ascension, J2000, degrees |
| `dec` | catalog declination, J2000, degrees |
| `sky_group` | friends-of-friends group id at 0.5 arcsec; a group is one object where objects are counted |
| `y` | training label: 1 spectroscopic LRD, 0 comparison object, empty for ambiguous or unlabeled |
| `picked` | selected by the union of the seven rules (flags of record, full band set) |
| `in_any_list` | member of any input list or catalog |
| `score_mean` | out-of-fold ranking score, mean over the 48 bags of the fold model |
| `score_std` | standard deviation of the 48 bag outputs |
| `t_burden` | the fold's equal-burden threshold |
| `t_neg1` | the fold's 1 percent comparison-set threshold |
| `t_05` | the fold's 0.5 percent training-row threshold |
| `sel_burden` | score_mean at or above t_burden |
| `sel_neg1` | score_mean at or above t_neg1 |
| `sel_05` | score_mean at or above t_05 |
| `sel_region_matched` | in the top K of the out-of-fold ranking within its region, K the union of seven's in-support row count there (the primary comparison) |
| `sel_region_matched_eight` | the same with K the union of eight's row count in the region |
| `sel_barro24b` | selected by the eighth module, barro24b (Barro et al. 2024a), coded after the freeze |
