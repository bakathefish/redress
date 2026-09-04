
import numpy as np
import pandas as pd
import pytest

from redress import contracts as con
from redress import splits as sp


def valid_photometry(n=4) -> pd.DataFrame:
    """A minimal fully-valid photometry table with two bands covered."""
    df = pd.DataFrame(
        {
            "object_id": [f"src{i}" for i in range(n)],
            "field": ["ceers"] * n,
            "ra_deg": np.linspace(214.8, 214.9, n),
            "dec_deg": np.linspace(52.7, 52.8, n),
            "source_catalog": ["dja-test-v0"] * n,
        }
    )
    for band in ("f150w", "f444w"):
        df[f"f_{band}_ujy"] = np.linspace(0.5, 2.0, n)
        df[f"e_{band}_ujy"] = 0.05
        df[f"cov_{band}"] = True
    # one band entirely uncovered on row 0 — the legal missingness pattern
    df.loc[0, "f_f150w_ujy"] = np.nan
    df.loc[0, "e_f150w_ujy"] = np.nan
    df.loc[0, "cov_f150w"] = False
    return df


def valid_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "field": ["ceers", "ceers", "goods-s"],
            "object_id": ["src1", "src2", "src9"],
            "label_lrd": np.array([1, 0, -1], dtype=np.int8),
            "label_source": ["perez-gonzalez-2026", "ginolfi-2026", "unlabeled"],
            "z_spec": [5.3, 4.1, np.nan],
            "targeting_program": ["rubies", "wide", ""],
            "grism_unbiased": [False, False, False],
        }
    )


# ---------------------------------------------------------------- photometry: valid

def test_valid_photometry_passes():
    con.validate_photometry_table(valid_photometry())  # no raise


def test_negative_flux_is_legal():
    df = valid_photometry()
    df.loc[1, "f_f444w_ujy"] = -0.3  # forced photometry scatters below zero
    con.validate_photometry_table(df)  # no raise


# ---------------------------------------------------------------- photometry: violations

def test_missing_required_column():
    df = valid_photometry().drop(columns=["ra_deg"])
    with pytest.raises(con.ContractError, match="ra_deg"):
        con.validate_photometry_table(df)


def test_no_photometry_bands_at_all():
    df = valid_photometry()[["object_id", "field", "ra_deg", "dec_deg", "source_catalog"]]
    with pytest.raises(con.ContractError, match="band"):
        con.validate_photometry_table(df)


def test_unknown_band_rejected():
    df = valid_photometry()
    df["f_f999x_ujy"] = 1.0
    df["e_f999x_ujy"] = 0.1
    df["cov_f999x"] = True
    with pytest.raises(con.ContractError, match="f999x"):
        con.validate_photometry_table(df)


def test_incomplete_band_triplet():
    df = valid_photometry().drop(columns=["cov_f444w"])
    with pytest.raises(con.ContractError, match="f444w"):
        con.validate_photometry_table(df)


def test_ra_out_of_range():
    df = valid_photometry()
    df.loc[2, "ra_deg"] = 361.0
    with pytest.raises(con.ContractError, match="ra_deg"):
        con.validate_photometry_table(df)


def test_nan_position_rejected():
    df = valid_photometry()
    df.loc[1, "dec_deg"] = np.nan
    with pytest.raises(con.ContractError, match="dec_deg"):
        con.validate_photometry_table(df)


def test_covered_but_nan_flux_rejected():
    # cov=True with NaN flux is a violation, "not a shrug" (DATA_CONTRACTS §1).
    df = valid_photometry()
    df.loc[2, "f_f444w_ujy"] = np.nan
    with pytest.raises(con.ContractError, match="cov_f444w"):
        con.validate_photometry_table(df)


def test_uncovered_but_finite_flux_rejected():
    df = valid_photometry()
    df.loc[0, "f_f150w_ujy"] = 1.0  # cov=False but a flux present
    with pytest.raises(con.ContractError, match="cov_f150w"):
        con.validate_photometry_table(df)


def test_nonpositive_error_rejected():
    df = valid_photometry()
    df.loc[3, "e_f444w_ujy"] = 0.0
    with pytest.raises(con.ContractError, match="e_f444w_ujy"):
        con.validate_photometry_table(df)


def test_sentinel_value_rejected():
    # -99 style sentinels must have been converted at ingest.
    df = valid_photometry()
    df.loc[1, "f_f150w_ujy"] = -99.0
    with pytest.raises(con.ContractError, match="sentinel"):
        con.validate_photometry_table(df)


def test_duplicate_object_id_within_field_rejected():
    df = valid_photometry()
    df.loc[1, "object_id"] = "src0"
    with pytest.raises(con.ContractError, match="duplicate"):
        con.validate_photometry_table(df)


def test_unknown_field_slug_rejected():
    df = valid_photometry()
    df.loc[0, "field"] = "narnia"
    with pytest.raises(con.ContractError, match="narnia"):
        con.validate_photometry_table(df)


def test_custom_field_set_accepted():
    df = valid_photometry()
    df["field"] = "my-sim-field"
    con.validate_photometry_table(df, allowed_fields={"my-sim-field"})  # no raise


def test_string_false_coverage_rejected():
    # council r4b deciding test: astype(bool) would coerce "False" to True — a
    # silent uncovered->covered flip. Non-bool cov dtype must raise.
    df = valid_photometry()
    df["cov_f444w"] = df["cov_f444w"].astype(str)
    with pytest.raises(con.ContractError, match="cov_f444w"):
        con.validate_photometry_table(df)


# ---------------------------------------------------------------- labels

def test_valid_labels_pass():
    con.validate_label_table(valid_labels())  # no raise


def test_label_values_constrained():
    df = valid_labels()
    df.loc[0, "label_lrd"] = 2
    with pytest.raises(con.ContractError, match="label_lrd"):
        con.validate_label_table(df)


def test_negative_z_spec_rejected():
    df = valid_labels()
    df.loc[0, "z_spec"] = -1.0
    with pytest.raises(con.ContractError, match="z_spec"):
        con.validate_label_table(df)


def test_duplicate_label_rows_rejected():
    df = valid_labels()
    df.loc[1, ["field", "object_id"]] = ["ceers", "src1"]
    with pytest.raises(con.ContractError, match="duplicate"):
        con.validate_label_table(df)


def test_nonnumeric_z_spec_rejected():
    # council r4b deciding test: "abc" must raise, never coerce to NaN
    # (NaN means "no measurement", not "bad entry").
    df = valid_labels()
    df["z_spec"] = df["z_spec"].astype(object)
    df.loc[0, "z_spec"] = "abc"
    with pytest.raises(con.ContractError, match="z_spec"):
        con.validate_label_table(df)


def test_nan_label_join_keys_rejected():
    # council r4b deciding test: NaN keys poison joins and leakage guards.
    df = valid_labels()
    df.loc[0, "object_id"] = np.nan
    with pytest.raises(con.ContractError, match="join key"):
        con.validate_label_table(df)


def test_label_field_slug_validated_against_registry():
    # council r4b deciding test: labels obey the same field registry photometry does.
    df = valid_labels()
    df.loc[0, "field"] = "narnia"
    with pytest.raises(con.ContractError, match="narnia"):
        con.validate_label_table(df)


def test_nonbool_grism_flag_rejected():
    # same defect class as the cov counter, applied to the label table's flag
    df = valid_labels()
    df["grism_unbiased"] = df["grism_unbiased"].astype(str)
    with pytest.raises(con.ContractError, match="grism_unbiased"):
        con.validate_label_table(df)


# ---------------------------------------------------------------- forbidden features

def test_forbidden_list_blocks_label_adjacent_columns():
    bad = ["f_f444w_ujy", "z_spec"]
    with pytest.raises(sp.LeakageError):
        sp.assert_no_forbidden_features(bad, con.FORBIDDEN_DETECTOR_FEATURES)
    for name in ("label_lrd", "targeting_program", "grism_unbiased", "spec_snr",
                 "fwhm_ha", "ew_ha", "line_flux_ha"):
        with pytest.raises(sp.LeakageError):
            sp.assert_no_forbidden_features([name], con.FORBIDDEN_DETECTOR_FEATURES)


def test_forbidden_list_allows_photometry_and_zphot():
    ok = ["f_f444w_ujy", "e_f150w_ujy", "cov_f770w", "z_phot", "beta_opt",
          "compactness_f444w"]
    sp.assert_no_forbidden_features(ok, con.FORBIDDEN_DETECTOR_FEATURES)  # no raise
