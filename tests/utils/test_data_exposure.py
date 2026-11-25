from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock
import numpy as np
import pandas as pd
from oasis_data_manager.filestore.backends.local import LocalStorage

from oasislmf.utils.data import (
    get_exposure_data,
    local_oed_files_as_paths,
    prepare_account_df,
    prepare_location_df,
    prepare_oed_exposure,
    prepare_reinsurance_df,
)


class TestPrepareLocationDf(TestCase):

    def test_loc_idx_column(self):
        location_df = pd.DataFrame(
            {
                "PortNumber": [1, 1, 2],
                "AccNumber": ["A11111", "A11111", "A22222"],
                "LocNumber": [10001, 10002, 10003],
                "BuildingTIV": [500000, 750000, 1200000],
            }
        )
        location_df.index = pd.Index([10, 20, 30], name="original_index")
        result = prepare_location_df(location_df)
        self.assertIn("loc_idx", result.columns)
        self.assertEqual(result["loc_idx"].tolist(), location_df.index.tolist())


class TestPrepareAccountDf(TestCase):

    def test_acc_idx_column(self):
        account_df = pd.DataFrame(
            {
                "PortNumber": [1, 2],
                "AccNumber": ["A11111", "A22222"],
                "PolNumber": ["P100", "P200"],
            }
        )
        account_df.index = pd.Index([100, 200], name="original_index")
        result = prepare_account_df(account_df)
        self.assertIn("acc_idx", result.columns)
        self.assertEqual(result["acc_idx"].tolist(), account_df.index.tolist())

    def test_layer_number_default(self):
        account_df = pd.DataFrame(
            {"PortNumber": [1], "AccNumber": ["A11111"], "PolNumber": ["P100"]}
        )
        result = prepare_account_df(account_df)
        self.assertIn("LayerNumber", result.columns)
        self.assertEqual(result["LayerNumber"].iloc[0], 1)

    def test_layer_id(self):
        account_df = pd.DataFrame(
            {
                "PortNumber": [1, 1, 1],
                "AccNumber": ["A11111", "A11111", "A11111"],
                "PolNumber": ["P100", "P100", "P100"],
                "LayerNumber": [1, 1, 2],
            }
        )

        result = prepare_account_df(account_df)
        self.assertIn("layer_id", result.columns)
        self.assertEqual(result["layer_id"].dtype, np.uint32)
        self.assertEqual(result["layer_id"].iloc[0], result["layer_id"].iloc[1])
        self.assertNotEqual(result["layer_id"].iloc[1], result["layer_id"].iloc[2])


class TestPrepareReinsuranceDf(TestCase):

    def test_default_columns(self):
        ri_info = pd.DataFrame({"ReinsNumber": [1], "RiskLevel": ["SEL"]})
        ri_scope = pd.DataFrame({"ReinsNumber": [1]})

        result_info, result_scope = prepare_reinsurance_df(ri_info, ri_scope)
        self.assertIn("CededPercent", result_info.columns)
        self.assertIn("RiskLimit", result_info.columns)
        self.assertIn("OccLimit", result_info.columns)
        self.assertIn("TreatyShare", result_info.columns)
        self.assertIn("PortNumber", result_scope.columns)
        self.assertIn("AccNumber", result_scope.columns)

    def test_default_values(self):
        ri_info = pd.DataFrame({"ReinsNumber": [1], "RiskLevel": ["SEL"]})
        ri_scope = pd.DataFrame({"ReinsNumber": [1]})

        result_info, result_scope = prepare_reinsurance_df(ri_info, ri_scope)
        self.assertEqual(result_info["CededPercent"].iloc[0], 1.0)
        self.assertEqual(result_info["TreatyShare"].iloc[0], 1.0)
        self.assertEqual(result_info["AttachmentBasis"].iloc[0], "LO")
        self.assertEqual(result_scope["CededPercent"].iloc[0], 1.0)

    def test_fill_empty_and_fillna_defaults(self):
        ri_info = pd.DataFrame(
            {
                "ReinsNumber": [1],
                "RiskLevel": [""],
                "CededPercent": [np.nan],
            }
        )
        ri_scope = pd.DataFrame(
            {
                "ReinsNumber": [1],
                "CededPercent": [np.nan],
            }
        )
        result_info, result_scope = prepare_reinsurance_df(ri_info, ri_scope)
        self.assertEqual(result_info["RiskLevel"].iloc[0], "SEL")
        self.assertEqual(result_info["CededPercent"].iloc[0], 1.0)
        self.assertEqual(result_scope["CededPercent"].iloc[0], 1.0)


class TestPrepareOedExposure(TestCase):

    def test_location_data(self):
        mock_exposure = Mock()
        mock_location = Mock()
        mock_location.dataframe = pd.DataFrame(
            {"PortNumber": [1], "LocNumber": [10001]}
        )
        mock_exposure.location = mock_location
        mock_exposure.account = None
        mock_exposure.ri_info = None
        mock_exposure.ri_scope = None
        mock_exposure.get_subject_at_risk_source.return_value.dataframe = None

        prepare_oed_exposure(mock_exposure)
        self.assertIn("loc_idx", mock_exposure.location.dataframe.columns)

    def test_account_data(self):
        mock_exposure = Mock()
        mock_account = Mock()
        mock_account.dataframe = pd.DataFrame(
            {"PortNumber": [1], "AccNumber": ["A11111"], "PolNumber": ["P100"]}
        )
        mock_exposure.location = None
        mock_exposure.account = mock_account
        mock_exposure.ri_info = None
        mock_exposure.ri_scope = None
        mock_exposure.get_subject_at_risk_source.return_value.dataframe = None
        prepare_oed_exposure(mock_exposure)
        self.assertIn("acc_idx", mock_exposure.account.dataframe.columns)
        self.assertIn("layer_id", mock_exposure.account.dataframe.columns)


EXPOSURE_RUN_DIR = Path(__file__).resolve().parents[1] / "assets" / "test_exposure_run"


def test_local_oed_files_as_paths():
    location = str(EXPOSURE_RUN_DIR / "location.csv")
    config = {"location": location, "account": "https://example.com/account.csv", "ri_info": None, "check_oed": False}
    assert local_oed_files_as_paths(config) is config
    assert config == {"location": Path(location), "account": "https://example.com/account.csv", "ri_info": None, "check_oed": False}


def test_get_exposure_data_reads_string_paths_without_storage_url(monkeypatch):
    """OED files given as strings are not read through ``LocalStorage.get_storage_url``.

    That URL is ``file://C:\\...`` on Windows, whose path part is empty; the fake below fails the same way on any OS.
    """

    def windows_storage_url(self, filename=None, **kwargs):
        raise OSError(f"<urlopen error [WinError 3] The system cannot find the path specified: ''> ({filename})")

    monkeypatch.setattr(LocalStorage, "get_storage_url", windows_storage_url)
    config = {
        "location": str(EXPOSURE_RUN_DIR / "location.csv"),
        "account": str(EXPOSURE_RUN_DIR / "account.csv"),
        "check_oed": False,
        "use_field": True,
    }
    step = SimpleNamespace(kwargs={}, get_exposure_data_config=lambda: dict(config))

    exposure = get_exposure_data(step)

    assert len(exposure.location.dataframe) == len(pd.read_csv(config["location"]))
    assert len(exposure.account.dataframe) == len(pd.read_csv(config["account"]))
