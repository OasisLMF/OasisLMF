from unittest import TestCase
from unittest.mock import Mock
import numpy as np
import pandas as pd

from oasislmf.utils.data import (
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
        result = prepare_location_df(location_df)
        self.assertIn("loc_idx", result.columns)
        self.assertEqual(result["loc_idx"].tolist(), [0, 1, 2])

    def test_default_field_types(self):
        location_df = pd.DataFrame(
            {
                "PortNumber": [1],
                "AccNumber": ["A11111"],
                "LocNumber": [10001],
                "BIWaitingPeriod": [30],
                "BIPOI": [50000],
            }
        )

        result = prepare_location_df(location_df)
        self.assertIn("BIWaitingPeriodType", result.columns)
        self.assertEqual(result["BIWaitingPeriodType"].iloc[0], 3)
        self.assertIn("BIPOIType", result.columns)
        self.assertEqual(result["BIPOIType"].iloc[0], 3)


class TestPrepareAccountDf(TestCase):

    def test_acc_idx_column(self):
        account_df = pd.DataFrame(
            {
                "PortNumber": [1, 2],
                "AccNumber": ["A11111", "A22222"],
                "PolNumber": ["P100", "P200"],
            }
        )

        result = prepare_account_df(account_df)
        self.assertIn("acc_idx", result.columns)
        self.assertEqual(result["acc_idx"].tolist(), [0, 1])

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
                "PortNumber": [1, 1, 2],
                "AccNumber": ["A11111", "A11111", "A22222"],
                "PolNumber": ["P100", "P100", "P200"],
                "LayerNumber": [1, 2, 1],
            }
        )

        result = prepare_account_df(account_df)
        self.assertIn("layer_id", result.columns)
        self.assertEqual(result["layer_id"].dtype, np.uint32)


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
