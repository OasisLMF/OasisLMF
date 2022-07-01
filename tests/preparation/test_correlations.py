"""
This file tests the mapping of the correlation data between supported perils and correlation settings
"""
import os
from unittest import main, TestCase

import pandas as pd

from oasislmf.preparation.correlations import map_data, get_correlation_input_items, get_model_settings


META_PATH = os.path.realpath(__file__).replace("test_correlations.py", "meta_data/")


class TestMapData(TestCase):

    def setUp(self) -> None:
        settings_path = META_PATH + "model_settings.json"
        self.model_settings = get_model_settings(settings_path)

    def tearDown(self) -> None:
        pass

    def test_map_data(self):
        mapped_data = map_data(data=self.model_settings)
        self.assertEqual(EXPECTED_MAPPED_DATA, mapped_data.to_dict('records'))

    def test_get_correlation_input_items(self):
        gul_path = META_PATH + "gul_inputs_df.csv"
        settings_path = META_PATH + "model_settings.json"

        gul_inputs_df = pd.read_csv(gul_path)
        correlation_df = get_correlation_input_items(model_settings_path=settings_path, gul_inputs_df=gul_inputs_df)
        correlation_df_check = pd.read_csv(f"{META_PATH}correlation_df.csv")

        correlation_df_check.equals(correlation_df)


EXPECTED_MAPPED_DATA = [
    {'id': 'WSS', 'desc': 'Single Peril: Storm Surge', 'peril_correlation_group': 1, 'correlation_value': '0.7'},
    {'id': 'WTC', 'desc': 'Single Peril: Tropical Cyclone', 'peril_correlation_group': 2, 'correlation_value': '0.5'},
]


if __name__ == "__main__":
    main()
