"""
This file tests the mapping of the correlation data between supported perils and correlation settings
"""
import os
from unittest import TestCase, main

from oasislmf.preparation.correlations import map_data
from ods_tools.oed.setting_schema import ModelSettingSchema

META_PATH = os.path.realpath(__file__).replace("test_correlations.py", "meta_data/")


class TestMapData(TestCase):

    def setUp(self) -> None:
        settings_path = META_PATH + "model_settings.json"
        self.model_settings = ModelSettingSchema().get(settings_path)

    def tearDown(self) -> None:
        pass

    def test_map_data(self):
        expected_mapped_data = [
            {
                'id': 'WSS',
                'desc': 'Single Peril: Storm Surge',
                'peril_correlation_group': 1,
                'damage_correlation_value': '0.7',
                'hazard_correlation_value': '0.0',
            },
            {
                'id': 'WTC',
                'desc': 'Single Peril: Tropical Cyclone',
                'peril_correlation_group': 2,
                'damage_correlation_value': '0.5',
                'hazard_correlation_value': '0.3',
            },
        ]

        mapped_data = map_data(data=self.model_settings, logger=None)
        self.assertEqual(expected_mapped_data, mapped_data.to_dict('records'))


if __name__ == "__main__":
    main()
