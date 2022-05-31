"""
This file tests the functions and properties for the class ModelSettings for managing the data around model settings.
"""
import json
import os
from unittest import TestCase, main
from unittest.mock import patch, PropertyMock

from oasislmf.pytools.data_layer.oasis_files.model_settings import ModelSettings

EXAMPLE_DATA = {
    "model_settings": {
        "event_set": {
            "name": "Event Set",
            "desc": "Piwind Event Set selection",
            "default": "p",
            "options": [
                {"id": "p", "desc": "Probabilistic"}
            ]
         },
        "event_occurrence_id": {
            "name": "Occurrence Set",
            "desc": "PiWind Occurrence selection",
            "default": "lt",
            "options": [
                {"id": "lt", "desc": "Long Term"}
            ]
        }
     },
    "lookup_settings": {
        "supported_perils": [
           {"id": "WSS", "desc": "Single Peril: Storm Surge", "peril_correlation_group": 1},
           {"id": "WTC", "desc": "Single Peril: Tropical Cyclone", "peril_correlation_group": 2},
           {"id": "WW1", "desc": "Group Peril: Windstorm with storm surge", "peril_correlation_group": 3},
           {"id": "WW2", "desc": "Group Peril: Windstorm w/o storm surge", "peril_correlation_group": 4}
        ]
    },
    "data_settings": {
	    "group_fields": ["PortNumber", "AccNumber", "LocNumber"]
    },
    "correlation_ settings": [
        {"peril_correlation_group": 1, "correlation_value": 0.5},
        {"peril_correlation_group": 2, "correlation_value": 0.2}
    ]
}

MISSING_EXAMPLE_DATA = {
    "model_settings": {
        "event_set": {
            "name": "Event Set",
            "desc": "Piwind Event Set selection",
            "default": "p",
            "options": [
                {"id": "p", "desc": "Probabilistic"}
            ]
         },
        "event_occurrence_id": {
            "name": "Occurrence Set",
            "desc": "PiWind Occurrence selection",
            "default": "lt",
            "options": [
                {"id": "lt", "desc": "Long Term"}
            ]
        }
     },
    "lookup_settings": {
        "supported_perils": [
           {"id": "WSS", "desc": "Single Peril: Storm Surge"},
           {"id": "WTC", "desc": "Single Peril: Tropical Cyclone"},
           {"id": "WW1", "desc": "Group Peril: Windstorm with storm surge"},
           {"id": "WW2", "desc": "Group Peril: Windstorm w/o storm surge"}
        ]
    }
}


class TestModelSettings(TestCase):

    STASH_PATH = "./test.json"
    FILE_STASH_PATH = "./path.txt"

    def setUp(self) -> None:
        self.test = ModelSettings()

    def tearDown(self) -> None:
        if os.path.exists(self.STASH_PATH):
            os.remove(self.STASH_PATH)
        if os.path.exists(self.FILE_STASH_PATH):
            os.remove(self.FILE_STASH_PATH)

    def write_json_data(self, data: dict = EXAMPLE_DATA) -> None:
        with open(self.STASH_PATH, "w") as file:
            file.write(json.dumps(data))

    def test___init__(self):
        test = ModelSettings()
        self.assertEqual(None, test._data)
        self.assertEqual(None, test._correlation_map)

    @patch("oasislmf.pytools.data_layer.oasis_files.model_settings.ModelSettings.conversion_table", new_callable=PropertyMock)
    @patch("oasislmf.pytools.data_layer.oasis_files.model_settings.ModelSettings.correlation_map", new_callable=PropertyMock)
    def test_get_correlation_value(self, mock_correlation_map, mock_conversion_table):
        mock_conversion_table.return_value = CONVERSION_TABLE_EXAMPLE_DATA
        mock_correlation_map.return_value = MAPPED_EXAMPLE_DATA
        self.assertEqual(0.0, self.test.get_correlation_value(peril_correlation_group=3))
        self.assertEqual(0.5, self.test.get_correlation_value(peril_correlation_group=1))
        self.assertEqual(0.0, self.test.get_correlation_value(peril_correlation_group=100000))
        self.assertEqual(0.0, self.test.get_correlation_value(peril_id="WW1"))
        self.assertEqual(0.5, self.test.get_correlation_value(peril_id="WSS"))
        self.assertEqual(0.0, self.test.get_correlation_value(peril_id="testing"))

    @patch("oasislmf.pytools.data_layer.oasis_files.model_settings.ModelSettings.file_directory", new_callable=PropertyMock)
    def test_load_data(self, mock_file_directory):
        mock_file_directory.return_value = self.STASH_PATH
        self.write_json_data()
        self.assertEqual(EXAMPLE_DATA, self.test._load_data())

    @patch("oasislmf.pytools.data_layer.oasis_files.model_settings.ModelSettings.file_directory_stash", new_callable=PropertyMock)
    def test_define_path(self, mock_file_directory_stash):
        test_path = "test/to/path.txt"
        mock_file_directory_stash.return_value = self.FILE_STASH_PATH

        with self.assertRaises(FileNotFoundError) as error:
            _ = self.test.file_directory
        expected_message = "stash file not found please implement the defile_path function before trying to read model settings data"
        self.assertEqual(expected_message, str(error.exception))

        ModelSettings.define_path(file_path=test_path)
        self.assertEqual(test_path, self.test.file_directory)

    @patch("oasislmf.pytools.data_layer.oasis_files.model_settings.ModelSettings.file_directory", new_callable=PropertyMock)
    def test_data(self, mock_file_directory):
        mock_file_directory.return_value = self.STASH_PATH
        self.write_json_data()

        self.assertEqual(None, self.test._data)
        self.assertEqual(EXAMPLE_DATA, self.test.data)
        self.assertEqual(EXAMPLE_DATA, self.test._data)

    @patch("oasislmf.pytools.data_layer.oasis_files.model_settings.ModelSettings.file_directory", new_callable=PropertyMock)
    def test_correlation_map(self, mock_file_directory):
        mock_file_directory.return_value = self.STASH_PATH
        self.write_json_data()
        self.assertEqual(MAPPED_EXAMPLE_DATA, self.test.correlation_map)
        self.assertEqual(CONVERSION_TABLE_EXAMPLE_DATA, self.test.conversion_table)

        self.write_json_data(data=MISSING_EXAMPLE_DATA)
        self.test._map_data()
        self.assertEqual(MAPPED_EXAMPLE_DATA, self.test.correlation_map)
        self.assertEqual(CONVERSION_TABLE_EXAMPLE_DATA, self.test.conversion_table)


MAPPED_EXAMPLE_DATA = {
    1: {'id': 'WSS', 'desc': 'Single Peril: Storm Surge', 'peril_correlation_group': 1, 'correlation_value': 0.5},
    2: {'id': 'WTC', 'desc': 'Single Peril: Tropical Cyclone', 'peril_correlation_group': 2, 'correlation_value': 0.2},
    3: {'id': 'WW1', 'desc': 'Group Peril: Windstorm with storm surge', 'peril_correlation_group': 3},
    4: {'id': 'WW2', 'desc': 'Group Peril: Windstorm w/o storm surge', 'peril_correlation_group': 4}
}

CONVERSION_TABLE_EXAMPLE_DATA = {
    'WSS': 1,
    'WTC': 2,
    'WW1': 3,
    'WW2': 4
}


if __name__ == "__main__":
    main()
