from unittest import main, TestCase

from oasislmf.preparation.gul_inputs import map_data
import os


class TestMapData(TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_map_data(self):
        mapped_data = map_data(data=INPUT_DATA)
        self.assertEqual(EXPECTED_MAPPED_DATA, mapped_data.to_dict('records'))


EXPECTED_MAPPED_DATA = [
    {'id': 'WSS', 'desc': 'Single Peril: Storm Surge', 'peril_correlation_group': 1, 'correlation_value': '0.7'},
    {'id': 'WTC', 'desc': 'Single Peril: Tropical Cyclone', 'peril_correlation_group': 2, 'correlation_value': '0.5'},
    {'id': 'WTF', 'desc': 'Single Peril: Tropical Flood', 'peril_correlation_group': 2, 'correlation_value': '0.5'}
]


INPUT_DATA = {
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
           {"id": "WSS", "desc": "Single Peril: Storm Surge", "peril_correlation_group":  1},
           {"id": "WTC", "desc": "Single Peril: Tropical Cyclone", "peril_correlation_group":  2},
           {"id": "WW1", "desc": "Group Peril: Windstorm with storm surge", "peril_correlation_group":  3},
           {"id": "WW2", "desc": "Group Peril: Windstorm w/o storm surge", "peril_correlation_group":  4},
           {"id": "WTF", "desc": "Single Peril: Tropical Flood", "peril_correlation_group":  2}
        ]
    },
    "correlation_settings": [
      {"peril_correlation_group":  1, "correlation_value":  "0.7"},
      {"peril_correlation_group":  2, "correlation_value":  "0.5"}
    ]
}


if __name__ == "__main__":
    main()
