import os 
from unittest import TestCase, main
from unittest.mock import patch, PropertyMock

from pandas import DataFrame

from oasislmf.pytools.getmodel.get_model_process import GetModelProcess


class TestGetModelProcess(TestCase):

    def setUp(self) -> None:
        """
        Order of operations:

        merge_model_with_footprint
        merge_complex_items
        merge_vulnerabilities
        merge_damage_bin_dict
        """
        self.test = GetModelProcess(data_path=".")
        script_dir =  os.path.dirname(os.path.realpath(__file__))
        os.chdir(script_dir)


    # def test__data_loading(self):
    #     self.assertEqual(DataFrame, type(self.test.damage_bin.value))
    #     self.assertEqual(DataFrame, type(self.test.footprint.value))
    #     self.assertEqual(DataFrame, type(self.test.vulnerabilities.value))

    def test_stream_events(self):
        test = GetModelProcess(data_path="./static/", events=DataFrame([{"one": 1, "two": 2}, {"one": 1, "two": 2}]))

        self.assertEqual(DataFrame, type(test.events.value))
        self.assertEqual([{'one': 1, 'two': 2}, {'one': 1, 'two': 2}], list(test.events.value.T.to_dict().values()))

    def test_stream_type(self):
        self.assertEqual(b'\x01\x00\x00\x00', self.test.STREAM_HEADER)
        self.test.stream_type = 2
        self.assertEqual(b'\x02\x00\x00\x00', self.test.STREAM_HEADER)

    @patch("oasislmf.pytools.getmodel.get_model_process.GetModelProcess.footprint", new_callable=PropertyMock)
    @patch("oasislmf.pytools.getmodel.get_model_process.GetModelProcess.events", new_callable=PropertyMock)
    def test_merge_model_with_footprint(self, mock_events, mock_footprint):
        mock_events.return_value.value = DataFrame([
            {"event_id": 1},
            {"event_id": 2},
            {"event_id": 3},
            {"event_id": 4}
        ])

        # each event area peril id should sum to one over the intensity bins
        mock_footprint.return_value.value = DataFrame([
            {"event_id": 1, "areaperil_id": 10, "intensity_bin_id": 1, "probability": 0.47},
            {"event_id": 1, "areaperil_id": 10, "intensity_bin_id": 2, "probability": 0.53},

            {"event_id": 2, "areaperil_id": 20, "intensity_bin_id": 1, "probability": 0.30},
            {"event_id": 2, "areaperil_id": 20, "intensity_bin_id": 2, "probability": 0.70},

            {"event_id": 3, "areaperil_id": 30, "intensity_bin_id": 3, "probability": 0.28},
            {"event_id": 3, "areaperil_id": 30, "intensity_bin_id": 4, "probability": 0.72},

            {"event_id": 4, "areaperil_id": 40, "intensity_bin_id": 4, "probability": 0.81},
            {"event_id": 4, "areaperil_id": 40, "intensity_bin_id": 5, "probability": 0.19}
        ])

        self.test.merge_model_with_footprint()

        self.assertEqual([
            {'event_id': 1, 'order': 0, 'area_peril_id': 10, 'intensity_bin_id': 1, 'footprint_probability': 0.47},
            {'event_id': 1, 'order': 0, 'area_peril_id': 10, 'intensity_bin_id': 2, 'footprint_probability': 0.53},

            {'event_id': 2, 'order': 1, 'area_peril_id': 20, 'intensity_bin_id': 1, 'footprint_probability': 0.3},
            {'event_id': 2, 'order': 1, 'area_peril_id': 20, 'intensity_bin_id': 2, 'footprint_probability': 0.7},

            {'event_id': 3, 'order': 2, 'area_peril_id': 30, 'intensity_bin_id': 3, 'footprint_probability': 0.28},
            {'event_id': 3, 'order': 2, 'area_peril_id': 30, 'intensity_bin_id': 4, 'footprint_probability': 0.72},

            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'intensity_bin_id': 4, 'footprint_probability': 0.81},
            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'intensity_bin_id': 5, 'footprint_probability': 0.19}
        ],
            list(self.test.model.T.to_dict().values())
        )

    @patch("oasislmf.pytools.getmodel.get_model_process.GetModelProcess.vulnerabilities", new_callable=PropertyMock)
    def test_merge_vulnerabilities(self, mock_vulnerabilities):
        self.test.model = DataFrame([
            {'event_id': 1, 'order': 0, 'area_peril_id': 10, 'intensity_bin_id': 1, 'footprint_probability': 0.47},
            {'event_id': 1, 'order': 0, 'area_peril_id': 10, 'intensity_bin_id': 2, 'footprint_probability': 0.53},

            {'event_id': 2, 'order': 1, 'area_peril_id': 20, 'intensity_bin_id': 1, 'footprint_probability': 0.3},
            {'event_id': 2, 'order': 1, 'area_peril_id': 20, 'intensity_bin_id': 2, 'footprint_probability': 0.7},

            {'event_id': 3, 'order': 2, 'area_peril_id': 30, 'intensity_bin_id': 3, 'footprint_probability': 0.28},
            {'event_id': 3, 'order': 2, 'area_peril_id': 30, 'intensity_bin_id': 4, 'footprint_probability': 0.72},

            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'intensity_bin_id': 4, 'footprint_probability': 0.81},
            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'intensity_bin_id': 5, 'footprint_probability': 0.19}
        ])

        # vulnerability and intensity combination needs to sum to one across the damage bins
        mock_vulnerabilities.return_value.value = DataFrame([
            {"vulnerability_id": 1, "intensity_bin_id": 1, "damage_bin_id": 1, "probability": 0.45},
            {"vulnerability_id": 1, "intensity_bin_id": 2, "damage_bin_id": 2, "probability": 0.65},

            {"vulnerability_id": 2, "intensity_bin_id": 3, "damage_bin_id": 1, "probability": 0.78},
            {"vulnerability_id": 2, "intensity_bin_id": 4, "damage_bin_id": 2, "probability": 0.22},

            {"vulnerability_id": 3, "intensity_bin_id": 1, "damage_bin_id": 1, "probability": 0.89},
            {"vulnerability_id": 3, "intensity_bin_id": 4, "damage_bin_id": 3, "probability": 0.11},

            {"vulnerability_id": 4, "intensity_bin_id": 2, "damage_bin_id": 1, "probability": 0.35},
            {"vulnerability_id": 4, "intensity_bin_id": 5, "damage_bin_id": 3, "probability": 0.65},
        ])

        self.test.merge_vulnerabilities()

        self.assertEqual([
            {'event_id': 2.0, 'order': 1.0, 'area_peril_id': 20.0, 'footprint_probability': 0.3, 'vulnerability_id': 3.0, 'damage_bin_id': 1.0, 'vulnerability_probability': 0.89},
            {'event_id': 2.0, 'order': 1.0, 'area_peril_id': 20.0, 'footprint_probability': 0.7, 'vulnerability_id': 4.0, 'damage_bin_id': 1.0, 'vulnerability_probability': 0.35},
            {'event_id': 3.0, 'order': 2.0, 'area_peril_id': 30.0, 'footprint_probability': 0.28, 'vulnerability_id': 2.0, 'damage_bin_id': 1.0, 'vulnerability_probability': 0.78},
            {'event_id': 4.0, 'order': 3.0, 'area_peril_id': 40.0, 'footprint_probability': 0.81, 'vulnerability_id': 3.0, 'damage_bin_id': 3.0, 'vulnerability_probability': 0.11},
            {'event_id': 4.0, 'order': 3.0, 'area_peril_id': 40.0, 'footprint_probability': 0.19, 'vulnerability_id': 4.0, 'damage_bin_id': 3.0, 'vulnerability_probability': 0.65}
        ],
            list(self.test.model.T.to_dict().values())
        )

    @patch("oasislmf.pytools.getmodel.get_model_process.GetModelProcess.items", new_callable=PropertyMock)
    def test_filter_footprint(self, mock_items):
        self.test.model = DataFrame([
            {'event_id': 1, 'order': 0, 'area_peril_id': 10, 'footprint_probability': 0.47, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.45},
            {'event_id': 1, 'order': 0, 'area_peril_id': 10, 'footprint_probability': 0.47, 'vulnerability_id': 3, 'damage_bin_id': 1, 'vulnerability_probability': 0.89},
            {'event_id': 2, 'order': 1, 'area_peril_id': 20, 'footprint_probability': 0.3, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.45},
            {'event_id': 2, 'order': 1, 'area_peril_id': 20, 'footprint_probability': 0.3, 'vulnerability_id': 3, 'damage_bin_id': 1, 'vulnerability_probability': 0.89},
            {'event_id': 1, 'order': 0, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.0},
            {'event_id': 1, 'order': 0, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 1, 'damage_bin_id': 2, 'vulnerability_probability': 0.65},
            {'event_id': 1, 'order': 0, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 4, 'damage_bin_id': 1, 'vulnerability_probability': 0.35},
            {'event_id': 2, 'order': 1, 'area_peril_id': 20, 'footprint_probability': 0.7, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.0},
            {'event_id': 2, 'order': 1, 'area_peril_id': 20, 'footprint_probability': 0.7, 'vulnerability_id': 1, 'damage_bin_id': 2, 'vulnerability_probability': 0.65},
            {'event_id': 2, 'order': 1, 'area_peril_id': 20, 'footprint_probability': 0.7, 'vulnerability_id': 4, 'damage_bin_id': 1, 'vulnerability_probability': 0.35},
            {'event_id': 3, 'order': 2, 'area_peril_id': 30, 'footprint_probability': 0.28, 'vulnerability_id': 2, 'damage_bin_id': 1, 'vulnerability_probability': 0.78},
            {'event_id': 3, 'order': 2, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 2, 'damage_bin_id': 1, 'vulnerability_probability': 0.0},
            {'event_id': 3, 'order': 2, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 2, 'damage_bin_id': 2, 'vulnerability_probability': 0.22},
            {'event_id': 3, 'order': 2, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 1, 'vulnerability_probability': 0.0},
            {'event_id': 3, 'order': 2, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 2, 'vulnerability_probability': 0.0},
            {'event_id': 3, 'order': 2, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 3, 'vulnerability_probability': 0.11},
            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'footprint_probability': 0.81, 'vulnerability_id': 2, 'damage_bin_id': 1, 'vulnerability_probability': 0.0},
            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'footprint_probability': 0.81, 'vulnerability_id': 2, 'damage_bin_id': 2, 'vulnerability_probability': 0.22},
            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'footprint_probability': 0.81, 'vulnerability_id': 3, 'damage_bin_id': 1, 'vulnerability_probability': 0.0},
            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'footprint_probability': 0.81, 'vulnerability_id': 3, 'damage_bin_id': 2, 'vulnerability_probability': 0.0},
            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'footprint_probability': 0.81, 'vulnerability_id': 3, 'damage_bin_id': 3, 'vulnerability_probability': 0.11},
            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 1, 'vulnerability_probability': 0.0},
            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 2, 'vulnerability_probability': 0.0},
            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 3, 'vulnerability_probability': 0.65}
        ])

        mock_items.return_value.value = DataFrame([
            {"item_id": 1, "coverage_id": 1, "areaperil_id": 10, "vulnerability_id": 1, "group_id": 1},
            {"item_id": 2, "coverage_id": 1, "areaperil_id": 20, "vulnerability_id": 2, "group_id": 1},
            {"item_id": 3, "coverage_id": 2, "areaperil_id": 30, "vulnerability_id": 3, "group_id": 2},
            {"item_id": 4, "coverage_id": 2, "areaperil_id": 40, "vulnerability_id": 4, "group_id": 2},
        ])
        self.test.filter_footprint()

        self.assertEqual([
            {'event_id': 1, 'order': 0.0, 'area_peril_id': 10, 'footprint_probability': 0.47, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.45},
            {'event_id': 1, 'order': 0.0, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.0},
            {'event_id': 1, 'order': 0.0, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 1, 'damage_bin_id': 2, 'vulnerability_probability': 0.65},
            {'event_id': 3, 'order': 2.0, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 1, 'vulnerability_probability': 0.0},
            {'event_id': 3, 'order': 2.0, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 2, 'vulnerability_probability': 0.0},
            {'event_id': 3, 'order': 2.0, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 3, 'vulnerability_probability': 0.11},
            {'event_id': 4, 'order': 3.0, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 1, 'vulnerability_probability': 0.0},
            {'event_id': 4, 'order': 3.0, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 2, 'vulnerability_probability': 0.0},
            {'event_id': 4, 'order': 3.0, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 3, 'vulnerability_probability': 0.65}
        ],
            list(self.test.model.T.to_dict().values())
        )

    @patch("oasislmf.pytools.getmodel.get_model_process.GetModelProcess.damage_bin", new_callable=PropertyMock)
    def test_merge_damage_bin_dict(self, mock_damage_bin):
        self.test.model = DataFrame([
            {'event_id': 1, 'order': 0, 'area_peril_id': 10, 'footprint_probability': 0.47, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.45},
            {'event_id': 1, 'order': 0, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.0},
            {'event_id': 1, 'order': 0, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 1, 'damage_bin_id': 2, 'vulnerability_probability': 0.65},
            {'event_id': 3, 'order': 2, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 1, 'vulnerability_probability': 0.0},
            {'event_id': 3, 'order': 2, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 2, 'vulnerability_probability': 0.0},
            {'event_id': 3, 'order': 2, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 3, 'vulnerability_probability': 0.11},
            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 1, 'vulnerability_probability': 0.0},
            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 2, 'vulnerability_probability': 0.0},
            {'event_id': 4, 'order': 3, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 3, 'vulnerability_probability': 0.65}
        ])

        mock_damage_bin.return_value.value = DataFrame([
            {"bin_index": 1, "bin_from": 0.0, "bin_to": 0.1, "interpolation": 0.05, "interval_type": 1203},
            {"bin_index": 2, "bin_from": 0.1, "bin_to": 0.2, "interpolation": 0.15, "interval_type": 1200},
            {"bin_index": 3, "bin_from": 0.2, "bin_to": 0.3, "interpolation": 0.25, "interval_type": 1202},
            {"bin_index": 4, "bin_from": 0.3, "bin_to": 0.4, "interpolation": 0.35, "interval_type": 1202}
        ])

        self.test.merge_damage_bin_dict()

        self.assertEqual([
            {'event_id': 1, 'area_peril_id': 10, 'footprint_probability': 0.47, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.45, 'interpolation': 0.05},
            {'event_id': 1, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.0, 'interpolation': 0.05},
            {'event_id': 1, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 1, 'damage_bin_id': 2, 'vulnerability_probability': 0.65, 'interpolation': 0.15},
            {'event_id': 3, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 1, 'vulnerability_probability': 0.0, 'interpolation': 0.05},
            {'event_id': 3, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 2, 'vulnerability_probability': 0.0, 'interpolation': 0.15},
            {'event_id': 3, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 3, 'vulnerability_probability': 0.11, 'interpolation': 0.25},
            {'event_id': 4, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 1, 'vulnerability_probability': 0.0, 'interpolation': 0.05},
            {'event_id': 4, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 2, 'vulnerability_probability': 0.0, 'interpolation': 0.15},
            {'event_id': 4, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 3, 'vulnerability_probability': 0.65, 'interpolation': 0.25}
        ],
            list(self.test.model.T.to_dict().values())
        )

    def test_calculate_probability_of_damage(self):
        self.test.model = DataFrame([
            {'event_id': 1, 'area_peril_id': 10, 'footprint_probability': 0.47, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.45, 'interpolation': 0.05},
            {'event_id': 1, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.0, 'interpolation': 0.05},
            {'event_id': 1, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 1, 'damage_bin_id': 2, 'vulnerability_probability': 0.65, 'interpolation': 0.15},
            {'event_id': 3, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 1, 'vulnerability_probability': 0.0, 'interpolation': 0.05},
            {'event_id': 3, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 2, 'vulnerability_probability': 0.0, 'interpolation': 0.15},
            {'event_id': 3, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 3, 'vulnerability_probability': 0.11, 'interpolation': 0.25},
            {'event_id': 4, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 1, 'vulnerability_probability': 0.0, 'interpolation': 0.05},
            {'event_id': 4, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 2, 'vulnerability_probability': 0.0, 'interpolation': 0.15},
            {'event_id': 4, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 3, 'vulnerability_probability': 0.65, 'interpolation': 0.25}
        ])

        self.test.calculate_probability_of_damage()

        self.assertEqual([
            {'event_id': 1, 'area_peril_id': 10, 'footprint_probability': 0.47, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.45, 'interpolation': 0.05, 'prob_to': 0.2115},
            {'event_id': 1, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.0, 'interpolation': 0.05, 'prob_to': 0.0},
            {'event_id': 1, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 1, 'damage_bin_id': 2, 'vulnerability_probability': 0.65, 'interpolation': 0.15, 'prob_to': 0.34450000000000003},
            {'event_id': 3, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 1, 'vulnerability_probability': 0.0, 'interpolation': 0.05, 'prob_to': 0.0},
            {'event_id': 3, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 2, 'vulnerability_probability': 0.0, 'interpolation': 0.15, 'prob_to': 0.0},
            {'event_id': 3, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 3, 'vulnerability_probability': 0.11, 'interpolation': 0.25, 'prob_to': 0.07919999999999999},
            {'event_id': 4, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 1, 'vulnerability_probability': 0.0, 'interpolation': 0.05, 'prob_to': 0.0},
            {'event_id': 4, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 2, 'vulnerability_probability': 0.0, 'interpolation': 0.15, 'prob_to': 0.0},
            {'event_id': 4, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 3, 'vulnerability_probability': 0.65, 'interpolation': 0.25, 'prob_to': 0.12350000000000001}
        ],
            list(self.test.model.T.to_dict().values())
        )

    def test_define_columns_for_saving(self):
        self.test.model = DataFrame([
            {'event_id': 1, 'area_peril_id': 10, 'footprint_probability': 0.47, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.45, 'interpolation': 0.05, 'prob_to': 0.2115},
            {'event_id': 1, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 1, 'damage_bin_id': 1, 'vulnerability_probability': 0.0, 'interpolation': 0.05, 'prob_to': 0.0},
            {'event_id': 1, 'area_peril_id': 10, 'footprint_probability': 0.53, 'vulnerability_id': 1, 'damage_bin_id': 2, 'vulnerability_probability': 0.65, 'interpolation': 0.15, 'prob_to': 0.34450000000000003},
            {'event_id': 3, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 1, 'vulnerability_probability': 0.0, 'interpolation': 0.05, 'prob_to': 0.0},
            {'event_id': 3, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 2, 'vulnerability_probability': 0.0, 'interpolation': 0.15, 'prob_to': 0.0},
            {'event_id': 3, 'area_peril_id': 30, 'footprint_probability': 0.72, 'vulnerability_id': 3, 'damage_bin_id': 3, 'vulnerability_probability': 0.11, 'interpolation': 0.25, 'prob_to': 0.07919999999999999},
            {'event_id': 4, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 1, 'vulnerability_probability': 0.0, 'interpolation': 0.05, 'prob_to': 0.0},
            {'event_id': 4, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 2, 'vulnerability_probability': 0.0, 'interpolation': 0.15, 'prob_to': 0.0},
            {'event_id': 4, 'area_peril_id': 40, 'footprint_probability': 0.19, 'vulnerability_id': 4, 'damage_bin_id': 3, 'vulnerability_probability': 0.65, 'interpolation': 0.25, 'prob_to': 0.12350000000000001}
        ])

        self.test.define_columns_for_saving()

        self.assertEqual([
            {'event_id': 1, 'areaperil_id': 10, 'vulnerability_id': 1, 'bin_index': 1, 'prob_to': 0.2115, 'bin_mean': 0.05},
            {'event_id': 1, 'areaperil_id': 10, 'vulnerability_id': 1, 'bin_index': 1, 'prob_to': 0.0, 'bin_mean': 0.05},
            {'event_id': 1, 'areaperil_id': 10, 'vulnerability_id': 1, 'bin_index': 2, 'prob_to': 0.34450000000000003, 'bin_mean': 0.15},
            {'event_id': 3, 'areaperil_id': 30, 'vulnerability_id': 3, 'bin_index': 1, 'prob_to': 0.0, 'bin_mean': 0.05},
            {'event_id': 3, 'areaperil_id': 30, 'vulnerability_id': 3, 'bin_index': 2, 'prob_to': 0.0, 'bin_mean': 0.15},
            {'event_id': 3, 'areaperil_id': 30, 'vulnerability_id': 3, 'bin_index': 3, 'prob_to': 0.07919999999999999, 'bin_mean': 0.25},
            {'event_id': 4, 'areaperil_id': 40, 'vulnerability_id': 4, 'bin_index': 1, 'prob_to': 0.0, 'bin_mean': 0.05},
            {'event_id': 4, 'areaperil_id': 40, 'vulnerability_id': 4, 'bin_index': 2, 'prob_to': 0.0, 'bin_mean': 0.15},
            {'event_id': 4, 'areaperil_id': 40, 'vulnerability_id': 4, 'bin_index': 3, 'prob_to': 0.12350000000000001, 'bin_mean': 0.25}
        ],
            list(self.test.model.T.to_dict().values())
        )

    @patch("oasislmf.pytools.getmodel.get_model_process.GetModelProcess.items", new_callable=PropertyMock)
    @patch("oasislmf.pytools.getmodel.get_model_process.GetModelProcess.footprint", new_callable=PropertyMock)
    @patch("oasislmf.pytools.getmodel.get_model_process.GetModelProcess.events", new_callable=PropertyMock)
    @patch("oasislmf.pytools.getmodel.get_model_process.GetModelProcess.vulnerabilities", new_callable=PropertyMock)
    @patch("oasislmf.pytools.getmodel.get_model_process.GetModelProcess.damage_bin", new_callable=PropertyMock)
    def test_full_run(self, mock_damage_bin, mock_vulnerabilities, mock_events, mock_footprint, mock_items):

        mock_damage_bin.return_value.value = DataFrame([
            {"bin_index": 1, "bin_from": 0.0, "bin_to": 0.1, "interpolation": 0.05, "interval_type": 1203},
            {"bin_index": 2, "bin_from": 0.1, "bin_to": 0.2, "interpolation": 0.15, "interval_type": 1200},
            {"bin_index": 3, "bin_from": 0.2, "bin_to": 0.3, "interpolation": 0.25, "interval_type": 1202},
            {"bin_index": 4, "bin_from": 0.3, "bin_to": 0.4, "interpolation": 0.35, "interval_type": 1202}
        ])

        # vulnerability and intensity combination needs to sum to one across the damage bins
        mock_vulnerabilities.return_value.value = DataFrame([
            {"vulnerability_id": 1, "intensity_bin_id": 1, "damage_bin_id": 1, "probability": 0.45},
            {"vulnerability_id": 1, "intensity_bin_id": 2, "damage_bin_id": 2, "probability": 0.65},

            {"vulnerability_id": 2, "intensity_bin_id": 3, "damage_bin_id": 1, "probability": 0.78},
            {"vulnerability_id": 2, "intensity_bin_id": 4, "damage_bin_id": 2, "probability": 0.22},

            {"vulnerability_id": 3, "intensity_bin_id": 1, "damage_bin_id": 1, "probability": 0.89},
            {"vulnerability_id": 3, "intensity_bin_id": 4, "damage_bin_id": 3, "probability": 0.11},

            {"vulnerability_id": 4, "intensity_bin_id": 2, "damage_bin_id": 1, "probability": 0.35},
            {"vulnerability_id": 4, "intensity_bin_id": 5, "damage_bin_id": 3, "probability": 0.65},
        ])

        mock_events.return_value.value = DataFrame([
            {"event_id": 1},
            {"event_id": 2},
            {"event_id": 3},
            {"event_id": 4}
        ])

        # each event area peril id should sum to one over the intensity bins
        mock_footprint.return_value.value = DataFrame([
            {"event_id": 1, "areaperil_id": 10, "intensity_bin_id": 1, "probability": 0.47},
            {"event_id": 1, "areaperil_id": 10, "intensity_bin_id": 2, "probability": 0.53},

            {"event_id": 2, "areaperil_id": 20, "intensity_bin_id": 1, "probability": 0.30},
            {"event_id": 2, "areaperil_id": 20, "intensity_bin_id": 2, "probability": 0.70},

            {"event_id": 3, "areaperil_id": 30, "intensity_bin_id": 3, "probability": 0.28},
            {"event_id": 3, "areaperil_id": 30, "intensity_bin_id": 4, "probability": 0.72},

            {"event_id": 4, "areaperil_id": 40, "intensity_bin_id": 4, "probability": 0.81},
            {"event_id": 4, "areaperil_id": 40, "intensity_bin_id": 5, "probability": 0.19}
        ])

        mock_items.return_value.value = DataFrame([
            {"item_id": 1, "coverage_id": 1, "areaperil_id": 10, "vulnerability_id": 1, "group_id": 1.0},
            {"item_id": 2, "coverage_id": 1, "areaperil_id": 20, "vulnerability_id": 2, "group_id": 1.0},
            {"item_id": 3, "coverage_id": 2, "areaperil_id": 30, "vulnerability_id": 3, "group_id": 2.0},
            {"item_id": 4, "coverage_id": 2, "areaperil_id": 40, "vulnerability_id": 4, "group_id": 2.0},
        ])

        """
        The commented out code block below writes the dataframes defined in this function to files.
        """
        # mock_footprint.return_value.value.to_csv("./footprint.parquet", index=False)
        # mock_events.return_value.value.to_csv("./events.csv", index=False)
        # mock_vulnerabilities.return_value.value.to_csv("./vulnerability.csv", index=False)
        # mock_damage_bin.return_value.value.to_csv("./damage_bin_dict.parquet", index=False)

        # mock_footprint.return_value.value.to_parquet("./footprint.parquet", index=False)
        # mock_events.return_value.value.to_parquet("./events.csv", index=False)
        # mock_vulnerabilities.return_value.value.to_parquet("./vulnerability.parquet", index=False)
        # mock_damage_bin.return_value.value.to_parquet("./damage_bin_dict.parquet", index=False)

        self.test.run()

        self.assertEqual([{'event_id': 1.0, 'areaperil_id': 10.0, 'vulnerability_id': 1.0, 'bin_index': 1.0, 'prob_to': 0.2115, 'bin_mean': 0.05},
                          {'event_id': 1.0, 'areaperil_id': 10.0, 'vulnerability_id': 1.0, 'bin_index': 1.0, 'prob_to': 0.2115, 'bin_mean': 0.05},
                          {'event_id': 1.0, 'areaperil_id': 10.0, 'vulnerability_id': 1.0, 'bin_index': 2.0, 'prob_to': 0.556, 'bin_mean': 0.15},
                          {'event_id': 3.0, 'areaperil_id': 30.0, 'vulnerability_id': 3.0, 'bin_index': 1.0, 'prob_to': 0.0, 'bin_mean': 0.05},
                          {'event_id': 3.0, 'areaperil_id': 30.0, 'vulnerability_id': 3.0, 'bin_index': 2.0, 'prob_to': 0.0, 'bin_mean': 0.15},
                          {'event_id': 3.0, 'areaperil_id': 30.0, 'vulnerability_id': 3.0, 'bin_index': 3.0, 'prob_to': 0.07919999999999999, 'bin_mean': 0.25},
                          {'event_id': 4.0, 'areaperil_id': 40.0, 'vulnerability_id': 4.0, 'bin_index': 1.0, 'prob_to': 0.0, 'bin_mean': 0.05},
                          {'event_id': 4.0, 'areaperil_id': 40.0, 'vulnerability_id': 4.0, 'bin_index': 2.0, 'prob_to': 0.0, 'bin_mean': 0.15},
                          {'event_id': 4.0, 'areaperil_id': 40.0, 'vulnerability_id': 4.0, 'bin_index': 3.0, 'prob_to': 0.12350000000000001, 'bin_mean': 0.25}
        ],
            list(self.test.model.T.to_dict().values())
        )


if __name__ == "__main__":
    main()
