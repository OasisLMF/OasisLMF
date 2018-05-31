# (c) 2013-2016 Oasis LMF Ltd.  Software provided for early adopter evaluation only.
import inspect
import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

SRC_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'src'))

if os.path.exists(SRC_PATH):
    sys.path.append(SRC_PATH)

import oasis_utils

from keys_server import (
    AreaPerilLookup,
    VulnerabilityLookup
)

class LookupTests(unittest.TestCase):

    area_peril_lookup_test_data = [
        {"AREA_PERIL_ID": 1, "LAT1": 0.0, "LON1": 0.0, "LAT2": 1.0, "LON2": 0.0, "LAT3": 0.0, "LON3": 1.0, "LAT4": 1.0, "LON4": 1.0},
        {"AREA_PERIL_ID": 2, "LAT1": 1.0, "LON1": 0.0, "LAT2": 2.0, "LON2": 0.0, "LAT3": 1.0, "LON3": 1.0, "LAT4": 2.0, "LON4": 1.0},
        {"AREA_PERIL_ID": 3, "LAT1": 0.0, "LON1": 1.0, "LAT2": 1.0, "LON2": 1.0, "LAT3": 0.0, "LON3": 2.0, "LAT4": 1.0, "LON4": 2.0}
    ]

    vulnerability_lookup_test_data = [
        {"VULNERABILITY_ID": 1, "COVERAGE": 1, "CLASS_1": "A"},
        {"VULNERABILITY_ID": 2, "COVERAGE": 2, "CLASS_1": "A"},
        {"VULNERABILITY_ID": 3, "COVERAGE": 3, "CLASS_1": "A"}
    ]

    def test_validate_lat(self):
        lookup = AreaPerilLookup()
        assert lookup.validate_lat(-90.0)
        assert lookup.validate_lat(-10.0)
        assert lookup.validate_lat(0)
        assert lookup.validate_lat(10.0)
        assert lookup.validate_lat(90.0)
        assert lookup.validate_lat('90.0')
        assert not lookup.validate_lat('abc')
        assert not lookup.validate_lat(-90.01)
        assert not lookup.validate_lat(300.0)
        assert not lookup.validate_lat(-300.0)

    def test_area_peril_lookup_success(self):
        lookup = AreaPerilLookup()
        lookup.load_lookup_data(self.area_peril_lookup_test_data)

        location = {'id': 1, 'lat': 0.5, 'lon': 0.5}
        result = lookup.do_lookup_location(location)

        assert result['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert result['area_peril_id'] == 1

    def test_area_peril_lookup_corners_1(self):
        lookup = AreaPerilLookup()
        lookup.load_lookup_data(self.area_peril_lookup_test_data)

        location = {'id': 1, 'lat': 0.0, 'lon': 0.0}
        result = lookup.do_lookup_location(location)
        assert result['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert result['area_peril_id'] == 1

        location = {'id': 1, 'lat': 1.0, 'lon': 0.0}
        result = lookup.do_lookup_location(location)
        assert result['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert result['area_peril_id'] == 1

        location = {'id': 1, 'lat': 0.0, 'lon': 1.0}
        result = lookup.do_lookup_location(location)
        assert result['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert result['area_peril_id'] == 1

        location = {'id': 1, 'lat': 1.0, 'lon': 1.0}
        result = lookup.do_lookup_location(location)
        assert result['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert result['area_peril_id'] == 1

    def test_area_peril_lookup_corners_2(self):
        lookup = AreaPerilLookup()
        lookup.load_lookup_data(self.area_peril_lookup_test_data)

        location = {'id': 1, 'lat': 1.0, 'lon': 0.0}
        result = lookup.do_lookup_location(location)
        assert result['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert result['area_peril_id'] == 1

        location = {'id': 1, 'lat': 1.0, 'lon': 1.0}
        result = lookup.do_lookup_location(location)
        assert result['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert result['area_peril_id'] == 1

        location = {'id': 1, 'lat': 2.0, 'lon': 0.0}
        result = lookup.do_lookup_location(location)
        assert result['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert result['area_peril_id'] == 2

        location = {'id': 1, 'lat': 2.0, 'lon': 1.0}
        result = lookup.do_lookup_location(location)
        assert result['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert result['area_peril_id'] == 2

    def test_area_peril_lookup_nomatch(self):
        lookup = AreaPerilLookup()
        lookup.load_lookup_data(self.area_peril_lookup_test_data)

        location = {'id': 1, 'lat': 10.0, 'lon': 1.0}
        result = lookup.do_lookup_location(location)

        assert result['status'] == oasis_utils.KEYS_STATUS_NOMATCH
        assert result['area_peril_id'] is None

    def test_area_peril_lookup_invalid_lat_1(self):
        lookup = AreaPerilLookup()

        location = {'id': 1, 'lat': 'abc', 'lon': 1.0}
        result = lookup.do_lookup_location(location)

        assert result['status'] == oasis_utils.KEYS_STATUS_FAIL
        assert result['area_peril_id'] is None

    def test_area_peril_lookup_invalid_lat_2(self):
        lookup = AreaPerilLookup()

        location = {'id': 1, 'lat': -100.0, 'lon': 1.0}
        result = lookup.do_lookup_location(location)

        assert result['status'] == oasis_utils.KEYS_STATUS_FAIL
        assert result['area_peril_id'] is None

    def test_area_peril_lookup_invalid_lat_3(self):
        lookup = AreaPerilLookup()

        location = {'id': 1, 'lat': 100.0, 'lon': 1.0}
        result = lookup.do_lookup_location(location)

        assert result['status'] == oasis_utils.KEYS_STATUS_FAIL
        assert result['area_peril_id'] is None

    def test_area_peril_lookup_invalid_lon_1(self):
        lookup = AreaPerilLookup()

        location = {'id': 1, 'lat': 1.0, 'lon': 'abc'}
        result = lookup.do_lookup_location(location)

        assert result['status'] == oasis_utils.KEYS_STATUS_FAIL
        assert result['area_peril_id'] is None

    def test_area_peril_lookup_invalid_lon_2(self):
        lookup = AreaPerilLookup()

        location = {'id': 1, 'lat': 10.0, 'lon': -190.0}
        result = lookup.do_lookup_location(location)

        assert result['status'] == oasis_utils.KEYS_STATUS_FAIL
        assert result['area_peril_id'] is None

    def test_area_peril_lookup_invalid_lon_3(self):
        lookup = AreaPerilLookup()

        location = {'id': 1, 'lat': 10.0, 'lon': 190.0}
        result = lookup.do_lookup_location(location)

        assert result['status'] == oasis_utils.KEYS_STATUS_FAIL
        assert result['area_peril_id'] is None

    def test_vulnerability_lookup_success(self):
        lookup = VulnerabilityLookup()
        lookup.load_lookup_data(self.vulnerability_lookup_test_data)

        location = {'id': 1, "coverage": 1, "class_1": "A", "class_2": "B"}
        result = lookup.do_lookup_location(location)

        assert result['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert result['vulnerability_id'] == 1

        location = {'id': 2, "coverage": 2, "class_1": "A", "class_2": "B"}
        result = lookup.do_lookup_location(location)

        assert result['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert result['vulnerability_id'] == 2

        location = {'id': 3, "coverage": 3, "class_1": "A", "class_2": "B"}
        result = lookup.do_lookup_location(location)

        assert result['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert result['vulnerability_id'] == 3

    def test_vulnerability_lookup_nomatch(self):
        lookup = VulnerabilityLookup()
        lookup.load_lookup_data(self.vulnerability_lookup_test_data)

        location = {'id': 1, "coverage": 1, "class_1": "B", "class_2": "B"}
        result = lookup.do_lookup_location(location)

        assert result['status'] == oasis_utils.KEYS_STATUS_NOMATCH
        assert result['vulnerability_id'] is None

        location = {'id': 2, "coverage": 5, "class_1": "A", "class_2": "B"}
        result = lookup.do_lookup_location(location)

        assert result['status'] == oasis_utils.KEYS_STATUS_NOMATCH
        assert result['vulnerability_id'] is None

if __name__ == '__main__':
    unittest.main()
