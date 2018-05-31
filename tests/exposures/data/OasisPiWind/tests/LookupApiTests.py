# (c) 2013-2016 Oasis LMF Ltd.  Software provided for early adopter evaluation only.
import inspect
import json
import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'src'))
sys.path.append(SRC_DIR)

import oasis_utils

from keys_server import PiWindKeysLookup

from oasis_keys_server import app

class LookupApiTests(unittest.TestCase):
    '''
    Area peril and vulnerability tests.
    '''

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

    def setUp(self):

        app.DO_GZIP_RESPONSE = False
        app.APP.config['TESTING'] = True

        app.keys_lookup = PiWindKeysLookup()

        app.keys_lookup.area_peril_lookup.load_lookup_data(self.area_peril_lookup_test_data)

        app.keys_lookup.vulnerability_lookup.load_lookup_data(self.vulnerability_lookup_test_data)

        self.app = app.APP.test_client()

    def test_one_match_json(self):

        locations = [{'id': 1, 'lat': 1.0, 'lon': 1.0, "coverage": 1, "class_1": "A", "class_2": "B"}]
        response = self.post_json(locations)
        assert(response.status_code == 200)

        locations = json.loads(response.data.decode('utf-8'))['items']

        assert len(locations) == 1
        assert locations[0]['id'] == 1
        assert locations[0]['coverage'] == 1
        assert locations[0]['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert locations[0]['area_peril_id'] == 1

    def test_two_match_json(self):

        locations = [
            {'id': 1, 'lat': 0.5, 'lon': 0.5, "coverage": 1, "class_1": "A", "class_2": "B"}, 
            {'id': 2, 'lat': 1.5, 'lon': 0.5, "coverage": 1, "class_1": "A", "class_2": "B"}
        ]
        response = self.post_json(locations)
        assert response.status_code == 200

        locations = json.loads(response.data.decode('utf-8'))['items']

        assert len(locations) == 2
        assert locations[0]['id'] == 1
        assert locations[0]['coverage'] == 1
        assert locations[0]['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert locations[0]['area_peril_id'] == 1
        assert locations[1]['id'] == 2
        assert locations[1]['coverage'] == 1
        assert locations[1]['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert locations[1]['area_peril_id'] == 2

    def test_one_nomatch_json(self):

        locations = [{'id': 1, 'lat': 10.0, 'lon': 1.0, "coverage": 1, "class_1": "A", "class_2": "B"}]
        response = self.post_json(locations)
        assert response.status_code == 200

        locations = json.loads(response.data.decode('utf-8'))['items']

        assert len(locations) == 1
        assert locations[0]['id'] == 1
        assert locations[0]['coverage'] == 1
        assert locations[0]['status'] == oasis_utils.KEYS_STATUS_NOMATCH
        assert locations[0]['area_peril_id'] is None

    def test_one_fail_json(self):

        locations = [{'id': 1, 'lat': 500.0, 'lon': 1.0, "coverage": 1, "class_1": "A", "class_2": "B"}]
        response = self.post_json(locations)
        assert response.status_code == 200

        locations = json.loads(response.data.decode('utf-8'))['items']

        assert len(locations) == 1
        assert locations[0]['id'] == 1
        assert locations[0]['coverage'] == 1
        assert locations[0]['status'] == oasis_utils.KEYS_STATUS_FAIL
        assert locations[0]['area_peril_id'] is None

    def test_one_match_csv(self):

        locations = [{'id': 1, 'lat': 1.0, 'lon': 1.0, "coverage": 1, "class_1": "A", "class_2": "B"}]
        response = self.post_csv(locations)
        assert response.status_code == 200

        locations = json.loads(response.data.decode('utf-8'))['items']
        
        assert len(locations) == 1
        assert locations[0]['id'] == 1
        assert locations[0]['coverage'] == 1
        assert locations[0]['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert locations[0]['area_peril_id'] == 1

    def test_two_match_csv(self):

        locations = [
            {'id': 1, 'lat': 0.5, 'lon': 0.5, "coverage": 1, "class_1": "A", "class_2": "B"}, 
            {'id': 2, 'lat': 1.5, 'lon': 0.5, "coverage": 1, "class_1": "A", "class_2": "B"}
        ]
        response = self.post_csv(locations)
        assert response.status_code == 200

        locations = json.loads(response.data.decode('utf-8'))['items']

        assert len(locations) == 2
        assert locations[0]['id'] == 1
        assert locations[0]['coverage'] == 1
        assert locations[0]['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert locations[0]['area_peril_id'] == 1
        assert locations[1]['id'] == 2
        assert locations[1]['coverage'] == 1
        assert locations[1]['status'] == oasis_utils.KEYS_STATUS_SUCCESS
        assert locations[1]['area_peril_id'] == 2

    def test_one_nomatch_csv(self):

        locations = [{'id': 1, 'lat': 10.0, 'lon': 1.0, "coverage": 1, "class_1": "A", "class_2": "B"}]
        response = self.post_csv(locations)
        locations = json.loads(response.data.decode('utf-8'))['items']

        assert len(locations) == 1
        assert locations[0]['status'] == oasis_utils.KEYS_STATUS_NOMATCH 
        assert locations[0]['area_peril_id'] is None

    def test_one_fail_csv(self):

        locations = [{'id': 1, 'lat': 500.0, 'lon': 1.0, "coverage": 1, "class_1": "A", "class_2": "B"}]
        response = self.post_csv(locations)
        assert response.status_code == 200

        locations = json.loads(response.data.decode('utf-8'))['items']

        assert len(locations) == 1
        assert locations[0]['id'] == 1
        assert locations[0]['coverage'] == 1
        assert locations[0]['status'] == oasis_utils.KEYS_STATUS_FAIL
        assert locations[0]['area_peril_id'] is None

    def test_invalid_content_type(self):

        headers = [('Content-Type', 'text/abc')]
        response = self.app.post(os.path.join(app.SERVICE_BASE_URL, 'get_keys'), headers=headers, data="")

        assert response.status_code == 500

    def post_csv(self, data):

        csv_data = ""
        csv_data = "ID,LAT,LON,COVERAGE,CLASS_1,CLASS_2\n"
        for item in data:
            csv_data = csv_data + \
                str(item['id']) + ',' + str(item['lat']) + ',' + str(item['lon']) + ',' + \
                str(item['coverage']) + ',' + str(item['class_1']) + ',' + str(item['class_2']) + '\n'

        headers = [('Content-Type', oasis_utils.HTTP_REQUEST_CONTENT_TYPE_CSV)]
        headers.append(('Content-Length', len(csv_data)))
        return self.app.post(os.path.join(app.SERVICE_BASE_URL, 'get_keys'), headers=headers, data=csv_data)

    def post_json(self, data):

        _data = map(lambda rec: dict((k.upper(), rec[k]) for k in rec), data)
        def obj_dict(obj):
            return obj.__dict__
        json_data = json.dumps(_data, default=obj_dict).encode('utf-8')
        json_data_length = len(json_data)
        headers = [('Content-Type', oasis_utils.HTTP_REQUEST_CONTENT_TYPE_JSON)]
        headers.append(('Content-Length', json_data_length))
        return self.app.post(os.path.join(app.SERVICE_BASE_URL, 'get_keys'), headers=headers, data=json_data)

if __name__ == '__main__':
    unittest.main()
