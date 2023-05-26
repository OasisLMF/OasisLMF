import os
import sys
import io
import json
import logging

import pathlib
import tarfile
import pandas as pd
from requests import HTTPError
from requests.exceptions import HTTPError

from datetime import datetime
from io import StringIO
from tempfile import TemporaryDirectory

import unittest
from unittest.mock import Mock, MagicMock
from hypothesis import given, settings
from hypothesis import strategies as st

from posixpath import join as urljoin
from oasislmf.platform_api.client import (
    OasisException,
    ApiEndpoint,
    JsonEndpoint,
    FileEndpoint,
    API_models,
    API_portfolios,
    API_datafiles,
    API_analyses,
    APIClient,
)


from oasislmf.platform_api.session import APISession
from requests_toolbelt import MultipartEncoder

import responses
import requests

settings.register_profile("ci", max_examples=10)
settings.load_profile("ci")

PIWIND_EXP_URL = 'https://raw.githubusercontent.com/OasisLMF/OasisPiWind/develop/tests/inputs'
CONTENT_MAP = {
    'parquet': 'application/octet-stream',
    'pq': 'application/octet-stream',
    'csv': 'text/csv',
    'gz': 'application/gzip',
    'zip': 'application/zip',
    'bz2': 'application/x-bzip2',
}


@responses.activate
def create_api_session(url):
    responses.get(
        url=f'{url}/healthcheck/',
        json={"status": "OK"})

    responses.post(
        url=f'{url}/access_token/',
        json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
        headers={"authorization": "Bearer acc_tkn"})
    return APISession(url, 'testUser', 'testPass')


class TestApiEndpoint(unittest.TestCase):
    def setUp(self):
        self.session = MagicMock()
        self.url_endpoint = "http://example.com/api"
        self.api = ApiEndpoint(self.session, self.url_endpoint)

    def test_get_with_id(self):
        ID = 1
        self.api.get(ID)
        self.session.get.assert_called_with('{}/{}/'.format(self.url_endpoint, ID))

    def test_get_without_id(self):
        self.api.get()
        self.session.get.assert_called_with(self.url_endpoint)

    def test_delete(self):
        ID = 1
        self.api.delete(ID)
        self.session.delete.assert_called_with('{}/{}/'.format(self.url_endpoint, ID))

    @given(metadata=st.dictionaries(
        keys=st.from_regex(r"^[a-zA-Z0-9_-]+$"),
        values=st.from_regex(r"^[a-zA-Z0-9_-]+$")
    ))
    def test_search(self, metadata):
        search_string = '?' if metadata else ''
        for key, value in metadata.items():
            search_string += f'{key}={value}&'
        expected_url = '{}{}'.format(self.url_endpoint, search_string.rstrip('&'))
        self.api.search(metadata)
        self.session.get.assert_called_with(expected_url)

    @given(data=st.dictionaries(keys=st.text(), values=st.text()))
    def test_create(self, data):
        self.api.create(data)
        self.session.post.assert_called_with(self.url_endpoint, json=data)


class JsonEndpointTests(unittest.TestCase):

    def setUp(self):
        assert responses, 'responses package required to run'
        self.url_endpoint = 'http://example.com/api'
        self.url_resource = 'resource'
        self.session = create_api_session(self.url_endpoint)
        self.api = JsonEndpoint(self.session, self.url_endpoint, self.url_resource)
        self.headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        responses.start()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        responses.stop()
        responses.reset()

    def test_build_url(self):
        ID = 123
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        result = self.api._build_url(ID)
        self.assertEqual(result, expected_url)

    @given(ID=st.integers(min_value=1))
    def test_get(self, ID):
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.get(url=expected_url, json=[])

        rsp = self.api.get(ID)
        self.assertEqual(rsp.url, expected_url)

    @given(ID=st.integers(min_value=1), data=st.dictionaries(keys=st.text(), values=st.text()))
    def test_post(self, ID, data):
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.post(url=expected_url, headers=self.headers)

        rsp = self.api.post(ID, data)
        self.assertEqual(rsp.url, expected_url)

    @given(ID=st.integers(min_value=1))
    def test_delete(self, ID):
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.delete(url=expected_url, headers=self.headers)

        rsp = self.api.delete(ID)
        self.assertEqual(rsp.url, expected_url)

    @given(
        ID=st.integers(min_value=1),
        file_path=st.from_regex(r"^[a-zA-Z0-9_-]+$"),
        overwrite=st.booleans(),
        payload=st.dictionaries(keys=st.text(min_size=1), values=st.text(min_size=1))
    )
    def test_download(self, ID, file_path, overwrite, payload):
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.get(url=expected_url, json=payload)

        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, file_path))
            rsp = self.api.download(ID, abs_fp, overwrite)

            with open(abs_fp, mode='r') as f:
                json_saved = json.load(f)

            assert rsp.url == expected_url
            assert rsp.json() == payload
            self.assertTrue(os.path.isfile(abs_fp))
            self.assertEqual(json_saved, payload)

    @given(ID=st.integers(min_value=1), file_path=st.from_regex(r"^[a-zA-Z0-9_-]+$"))
    def test_download__file_exists_exception_raised(self, ID, file_path):
        overwrite = False
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.get(url=expected_url, json={'key': 'value'})

        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, file_path))
            pathlib.Path(abs_fp).touch()

            with self.assertRaises(IOError) as context:
                self.api.download(ID, abs_fp, overwrite)
            exception = context.exception
            self.assertEqual(str(exception), f'Local file alreday exists: {abs_fp}')

    @given(ID=st.integers(min_value=1), file_path=st.from_regex(r"^[a-zA-Z0-9_-]+$"))
    def test_download__file_exists_and_overwritten(self, ID, file_path):
        overwrite = True
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.get(url=expected_url, json={'key': 'value'})

        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, file_path))
            pathlib.Path(abs_fp).touch()

            if overwrite is True:
                rsp = self.api.download(ID, abs_fp, overwrite)

                with open(abs_fp, mode='r') as f:
                    json_saved = json.load(f)

                assert rsp.url == expected_url
                self.assertTrue(os.path.isfile(abs_fp))
                self.assertEqual(json_saved, {'key': 'value'})


class FileEndpointTests(unittest.TestCase):

    def setUp(self):
        assert responses, 'responses package required to run'
        self.url_endpoint = 'http://example.com/api'
        self.url_resource = 'resource'
        self.session = create_api_session(self.url_endpoint)
        self.api = FileEndpoint(self.session, self.url_endpoint, self.url_resource)
        self.headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        self.parquet_data = requests.get(f'{PIWIND_EXP_URL}/SourceLocOEDPiWind10.parquet').content
        self.csv_data = requests.get(f'{PIWIND_EXP_URL}/SourceLocOEDPiWind10.csv').content
        responses.start()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        responses.stop()
        responses.reset()

    def test_build_url(self):
        ID = 123
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        result = self.api._build_url(ID)
        self.assertEqual(result, expected_url)

    def test_set_content_type__with_supported_extensions(self):

        for ext in CONTENT_MAP:
            file_path = f'file.{ext}'
            expected_content_type = CONTENT_MAP[ext]
            result = self.api._set_content_type(file_path)
            self.assertEqual(result, expected_content_type)

    def test_set_content_type__with_unsupported_extension(self):
        file_path = 'file.unknown'
        expected_content_type = 'text/csv'
        result = self.api._set_content_type(file_path)
        self.assertEqual(result, expected_content_type)

    @given(
        ID=st.integers(min_value=1),
        file_path=st.from_regex(r"^[a-zA-Z0-9_-]+$"),
        file_data=st.binary(min_size=1),
        file_ext=st.sampled_from([
            'parquet',
            'pq',
            'csv',
            'gz',
            'zip',
            'bz2',
        ])
    )
    def test_upload_file(self, ID, file_path, file_data, file_ext):
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        file_name = f'{file_path}.{file_ext}'
        request_resp_json = {
            "created": "2023-05-25T13:55:50.665455Z",
            "file": f"5ab983f8f9144b8bbad582367b043c1f.{file_ext}",
            "filename": file_name
        }

        responses.post(
            url=expected_url,
            headers={'accept': CONTENT_MAP[file_ext]},
            json=request_resp_json
        )

        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, file_name))
            with open(abs_fp, 'wb') as file:
                # Write the binary data to the file
                file.write(file_data)

            # Fire Upload call
            rsp = self.api.upload(ID, abs_fp, CONTENT_MAP[file_ext])

            # Load request data
            request = responses.calls[-1].request
            body = request.body
            req_filename, req_file_hl, req_content_type = body.fields['file']

            # Check request data
            self.assertTrue(isinstance(body, MultipartEncoder))
            self.assertEqual(request.url, expected_url)
            self.assertEqual(req_filename, file_name)
            self.assertEqual(req_content_type, CONTENT_MAP[file_ext])
            self.assertEqual(req_file_hl.read(), file_data)
            self.assertEqual(req_file_hl.name, os.path.join(d, file_name))

            # check response return
            self.assertEqual(rsp.json(), request_resp_json)

    @given(
        ID=st.integers(min_value=1),
        file_path=st.from_regex(r"^[a-zA-Z0-9_-]+$"),
        overwrite=st.booleans(),
    )
    def test_download__parquet_data(self, ID, file_path, overwrite):
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        file_name = f'{file_path}.parquet'

        responses.get(
            expected_url,
            body=self.parquet_data,
            content_type=CONTENT_MAP['parquet'],
            stream=True
        )

        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, file_name))
            rsp = self.api.download(ID, abs_fp, overwrite)
            with open(abs_fp, 'rb') as file:
                saved_data = file.read()

            assert rsp.url == expected_url
            assert saved_data == self.parquet_data
            self.assertTrue(os.path.isfile(abs_fp))

    @given(
        ID=st.integers(min_value=1),
        file_path=st.from_regex(r"^[a-zA-Z0-9_-]+$"),
        overwrite=st.booleans(),
    )
    def test_download__csv_data(self, ID, file_path, overwrite):
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        file_name = f'{file_path}.csv'

        responses.get(
            expected_url,
            body=self.csv_data,
            content_type=CONTENT_MAP['csv'],
            stream=True
        )

        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, file_name))
            rsp = self.api.download(ID, abs_fp, overwrite)
            with open(abs_fp, 'rb') as file:
                saved_data = file.read()

            assert rsp.url == expected_url
            assert saved_data == self.csv_data
            self.assertTrue(os.path.isfile(abs_fp))

    @given(ID=st.integers(min_value=1), file_path=st.from_regex(r"^[a-zA-Z0-9_-]+$"))
    def test_download__file_exists_exception_raised(self, ID, file_path):
        overwrite = False
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.get(url=expected_url, json={'key': 'value'})

        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, file_path))
            pathlib.Path(abs_fp).touch()

            with self.assertRaises(IOError) as context:
                self.api.download(ID, abs_fp, overwrite)
            exception = context.exception
            self.assertEqual(str(exception), f'Local file alreday exists: {abs_fp}')

    @given(ID=st.integers(min_value=1), file_path=st.from_regex(r"^[a-zA-Z0-9_-]+$"))
    def test_download__file_exists_and_overwritten(self, ID, file_path):
        overwrite = True
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.get(
            expected_url,
            body=self.csv_data,
            content_type=CONTENT_MAP['csv'],
            stream=True
        )

        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, file_path))
            pathlib.Path(abs_fp).touch()

            if overwrite is True:
                rsp = self.api.download(ID, abs_fp, overwrite)

            with open(abs_fp, 'rb') as file:
                saved_data = file.read()

                assert rsp.url == expected_url
                assert saved_data == self.csv_data
                self.assertTrue(os.path.isfile(abs_fp))

    @given(ID=st.integers(min_value=1))
    def test_get(self, ID):
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.get(url=expected_url, body=b'some data')

        rsp = self.api.get(ID)
        self.assertEqual(rsp.url, expected_url)

    @given(
        ID=st.integers(min_value=1), 
        data_object=st.text(), 
        content_type=st.sampled_from([
            'application/octet-stream',
            'application/octet-stream',
            'text/csv',
            'application/gzip',
            'application/zip',
            'application/x-bzip2',
            'qe3j3//da'
            '',
            None,
        ])
    )
    def test_post(self, ID, data_object, content_type):
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.post(url=expected_url)
        
        rsp = self.api.post(ID, data_object, content_type)
        request = responses.calls[-1].request
        req_type, req_data, req_content_type = request.body.fields['file']

        self.assertEqual(rsp.url, expected_url)
        self.assertEqual(req_type, 'data')
        self.assertEqual(req_data, data_object)
        self.assertEqual(req_content_type, content_type)
        
    def test_post__failed(self):
        expected_url = '{}/{}/{}'.format(self.url_endpoint, 2, self.url_resource)
        responses.post(url=expected_url, json={"error": "not found"}, status=404,)

        with self.assertRaises(HTTPError) as context:
            rsp = self.api.post(2, b'some data', 'application/octet-stream')

        exception = context.exception
        expected_msg = '404 Client Error: Not Found for url: http://example.com/api/2/resource'
        self.assertEqual(str(exception), expected_msg)

    @given(ID=st.integers(min_value=1))
    def test_delete(self, ID):
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.delete(url=expected_url)
        rsp = self.api.delete(ID)

        request = responses.calls[-1].request
        self.assertTrue(request.url, expected_url) 
        self.assertTrue(rsp.ok) 





    def test_get_dataframe__from_csv_data(self):
        ID = 123
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.get(
            expected_url,
            body=self.csv_data,
            content_type=CONTENT_MAP['csv'],
            stream=True
        )
        result_df = self.api.get_dataframe(ID)
        expected_df = pd.read_csv(io.BytesIO(self.csv_data), encoding='utf8')

        request = responses.calls[-1].request
        self.assertEqual(request.url, expected_url)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_get_dataframe__from_parquet_data(self):
        ID = 123
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.get(
            expected_url,
            body=self.parquet_data,
            content_type=CONTENT_MAP['pq'],
            stream=True
        )
        result_df = self.api.get_dataframe(ID)
        expected_df = pd.read_parquet(io.BytesIO(self.parquet_data))

        request = responses.calls[-1].request
        self.assertEqual(request.url, expected_url)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_get_dataframe__from_tar_data(self):
        ID = 123
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        tar_fp = os.path.join(os.path.dirname(__file__), 'data', 'analysis_1_output.tar')
        with open(tar_fp, mode='rb') as file: 
            tar_data = file.read()

        responses.get(
            expected_url,
            body=tar_data,
            content_type=CONTENT_MAP['gz'],
            stream=True)

        result = self.api.get_dataframe(ID)
        expected_files = [
            'gul_S1_aalcalc.csv', 
            'gul_S1_eltcalc.csv',
            'gul_S1_summary-info.csv',
            'gul_S1_melt.parquet',
            'gul_S1_mplt.parquet',
            'gul_S1_palt.parquet',
            'gul_S1_qelt.parquet',
            'gul_S1_qplt.parquet',
            'gul_S1_selt.parquet',
            'gul_S1_splt.parquet'
        ]
        
        for df in expected_files:
            self.assertTrue(isinstance(result[df], pd.DataFrame))
            self.assertTrue(len(result[df].index) > 0)

    @given(content_type=st.from_regex(r"^[a-zA-Z0-9_-]+$"))
    def test_get_dataframe__with_unsupported_file_type(self, content_type):
        ID = 123
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.get(
            expected_url,
            body=self.parquet_data,
            content_type=content_type,
            stream=True)
 
        with self.assertRaises(OasisException) as context:
            result = self.api.get_dataframe(ID)

        exception = context.exception
        expected_msg = f'Unsupported filetype for Dataframe conversion: {content_type}'
        self.assertEqual(str(exception), expected_msg)

    def test_post_dataframe(self):
        ID = 1
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        expected_df = pd.read_csv(io.BytesIO(self.csv_data), encoding='utf8')
        request_resp_json = {
            "created": "2023-05-25T13:55:50.665455Z",
            "file": f"5ab983f8f9144b8bbad582367b043c1f.csv",
            "filename": 'data'
        }

        responses.post(
            url=expected_url,
            headers={'accept': 'text/csv'},
            json=request_resp_json)

        rsp = self.api.post_dataframe(ID, expected_df)
        self.assertEqual(rsp.json(), request_resp_json)
        self.assertEqual(rsp.url, expected_url)

        request = responses.calls[-1].request
        requ_type, requ_data, requ_content_type = request.body.fields['file']
        requ_data.seek(0)
        result_df = pd.read_csv(requ_data)

        self.assertEqual(requ_type, 'data')
        self.assertEqual(requ_content_type, 'text/csv')
        pd.testing.assert_frame_equal(result_df, expected_df)


class APIModelsTests(unittest.TestCase):
    def setUp(self):
        assert responses, 'responses package required to run'
        self.session = create_api_session('http://example.com/api')
        self.url_endpoint = urljoin(self.session.url_base, 'models/')
        self.api = API_models(self.session, self.url_endpoint)
        responses.start()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        responses.stop()
        responses.reset()

    def test_endpoint_setup(self):
        endpoint_list = [
            'resource_file',
            'settings',
            'versions',
            'chunking_configuration',
            'scaling_configuration',
        ]
        for endpoint in endpoint_list:
           OasisAPI_obj = getattr(self.api, endpoint)
           self.assertEqual(OasisAPI_obj.url_resource, f'{endpoint}/')
           self.assertEqual(OasisAPI_obj.url_endpoint, self.url_endpoint)
           self.assertTrue(isinstance(OasisAPI_obj, (
                ApiEndpoint, 
                FileEndpoint, 
                JsonEndpoint
            )))

    @given(
        supplier_id=st.text(), 
        model_id=st.text(), 
        version_id=st.text(), 
        data_files=st.lists(st.integers() | st.text())
    )
    def test_model_create(self, supplier_id, model_id, version_id, data_files):
        expected_url = self.url_endpoint
        expected_data = {
            "supplier_id": supplier_id,
            "model_id": model_id,
            "version_id": version_id,
            "data_files": data_files
        }
        json_rsp = {
            "id": 2,
            "supplier_id": supplier_id,
            "model_id": model_id,
            "version_id": version_id,
            "created": "2023-05-26T05:19:10.774210Z",
            "modified": "2023-05-26T05:19:10.774210Z",
            "data_files": data_files,
            "resource_file": "http://localhost:8000/v1/models/2/resource_file/",
            "settings": "http://localhost:8000/v1/models/2/settings/",
            "versions": "http://localhost:8000/v1/models/2/versions/"
        }

        responses.post(url=expected_url, json=json_rsp)
        result = self.api.create(supplier_id, model_id, version_id, data_files)
        self.assertEqual(result.json(), json_rsp)
        
        request = responses.calls[-1].request
        self.assertEqual(request.headers['content-type'], 'application/json')
        self.assertEqual(json.loads(request.body), expected_data)


    @given(
        ID=st.integers(min_value=1),
        supplier_id=st.text(), 
        model_id=st.text(), 
        version_id=st.text(), 
        data_files=st.lists(st.integers() | st.text())
    )
    def test_model_update(self, ID, supplier_id, model_id, version_id, data_files):
        expected_url = f'{self.url_endpoint}{ID}/'
        expected_data = {
            "supplier_id": supplier_id,
            "model_id": model_id,
            "version_id": version_id,
            "data_files": data_files
        }
        json_rsp = {
            "id": ID,
            "supplier_id": supplier_id,
            "model_id": model_id,
            "version_id": version_id,
            "created": "2023-05-26T05:19:10.774210Z",
            "modified": "2023-05-26T05:19:10.774210Z",
            "data_files": data_files,
            "resource_file": f"http://localhost:8000/v1/models/{ID}/resource_file/",
            "settings": f"http://localhost:8000/v1/models/{ID}/settings/",
            "versions": f"http://localhost:8000/v1/models/{ID}/versions/"
        }

        responses.put(url=expected_url, json=json_rsp)
        result = self.api.update(ID, supplier_id, model_id, version_id, data_files)
        self.assertEqual(result.json(), json_rsp)
        
        request = responses.calls[-1].request
        self.assertEqual(request.headers['content-type'], 'application/json')
        self.assertEqual(json.loads(request.body), expected_data)

    @given(st.integers(min_value=1))
    def test_get_data_files(self, ID):
        expected_url = '{}{}/data_files'.format(self.url_endpoint, ID)
        responses.get(url=expected_url)
        result = self.api.data_files(ID)
        self.assertTrue(result.ok)


class APIPortfoliosTests(unittest.TestCase):
    def setUp(self):
        assert responses, 'responses package required to run'
        self.session = create_api_session('http://example.com/api')
        self.url_endpoint = urljoin(self.session.url_base, 'portfolios/')
        self.api = API_portfolios(self.session, self.url_endpoint)
        responses.start()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        responses.stop()
        responses.reset()

    def test_endpoint_setup(self):
        endpoint_list = [
            'location_file',
            'accounts_file',
            'reinsurance_info_file',
            'reinsurance_scope_file',
            'storage_links',
        ]
        for endpoint in endpoint_list:
           OasisAPI_obj = getattr(self.api, endpoint)
           self.assertEqual(OasisAPI_obj.url_resource, f'{endpoint}/')
           self.assertEqual(OasisAPI_obj.url_endpoint, self.url_endpoint)
           self.assertTrue(isinstance(OasisAPI_obj, (
                ApiEndpoint, 
                FileEndpoint, 
                JsonEndpoint
            )))


    @given(st.text())
    def test_create(self, name):
        ID = 2
        expected_url = self.url_endpoint
        expected_data = {"name": name}
        json_rsp = {
              "id": ID,
              "name": "string",
              "created": "2023-05-26T06:48:52.524821Z",
              "modified": "2023-05-26T06:48:52.524821Z",
              "location_file": None,
              "accounts_file": None,
              "reinsurance_info_file": None,
              "reinsurance_scope_file": None,
              "storage_links": f"http://localhost:8000/v1/portfolios/{ID}/storage_links/"
        }

        responses.post(url=expected_url, json=json_rsp)
        result = self.api.create(name)
        self.assertEqual(result.json(), json_rsp)
        
        request = responses.calls[-1].request
        self.assertEqual(request.headers['content-type'], 'application/json')
        self.assertEqual(json.loads(request.body), expected_data)


    @given(st.integers(min_value=1), st.text())
    def test_update(self, ID, name):
        expected_url = f'{self.url_endpoint}{ID}/'
        expected_data = {"name": name}

        responses.put(url=expected_url)
        result = self.api.update(ID, name)

        request = responses.calls[-1].request
        self.assertEqual(request.headers['content-type'], 'application/json')
        self.assertEqual(json.loads(request.body), expected_data)

    @given(st.integers(min_value=1), st.text(), st.integers(min_value=1))
    def test_create_analyses(self, ID, name, model_id):
        expected_url = '{}{}/create_analysis/'.format(self.url_endpoint, ID)
        expected_data = {"name": name, "model": model_id}

        responses.post(url=expected_url)
        result = self.api.create_analyses(ID, name, model_id)
        request = responses.calls[-1].request
        self.assertEqual(request.headers['content-type'], 'application/json')
        self.assertEqual(json.loads(request.body), expected_data)


class APIAnalysesTests(unittest.TestCase):

    def setUp(self):
        assert responses, 'responses package required to run'
        self.session = create_api_session('http://example.com/api')
        self.url_endpoint = urljoin(self.session.url_base, 'analyses/')
        self.api = API_analyses(self.session, self.url_endpoint)
        responses.start()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        responses.stop()
        responses.reset()


    def test_endpoint_setup(self):
        endpoint_list = [
            'lookup_errors_file',
            'lookup_success_file',
            'lookup_validation_file',
            'summary_levels_file',
            'input_file',
            'input_generation_traceback_file',
            'output_file',
            'run_traceback_file',
            'run_log_file',
            'settings_file',
            'settings',
        ]
        for endpoint in endpoint_list:
           OasisAPI_obj = getattr(self.api, endpoint)
           self.assertEqual(OasisAPI_obj.url_resource, f'{endpoint}/')
           self.assertEqual(OasisAPI_obj.url_endpoint, self.url_endpoint)
           self.assertTrue(isinstance(OasisAPI_obj, (
                ApiEndpoint, 
                FileEndpoint, 
                JsonEndpoint
            )))

    @given(st.text(), st.integers(min_value=1), st.integers(min_value=1), st.lists(st.integers() | st.text(), min_size=0, max_size=10))
    def test_create(self, name, portfolio_id, model_id, data_files):
        ID = 2
        expected_url = self.url_endpoint
        expected_data = {
            "name": name,
            "portfolio": portfolio_id,
            "model": model_id,
            "complex_model_data_files": data_files
        }
        json_rsp = {
            "created": "2023-05-26T07:11:08.140539Z",
            "modified": "2023-05-26T07:11:08.140539Z",
            "name": "string",
            "id": ID,
            "portfolio": portfolio_id,
            "model": model_id,
            "status": "NEW",
            "task_started": None,
            "task_finished": None,
            "complex_model_data_files": data_files,
            "input_file": None,
            "settings_file": None,
            "settings": None,
            "lookup_errors_file": None,
            "lookup_success_file": None,
            "lookup_validation_file": None,
            "summary_levels_file": None,
            "input_generation_traceback_file": None,
            "output_file": None,
            "run_traceback_file": None,
            "run_log_file": None,
            "storage_links": f"http://localhost:8000/v1/analyses/{ID}/storage_links/"
        }
        responses.post(url=expected_url, json=json_rsp)
        result = self.api.create(name, portfolio_id, model_id, data_files)
        self.assertEqual(result.json(), json_rsp)
        
        request = responses.calls[-1].request
        self.assertEqual(request.headers['content-type'], 'application/json')
        self.assertEqual(json.loads(request.body), expected_data)


    @given(st.integers(min_value=1), st.text(), st.integers(min_value=1), st.integers(min_value=1), st.lists(st.integers() | st.text(), min_size=0, max_size=10))
    def test_update(self, ID, name, portfolio_id, model_id, data_files):
        expected_url = f'{self.url_endpoint}{ID}/'
        expected_data = {
            "name": name,
            "portfolio": portfolio_id,
            "model": model_id,
            "complex_model_data_files": data_files
        }
        responses.put(url=expected_url)
        result = self.api.update(ID, name, portfolio_id, model_id, data_files)

        request = responses.calls[-1].request
        self.assertEqual(request.headers['content-type'], 'application/json')
        self.assertEqual(json.loads(request.body), expected_data)


    @given(st.integers(min_value=1))
    def test_status(self, ID):
        expected_url = f'{self.url_endpoint}{ID}/'
        expected_status = 'INPUTS_GENERATION_STARTED'
        json_rsp = {
            "id": ID,
            "status": expected_status,
        }
        responses.get(url=expected_url, json=json_rsp)
        result = self.api.status(ID)
        self.assertEqual(result, expected_status)

    @given(st.integers(min_value=1))
    def test_generate(self, ID):
        expected_url = f'{self.url_endpoint}{ID}/generate_inputs/'
        responses.post(url=expected_url)
        result = self.api.generate(ID)
        self.assertTrue(result.ok)

    @given(st.integers(min_value=1))
    def test_run(self, ID):
        expected_url = f'{self.url_endpoint}{ID}/run/'
        responses.post(url=expected_url)
        result = self.api.run(ID)
        self.assertTrue(result.ok)

    @given(st.integers(min_value=1))
    def test_cancel_analysis_run(self, ID):
        expected_url = f'{self.url_endpoint}{ID}/cancel_analysis_run/'
        responses.post(url=expected_url)
        result = self.api.cancel_analysis_run(ID)
        self.assertTrue(result.ok)

    @given(st.integers(min_value=1))
    def test_cancel_generate_inputs(self, ID):
        expected_url = f'{self.url_endpoint}{ID}/cancel_generate_inputs/'
        responses.post(url=expected_url)
        result = self.api.cancel_generate_inputs(ID)
        self.assertTrue(result.ok)

    @given(st.integers(min_value=1))
    def test_cancel(self, ID):
        expected_url = f'{self.url_endpoint}{ID}/cancel/'
        responses.post(url=expected_url)
        result = self.api.cancel(ID)
        self.assertTrue(result.ok)

    @given(st.integers(min_value=1))
    def test_copy(self, ID):
        expected_url = f'{self.url_endpoint}{ID}/copy/'
        responses.post(url=expected_url)
        result = self.api.copy(ID)
        self.assertTrue(result.ok)

    @given(st.integers(min_value=1))
    def test_data_files(self, ID):
        expected_url = f'{self.url_endpoint}{ID}/data_files/'
        responses.get(url=expected_url)
        result = self.api.data_files(ID)
        self.assertTrue(result.ok)

    @given(st.integers(min_value=1))
    def test_storage_links(self, ID):
        expected_url = f'{self.url_endpoint}{ID}/storage_links/'
        responses.get(url=expected_url)
        result = self.api.storage_links(ID)
        self.assertTrue(result.ok)

    @given(st.integers(min_value=1))
    def test_sub_task_list(self, ID):
        expected_url = f'{self.url_endpoint}{ID}/sub_task_list/'
        responses.get(url=expected_url)
        result = self.api.sub_task_list(ID)
        self.assertTrue(result.ok)


# ----------------- Working up to here ----------------------------------------#





#
#
#
#
#
# V1
#
#
# class APIClientTests(unittest.TestCase):
# @classmethod
# def setUpClass(cls):
# cls.logger = logging.getLogger(__name__)
# cls.api_url = 'http://localhost:8000'
# cls.api_ver = 'V1'
# cls.username = 'admin'
# cls.password = 'password'
# cls.timeout = 25
##
# def setUp(self):
# self.api_session_mock = MagicMock()
# self.api_models_mock = MagicMock()
# self.api_portfolios_mock = MagicMock()
# self.api_analyses_mock = MagicMock()
# self.api_data_files_mock = MagicMock()
##
# self.api_session_mock.get.return_value = MagicMock()
# self.api_session_mock.post.return_value = MagicMock()
##
# self.api_models_mock.create.return_value = MagicMock(json=MagicMock(return_value={'id': 1}))
# self.api_portfolios_mock.create.return_value = MagicMock(json=MagicMock(return_value={'id': 2}))
##
# self.client = APIClient(
# api_url=self.api_url,
# api_ver=self.api_ver,
# username=self.username,
# password=self.password,
# timeout=self.timeout,
# logger=self.logger
# )
# self.client.api = self.api_session_mock
# self.client.models = self.api_models_mock
# self.client.portfolios = self.api_portfolios_mock
# self.client.analyses = self.api_analyses_mock
# self.client.data_files = self.api_data_files_mock
##
# @given(st.text())
# def test_oed_peril_codes(self, response_text):
# expected_url = '{}oed_peril_codes/'.format(self.api_url)
##
# self.api_session_mock.get.return_value.text = response_text
##
# result = self.client.oed_peril_codes()
##
# self.api_session_mock.get.assert_called_with(expected_url)
# self.assertEqual(result, self.api_session_mock.get.return_value.text)
##
# @given(st.text())
# def test_server_info(self, response_text):
# expected_url = '{}server_info/'.format(self.api_url)
##
# self.api_session_mock.get.return_value.text = response_text
##
# result = self.client.server_info()
##
# self.api_session_mock.get.assert_called_with(expected_url)
# self.assertEqual(result, self.api_session_mock.get.return_value.text)
##
# @given(st.text())
# def test_healthcheck(self, response_text):
# expected_url = '{}healthcheck/'.format(self.api_url)
##
# self.api_session_mock.get.return_value.text = response_text
##
# result = self.client.healthcheck()
##
# self.api_session_mock.get.assert_called_with(expected_url)
# self.assertEqual(result, self.api_session_mock.get.return_value.text)
##
#
#
#
# V2
#
# class APIClientTests(unittest.TestCase):
# @patch('your_module.APISession')
# def setUp(self, MockAPISession):
# self.api_url = 'http://localhost:8000'
# self.api_ver = 'V1'
# self.username = 'admin'
# self.password = 'password'
# self.timeout = 25
# self.logger = MagicMock(spec=logging.Logger)
##
# self.api = MockAPISession.return_value
# self.api.url_base = self.api_url
##
# self.client = APIClient(
# api_url=self.api_url,
# api_ver=self.api_ver,
# username=self.username,
# password=self.password,
# timeout=self.timeout,
# logger=self.logger
# )
##
# def test_oed_peril_codes(self):
# expected_url = '{}/oed_peril_codes/'.format(self.api.url_base)
# self.api.get.return_value.ok = True
##
# result = self.client.oed_peril_codes()
##
# self.api.get.assert_called_with(expected_url)
# self.assertEqual(result, self.api.get.return_value)
##
# def test_server_info(self):
# expected_url = '{}/server_info/'.format(self.api.url_base)
# self.api.get.return_value.ok = True
##
# result = self.client.server_info()
##
# self.api.get.assert_called_with(expected_url)
# self.assertEqual(result, self.api.get.return_value)
##
# def test_healthcheck(self):
# expected_url = '{}/healthcheck/'.format(self.api.url_base)
# self.api.get.return_value.ok = True
##
# result = self.client.healthcheck()
##
# self.api.get.assert_called_with(expected_url)
# self.assertEqual(result, self.api.get.return_value)
##
# def test_upload_inputs_create_portfolio(self):
# portfolio_name = 'Portfolio_01012021-120000'
# location_fp = '/path/to/location_file.csv'
# accounts_fp = '/path/to/accounts_file.csv'
# ri_info_fp = '/path/to/ri_info_file.csv'
# ri_scope_fp = '/path/to/ri_scope_file.csv'
##
# self.client.portfolios.create.return_value.json.return_value = {'id': 1}
# self.client.portfolios.location_file.upload.return_value.ok = True
# self.client.portfolios.accounts_file.upload.return_value.ok = True
# self.client.portfolios.reinsurance_info_file.upload.return_value.ok = True
# self.client.portfolios.reinsurance_scope_file.upload.return_value.ok = True
##
# result = self.client.upload_inputs(
# portfolio_name=portfolio_name,
# location_fp=location_fp,
# accounts_fp=accounts_fp,
# ri_info_fp=ri_info_fp,
# ri_scope_fp=ri_scope_fp
# )
##
# self.client.portfolios.create.assert_called_with(portfolio_name)
# self.client.portfolios.location_file.upload.assert_called_with(1, location_fp)
# self.client.portfolios.accounts_file.upload.assert_called_with(1, accounts_fp)
# self.client.portfolios.reinsurance_info_file.upload.assert_called_with(1, ri_info_fp)
# self.client.portfolios.reinsurance_scope_file.upload.assert_called_with(1, ri_scope_fp)
# self.assertEqual(result, {'id': 1})
#
#
# class APIClientTestCase(unittest.TestCase):
# @patch('mymodule.api_client.APISession')
# def test_init(self, mock_APISession):
# Mock the APISession class
# mock_api_session = Mock()
# mock_APISession.return_value = mock_api_session
##
# Create an instance of APIClient
# api_client = APIClient(api_url='http://localhost:8000', username='admin', password='password')
##
# Check that APISession is initialized with the correct arguments
# mock_APISession.assert_called_once_with('http://localhost:8000', 'admin', 'password', timeout=25)
##
# Check that the APISession instance is assigned to the api attribute
# self.assertEqual(api_client.api, mock_api_session)
##
# Additional assertions for other attributes if needed
##
# @given(text())
# @patch('mymodule.api_client.APISession.post')
# def test_create_model(self, model_data, mock_post):
# Mock the APISession post method
# mock_response = Mock()
# mock_response.json.return_value = {'id': 1, 'name': 'Model 1'}
# mock_post.return_value = mock_response
##
# Create an instance of APIClient
# api_client = APIClient(api_url='http://localhost:8000', username='admin', password='password')
##
# Call the create_model method
# model = api_client.create_model(model_data)
##
# Check that the post method is called with the correct arguments
# mock_post.assert_called_once_with('/models/', json=model_data)
##
# Check that the returned model object has the correct attributes
# self.assertEqual(model.id, 1)
# self.assertEqual(model.name, 'Model 1')
##
# @patch('mymodule.api_client.APISession.get')
# def test_get_model(self, mock_get):
# Mock the APISession get method
# mock_response = Mock()
# mock_response.json.return_value = {'id': 1, 'name': 'Model 1'}
# mock_get.return_value = mock_response
##
# Create an instance of APIClient
# api_client = APIClient(api_url='http://localhost:8000', username='admin', password='password')
##
# Call the get_model method
# model = api_client.get_model(1)
##
# Check that the get method is called with the correct arguments
# mock_get.assert_called_once_with('/models/1/')
##
# Check that the returned model object has the correct attributes
# self.assertEqual(model.id, 1)
# self.assertEqual(model.name, 'Model 1')
##
# @patch('mymodule.api_client.APISession.put')
# def test_update_model(self, mock_put):
# Mock the APISession put method
# mock_response = Mock()
# mock_response.json.return_value = {'id': 1, 'name': 'Updated Model'}
# mock_put.return_value = mock_response
##
# Create an instance of APIClient
# api_client = APIClient(api_url='http://localhost:8000', username='admin', password='password')
##
# Call the update_model method
# model = api_client.update_model(1, {'name': 'Updated Model'})
##
# Check that the put method is called with the correct arguments
# mock_put.assert_called_once_with('/models/1/', json={'name': 'Updated Model'})
##
# Check that the returned model object has the correct attributes
# self.assertEqual(model.id, 1)
# self.assertEqual(model.name, 'Updated Model')
##
# @patch('mymodule.api_client.APISession.delete')
# def test_delete_model(self, mock_delete):
# Create an instance of APIClient
# api_client = APIClient(api_url='http://localhost:8000', username='admin', password='password')
##
# Call the delete_model method
# api_client.delete_model(1)
##
# Check that the delete method is called with the correct arguments
# mock_delete.assert_called_once_with('/models/1/')
##
# @patch('mymodule.api_client.APISession.get')
# def test_get_model_error(self, mock_get):
# Mock the APISession get method to raise an HTTPError
# mock_response = Mock()
# mock_response.raise_for_status.side_effect = HTTPError('Not Found')
# mock_get.return_value = mock_response
##
# Create an instance of APIClient
# api_client = APIClient(api_url='http://localhost:8000', username='admin', password='password')
##
# Call the get_model method and check that it raises an exception
# with self.assertRaises(HTTPError):
# api_client.get_model(1)
##
# Check that the get method is called with the correct arguments
# mock_get.assert_called_once_with('/models/1/')
##
