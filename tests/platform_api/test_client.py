import os
import io
import json
import logging

import pathlib
import pandas as pd

from tempfile import TemporaryDirectory

import unittest
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

from posixpath import join as urljoin

import requests
from requests.exceptions import HTTPError
from requests_toolbelt import MultipartEncoder
import responses
from responses.registries import OrderedRegistry

from packaging import version
from importlib.metadata import version as get_version

from oasislmf.platform_api.session import APISession
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
    SettingTemplatesBaseEndpoint,
    SettingTemplatesEndpoint,
)


settings.register_profile("ci", max_examples=50, deadline=None)
settings.load_profile("ci")

PIWIND_EXP_URL = 'https://raw.githubusercontent.com/OasisLMF/OasisPiWind/main/tests/inputs'
CONTENT_MAP = {
    'parquet': 'application/octet-stream',
    'pq': 'application/octet-stream',
    'csv': 'text/csv',
    'gz': 'application/gzip',
    'zip': 'application/zip',
    'bz2': 'application/x-bzip2',
}


responses_ver = get_version("responses")
DISABLE_DATA_CHECKS = version.parse(responses_ver) >= version.parse("0.25.3")


@responses.activate
def create_api_session(url):
    responses.get(
        url=f'{url}/healthcheck/',
        json={"status": "OK"})

    responses.post(
        url=f'{url}/access_token/',
        json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
        headers={"authorization": "Bearer acc_tkn"})
    return APISession(url, auth_type='simple', username='testUser', password='testPass')


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

    def tearDown(self):
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

    @given(
        ID=st.integers(min_value=1),
        file_path=st.from_regex(r"^[a-zA-Z0-9_-]+$"),
        overwrite=st.booleans(),
        payload=st.dictionaries(keys=st.text(min_size=1), values=st.text(min_size=1))
    )
    def test_download__subdir_is_created(self, ID, file_path, overwrite, payload):
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        responses.get(url=expected_url, json=payload)

        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, 'sub-dir', file_path))
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

    def tearDown(self):
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
    @pytest.mark.skipif(DISABLE_DATA_CHECKS, reason="Test not compatible with responses > 0.25.3")
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

    @given(
        ID=st.integers(min_value=1),
        file_path=st.from_regex(r"^[a-zA-Z0-9_-]+$"),
        overwrite=st.booleans(),
    )
    def test_download__subdir_is_created(self, ID, file_path, overwrite):
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        file_name = f'{file_path}.csv'

        responses.get(
            expected_url,
            body=self.csv_data,
            content_type=CONTENT_MAP['csv'],
            stream=True
        )

        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, 'sub-dir', file_name))
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
    @pytest.mark.skipif(DISABLE_DATA_CHECKS, reason="Test not compatible with responses > 0.25.3")
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
            self.api.post(2, b'some data', 'application/octet-stream')

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
            self.api.get_dataframe(ID)

        exception = context.exception
        expected_msg = f'Unsupported filetype for Dataframe conversion: {content_type}'
        self.assertEqual(str(exception), expected_msg)

    @pytest.mark.skipif(DISABLE_DATA_CHECKS, reason="Test not compatible with responses > 0.25.3")
    def test_post_dataframe(self):
        ID = 1
        expected_url = '{}/{}/{}'.format(self.url_endpoint, ID, self.url_resource)
        expected_df = pd.read_csv(io.BytesIO(self.csv_data), encoding='utf8')
        request_resp_json = {
            "created": "2023-05-25T13:55:50.665455Z",
            "file": "5ab983f8f9144b8bbad582367b043c1f.csv",
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


class APIDatafilesTests(unittest.TestCase):

    def setUp(self):
        assert responses, 'responses package required to run'
        self.session = create_api_session('http://example.com/api')
        self.url_endpoint = urljoin(self.session.url_base, 'data_files/')
        self.api = API_datafiles(self.session, self.url_endpoint)
        responses.start()

    def tearDown(self):
        responses.stop()
        responses.reset()

    def test_endpoint_setup(self):
        endpoint_list = ['content']
        for endpoint in endpoint_list:
            OasisAPI_obj = getattr(self.api, endpoint)
            self.assertEqual(OasisAPI_obj.url_resource, f'{endpoint}/')
            self.assertEqual(OasisAPI_obj.url_endpoint, self.url_endpoint)
            self.assertTrue(isinstance(OasisAPI_obj, (
                ApiEndpoint,
                FileEndpoint,
                JsonEndpoint
            )))

    @given(st.text(), st.text())
    def test_create(self, file_description, file_category):
        ID = 2
        expected_url = self.url_endpoint
        expected_data = {
            "file_description": file_description,
            "file_category": file_category,
        }
        json_rsp = {
            "id": ID,
            "file_description": file_description,
            "file_category": file_category,
            "created": "2023-05-26T11:43:26.820340Z",
            "modified": "2023-05-26T11:43:26.820340Z",
            "file": None,
            "filename": None,
            "stored": None,
            "content_type": None
        }

        responses.post(url=expected_url, json=json_rsp)
        result = self.api.create(file_description, file_category)
        self.assertEqual(result.json(), json_rsp)

        request = responses.calls[-1].request
        self.assertEqual(request.headers['content-type'], 'application/json')
        self.assertEqual(json.loads(request.body), expected_data)

    @given(st.integers(min_value=1), st.text(), st.text())
    def test_update(self, ID, file_description, file_category):
        expected_url = f'{self.url_endpoint}{ID}/'
        expected_data = {
            "file_description": file_description,
            "file_category": file_category,
        }

        responses.put(url=expected_url)
        self.api.update(ID, file_description, file_category)

        request = responses.calls[-1].request
        self.assertEqual(request.headers['content-type'], 'application/json')
        self.assertEqual(json.loads(request.body), expected_data)


class SettingTemplatesBaseEndpointTest(unittest.TestCase):
    def setUp(self):
        assert responses, 'responses package required to run'
        self.url_endpoint = 'http://example.com/api'
        self.session = create_api_session(self.url_endpoint)
        self.url_resource = 'resource/'
        self.headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        self.api__noresource = SettingTemplatesBaseEndpoint(self.session, self.url_endpoint)
        self.api = SettingTemplatesBaseEndpoint(self.session, self.url_endpoint, self.url_resource)
        responses.start()

    def tearDown(self):
        responses.stop()
        responses.reset()

    def test_build_url(self):
        model_pk = 123
        ID = 456
        expected_url = '{}/{}/{}/{}/{}'.format(self.url_endpoint, model_pk,
                                               'setting_templates', ID, self.url_resource)
        result = self.api._build_url(model_pk, ID)
        self.assertEqual(result, expected_url)

    def test_buil_url__no_id(self):
        model_pk = 123
        expected_url = '{}/{}/{}/'.format(self.url_endpoint, model_pk,
                                          'setting_templates')
        result = self.api__noresource._build_url(model_pk)
        self.assertEqual(result, expected_url)

    @given(model_pk=st.integers(min_value=1), ID=st.integers(min_value=1))
    def test_get__resource(self, model_pk, ID):
        expected_url = '{}/{}/{}/{}/{}'.format(self.url_endpoint, model_pk,
                                               'setting_templates', ID,
                                               self.url_resource)
        responses.get(url=expected_url, json=[])

        rsp = self.api.get(model_pk, ID)
        self.assertEqual(rsp.url, expected_url)

    @given(model_pk=st.integers(min_value=1), ID=st.integers(min_value=1))
    def test_get__model_id(self, model_pk, ID):
        expected_url = '{}/{}/{}/{}'.format(self.url_endpoint, model_pk,
                                            'setting_templates', ID)
        responses.get(url=expected_url, json=[])
        logger = logging.getLogger(__name__)
        logger.info(f'expected_url: {expected_url}')

        rsp = self.api__noresource.get(model_pk, ID)
        logger.info(f'rsp_url: {rsp.url}')
        self.assertEqual(rsp.url, expected_url)

    @given(model_pk=st.integers(min_value=1))
    def test_get(self, model_pk):
        expected_url = '{}/{}/{}/'.format(self.url_endpoint, model_pk,
                                          'setting_templates')
        responses.get(url=expected_url, json=[])

        rsp = self.api__noresource.get(model_pk)
        self.assertEqual(rsp.url, expected_url)

    @given(model_pk=st.integers(min_value=1), ID=st.integers(min_value=1), data=st.dictionaries(keys=st.text(), values=st.text()))
    def test_post(self, model_pk, ID, data):
        expected_url = '{}/{}/{}/{}/{}'.format(self.url_endpoint, model_pk,
                                               'setting_templates', ID,
                                               self.url_resource)
        responses.post(url=expected_url, headers=self.headers)

        rsp = self.api.post(model_pk=model_pk, ID=ID, data=data)
        self.assertEqual(rsp.url, expected_url)

    @given(model_pk=st.integers(min_value=1), ID=st.integers(min_value=1))
    def test_delete(self, model_pk, ID):
        expected_url = '{}/{}/{}/{}/{}'.format(self.url_endpoint, model_pk,
                                               'setting_templates', ID,
                                               self.url_resource)
        responses.delete(url=expected_url, headers=self.headers)

        rsp = self.api.delete(model_pk, ID)
        self.assertEqual(rsp.url, expected_url)


class SettingTemplatesEndpointTest(unittest.TestCase):
    def setUp(self):
        assert responses, 'responses package required to run'
        self.url_endpoint = 'http://example.com/api'
        self.session = create_api_session(self.url_endpoint)
        self.api = SettingTemplatesEndpoint(self.session, self.url_endpoint)
        responses.start()

    def tearDown(self):
        responses.stop()
        responses.reset()

    def test_endpoint_setup(self):
        self.assertTrue(isinstance(self.api, SettingTemplatesBaseEndpoint))

        content_obj = getattr(self.api, 'content')
        self.assertEqual(content_obj.url_resource, 'content/')
        self.assertEqual(content_obj.url_endpoint, self.url_endpoint)
        self.assertTrue(isinstance(content_obj, SettingTemplatesBaseEndpoint))


class APIModelsTests(unittest.TestCase):
    def setUp(self):
        assert responses, 'responses package required to run'
        self.session = create_api_session('http://example.com/api')
        self.url_endpoint = urljoin(self.session.url_base, 'models/')
        self.api = API_models(self.session, self.url_endpoint)
        responses.start()

    def tearDown(self):
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

        # SettingTemplates url_resource not set
        self.assertTrue(isinstance(getattr(self.api, 'setting_templates'), SettingTemplatesEndpoint))

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

    def tearDown(self):
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
        self.api.update(ID, name)

        request = responses.calls[-1].request
        self.assertEqual(request.headers['content-type'], 'application/json')
        self.assertEqual(json.loads(request.body), expected_data)

    @given(st.integers(min_value=1), st.text(), st.integers(min_value=1))
    def test_create_analyses(self, ID, name, model_id):
        expected_url = '{}{}/create_analysis/'.format(self.url_endpoint, ID)
        expected_data = {"name": name, "model": model_id}

        responses.post(url=expected_url)
        self.api.create_analyses(ID, name, model_id)
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

    def tearDown(self):
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
        self.api.update(ID, name, portfolio_id, model_id, data_files)

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


class APIClientTests(unittest.TestCase):

    def setUp(self):
        assert responses, 'responses package required to run'
        self.api_url = 'http://example.com/api'
        self.api_ver = 'v1'
        self.username = 'testUser'
        self.password = 'testPass'
        self.timeout = 0.1
        self.logger = MagicMock(spec=logging.Logger)

        responses.start()
        responses.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})

        responses.post(
            url=f'{self.api_url}/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"})

        self.client = APIClient(
            api_url=self.api_url,
            api_ver=self.api_ver,
            auth_type="simple",
            username=self.username,
            password=self.password,
            timeout=self.timeout,
            logger=self.logger
        )

    def tearDown(self):
        responses.stop()
        responses.reset()

    def test_endpoint_setup(self):
        self.assertEqual(self.client.models.url_endpoint, f'{self.api_url}/{self.api_ver}/models/')
        self.assertEqual(self.client.portfolios.url_endpoint, f'{self.api_url}/{self.api_ver}/portfolios/')
        self.assertEqual(self.client.analyses.url_endpoint, f'{self.api_url}/{self.api_ver}/analyses/')
        self.assertEqual(self.client.data_files.url_endpoint, f'{self.api_url}/{self.api_ver}/data_files/')

        self.assertTrue(isinstance(self.client.models, API_models))
        self.assertTrue(isinstance(self.client.portfolios, API_portfolios))
        self.assertTrue(isinstance(self.client.analyses, API_analyses))
        self.assertTrue(isinstance(self.client.data_files, API_datafiles))

    def test_oed_peril_codes(self):
        expected_url = '{}/oed_peril_codes/'.format(self.api_url)
        responses.get(url=expected_url)
        result = self.client.oed_peril_codes()
        self.assertTrue(result.ok)

    def test_server_info(self):
        expected_url = '{}/server_info/'.format(self.api_url)
        responses.get(url=expected_url)
        result = self.client.server_info()
        self.assertTrue(result.ok)

    def test_healthcheck(self):
        expected_url = '{}/healthcheck/'.format(self.api_url)
        responses.get(url=expected_url)
        result = self.client.healthcheck()
        self.assertTrue(result.ok)

    def test_upload_inputs__create_portfolio(self):
        ID = 3
        portfolio_name = 'Portfolio_01012021-120000'
        location_fp = 'location_file.csv'
        accounts_fp = 'accounts_file.csv'
        ri_info_fp = 'ri_info_file.csv'
        ri_scope_fp = 'ri_scope_file.csv'

        # create fake data
        with TemporaryDirectory() as d:
            for exp_file in [location_fp, accounts_fp, ri_info_fp, ri_scope_fp]:
                abs_fp = os.path.realpath(os.path.join(d, exp_file))
                with open(abs_fp, 'wb') as file:
                    file.write(f'Dummy data for {abs_fp}'.encode('ascii'))

            # create fake responces
            responses.post(url=self.client.portfolios.url_endpoint, json={'id': ID})
            responses.post(url=self.client.portfolios.location_file._build_url(ID))
            responses.post(url=self.client.portfolios.accounts_file._build_url(ID))
            responses.post(url=self.client.portfolios.reinsurance_info_file._build_url(ID))
            responses.post(url=self.client.portfolios.reinsurance_scope_file._build_url(ID))

            result = self.client.upload_inputs(
                portfolio_name=portfolio_name,
                location_fp=os.path.join(d, location_fp),
                accounts_fp=os.path.join(d, accounts_fp),
                ri_info_fp=os.path.join(d, ri_info_fp),
                ri_scope_fp=os.path.join(d, ri_scope_fp)
            )
            self.assertEqual(result, {'id': ID})

    def test_upload_inputs__update_success(self):
        ID = 3
        location_fp = 'location_file.csv'

        # create fake data
        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, location_fp))
            with open(abs_fp, 'wb') as file:
                file.write(f'Dummy data for {abs_fp}'.encode('ascii'))

            # create fake responces
            responses.put(url=urljoin(self.client.portfolios.url_endpoint, f'{ID}/'), json={'id': ID})
            responses.post(url=self.client.portfolios.location_file._build_url(ID))

            result = self.client.upload_inputs(
                portfolio_id=ID,
                location_fp=os.path.join(d, location_fp),
            )
            self.assertEqual(result, {'id': ID})

    def test_upload_inputs__update_failed(self):
        ID = 3
        location_fp = 'location_file.csv'

        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, location_fp))
            with open(abs_fp, 'wb') as file:
                file.write(f'Dummy data for {abs_fp}'.encode('ascii'))

            responses.put(url=urljoin(self.client.portfolios.url_endpoint, f'{ID}/'), json={"error": "not found"}, status=404)
            responses.post(url=self.client.portfolios.location_file._build_url(ID))
            with self.assertRaises(OasisException):
                self.client.upload_inputs(
                    portfolio_id=ID,
                    location_fp=os.path.join(d, location_fp),
                )

    def test_upload_settings__as_dict(self):
        ID = 5
        settings = {'run': 'options', 'model_settings': '...etc..'}
        expected_url = self.client.analyses.settings._build_url(ID)
        responses.post(url=expected_url)

        self.client.upload_settings(ID, settings)
        request = responses.calls[-1].request
        self.assertEqual(json.loads(request.body), settings)

    def test_upload_settings__as_file(self):
        ID = 5
        settings_fp = 'settings.json'
        settings = {'run': 'options', 'model_settings': '...etc..'}
        expected_url = self.client.analyses.settings._build_url(ID)
        responses.post(url=expected_url)

        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, settings_fp))
            with open(abs_fp, 'w') as file:
                json.dump(settings, file)
            self.client.upload_settings(ID, abs_fp)

        request = responses.calls[-1].request
        self.assertEqual(json.loads(request.body), settings)

    def test_upload_settings__invalid(self):
        with self.assertRaises(TypeError):
            self.client.upload_settings(1, 1)

    def test_create_analysis__success(self):
        settings = {'run': 'options', 'model_settings': '...etc..'}
        settings_fp = 'settings.json'

        responses.post(url=self.client.analyses.url_endpoint, json={'id': 69})
        settings_expected_url = self.client.analyses.settings._build_url(69)
        responses.post(url=settings_expected_url)

        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, settings_fp))
            with open(abs_fp, 'w') as file:
                json.dump(settings, file)

            self.client.create_analysis(
                portfolio_id=42,
                model_id=5,
                analysis_name='my_analysis',
                analysis_settings_fp=abs_fp
            )

    def test_create_analysis__invalid(self):
        settings = {'run': 'options', 'model_settings': '...etc..'}
        settings_fp = 'settings.json'

        responses.post(url=self.client.analyses.url_endpoint,
                       json={'bad request': 'Portfolio not found'}, status=400)
        settings_expected_url = self.client.analyses.settings._build_url(69)
        responses.post(url=settings_expected_url)

        with TemporaryDirectory() as d:
            abs_fp = os.path.realpath(os.path.join(d, settings_fp))
            with open(abs_fp, 'w') as file:
                json.dump(settings, file)

            with self.assertRaises(OasisException):
                self.client.create_analysis(
                    portfolio_id=42,
                    model_id=5,
                )

    def test_run_generate__success(self):
        ID = 1
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/'
        exec_url = f'{expected_url}generate_inputs/'

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            rsps.post(exec_url, json={"id": ID, "status": "INPUTS_GENERATION_QUEUED"})
            rsps.get(expected_url, json={"id": ID, "status": "INPUTS_GENERATION_STARTED"})
            rsps.get(expected_url, json={"id": ID, "status": "INPUTS_GENERATION_STARTED"})
            rsps.get(expected_url, json={"id": ID, "status": "INPUTS_GENERATION_STARTED"})
            rsps.get(expected_url, json={"id": ID, "status": "READY"})
            result = self.client.run_generate(analysis_id=ID, poll_interval=0.1)

            self.assertTrue(result)
            self.logger.info.assert_any_call(f'Inputs Generation: Starting (id={ID})')
            self.logger.info.assert_any_call(f'Input Generation: Queued (id={ID})')
            self.logger.info.assert_any_call(f'Input Generation: Executing (id={ID})')
            self.logger.info.assert_any_call(f'Inputs Generation: Complete (id={ID})')

    def test_run_generate__cancelled(self):
        ID = 1
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/'
        exec_url = f'{expected_url}generate_inputs/'

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            rsps.post(exec_url, json={"id": ID, "status": "INPUTS_GENERATION_QUEUED"})
            rsps.get(expected_url, json={"id": ID, "status": "INPUTS_GENERATION_STARTED"})
            rsps.get(expected_url, json={"id": ID, "status": "INPUTS_GENERATION_CANCELLED"})
            result = self.client.run_generate(analysis_id=ID, poll_interval=0.1)

            self.assertFalse(result)
            self.logger.info.assert_any_call(f'Input Generation: Cancelled (id={ID})')

    def test_run_generate__exec_error(self):
        ID = 1
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/'
        exec_url = f'{expected_url}generate_inputs/'
        trace_url = f'{expected_url}input_generation_traceback_file/'
        trace_error_msg = 'run error logs'

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            rsps.post(exec_url, json={"id": ID, "status": "INPUTS_GENERATION_QUEUED"})
            rsps.get(expected_url, json={"id": ID, "status": "INPUTS_GENERATION_STARTED"})
            rsps.get(expected_url, json={"id": ID, "status": "INPUTS_GENERATION_ERROR"})
            rsps.get(trace_url, body=trace_error_msg)
            result = self.client.run_generate(analysis_id=ID, poll_interval=0.1)

            self.assertFalse(result)
            self.logger.error.assert_called_with(trace_error_msg)

    def test_run_generate__http_error(self):
        ID = 1
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/'
        exec_url = f'{expected_url}generate_inputs/'

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            rsps.post(exec_url, json={"id": ID, "status": "INPUTS_GENERATION_QUEUED"})
            rsps.get(expected_url, json={"id": ID, "status": "INPUTS_GENERATION_STARTED"})
            rsps.get(expected_url, json={"error": "Analysis not found"}, status=404)

            with self.assertRaises(OasisException):
                result = self.client.run_generate(analysis_id=ID, poll_interval=0.1)
                self.assertFalse(result)

    def test_run_generate__unknown_status(self):
        ID = 1
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/'
        exec_url = f'{expected_url}generate_inputs/'

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            rsps.post(exec_url, json={"id": ID, "status": "INPUTS_GENERATION_QUEUED"})
            rsps.get(expected_url, json={"id": ID, "status": "INPUTS_GENERATION_STARTED"})
            rsps.get(expected_url, json={"id": ID, "status": "SOME_NEW_STATUS"})

            with self.assertRaises(OasisException):
                result = self.client.run_generate(analysis_id=ID, poll_interval=0.1)
                self.assertFalse(result)

    def test_run_generate__with_subtasks_success(self):
        ID = 1
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/'
        exec_url = f'{expected_url}generate_inputs/'

        sub_task_data_fp = os.path.join(os.path.dirname(__file__), 'data', 'input_sub-tasks.json')
        sub_task_url = f'{expected_url}sub_task_list/'
        with open(sub_task_data_fp, mode='r') as f:
            sub_task_data = json.load(f)

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            rsps.post(exec_url, json={"id": ID, "status": "INPUTS_GENERATION_QUEUED", "sub_task_list": "http://some-url", 'run_mode': 'V2'})
            rsps.get(expected_url, json={"id": ID, "status": "INPUTS_GENERATION_STARTED", "sub_task_list": "http://some-url", 'run_mode': 'V2'})
            rsps.get(sub_task_url, json=sub_task_data)
            rsps.get(sub_task_url, json=sub_task_data)
            rsps.get(expected_url, json={"id": ID, "status": "READY"})
            result = self.client.run_generate(analysis_id=ID, poll_interval=0.1)
            self.assertTrue(result)

    def test_run_generate__with_subtasks_cancelled(self):
        ID = 1
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/'
        exec_url = f'{expected_url}generate_inputs/'

        sub_task_data_fp = os.path.join(os.path.dirname(__file__), 'data', 'input_sub-tasks.json')
        sub_task_url = f'{expected_url}sub_task_list/'
        with open(sub_task_data_fp, mode='r') as f:
            sub_task_data = json.load(f)

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            rsps.post(exec_url, json={"id": ID, "status": "INPUTS_GENERATION_QUEUED", "sub_task_list": "http://some-url", 'run_mode': 'V2'})
            rsps.get(expected_url, json={"id": ID, "status": "INPUTS_GENERATION_STARTED", "sub_task_list": "http://some-url", 'run_mode': 'V2'})
            rsps.get(sub_task_url, json=sub_task_data)
            rsps.get(sub_task_url, json=sub_task_data)
            rsps.get(expected_url, json={"id": ID, "status": "INPUTS_GENERATION_CANCELLED"})
            result = self.client.run_generate(analysis_id=ID, poll_interval=0.1)
            self.assertFalse(result)

    def test_run_analysis__success(self):
        ID = 1
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/'
        expected_settings = {'model_settings': ".. settings here .."}
        expected_settings_url = self.client.analyses.settings._build_url(ID)
        exec_url = f'{expected_url}run/'

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            rsps.post(expected_settings_url)
            rsps.post(exec_url, json={"id": ID, "status": "RUN_QUEUED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_QUEUED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_STARTED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_STARTED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_STARTED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_COMPLETED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_COMPLETED"})
            result = self.client.run_analysis(analysis_id=ID, poll_interval=0.1, analysis_settings_fp=expected_settings)

            self.assertTrue(result)
            self.logger.info.assert_any_call(f'Analysis Run: Starting (id={ID})')
            self.logger.info.assert_any_call(f'Analysis Run: Queued (id={ID})')
            self.logger.info.assert_any_call(f'Analysis Run: Executing (id={ID})')
            self.logger.info.assert_any_call(f'Analysis Run: Complete (id={ID})')

    def test_run_analysis__cancelled(self):
        ID = 1
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/'
        exec_url = f'{expected_url}run/'

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            rsps.post(exec_url, json={"id": ID, "status": "RUN_QUEUED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_STARTED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_CANCELLED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_CANCELLED"})
            result = self.client.run_analysis(analysis_id=ID, poll_interval=0.1)

            self.assertFalse(result)
            self.logger.info.assert_any_call(f'Analysis Run: Cancelled (id={ID})')

    def test_run_analysis__exec_error(self):
        ID = 1
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/'
        exec_url = f'{expected_url}run/'
        trace_url = f'{expected_url}run_traceback_file/'
        trace_error_msg = 'run error logs'

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            rsps.post(exec_url, json={"id": ID, "status": "RUN_QUEUED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_QUEUED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_STARTED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_ERROR"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_ERROR"})
            rsps.get(trace_url, body=trace_error_msg)
            result = self.client.run_analysis(analysis_id=ID, poll_interval=0.1)

            self.assertFalse(result)
            self.logger.error.assert_called_with(f"Analysis Run: Failed (id={ID})\n\nServer logs:\n{trace_error_msg}")

    def test_run_analysis__unknown_status(self):
        ID = 1
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/'
        exec_url = f'{expected_url}run/'

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            rsps.post(exec_url, json={"id": ID, "status": "RUN_QUEUED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_STARTED"})
            rsps.get(expected_url, json={"id": ID, "status": "SOME_NEW_STATUS"})

            with self.assertRaises(OasisException):
                result = self.client.run_analysis(analysis_id=ID, poll_interval=0.1)
                self.assertFalse(result)

    def test_run_analysis__http_error(self):
        ID = 1
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/'
        exec_url = f'{expected_url}run/'

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            rsps.post(exec_url, json={"error": "Analysis ID not fount"}, status=404)
            with self.assertRaises(OasisException):
                self.client.run_analysis(analysis_id=ID, poll_interval=0.1)

    def test_run_analysis__with_subtasks_success(self):
        ID = 1
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/'
        exec_url = f'{expected_url}run/'

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            rsps.post(exec_url, json={"id": ID, "status": "RUN_QUEUED", "sub_task_list": "http://some-url", 'run_mode': 'V2'})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_QUEUED", "sub_task_list": "http://some-url", 'run_mode': 'V2'})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_STARTED", "sub_task_list": "http://some-url", 'run_mode': 'V2'})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_COMPLETED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_COMPLETED"})
            result = self.client.run_analysis(analysis_id=ID, poll_interval=0.1)
            self.assertTrue(result)

    def test_run_analysis__with_subtasks_cancelled(self):
        ID = 1
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/'
        exec_url = f'{expected_url}run/'

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            rsps.post(exec_url, json={"id": ID, "status": "RUN_QUEUED", "sub_task_list": "http://some-url", 'run_mode': 'V2'})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_STARTED", "sub_task_list": "http://some-url", 'run_mode': 'V2'})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_CANCELLED"})
            rsps.get(expected_url, json={"id": ID, "status": "RUN_CANCELLED"})
            result = self.client.run_analysis(analysis_id=ID, poll_interval=0.1)
            self.assertFalse(result)

    def test_cancel_generate__success(self):
        ID = 3
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/cancel_generate_inputs/'
        responses.post(url=expected_url)
        self.client.cancel_generate(ID)

    def test_cancel_generate__error(self):
        ID = 3
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/cancel_generate_inputs/'
        responses.post(url=expected_url, json={'Error': "not found"}, status=404)
        with self.assertRaises(OasisException):
            self.client.cancel_generate(ID)

    def test_cancel_analysis__success(self):
        ID = 66
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/cancel_analysis_run/'
        responses.post(url=expected_url)
        self.client.cancel_analysis(ID)

    def test_cancel_analysis__error(self):
        ID = 66
        expected_url = f'{self.client.analyses.url_endpoint}{ID}/cancel_analysis_run/'
        responses.post(url=expected_url, json={'Error': "not found"}, status=404)
        with self.assertRaises(OasisException):
            self.client.cancel_analysis(ID)

    def test_download_output__success(self):
        ID = 33
        tar_fp = os.path.join(os.path.dirname(__file__), 'data', 'analysis_1_output.tar')
        with open(tar_fp, mode='rb') as file:
            tar_data = file.read()

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            expected_url_get = f'{self.client.analyses.url_endpoint}{ID}/output_file/'
            rsps.get(
                expected_url_get,
                body=tar_data,
                content_type=CONTENT_MAP['gz'],
                stream=True)

            expected_url_delete = f'{self.client.analyses.url_endpoint}{ID}/'
            rsps.delete(expected_url_delete)

            with TemporaryDirectory() as d:
                tar_filename = 'test_outputs.tar'
                abs_fp = os.path.realpath(os.path.join(d, tar_filename))
                pathlib.Path(abs_fp).touch()
                self.client.download_output(ID, download_path=d, filename=tar_filename, clean_up=True, overwrite=True)
                self.assertTrue(os.path.isfile(abs_fp))

    def test_download_output__error(self):
        ID = 33

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            expected_url_get = f'{self.client.analyses.url_endpoint}{ID}/output_file/'
            rsps.get(expected_url_get, json={'Error': 'Analysis not found'}, status=404)
            with self.assertRaises(OasisException):
                self.client.download_output(ID)
