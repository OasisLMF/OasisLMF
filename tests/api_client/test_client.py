from __future__ import unicode_literals

import json
import os
import io
import string
from random import choice
from tempfile import NamedTemporaryFile
from backports.tempfile import TemporaryDirectory
from unittest import TestCase

import responses
from hypothesis import given
from hypothesis.strategies import integers
from mock import patch, Mock
from pathlib2 import Path
from requests import RequestException

from oasislmf.api_client.client import OasisAPIClient
from oasislmf.model_execution.files import TAR_FILE
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.status import STATUS_FAILURE, STATUS_PENDING, STATUS_SUCCESS


class ClientHealthCheck(TestCase):
    @given(integers(min_value=1, max_value=5))
    def test_heath_check_raise_an_exception_on_each_call___result_is_false(self, max_attempts):
        with patch('requests.get', Mock(side_effect=RequestException())) as get_mock:
            client = OasisAPIClient('http://localhost:8001')

            result = client.health_check(max_attempts, retry_delay=0)

            self.assertFalse(result)
            self.assertEqual(max_attempts, get_mock.call_count)
            call_urls = [args[0] for args in get_mock.call_args_list]
            self.assertEqual(
                [('http://localhost:8001/healthcheck',) for i in range(max_attempts)],
                call_urls,
            )

    @given(integers(min_value=1, max_value=5))
    def test_heath_check_returns_non_200_on_each_call___result_is_false(self, max_attempts):
        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET, 'http://localhost:8001/healthcheck', status=404)

            client = OasisAPIClient('http://localhost:8001')

            result = client.health_check(max_attempts, retry_delay=0)

            self.assertFalse(result)
            self.assertEqual(max_attempts, len(rsps.calls))
            self.assertEqual(
                ['http://localhost:8001/healthcheck' for i in range(max_attempts)],
                [call.request.url for call in rsps.calls],
            )

    def test_heath_check_returns_200___result_is_true(self):
        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET, 'http://localhost:8001/healthcheck', status=200)

            client = OasisAPIClient('http://localhost:8001')

            result = client.health_check()

            self.assertTrue(result)
            self.assertEqual(1, len(rsps.calls))
            self.assertEqual('http://localhost:8001/healthcheck', rsps.calls[0].request.url)


class RunAnalysis(TestCase):
    def test_request_response_is_not_ok___exception_is_raised(self):
        client = OasisAPIClient('http://localhost:8001')

        with responses.RequestsMock() as rsps:
            rsps.add(responses.POST, 'http://localhost:8001/analysis/foo', status=400)

            with self.assertRaises(OasisException):
                client.run_analysis({'analysis': 'data'}, 'foo')

            self.assertEqual(1, len(rsps.calls))
            self.assertEqual('http://localhost:8001/analysis/foo', rsps.calls[0].request.url)
            self.assertEqual({'analysis': 'data'}, json.loads(rsps.calls[0].request.body.decode()))

    def test_request_response_is_ok___response_location_is_returned(self):
        client = OasisAPIClient('http://localhost:8001')

        with responses.RequestsMock() as rsps:
            rsps.add(responses.POST, 'http://localhost:8001/analysis/foo', status=200, body=b'{"location": "output-location"}')

            self.assertEqual('output-location', client.run_analysis({'analysis': 'data'}, 'foo'))


class RunAnalysisAndPoll(TestCase):
    analysis_status_location = 'analysis_status_location'
    analysis_output_location = 'analysis_output_location'
    input_location = 'input_location'
    output_location = 'input_location'
    analysis_settings = {'analysis': 'settings'}

    def get_mocked_client(self):
        client = OasisAPIClient('http://localhost:8001')
        client.run_analysis = Mock(return_value=self.analysis_status_location)
        client.get_analysis_status = Mock(return_value=(STATUS_SUCCESS, self.analysis_output_location))
        client.download_outputs = Mock()
        client.delete_exposure = Mock()
        client.delete_outputs = Mock()
        return client

    def test_data_is_ready_at_first_status_call___data_is_downloaded_and_cleaned_correctly_never_sleeping(self):
        with patch('oasislmf.api_client.client.time.sleep') as sleep_mock:
            client = self.get_mocked_client()

            client.run_analysis_and_poll(self.analysis_settings, self.input_location, self.output_location)

            client.run_analysis.assert_called_once_with(self.analysis_settings, self.input_location)
            client.get_analysis_status.assert_called_once_with(self.analysis_status_location)
            sleep_mock.assert_not_called()
            client.download_outputs.assert_called_once_with(self.analysis_output_location, os.path.join(self.output_location, self.analysis_output_location + '.tar.gz'))
            client.delete_exposure.assert_called_once_with(self.input_location)
            client.delete_outputs.assert_called_once_with(self.analysis_output_location)

    @given(integers(min_value=1, max_value=10), integers(min_value=1, max_value=10))
    def test_data_is_ready_after_multiple_status_calls___data_is_downloaded_and_ceaned_correctly_sleeping_after_every_pending_call(self, pending_calls, poll_time):
        with patch('oasislmf.api_client.client.time.sleep') as sleep_mock:
            client = self.get_mocked_client()
            client.get_analysis_status = Mock(side_effect=[(STATUS_PENDING, '')] * pending_calls + [(STATUS_SUCCESS, self.analysis_output_location)])

            client.run_analysis_and_poll(self.analysis_settings, self.input_location, self.output_location, analysis_poll_interval=poll_time)

            client.run_analysis.assert_called_once_with(self.analysis_settings, self.input_location)

            self.assertEqual(pending_calls + 1, client.get_analysis_status.call_count)
            self.assertEqual([(self.analysis_status_location, )] * (pending_calls + 1), [args[0] for args in client.get_analysis_status.call_args_list])

            self.assertEqual(pending_calls, sleep_mock.call_count)
            self.assertEqual([(poll_time, )] * pending_calls, [args[0] for args in sleep_mock.call_args_list])

            client.download_outputs.assert_called_once_with(self.analysis_output_location, os.path.join(self.output_location, self.analysis_output_location + '.tar.gz'))
            client.delete_exposure.assert_called_once_with(self.input_location)
            client.delete_outputs.assert_called_once_with(self.analysis_output_location)


class GetAnalysisStatus(TestCase):
    def test_request_is_not_ok___exception_is_raised(self):
        client = OasisAPIClient('http://localhost:8001')

        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET, 'http://localhost:8001/analysis_status/foo', status=400)

            with self.assertRaises(OasisException):
                client.get_analysis_status('foo')

    def test_request_status_is_failure___exception_is_raised(self):
        client = OasisAPIClient('http://localhost:8001')

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                'http://localhost:8001/analysis_status/foo',
                status=200,
                body=json.dumps({'status': STATUS_FAILURE, 'message': 'oops'}).encode()
            )

            with self.assertRaises(OasisException):
                client.get_analysis_status('foo')

    def test_request_status_is_pending___result_status_is_pending_location_is_empty(self):
        client = OasisAPIClient('http://localhost:8001')

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                'http://localhost:8001/analysis_status/foo',
                status=200,
                body=json.dumps({'status': STATUS_PENDING}).encode()
            )

            status, outputs_location = client.get_analysis_status('foo')

            self.assertEqual(status, STATUS_PENDING)
            self.assertEqual(outputs_location, '')

    def test_request_status_is_success___result_status_is_success_location_is_from_response_body(self):
        client = OasisAPIClient('http://localhost:8001')

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                'http://localhost:8001/analysis_status/foo',
                status=200,
                body=json.dumps({'status': STATUS_SUCCESS, 'outputs_location': 'outputs-location'}).encode()
            )

            status, outputs_location = client.get_analysis_status('foo')

            self.assertEqual(status, STATUS_SUCCESS)
            self.assertEqual(outputs_location, 'outputs-location')


class DeleteResource(TestCase):
    def test_response_is_not_ok___warning_is_logged(self):
        client = OasisAPIClient('http://localhost:8001', Mock())

        with responses.RequestsMock() as rsps:
            rsps.add(responses.DELETE, 'http://localhost:8001/foo', status=400)

            client.delete_resource('foo')

            client._logger.warning.assert_called_with("DELETE http://localhost:8001/foo failed: 400")

    def test_response_is_ok___info_is_logged(self):
        client = OasisAPIClient('http://localhost:8001', Mock())

        with responses.RequestsMock() as rsps:
            rsps.add(responses.DELETE, 'http://localhost:8001/foo', status=200)

            client.delete_resource('foo')

            client._logger.info.assert_called_with('Deleted http://localhost:8001/foo')


class DeleteOutputs(TestCase):
    def test_delete_resource_is_called_with_the_correct_parameters(self):
        client = OasisAPIClient('http://localhost:8001')
        client.delete_resource = Mock()

        client.delete_outputs('foo')

        client.delete_resource.assert_called_once_with('/outputs/foo')


class DeleteExposures(TestCase):
    def test_delete_resource_is_called_with_the_correct_parameters(self):
        client = OasisAPIClient('http://localhost:8001')
        client.delete_resource = Mock()

        client.delete_exposure('foo')

        client.delete_resource.assert_called_once_with('/exposure/foo')


class DownloadResource(TestCase):
    def test_local_file_already_exists___exception_is_raised_and_file_is_unchanged(self):
        client = OasisAPIClient('http://localhost:8001')

        with NamedTemporaryFile('w+') as f:
            f.write('foobarboo')
            f.flush()

            with self.assertRaises(OasisException):
                client.download_resource('foo', f.name)

            f.seek(0)
            self.assertEqual('foobarboo', f.read())

    def test_response_is_not_ok___exception_is_raised_and_file_is_not_created(self):
        client = OasisAPIClient('http://localhost:8001')

        with NamedTemporaryFile('w') as f:
            local_filename = f.name

        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET, 'http://localhost:8001/foo', status=400)

            with self.assertRaises(OasisException):
                client.download_resource('foo', local_filename)

            self.assertFalse(os.path.exists(local_filename))

    def test_response_is_ok___file_is_created_with_correct_content(self):
        client = OasisAPIClient('http://localhost:8001')
        client.DOWNLOAD_CHUCK_SIZE_IN_BYTES = 10

        expected_content = ''.join(choice(string.ascii_letters) for i in range(2 * client.DOWNLOAD_CHUCK_SIZE_IN_BYTES)).encode()

        with NamedTemporaryFile('w') as f:
            local_filename = f.name

        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET, 'http://localhost:8001/foo', status=200, body=expected_content)

            client.download_resource('foo', local_filename)

            self.assertTrue(os.path.exists(local_filename))

            with io.open(local_filename, 'rb') as f:
                dld_content = f.read()

            os.remove(local_filename)
            self.assertEqual(expected_content, dld_content)


class DownloadOutputs(TestCase):
    def test_download_resource_is_called_with_the_correct_parameters(self):
        client = OasisAPIClient('http://localhost:8001')
        client.download_resource = Mock()

        client.download_outputs('foo', 'local_filename')

        client.download_resource.assert_called_once_with('/outputs/foo', 'local_filename')


class DownloadExposures(TestCase):
    def test_download_resource_is_called_with_the_correct_parameters(self):
        client = OasisAPIClient('http://localhost:8001')
        client.download_resource = Mock()

        client.download_exposure('foo', 'local_filename')

        client.download_resource.assert_called_once_with('/exposure/foo', 'local_filename')


def fake_build_tar_fn(d):
    with io.open(os.path.join(d, TAR_FILE), 'wb') as f:
        f.write(''.join(choice(string.ascii_letters) for i in range(100)).encode())


class UploadInputsFromDirectory(TestCase):
    @patch('oasislmf.api_client.client.create_binary_tar_file')
    @patch('oasislmf.api_client.client.create_binary_files')
    @patch('oasislmf.api_client.client.check_conversion_tools')
    @patch('oasislmf.api_client.client.check_inputs_directory')
    def test_do_build_is_false___bin_building_functions_are_not_called(self, check_mock, check_tools_mock, create_bin_mock, create_tar_mock):
        with TemporaryDirectory() as d, responses.RequestsMock() as rsps:
            Path(os.path.join(d, TAR_FILE)).touch()
            rsps.add(responses.POST, url='http://localhost:8001/exposure', body=json.dumps({'exposures': [{'location': 'exposure_location'}]}).encode())

            client = OasisAPIClient('http://localhost:8001')
            client.upload_inputs_from_directory(d, do_build=False)

            check_mock.assert_not_called()
            check_tools_mock.assert_not_called()
            create_bin_mock.assert_not_called()
            create_tar_mock.assert_not_called()

    @patch('oasislmf.api_client.client.create_binary_tar_file')
    @patch('oasislmf.api_client.client.create_binary_files')
    @patch('oasislmf.api_client.client.check_conversion_tools')
    @patch('oasislmf.api_client.client.check_inputs_directory')
    def test_do_build_is_true_do_il_is_false___bin_building_functions_are_called_with_correct_args(self, check_mock, check_tools_mock, create_bin_mock, create_tar_mock):
        create_tar_mock.side_effect = fake_build_tar_fn

        with TemporaryDirectory() as d, responses.RequestsMock() as rsps:
            rsps.add(responses.POST, url='http://localhost:8001/exposure', body=json.dumps({'exposures': [{'location': 'exposure_location'}]}).encode())

            client = OasisAPIClient('http://localhost:8001')
            client.upload_inputs_from_directory(d, do_build=True)

            check_mock.assert_called_once_with(d, do_il=False)
            check_tools_mock.assert_called_once_with(do_il=False)
            create_bin_mock.assert_called_once_with(d, d, do_il=False)
            create_tar_mock.assert_called_once_with(d)

    @patch('oasislmf.api_client.client.create_binary_tar_file')
    @patch('oasislmf.api_client.client.create_binary_files')
    @patch('oasislmf.api_client.client.check_conversion_tools')
    @patch('oasislmf.api_client.client.check_inputs_directory')
    def test_do_build_is_true_do_il_is_true___bin_building_functions_are_called_with_correct_args(self, check_mock, check_tools_mock, create_bin_mock, create_tar_mock):
        create_tar_mock.side_effect = fake_build_tar_fn

        with TemporaryDirectory() as d, responses.RequestsMock() as rsps:
            rsps.add(responses.POST, url='http://localhost:8001/exposure', body=json.dumps({'exposures': [{'location': 'exposure_location'}]}).encode())

            client = OasisAPIClient('http://localhost:8001')
            client.upload_inputs_from_directory(d, do_build=True, do_il=True)

            check_mock.assert_called_once_with(d, do_il=True)
            check_tools_mock.assert_called_once_with(do_il=True)
            create_bin_mock.assert_called_once_with(d, d, do_il=True)
            create_tar_mock.assert_called_once_with(d)

    @patch('oasislmf.api_client.client.create_binary_tar_file')
    @patch('oasislmf.api_client.client.create_binary_files')
    @patch('oasislmf.api_client.client.check_conversion_tools')
    @patch('oasislmf.api_client.client.check_inputs_directory')
    def test_do_build_is_truebin_dir_is_supplied___bin_building_functions_are_called_with_correct_args(self, check_mock, check_tools_mock, create_bin_mock, create_tar_mock):
        create_tar_mock.side_effect = fake_build_tar_fn

        with TemporaryDirectory() as d, responses.RequestsMock() as rsps:
            bin_dir = os.path.join(d, 'alt_bin_dir')
            os.mkdir(bin_dir)

            rsps.add(responses.POST, url='http://localhost:8001/exposure', body=json.dumps({'exposures': [{'location': 'exposure_location'}]}).encode())

            client = OasisAPIClient('http://localhost:8001')
            client.upload_inputs_from_directory(d, bin_directory=bin_dir, do_build=True)

            check_mock.assert_called_once_with(d, do_il=False)
            check_tools_mock.assert_called_once_with(do_il=False)
            create_bin_mock.assert_called_once_with(d, bin_dir, do_il=False)
            create_tar_mock.assert_called_once_with(bin_dir)

    def test_tar_file_exists___correct_file_is_posted(self):
        with TemporaryDirectory() as d, responses.RequestsMock() as rsps:
            Path(os.path.join(d, TAR_FILE)).touch()

            rsps.add(responses.POST, url='http://localhost:8001/exposure', body=json.dumps({'exposures': [{'location': 'exposure_location'}]}).encode())

            client = OasisAPIClient('http://localhost:8001')
            client.upload_inputs_from_directory(d)

            self.assertEqual(1, len(rsps.calls))

            request = rsps.calls[0].request
            self.assertEqual(request.url, 'http://localhost:8001/exposure')

            multipart_data = request.body.fields['file']
            self.assertEqual('inputs.tar.gz', multipart_data[0])
            self.assertEqual(os.path.join(d, TAR_FILE), multipart_data[1].name)
            self.assertEqual('text/plain', multipart_data[2])

    def test_tar_file_exists_in_specified_bin_dir___correct_file_is_posted(self):
        with TemporaryDirectory() as d, responses.RequestsMock() as rsps:
            bin_dir = os.path.join(d, 'alt_bin_dir')
            os.mkdir(bin_dir)
            Path(os.path.join(bin_dir, TAR_FILE)).touch()

            rsps.add(responses.POST, url='http://localhost:8001/exposure', body=json.dumps({'exposures': [{'location': 'exposure_location'}]}).encode())

            client = OasisAPIClient('http://localhost:8001')
            client.upload_inputs_from_directory(d, bin_directory=bin_dir)

            self.assertEqual(1, len(rsps.calls))

            request = rsps.calls[0].request
            self.assertEqual(request.url, 'http://localhost:8001/exposure')

            multipart_data = request.body.fields['file']
            self.assertEqual('inputs.tar.gz', multipart_data[0])
            self.assertEqual(os.path.join(bin_dir, TAR_FILE), multipart_data[1].name)
            self.assertEqual('text/plain', multipart_data[2])

    def test_tar_file_exists___exposure_location_from_result_is_returned(self):
        with TemporaryDirectory() as d, responses.RequestsMock() as rsps:
            Path(os.path.join(d, TAR_FILE)).touch()

            rsps.add(responses.POST, url='http://localhost:8001/exposure', body=json.dumps({'exposures': [{'location': 'exposure_location'}]}).encode())

            client = OasisAPIClient('http://localhost:8001')
            result = client.upload_inputs_from_directory(d)

            self.assertEqual('exposure_location', result)

    def test_response_from_server_is_not_ok___oasis_error_is_raised(self):
        with TemporaryDirectory() as d, responses.RequestsMock() as rsps:
            Path(os.path.join(d, TAR_FILE)).touch()

            rsps.add(responses.POST, status=400, url='http://localhost:8001/exposure', body=json.dumps({'exposures': [{'location': 'exposure_location'}]}).encode())

            client = OasisAPIClient('http://localhost:8001')

            with self.assertRaises(OasisException):
                client.upload_inputs_from_directory(d)

    @patch('oasislmf.api_client.client.cleanup_bin_directory')
    def test_do_clean_is_false___clean_bin_directory_is_not_called(self, clean_mock):
        with TemporaryDirectory() as d, responses.RequestsMock() as rsps:
            Path(os.path.join(d, TAR_FILE)).touch()

            rsps.add(responses.POST, url='http://localhost:8001/exposure', body=json.dumps({'exposures': [{'location': 'exposure_location'}]}).encode())

            client = OasisAPIClient('http://localhost:8001')
            client.upload_inputs_from_directory(d)

            clean_mock.assert_not_called()

    @patch('oasislmf.api_client.client.cleanup_bin_directory')
    def test_do_clean_is_true___clean_bin_directory_is_called_on_the_correct_directory(self, clean_mock):
        with TemporaryDirectory() as d, responses.RequestsMock() as rsps:
            Path(os.path.join(d, TAR_FILE)).touch()

            rsps.add(responses.POST, url='http://localhost:8001/exposure', body=json.dumps({'exposures': [{'location': 'exposure_location'}]}).encode())

            client = OasisAPIClient('http://localhost:8001')
            client.upload_inputs_from_directory(d, do_clean=True)

            clean_mock.assert_called_once_with(d)

    @patch('oasislmf.api_client.client.cleanup_bin_directory')
    def test_do_clean_is_true_and_bin_dir_is_specified___clean_bin_directory_is_called_on_the_correct_directory(self, clean_mock):
        with TemporaryDirectory() as d, responses.RequestsMock() as rsps:
            bin_dir = os.path.join(d, 'alt_bin_dir')
            os.mkdir(bin_dir)
            Path(os.path.join(bin_dir, TAR_FILE)).touch()

            rsps.add(responses.POST, url='http://localhost:8001/exposure', body=json.dumps({'exposures': [{'location': 'exposure_location'}]}).encode())

            client = OasisAPIClient('http://localhost:8001')
            client.upload_inputs_from_directory(d, bin_directory=bin_dir, do_clean=True)

            clean_mock.assert_called_once_with(bin_dir)
