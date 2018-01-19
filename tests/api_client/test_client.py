import json
from itertools import chain
from pathlib2 import Path
from backports.tempfile import TemporaryDirectory
from unittest import TestCase

import os
import responses
from hypothesis import given
from hypothesis.strategies import integers
from mock import patch, Mock
from requests import RequestException

from oasislmf.api_client.client import OasisAPIClient
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


class CheckInputDirectory(TestCase):
    def test_tar_file_already_exists___exception_is_raised(self):
        with TemporaryDirectory() as d:
            client = OasisAPIClient('http://localhost:8001')

            Path(os.path.join(d, client.TAR_FILE)).touch()
            with self.assertRaises(OasisException):
                client.check_inputs_directory(d, False)

    def test_do_il_is_false_non_il_input_files_are_missing__exception_is_raised(self):
        with TemporaryDirectory() as d:
            client = OasisAPIClient('http://localhost:8001')

            for p in client.IL_INPUTS_FILES:
                Path(os.path.join(d, p + '.csv')).touch()

            with self.assertRaises(OasisException):
                client.check_inputs_directory(d, False)

    def test_do_is_is_false_non_il_input_files_are_present___no_exception_is_raised(self):
        with TemporaryDirectory() as d:
            client = OasisAPIClient('http://localhost:8001')

            for p in client.GUL_INPUTS_FILES:
                Path(os.path.join(d, p + '.csv')).touch()

            try:
                client.check_inputs_directory(d, False)
            except Exception as e:
                self.fail('Exception was raised {}: {}'.format(type(e), e))

    def test_do_il_is_true_all_input_files_are_missing__exception_is_raised(self):
        with TemporaryDirectory() as d:
            client = OasisAPIClient('http://localhost:8001')

            with self.assertRaises(OasisException):
                client.check_inputs_directory(d, True)

    def test_do_il_is_true_gul_input_files_are_missing__exception_is_raised(self):
        with TemporaryDirectory() as d:
            client = OasisAPIClient('http://localhost:8001')

            for p in client.IL_INPUTS_FILES:
                Path(os.path.join(d, p + '.csv')).touch()

            with self.assertRaises(OasisException):
                client.check_inputs_directory(d, True)

    def test_do_il_is_true_il_input_files_are_missing__exception_is_raised(self):
        with TemporaryDirectory() as d:
            client = OasisAPIClient('http://localhost:8001')

            for p in client.GUL_INPUTS_FILES:
                Path(os.path.join(d, p + '.csv')).touch()

            with self.assertRaises(OasisException):
                client.check_inputs_directory(d, True)

    def test_do_il_is_true_all_input_files_are_present___no_exception_is_raised(self):
        with TemporaryDirectory() as d:
            client = OasisAPIClient('http://localhost:8001')

            for p in chain(client.GUL_INPUTS_FILES, client.IL_INPUTS_FILES):
                Path(os.path.join(d, p + '.csv')).touch()

            try:
                client.check_inputs_directory(d, True)
            except Exception as e:
                self.fail('Exception was raised {}: {}'.format(type(e), e))

    def test_do_il_is_false_il_bin_files_are_present___no_exception_is_raised(self):
        with TemporaryDirectory() as d:
            client = OasisAPIClient('http://localhost:8001')

            for p in chain(client.GUL_INPUTS_FILES, client.IL_INPUTS_FILES):
                Path(os.path.join(d, p + '.csv')).touch()

            for p in chain(client.IL_INPUTS_FILES):
                Path(os.path.join(d, p + '.bin')).touch()

            try:
                client.check_inputs_directory(d, False)
            except Exception as e:
                self.fail('Exception was raised {}: {}'.format(type(e), e))

    def test_do_il_is_false_gul_bin_files_are_present___exception_is_raised(self):
        with TemporaryDirectory() as d:
            client = OasisAPIClient('http://localhost:8001')

            for p in chain(client.GUL_INPUTS_FILES, client.IL_INPUTS_FILES):
                Path(os.path.join(d, p + '.csv')).touch()

            for p in chain(client.GUL_INPUTS_FILES):
                Path(os.path.join(d, p + '.bin')).touch()

            with self.assertRaises(OasisException):
                client.check_inputs_directory(d, False)

    def test_do_il_is_true_gul_bin_files_are_present___exception_is_raised(self):
        with TemporaryDirectory() as d:
            client = OasisAPIClient('http://localhost:8001')

            for p in chain(client.GUL_INPUTS_FILES, client.IL_INPUTS_FILES):
                Path(os.path.join(d, p + '.csv')).touch()

            for p in chain(client.GUL_INPUTS_FILES):
                Path(os.path.join(d, p + '.bin')).touch()

            with self.assertRaises(OasisException):
                client.check_inputs_directory(d, True)

    def test_do_il_is_true_il_bin_files_are_present___exception_is_raised(self):
        with TemporaryDirectory() as d:
            client = OasisAPIClient('http://localhost:8001')

            for p in chain(client.GUL_INPUTS_FILES, client.IL_INPUTS_FILES):
                Path(os.path.join(d, p + '.csv')).touch()

            for p in chain(client.IL_INPUTS_FILES):
                Path(os.path.join(d, p + '.bin')).touch()

            with self.assertRaises(OasisException):
                client.check_inputs_directory(d, True)

    def test_do_il_is_true_no_bin_files_are_present___no_exception_is_raised(self):
        with TemporaryDirectory() as d:
            client = OasisAPIClient('http://localhost:8001')

            for p in chain(client.GUL_INPUTS_FILES, client.IL_INPUTS_FILES):
                Path(os.path.join(d, p + '.csv')).touch()

            for p in chain(client.IL_INPUTS_FILES):
                Path(os.path.join(d, p + '.bin')).touch()

            try:
                client.check_inputs_directory(d, False)
            except Exception as e:
                self.fail('Exception was raised {}: {}'.format(type(e), e))


class CheckConversionTools(TestCase):
    def test_conversion_tools_are_set_correctly(self):
        client = OasisAPIClient('http://localhost:8001')

        self.assertEqual(
            client.CONVERSION_TOOLS,
            {
                'coverages': 'coveragetobin',
                'events': 'evetobin',
                'fm_policytc': 'fmpolicytctobin',
                'fm_profile': 'fmprofiletobin',
                'fm_programme': 'fmprogrammetobin',
                'fm_xref': 'fmxreftobin',
                'fmsummaryxref': 'fmsummaryxreftobin',
                'gulsummaryxref': 'gulsummaryxreftobin',
                'items': "itemtobin"
            }
        )

    def test_conversion_tools_are_empty___result_is_true(self):
        client = OasisAPIClient('http://localhost:8001')
        client.CONVERSION_TOOLS = {}

        self.assertTrue(client.check_conversion_tools())

    def test_conversion_tools_all_exist___result_is_true(self):
        client = OasisAPIClient('http://localhost:8001')
        client.CONVERSION_TOOLS = {
            'py': 'python'
        }

        self.assertTrue(client.check_conversion_tools())

    def test_some_conversion_tools_are_missing___error_is_raised(self):
        client = OasisAPIClient('http://localhost:8001')
        client.CONVERSION_TOOLS = {
            'py': 'python',
            'missing': 'missing_executable_foo_bar_boo',
        }

        with self.assertRaises(OasisException):
            client.check_conversion_tools()


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
