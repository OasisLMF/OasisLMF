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
