from unittest import TestCase

import responses
from hypothesis import given
from hypothesis.strategies import integers
from mock import patch, Mock
from requests import RequestException

from oasislmf.api_client.client import OasisAPIClient


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
