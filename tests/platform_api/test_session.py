import unittest
from unittest.mock import Mock, patch, ANY
from requests.exceptions import HTTPError, ConnectionError, ReadTimeout
from requests_toolbelt import MultipartEncoder
from requests import Session
from pathlib import Path

from oasislmf.platform_api.session import APISession, OasisException


class APISessionTests(unittest.TestCase):
    def setUp(self):
        self.api_url = "http://example.com/api/"
        self.username = "testuser"
        self.password = "testpass"
        self.timeout = 0.1
        self.retries = 5
        self.retry_delay = 0.1
        self.request_interval = 0.02
        self.logger = Mock()
        self.access_token = 'eFppNWAuP4ZL2nVrYvLD90jsjd5XV8H17CyBLPg1'
        self.refresh_token = 'QFu35v4lEihQahcWVzwJcy7ehjtABuo2b67jkmwI'

        # Mock connection
        response_mock = Mock()
        response_mock.json.return_value = {
            'access_token': self.access_token,
            'refresh_token': self.refresh_token
        }
        response_mock.raise_for_status.return_value = None
        post_mock = Mock(return_value=response_mock)

        response_healthcheck = Mock()
        response_healthcheck.son.return_value = {'status': 'OK'}
        response_healthcheck.raise_for_status.return_value = None
        mock_healthcheck = Mock(return_value=response_healthcheck)

        with patch.object(Session, 'post', post_mock), \
                patch.object(APISession, 'health_check', mock_healthcheck):
            self.session = APISession(
                self.api_url,
                auth_type="simple",
                username=self.username,
                password=self.password,
                timeout=self.timeout,
                retries=self.retries,
                retry_delay=self.retry_delay,
                request_interval=self.request_interval,
                logger=self.logger
            )

    def test_access_token__success(self):
        reissued_access_token = 'NEW_ACCESS_TOKEN'
        reissued_refresh_token = 'NEW_REFRESH_TOKEN'

        response_mock = Mock()
        response_mock.json.return_value = {
            'access_token': reissued_access_token,
            'refresh_token': reissued_refresh_token
        }
        response_mock.raise_for_status.return_value = None

        post_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'post', post_mock):
            self.session._APISession__get_access_token()

        post_mock.assert_called_once_with(
            'http://example.com/api/access_token/',
            json={"username": self.username, "password": self.password},
            timeout=self.timeout
        )
        self.assertEqual(self.session.tkn_access, reissued_access_token)
        self.assertEqual(self.session.tkn_refresh, reissued_refresh_token)
        self.assertEqual(self.session.headers['authorization'], f'Bearer {reissued_access_token}')

    def test_access_token__error(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=500))

        post_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'post', post_mock):
            with self.assertRaises(OasisException):
                self.session._APISession__get_access_token()

        post_mock.assert_called_once_with(
            'http://example.com/api/access_token/',
            json={"username": self.username, "password": self.password}
        )

    def test_refresh_token__success(self):
        response_mock = Mock()
        response_mock.json.return_value = {
            'access_token': 'new_access_token',
            'refresh_token': 'new_refresh_token'
        }
        response_mock.raise_for_status.return_value = None

        post_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'post', post_mock):
            self.session._refresh_token()

        post_mock.assert_called_once_with(
            'http://example.com/api/refresh_token/',
            timeout=self.timeout
        )
        self.assertEqual(self.session.tkn_access, 'new_access_token')
        self.assertEqual(self.session.tkn_refresh, 'new_refresh_token')
        self.assertEqual(self.session.headers['authorization'], 'Bearer new_access_token')

    def test_refresh_token__error(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=500))

        post_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'post', post_mock):
            with self.assertRaises(OasisException):
                self.session._refresh_token()

        post_mock.assert_called_with(
            'http://example.com/api/refresh_token/',
            timeout=self.timeout
        )

    def test_health_check__success(self):
        response_mock = Mock()
        response_mock.raise_for_status.return_value = None
        get_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'get', get_mock):
            self.session.health_check()

        get_mock.assert_called_once_with('http://example.com/api/healthcheck/')

    def test_health_check__error(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=500))
        get_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'get', get_mock):
            with self.assertRaises(OasisException):
                self.session.health_check()

        get_mock.assert_called_once_with('http://example.com/api/healthcheck/')

    def test_get_with_token_timeout__recovered(self):
        # call failed
        timeout_response_mock = Mock()
        timeout_response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=401))

        # Call ok
        success_response_mock = Mock()
        success_response_mock.raise_for_status.return_value = None
        get_mock = Mock(side_effect=[
            timeout_response_mock,
            timeout_response_mock,
            success_response_mock,
        ])

        # Get new token
        refresh_token_mock = Mock()
        self.session.tkn_refresh = 'recovered_refresh_token'

        with patch.object(Session, 'get', get_mock), \
                patch.object(APISession, '_refresh_token', refresh_token_mock):
            self.session.get('http://example.com/api/resource/')

        get_mock.assert_called_with(
            'http://example.com/api/resource/',
            timeout=self.timeout
        )
        assert get_mock.call_count == 3
        assert refresh_token_mock.call_count == 2
        self.assertEqual(self.session.tkn_refresh, 'recovered_refresh_token')

    def test_get_with_token_timeout__not_recovered(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=401))

        get_mock = Mock(return_value=response_mock)
        refresh_token_mock = Mock()

        self.session.tkn_refresh = 'refresh_token'

        with patch.object(Session, 'get', get_mock), \
                patch.object(APISession, '_refresh_token', refresh_token_mock):
            with self.assertRaises(OasisException):
                self.session.get('http://example.com/api/resource/')

        get_mock.assert_called_with(
            'http://example.com/api/resource/',
            timeout=self.timeout
        )
        self.assertEqual(get_mock.call_count, self.retries)

    def test_unrecoverable_error(self):
        response_mock = Mock()
        response_mock.status_code = 500
        response_mock.url = 'http://example.com/api/resource/'
        response_mock.text = 'Internal Server Error'

        logger_msg = 'msg_for_unrecoverable_error'
        error_mock = Mock(response=response_mock)
        error_msg = f'api error: 500, url: http://example.com/api/resource/, msg: Internal Server Error, {error_mock.__class__.__name__}: {error_mock}'

        with self.assertRaises(OasisException) as context:
            self.session.unrecoverable_error(error_mock, msg=logger_msg)

        exception = context.exception
        self.assertEqual(exception.__str__(), error_msg)
        self.assertEqual(exception.original_exception, error_mock)

    def test_upload__success(self):
        response_mock = Mock()
        response_mock.raise_for_status.return_value = None
        post_mock = Mock(return_value=response_mock)

        base_path = Path(__file__).parents[2]
        loc_filepath = Path.joinpath(base_path, 'validation', 'insurance', 'location.csv')
        content_type = 'text/csv'

        m = MultipartEncoder(fields={'file': (loc_filepath.name, loc_filepath.open(mode='rb'), content_type)})
        with patch.object(Session, 'post', post_mock):
            self.session.upload(
                'http://example.com/api/upload/',
                str(loc_filepath),
                content_type
            )

        post_mock.assert_called_once_with(
            'http://example.com/api/upload/',
            data=ANY,
            headers={'Content-Type': ANY},
            timeout=self.timeout
        )

        data_found = post_mock.call_args.kwargs['data']
        filename_found = data_found.fields['file'][0]
        filepath_found = data_found.fields['file'][1].name
        content_found = data_found.fields['file'][2]

        self.assertEqual(filename_found, loc_filepath.name)
        self.assertEqual(filepath_found, str(loc_filepath))
        self.assertEqual(content_found, content_type)
        self.assertTrue(isinstance(data_found, MultipartEncoder))

    def test_upload__error_with_retry(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = ConnectionError()
        post_mock = Mock(return_value=response_mock)

        base_path = Path(__file__).parents[2]
        loc_filepath = Path.joinpath(base_path, 'validation', 'insurance', 'location.csv')
        content_type = 'text/csv'

        with patch.object(Session, 'post', post_mock):
            with self.assertRaises(OasisException):
                self.session.upload(
                    'http://example.com/api/upload/',
                    str(loc_filepath),
                    content_type
                )

        post_mock.assert_called_with(
            'http://example.com/api/upload/',
            data=ANY,
            headers={'Content-Type': ANY},
            timeout=self.timeout
        )
        self.assertEqual(post_mock.call_count, self.retries)

    def test_upload__error_unrecoverable(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=404))
        post_mock = Mock(return_value=response_mock)

        base_path = Path(__file__).parents[2]
        loc_filepath = Path.joinpath(base_path, 'validation', 'insurance', 'location.csv')
        content_type = 'text/csv'

        with patch.object(Session, 'post', post_mock):
            with self.assertRaises(HTTPError):
                self.session.upload(
                    'http://example.com/api/upload/',
                    str(loc_filepath),
                    content_type
                )

        post_mock.assert_called_with(
            'http://example.com/api/upload/',
            data=ANY,
            headers={'Content-Type': ANY},
            timeout=self.timeout
        )
        self.assertEqual(post_mock.call_count, 1)

    def test_get__success(self):
        response_mock = Mock()
        response_mock.raise_for_status.return_value = None
        get_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'get', get_mock):
            self.session.get('http://example.com/api/resource/')

        get_mock.assert_called_once_with(
            'http://example.com/api/resource/',
            timeout=self.timeout
        )

    def test_get__error_with_retry(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = ConnectionError()

        get_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'get', get_mock):
            with self.assertRaises(OasisException):
                self.session.get('http://example.com/api/resource/')

        get_mock.assert_called_with(
            'http://example.com/api/resource/',
            timeout=self.timeout
        )
        self.assertEqual(get_mock.call_count, self.retries)

    def test_get__error_unrecoverable(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=404))

        get_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'get', get_mock):
            with self.assertRaises(HTTPError):
                self.session.get('http://example.com/api/resource/')

        get_mock.assert_called_with(
            'http://example.com/api/resource/',
            timeout=self.timeout
        )
        self.assertEqual(get_mock.call_count, 1)

    def test_post__success(self):
        response_mock = Mock()
        response_mock.raise_for_status.return_value = None

        post_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'post', post_mock):
            self.session.post('http://example.com/api/resource/', json={"data": "value"})

        post_mock.assert_called_once_with(
            'http://example.com/api/resource/',
            json={"data": "value"},
            timeout=self.timeout)

    def test_post__error_with_retry(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = ReadTimeout()
        post_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'post', post_mock):
            with self.assertRaises(OasisException):
                self.session.post('http://example.com/api/resource/', json={"data": "value"})

        post_mock.assert_called_with(
            'http://example.com/api/resource/',
            json={"data": "value"},
            timeout=self.timeout)
        self.assertEqual(post_mock.call_count, self.retries)

    def test_post__error_unrecoverable(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=404))
        post_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'post', post_mock):
            with self.assertRaises(HTTPError):
                self.session.post('http://example.com/api/resource/1/', json={"data": "value"})

        post_mock.assert_called_with(
            'http://example.com/api/resource/1/',
            json={"data": "value"},
            timeout=self.timeout)
        self.assertEqual(post_mock.call_count, 1)

    def test_delete__success(self):
        response_mock = Mock()
        response_mock.raise_for_status.return_value = None

        delete_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'delete', delete_mock):
            self.session.delete('http://example.com/api/resource/1/')

        delete_mock.assert_called_once_with(
            'http://example.com/api/resource/1/',
            timeout=self.timeout
        )

    def test_delete__error_with_retry(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=503))

        delete_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'delete', delete_mock):
            with self.assertRaises(OasisException):
                self.session.delete('http://example.com/api/resource/1/')

        delete_mock.assert_called_with(
            'http://example.com/api/resource/1/',
            timeout=self.timeout
        )
        self.assertEqual(delete_mock.call_count, self.retries)

    def test_delete__error_unrecoverable(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=404))

        delete_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'delete', delete_mock):
            with self.assertRaises(HTTPError):
                self.session.delete('http://example.com/api/resource/1/')

        delete_mock.assert_called_once_with(
            'http://example.com/api/resource/1/',
            timeout=self.timeout
        )

    def test_put__success(self):
        response_mock = Mock()
        response_mock.raise_for_status.return_value = None

        put_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'put', put_mock):
            self.session.put('http://example.com/api/resource/1/', json={"data": "value"})

        put_mock.assert_called_once_with(
            'http://example.com/api/resource/1/',
            json={"data": "value"},
            timeout=self.timeout
        )

    def test_put__error_with_retry(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = ConnectionError()

        put_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'put', put_mock):
            with self.assertRaises(OasisException):
                self.session.put('http://example.com/api/resource/1/', json={"data": "value"})

        put_mock.assert_called_with(
            'http://example.com/api/resource/1/',
            json={"data": "value"},
            timeout=self.timeout
        )
        self.assertEqual(put_mock.call_count, self.retries)

    def test_put__error_unrecoverable(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=404))

        put_mock = Mock(return_value=response_mock)

        with patch.object(Session, 'put', put_mock):
            with self.assertRaises(HTTPError):
                self.session.put('http://example.com/api/resource/1/', json={"data": "value"})

        put_mock.assert_called_once_with(
            'http://example.com/api/resource/1/',
            json={"data": "value"},
            timeout=self.timeout
        )
        self.assertEqual(put_mock.call_count, 1)
