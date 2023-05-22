import unittest
from unittest.mock import Mock, patch
from requests.exceptions import HTTPError, ConnectionError, ReadTimeout
from requests_toolbelt import MultipartEncoder

from your_module import APISession, OasisException


class APISessionTests(unittest.TestCase):
    def setUp(self):
        self.api_url = "http://example.com/api/"
        self.username = "testuser"
        self.password = "testpass"
        self.timeout = 25
        self.retries = 5
        self.retry_delay = 1
        self.request_interval = 0.02
        self.logger = Mock()
        self.session = APISession(
            self.api_url,
            self.username,
            self.password,
            timeout=self.timeout,
            retries=self.retries,
            retry_delay=self.retry_delay,
            request_interval=self.request_interval,
            logger=self.logger
        )

    def test_get_access_token_success(self):
        response_mock = Mock()
        response_mock.json.return_value = {
            'access_token': 'access_token',
            'refresh_token': 'refresh_token'
        }
        response_mock.raise_for_status.return_value = None

        post_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'post', post_mock):
            self.session._APISession__get_access_token(self.username, self.password)

        post_mock.assert_called_once_with(
            'http://example.com/api/access_token/',
            json={"username": self.username, "password": self.password}
        )
        self.assertEqual(self.session.tkn_access, 'access_token')
        self.assertEqual(self.session.tkn_refresh, 'refresh_token')
        self.assertEqual(self.session.headers['authorization'], 'Bearer access_token')

    def test_get_access_token_error(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=500))

        post_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'post', post_mock):
            with self.assertRaises(OasisException):
                self.session._APISession__get_access_token(self.username, self.password)

        post_mock.assert_called_once_with(
            'http://example.com/api/access_token/',
            json={"username": self.username, "password": self.password}
        )

    def test_refresh_token_success(self):
        response_mock = Mock()
        response_mock.json.return_value = {
            'access_token': 'new_access_token',
            'refresh_token': 'new_refresh_token'
        }
        response_mock.raise_for_status.return_value = None

        post_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'post', post_mock):
            self.session._refresh_token()

        post_mock.assert_called_once_with(
            'http://example.com/api/refresh_token/',
            timeout=self.timeout
        )
        self.assertEqual(self.session.tkn_access, 'new_access_token')
        self.assertEqual(self.session.tkn_refresh, 'new_refresh_token')
        self.assertEqual(self.session.headers['authorization'], 'Bearer new_access_token')

    def test_refresh_token_error(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=500))

        post_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'post', post_mock):
            with self.assertRaises(OasisException):
                self.session._refresh_token()

        post_mock.assert_called_once_with(
            'http://example.com/api/refresh_token/',
            timeout=self.timeout
        )

    def test_health_check_success(self):
        response_mock = Mock()
        response_mock.raise_for_status.return_value = None

        get_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'get', get_mock):
            self.session.health_check()

        get_mock.assert_called_once_with('http://example.com/api/healthcheck/')

    def test_health_check_error(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=500))

        get_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'get', get_mock):
            with self.assertRaises(OasisException):
                self.session.health_check()

        get_mock.assert_called_once_with('http://example.com/api/healthcheck/')

    def test_upload_success(self):
        response_mock = Mock()
        response_mock.raise_for_status.return_value = None

        post_mock = Mock(return_value=response_mock)
        open_mock = Mock()
        multipart_encoder_mock = Mock(content_type='multipart/form-data')

        with patch.object(APISession, 'post', post_mock), \
                patch('builtins.open', open_mock), \
                patch.object(MultipartEncoder, 'fields', {'file': ('filename.txt', b'filedata', 'text/plain')}), \
                patch.object(MultipartEncoder, 'content_type', multipart_encoder_mock.content_type):
            self.session.upload(
                'http://example.com/api/upload/',
                'path/to/file',
                'text/plain'
            )

        post_mock.assert_called_once_with(
            'http://example.com/api/upload/',
            data=multipart_encoder_mock,
            headers={'Content-Type': multipart_encoder_mock.content_type},
            timeout=self.timeout
        )
        open_mock.assert_called_once_with('path/to/file', 'rb')

    def test_upload_error(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = ConnectionError()

        post_mock = Mock(return_value=response_mock)
        open_mock = Mock()
        multipart_encoder_mock = Mock(content_type='multipart/form-data')

        with patch.object(APISession, 'post', post_mock), \
                patch('builtins.open', open_mock), \
                patch.object(MultipartEncoder, 'fields', {'file': ('filename.txt', b'filedata', 'text/plain')}), \
                patch.object(MultipartEncoder, 'content_type', multipart_encoder_mock.content_type):
            with self.assertRaises(ConnectionError):
                self.session.upload(
                    'http://example.com/api/upload/',
                    'path/to/file',
                    'text/plain'
                )

        post_mock.assert_called_once_with(
            'http://example.com/api/upload/',
            data=multipart_encoder_mock,
            headers={'Content-Type': multipart_encoder_mock.content_type},
            timeout=self.timeout
        )
        open_mock.assert_called_once_with('path/to/file', 'rb')

    # Add more unit tests for other methods as needed

    def test_get_with_token_timeout(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=401))

        get_mock = Mock(return_value=response_mock)
        refresh_token_mock = Mock()

        self.session.tkn_refresh = 'refresh_token'

        with patch.object(APISession, 'get', get_mock), \
                patch.object(APISession, '_refresh_token', refresh_token_mock):
            with self.assertRaises(OasisException):
                self.session.get('http://example.com/api/resource/')

        get_mock.assert_called_once_with(
            'http://example.com/api/resource/',
            timeout=self.timeout
        )
        refresh_token_mock.assert_called_once()


    def test_unrecoverable_error(self):
        response_mock = Mock()
        response_mock.status_code = 500
        response_mock.url = 'http://example.com/api/resource/'
        response_mock.text = 'Internal Server Error'

        error_mock = Mock(response=response_mock)
        error_msg = 'api error: 500, url: http://example.com/api/resource/, msg: Internal Server Error'

        with self.assertRaises(OasisException) as context:
            self.session.unrecoverable_error(error_mock)

        exception = context.exception
        self.assertEqual(exception.message, error_msg)
        self.assertEqual(exception.original_exception, error_mock)


 def test_get_success(self):
        response_mock = Mock()
        response_mock.raise_for_status.return_value = None

        get_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'get', get_mock):
            self.session.get('http://example.com/api/resource/')

        get_mock.assert_called_once_with(
            'http://example.com/api/resource/',
            timeout=self.timeout
        )

    def test_get_error(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = ConnectionError()

        get_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'get', get_mock):
            with self.assertRaises(ConnectionError):
                self.session.get('http://example.com/api/resource/')

        get_mock.assert_called_once_with(
            'http://example.com/api/resource/',
            timeout=self.timeout
        )

    def test_post_success(self):
        response_mock = Mock()
        response_mock.raise_for_status.return_value = None

        post_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'post', post_mock):
            self.session.post('http://example.com/api/resource/', json={"data": "value"})

        post_mock.assert_called_once_with(
            'http://example.com/api/resource/',
            json={"data": "value"},
            timeout=self.timeout
        )

    def test_post_error(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = ReadTimeout()

        post_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'post', post_mock):
            with self.assertRaises(ReadTimeout):
                self.session.post('http://example.com/api/resource/', json={"data": "value"})

        post_mock.assert_called_once_with(
            'http://example.com/api/resource/',
            json={"data": "value"},
            timeout=self.timeout
        )

    def test_delete_success(self):
        response_mock = Mock()
        response_mock.raise_for_status.return_value = None

        delete_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'delete', delete_mock):
            self.session.delete('http://example.com/api/resource/1/')

        delete_mock.assert_called_once_with(
            'http://example.com/api/resource/1/',
            timeout=self.timeout
        )

    def test_delete_error(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=403))

        delete_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'delete', delete_mock):
            with self.assertRaises(OasisException):
                self.session.delete('http://example.com/api/resource/1/')

        delete_mock.assert_called_once_with(
            'http://example.com/api/resource/1/',
            timeout=self.timeout
        )

    def test_put_success(self):
        response_mock = Mock()
        response_mock.raise_for_status.return_value = None

        put_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'put', put_mock):
            self.session.put('http://example.com/api/resource/1/', json={"data": "value"})

        put_mock.assert_called_once_with(
            'http://example.com/api/resource/1/',
            json={"data": "value"},
            timeout=self.timeout
        )

    def test_put_error(self):
        response_mock = Mock()
        response_mock.raise_for_status.side_effect = HTTPError(response=Mock(status_code=404))

        put_mock = Mock(return_value=response_mock)

        with patch.object(APISession, 'put', put_mock):
            with self.assertRaises(OasisException):
                self.session.put('http://example.com/api/resource/1/', json={"data": "value"})

        put_mock.assert_called_once_with(
            'http://example.com/api/resource/1/',
            json={"data": "value"},
            timeout=self.timeout
        )


if __name__ == '__main__':
    unittest.main()

