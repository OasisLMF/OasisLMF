import unittest
from unittest.mock import Mock, patch, ANY, call
from requests.exceptions import HTTPError, ConnectionError, ReadTimeout
from requests.auth import HTTPBasicAuth
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


# ---------------------------------------------------------------------------
# Helpers shared by the auth-mode tests below
# ---------------------------------------------------------------------------

def _ok_response(access='test_access', refresh='test_refresh'):
    r = Mock()
    r.raise_for_status.return_value = None
    r.json.return_value = {'access_token': access, 'refresh_token': refresh}
    return r


def _make_session(auth_type=None, post_response=None, server_auth_type=None, **init_kwargs):
    """
    Construct an APISession with all network calls mocked.
    Returns (session, post_mock, get_mock).
    """
    if post_response is None:
        post_response = _ok_response()

    server_info = Mock()
    server_info.status_code = 200
    server_info.json.return_value = {'config': {'API_AUTH_TYPE': server_auth_type}}

    post_mock = Mock(return_value=post_response)
    get_mock = Mock(return_value=server_info)

    with patch.object(APISession, 'health_check'), \
         patch.object(Session, 'post', post_mock), \
         patch.object(Session, 'get', get_mock):
        session = APISession(
            'http://example.com/api/',
            auth_type=auth_type,
            timeout=0.1,
            retries=3,
            retry_delay=0.1,
            **init_kwargs,
        )
    return session, post_mock, get_mock


# ---------------------------------------------------------------------------
# Init / auth-mode tests
# ---------------------------------------------------------------------------

class TestAPISessionInitDisabled(unittest.TestCase):

    def test_disabled_auth__no_credentials_required(self):
        session, post_mock, _ = _make_session(auth_type='disabled')
        self.assertEqual(session.auth_type, 'disabled')
        self.assertEqual(session.auth_credentials, {})
        self.assertIsNone(session.tkn_access)
        self.assertIsNone(session.tkn_refresh)
        post_mock.assert_not_called()

    def test_disabled_auth__header_unchanged(self):
        session, _, _ = _make_session(auth_type='disabled')
        self.assertEqual(session.headers['authorization'], '')


class TestAPISessionInitToken(unittest.TestCase):

    def setUp(self):
        self.access = 'direct_access_token'
        self.refresh = 'direct_refresh_token'
        self.session, self.post_mock, _ = _make_session(
            auth_type='token',
            access_token=self.access,
            refresh_token=self.refresh,
        )

    def test_token_auth__tokens_set_directly(self):
        self.assertEqual(self.session.tkn_access, self.access)
        self.assertEqual(self.session.tkn_refresh, self.refresh)

    def test_token_auth__authorization_header_set(self):
        self.assertEqual(self.session.headers['authorization'], f'Bearer {self.access}')

    def test_token_auth__no_http_call_for_token(self):
        # Session.post must not be called during __get_access_token for token auth
        self.post_mock.assert_not_called()


class TestAPISessionInitOIDC(unittest.TestCase):

    def setUp(self):
        self.client_id = 'my_client'
        self.client_secret = 'my_secret'
        self.session, self.post_mock, _ = _make_session(
            auth_type='oidc',
            client_id=self.client_id,
            client_secret=self.client_secret,
        )

    def test_oidc_auth__posts_to_access_token_endpoint(self):
        self.post_mock.assert_called_once_with(
            'http://example.com/api/access_token/',
            json={'client_id': self.client_id, 'client_secret': self.client_secret},
            timeout=0.1,
        )

    def test_oidc_auth__access_token_set(self):
        self.assertEqual(self.session.tkn_access, 'test_access')

    def test_oidc_auth__no_refresh_token(self):
        self.assertIsNone(self.session.tkn_refresh)

    def test_oidc_auth__authorization_header_set(self):
        self.assertEqual(self.session.headers['authorization'], 'Bearer test_access')


class TestAPISessionInitM2M(unittest.TestCase):

    def setUp(self):
        self.client_id = 'm2m_client'
        self.client_secret = 'm2m_secret'
        self.token_url = 'http://idp.example.com/token'
        self.session, self.post_mock, _ = _make_session(
            auth_type='m2m',
            client_id=self.client_id,
            client_secret=self.client_secret,
            token_url=self.token_url,
        )

    def test_m2m_auth__posts_to_token_url(self):
        self.post_mock.assert_called_once_with(
            self.token_url,
            data={'grant_type': 'client_credentials'},
            auth=ANY,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=0.1,
        )

    def test_m2m_auth__uses_basic_auth(self):
        auth_used = self.post_mock.call_args.kwargs['auth']
        self.assertIsInstance(auth_used, HTTPBasicAuth)
        self.assertEqual(auth_used.username, self.client_id)
        self.assertEqual(auth_used.password, self.client_secret)

    def test_m2m_auth__access_token_set(self):
        self.assertEqual(self.session.tkn_access, 'test_access')

    def test_m2m_auth__refresh_token_equals_access_token(self):
        self.assertEqual(self.session.tkn_refresh, self.session.tkn_access)

    def test_m2m_auth__authorization_header_set(self):
        self.assertEqual(self.session.headers['authorization'], 'Bearer test_access')

    def test_m2m_auth__with_scope(self):
        _, post_mock, _ = _make_session(
            auth_type='m2m',
            client_id=self.client_id,
            client_secret=self.client_secret,
            token_url=self.token_url,
            scope='openid profile',
        )
        post_mock.assert_called_once_with(
            self.token_url,
            data={'grant_type': 'client_credentials', 'scope': 'openid profile'},
            auth=ANY,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=0.1,
        )


# ---------------------------------------------------------------------------
# Credential validation error tests
# ---------------------------------------------------------------------------

class TestAPISessionInitCredentialErrors(unittest.TestCase):

    def _assert_raises_on_init(self, **init_kwargs):
        with patch.object(APISession, 'health_check'), \
             patch.object(Session, 'get'):
            with self.assertRaises(OasisException):
                APISession('http://example.com/api/', timeout=0.1, **init_kwargs)

    def test_simple_missing_password(self):
        self._assert_raises_on_init(auth_type='simple', username='user')

    def test_simple_missing_username(self):
        self._assert_raises_on_init(auth_type='simple', password='pass')

    def test_simple_no_credentials(self):
        self._assert_raises_on_init(auth_type='simple')

    def test_oidc_missing_client_secret(self):
        self._assert_raises_on_init(auth_type='oidc', client_id='cid')

    def test_oidc_missing_client_id(self):
        self._assert_raises_on_init(auth_type='oidc', client_secret='secret')

    def test_oidc_no_credentials(self):
        self._assert_raises_on_init(auth_type='oidc')

    def test_m2m_missing_token_url(self):
        self._assert_raises_on_init(auth_type='m2m', client_id='cid', client_secret='secret')

    def test_m2m_missing_client_secret(self):
        self._assert_raises_on_init(auth_type='m2m', client_id='cid', token_url='http://idp/token')

    def test_token_missing_refresh_token(self):
        self._assert_raises_on_init(auth_type='token', access_token='tok')

    def test_token_missing_access_token(self):
        self._assert_raises_on_init(auth_type='token', refresh_token='refresh')

    def test_unknown_auth_type(self):
        self._assert_raises_on_init(auth_type='ldap')


# ---------------------------------------------------------------------------
# _fetch_server_auth_type tests
# ---------------------------------------------------------------------------

class TestFetchServerAuthType(unittest.TestCase):

    def setUp(self):
        # Use a simple-auth session as the test subject
        self.session, _, _ = _make_session(
            auth_type='simple',
            username='user',
            password='pass',
        )

    def test_returns_auth_type_from_server_info(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {'config': {'API_AUTH_TYPE': 'oidc'}}

        with patch.object(Session, 'get', return_value=response):
            result = self.session._fetch_server_auth_type()

        self.assertEqual(result, 'oidc')

    def test_returns_none_when_server_returns_non_200(self):
        response = Mock()
        response.status_code = 500

        with patch.object(Session, 'get', return_value=response):
            result = self.session._fetch_server_auth_type()

        self.assertIsNone(result)

    def test_returns_none_on_connection_error(self):
        with patch.object(Session, 'get', side_effect=ConnectionError()):
            result = self.session._fetch_server_auth_type()

        self.assertIsNone(result)

    def test_returns_none_when_config_key_absent(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {}

        with patch.object(Session, 'get', return_value=response):
            result = self.session._fetch_server_auth_type()

        self.assertIsNone(result)


class TestAPISessionServerAuthTypeFallback(unittest.TestCase):

    def test_server_auth_type_used_when_auth_type_not_given(self):
        post_response = _ok_response()
        server_info = Mock()
        server_info.status_code = 200
        server_info.json.return_value = {'config': {'API_AUTH_TYPE': 'simple'}}

        with patch.object(APISession, 'health_check'), \
             patch.object(Session, 'post', return_value=post_response), \
             patch.object(Session, 'get', return_value=server_info):
            session = APISession(
                'http://example.com/api/',
                # auth_type intentionally omitted
                username='user',
                password='pass',
                timeout=0.1,
            )

        self.assertEqual(session.auth_type, 'simple')


# ---------------------------------------------------------------------------
# _refresh_token tests for oidc / m2m
# ---------------------------------------------------------------------------

class TestRefreshTokenOIDCAndM2M(unittest.TestCase):

    def _make_oidc_session(self):
        session, _, _ = _make_session(
            auth_type='oidc',
            client_id='cid',
            client_secret='secret',
        )
        return session

    def _make_m2m_session(self):
        session, _, _ = _make_session(
            auth_type='m2m',
            client_id='cid',
            client_secret='secret',
            token_url='http://idp.example.com/token',
        )
        return session

    def test_refresh_token__oidc_calls_get_access_token(self):
        session = self._make_oidc_session()
        get_access_mock = Mock()

        with patch.object(session, '_APISession__get_access_token', get_access_mock):
            session._refresh_token()

        get_access_mock.assert_called_once()

    def test_refresh_token__m2m_calls_get_access_token(self):
        session = self._make_m2m_session()
        get_access_mock = Mock()

        with patch.object(session, '_APISession__get_access_token', get_access_mock):
            session._refresh_token()

        get_access_mock.assert_called_once()

    def test_refresh_token__oidc_does_not_call_refresh_endpoint(self):
        session = self._make_oidc_session()
        get_access_mock = Mock()

        with patch.object(session, '_APISession__get_access_token', get_access_mock), \
             patch.object(Session, 'post') as post_mock:
            session._refresh_token()

        post_mock.assert_not_called()

    def test_refresh_token__m2m_does_not_call_refresh_endpoint(self):
        session = self._make_m2m_session()
        get_access_mock = Mock()

        with patch.object(session, '_APISession__get_access_token', get_access_mock), \
             patch.object(Session, 'post') as post_mock:
            session._refresh_token()

        post_mock.assert_not_called()


# ---------------------------------------------------------------------------
# __get_access_token tests for oidc / m2m / token
# ---------------------------------------------------------------------------

class TestGetAccessTokenOIDC(unittest.TestCase):

    def setUp(self):
        self.session, _, _ = _make_session(
            auth_type='oidc',
            client_id='cid',
            client_secret='secret',
        )

    def test_get_access_token__posts_client_credentials_to_api(self):
        new_response = _ok_response('new_access', 'ignored_refresh')

        with patch.object(Session, 'post', return_value=new_response) as post_mock:
            self.session._APISession__get_access_token()

        post_mock.assert_called_once_with(
            'http://example.com/api/access_token/',
            json={'client_id': 'cid', 'client_secret': 'secret'},
            timeout=0.1,
        )
        self.assertEqual(self.session.tkn_access, 'new_access')
        self.assertIsNone(self.session.tkn_refresh)

    def test_get_access_token__http_error_raises_oasis_exception(self):
        error_response = Mock()
        error_response.raise_for_status.side_effect = HTTPError(response=Mock(status_code=401))

        with patch.object(Session, 'post', return_value=error_response):
            with self.assertRaises(OasisException):
                self.session._APISession__get_access_token()


class TestGetAccessTokenM2M(unittest.TestCase):

    def setUp(self):
        self.token_url = 'http://idp.example.com/token'
        self.session, _, _ = _make_session(
            auth_type='m2m',
            client_id='m2m_cid',
            client_secret='m2m_secret',
            token_url=self.token_url,
        )

    def test_get_access_token__posts_to_token_url_with_basic_auth(self):
        new_response = _ok_response('new_m2m_access', 'ignored')

        with patch.object(Session, 'post', return_value=new_response) as post_mock:
            self.session._APISession__get_access_token()

        post_mock.assert_called_once_with(
            self.token_url,
            data={'grant_type': 'client_credentials'},
            auth=ANY,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=0.1,
        )
        auth_used = post_mock.call_args.kwargs['auth']
        self.assertIsInstance(auth_used, HTTPBasicAuth)
        self.assertEqual(auth_used.username, 'm2m_cid')
        self.assertEqual(auth_used.password, 'm2m_secret')

    def test_get_access_token__refresh_token_mirrors_access_token(self):
        new_response = _ok_response('fresh_access', 'ignored')

        with patch.object(Session, 'post', return_value=new_response):
            self.session._APISession__get_access_token()

        self.assertEqual(self.session.tkn_access, 'fresh_access')
        self.assertEqual(self.session.tkn_refresh, 'fresh_access')

    def test_get_access_token__http_error_raises_oasis_exception(self):
        error_response = Mock()
        error_response.raise_for_status.side_effect = HTTPError(response=Mock(status_code=401))

        with patch.object(Session, 'post', return_value=error_response):
            with self.assertRaises(OasisException):
                self.session._APISession__get_access_token()


class TestGetAccessTokenDirectToken(unittest.TestCase):

    def test_token_auth__tokens_assigned_without_http_call(self):
        session, post_mock, _ = _make_session(
            auth_type='token',
            access_token='direct_access',
            refresh_token='direct_refresh',
        )
        # post_mock was used during __init__ (healthcheck area) only;
        # the token path must not have added any calls to Session.post
        post_mock.assert_not_called()
        self.assertEqual(session.tkn_access, 'direct_access')
        self.assertEqual(session.tkn_refresh, 'direct_refresh')
        self.assertEqual(session.headers['authorization'], 'Bearer direct_access')
