from requests import codes as status
from requests import Session
from requests.adapters import HTTPAdapter
from requests_toolbelt import MultipartEncoder
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    ReadTimeout,
)

from posixpath import join as urljoin
import time
import os
import logging

from ..utils.exceptions import OasisException


class APISession(Session):
    def __init__(self, api_url, username, password, timeout=25, retries=5, retry_delay=1, request_interval=0.02, logger=None, **kwargs):
        super(APISession, self).__init__(**kwargs)
        self.logger = logger or logging.getLogger()

        # Extended class vars
        self.tkn_access = None
        self.tkn_refresh = None
        self.url_base = urljoin(api_url, '')
        self.timeout = timeout
        self.retry_max = 0
        self.retry_delay = retry_delay
        self.request_interval = request_interval
        self.headers = {
            'authorization': '',
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        self.adapters.clear()
        self.mount(self.url_base, HTTPAdapter(max_retries=self.retry_max))

        # Check connectivity & authentication
        self.health_check()
        self.retry_max = retries
        self.__get_access_token(username, password)

    def __get_access_token(self, username, password):
        try:
            url = urljoin(self.url_base, 'access_token/')
            r = self.post(url, json={"username": username, "password": password})
            r.raise_for_status()
            self.tkn_access = r.json()['access_token']
            self.tkn_refresh = r.json()['refresh_token']
            self.headers['authorization'] = 'Bearer {}'.format(self.tkn_access)
            return r
        except (TypeError, AttributeError, BytesWarning, HTTPError, ConnectionError, ReadTimeout) as e:
            err_msg = 'Authentication Error'
            raise OasisException(err_msg, e)

    def _refresh_token(self):
        try:
            self.headers['authorization'] = 'Bearer {}'.format(self.tkn_refresh)
            url = urljoin(self.url_base, 'refresh_token/')
            r = super(APISession, self).post(url, timeout=self.timeout)
            r.raise_for_status()

            self.tkn_access = r.json()['access_token']
            if 'refresh_token' in r.json():
                self.logger.debug(f"Refreshing access token")
                self.tkn_refresh = r.json()['refresh_token']
            self.headers['authorization'] = 'Bearer {}'.format(self.tkn_access)
            return r
        except (TypeError, AttributeError, BytesWarning, HTTPError, ConnectionError, ReadTimeout) as e:
            err_msg = 'Authentication Error: {}'.format(r.text)
            raise OasisException(err_msg, e)

    def unrecoverable_error(self, error, msg=None):
        err_r = error.response
        err_msg = 'api error: {}, url: {}, msg: {}'.format(err_r.status_code, err_r.url, err_r.text)
        self.logger.error(err_msg)
        if msg:
            self.logger.error(msg)

    # Connection Error Handlers
    def __recoverable(self, error, url, request, counter=1):
        self.logger.debug(f"Connection exception handler: {error}")
        if not (counter < self.retry_max):
            self.logger.debug("Max retries of '{}' reached".format(self.retry_max))
            raise OasisException(f'Failed to recover from error: {error}', error)

        if isinstance(error, (ConnectionError, ReadTimeout)):
            self.logger.debug(f"Recoverable error [{error}] from {request} {url}")
            self.logger.debug(f"Backoff timer: {self.retry_delay * counter}")

            # Reset HTTPAdapter & clear connection pool
            self.adapters.clear()
            self.mount(self.url_base, HTTPAdapter(max_retries=self.retry_max))
            time.sleep(self.retry_delay * counter)
            return True

        elif isinstance(error, HTTPError):
            http_err_code = error.response.status_code
            self.logger.debug(f"Recoverable error [{error}] from {request} {url}")

            if http_err_code in [502, 503, 504]:
                error = "HTTP {}".format(http_err_code)
                return True
            elif http_err_code in [401]:
                if self.tkn_refresh is not None:
                    self.logger.debug(f"requesting refresh token")
                    self._refresh_token()
                    return True
        return False

    # @oasis_log
    def health_check(self):
        """
        Checks the health of the server.

        """
        try:
            url = urljoin(self.url_base, 'healthcheck/')
            return super(APISession, self).get(url)
        except (TypeError, AttributeError, BytesWarning, HTTPError, ConnectionError, ReadTimeout) as e:
            err_msg = 'Health check failed: Unable to connect to {}'.format(self.url_base)
            raise OasisException(err_msg, e)

    def upload(self, url, filepath, content_type, **kwargs):
        counter = 0
        while True:
            counter += 1
            try:
                abs_fp = os.path.realpath(os.path.expanduser(filepath))
                m = MultipartEncoder(fields={'file': (os.path.basename(filepath), open(abs_fp, 'rb'), content_type)})
                r = super(APISession, self).post(url, data=m, headers={'Content-Type': m.content_type}, timeout=self.timeout, **kwargs)
                r.raise_for_status()
                time.sleep(self.request_interval)
            except (HTTPError, ConnectionError, ReadTimeout) as e:
                if self.__recoverable(e, url, 'GET', counter):
                    continue
                else:
                    self.logger.debug(f'Unrecoverable error: {e}')
                    raise e
            return r

    def get(self, url, **kwargs):
        counter = 0
        while True:
            counter += 1
            try:
                r = super(APISession, self).get(url, timeout=self.timeout, **kwargs)
                r.raise_for_status()
                time.sleep(self.request_interval)
            except (HTTPError, ConnectionError, ReadTimeout) as e:
                if self.__recoverable(e, url, 'GET', counter):
                    continue
                else:
                    self.logger.debug(f'Unrecoverable error: {e}')
                    raise e
            return r

    def post(self, url, **kwargs):
        counter = 0
        while True:
            counter += 1
            try:
                r = super(APISession, self).post(url, timeout=self.timeout, **kwargs)
                r.raise_for_status()
                time.sleep(self.request_interval)
            except (HTTPError, ConnectionError, ReadTimeout) as e:
                if self.__recoverable(e, url, 'POST', counter):
                    continue
                else:
                    self.logger.debug(f'Unrecoverable error: {e}')
                    raise e
            return r

    def delete(self, url, **kwargs):
        counter = 0
        while True:
            counter += 1
            try:
                r = super(APISession, self).delete(url, timeout=self.timeout, **kwargs)
                r.raise_for_status()
                time.sleep(self.request_interval)
            except (HTTPError, ConnectionError, ReadTimeout) as e:
                if self.__recoverable(e, url, 'DELETE', counter):
                    continue
                else:
                    self.logger.debug(f'Unrecoverable error: {e}')
                    raise e
            return r

    def put(self, url, **kwargs):
        counter = 0
        while True:
            counter += 1
            try:
                r = super(APISession, self).put(url, timeout=self.timeout, **kwargs)
                r.raise_for_status()
                time.sleep(self.request_interval)
            except (HTTPError, ConnectionError, ReadTimeout) as e:
                if self.__recoverable(e, url, 'OPTIONS', counter):
                    continue
                else:
                    self.logger.debug(f'Unrecoverable error: {e}')
                    raise e
            return r
