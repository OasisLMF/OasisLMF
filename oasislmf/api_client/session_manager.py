#!/bin/env python

import requests
from requests import codes as status 
from requests import Session
from requests.exceptions import *

from posixpath import join as urljoin
import time
import logging

from ..utils.exceptions import OasisException

class SessionManager(Session):
    def __init__(self, api_url, username, password, timeout=1, retries=3, retry_delay=1, logger=None, **kwargs):
        super(SessionManager, self).__init__(**kwargs)
        self.logger = logger or logging.getLogger()

        # Extended class vars
        self.tkn_access  = None
        self.tkn_refresh = None
        self.url_base    = urljoin(api_url, '')
        self.timeout     = timeout
        self.retry_max   = retries 
        self.retry_delay = retry_delay

        # Base Session class vars 
        self.headers = {
            'authorization': '',
            'accept': 'application/json',
            'content-type': 'application/json',
        }

        # Check connectivity & authentication
        self.health_check()
        self.__get_access_token(username, password) 


    def __get_access_token(self, username, password):
        url  = urljoin(self.url_base, 'access_token/')
        r = self.post(url, json={"username": username, "password": password})

        if r.ok:
            self.tkn_access  = r.json()['access_token']
            self.tkn_refresh = r.json()['refresh_token']
            self.headers['authorization'] = 'Bearer {}'.format(self.tkn_access)
        else:
            err_msg = 'Login failed: {}'.format(r.json())
            raise OasisException(err_msg)
        return r

    def _refresh_token(self):
        self.headers['authorization'] = 'Bearer {}'.format(self.tkn_refresh)
        url = urljoin(self.url_base, 'refresh_token/')
        r = super(SessionManager, self).post(url, timeout=self.timeout)
        if r.status_code == status.ok:
            self.tkn_access  = r.json()['access_token']
            self.headers['authorization'] = 'Bearer {}'.format(self.tkn_access)
        else:
            err_msg = 'Token refresh error: {}'.format(r.text)
            raise OasisException(err_msg)
        return r



    ## Connection Error Handler
    def __recoverable(self, error, url, request, counter=1):
        if not (counter < self.retry_max):
            self.logger.info("Max retries of '{}' reached".format(self.retry_max))
            return False

        if hasattr(error,'status_code'):
            self.logger.debug("Connection Error Handler: {}".format(error.status_code))
            if error.status_code in [502, 503, 504]:
                error = "HTTP {}".format(error.status_code)
            elif error.status_code in [401]:
                self._refresh_token()
                error = "HTTP {}".format(error.status_code)
            else:
                return False
        self.logger.debug("Recoverable error [{}] from {} {}, retry #{} in {}s".format(error, request, url, counter, self.retry_delay))
        time.sleep(self.retry_delay)
        return True




    #@oasis_log
    def health_check(self):
        """
        Checks the health of the server.

        """
        try:
            url = urljoin(self.url_base, 'healthcheck/')
            return super(SessionManager, self).get(url, timeout=self.timeout)
        except Exception as e:    
            err_msg = 'Health check failed: Unable to connect to {}'.format(self.url_base)
            raise OasisException(err_msg)

    def get(self, url, **kwargs):
        counter = 0
        while True:
            counter += 1
            try:
                r = super(SessionManager, self).get(url, timeout=self.timeout, **kwargs)
            except (HTTPError, ConnectionError, ReadTimeout) as e:
                r = e
                if self.__recoverable(r, url, 'GET', counter):
                    continue
            return r

    def post(self, url, **kwargs):
        counter = 0
        while True:
            counter += 1
            try:
                r = super(SessionManager, self).post(url, timeout=self.timeout, **kwargs)
            except (HTTPError, ConnectionError, ReadTimeout) as e:
                r = e
                if self.__recoverable(r, url, 'POST', counter):
                    continue
            return r

    def delete(self, url, **kwargs):
        counter = 0
        while True:
            counter += 1
            try:
                r = super(SessionManager, self).delete(url, timeout=self.timeout, **kwargs)
            except (HTTPError, ConnectionError, ReadTimeout) as e:
                r = e
                if self.__recoverable(r, url, 'DELETE', counter):
                    continue
            return r

    def put(self, url, **kwargs):
        counter = 0
        while True:
            counter += 1
            try:
                r = super(SessionManager, self).put(url, timeout=self.timeout, **kwargs)
            except (HTTPError, ConnectionError, ReadTimeout) as e:
                r = e
                if self.__recoverable(r, url, 'OPTIONS', counter):
                    continue
            return r
