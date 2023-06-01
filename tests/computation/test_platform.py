import unittest
from unittest.mock import MagicMock, patch, Mock
import pytest

import os

import logging

import oasislmf
from oasislmf.manager import OasisManager

from .data.common import *
from .data.platform_returns import * 
from .test_computation import ComputationChecker

import responses
from responses.registries import OrderedRegistry

class TestPlatformList(ComputationChecker):

    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()

        # Args
        cls.default_args = cls.manager._params_platform_list()

        # Tempfiles
        cls.tmp_files = cls.create_tmp_files(
            #[a for a in cls.default_args.keys() if 'csv' in a] +
            [a for a in cls.default_args.keys() if 'json' in a]
        )


    def add_connection_startup(self, responce_queue):
        responce_queue.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})

        responce_queue.post(
            url=f'{self.api_url}/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"})



    def setUp(self):
        assert responses, 'responses package required to run'
        self.api_url = 'http://example.com/api'

        responses.start()
        self.min_args = {
            'server_url': self.api_url,
        }


    def tearDown(self):
        responses.stop()
        responses.reset()

    @pytest.fixture(autouse=True)
    def logging_fixtures(self, caplog):
        self._caplog = caplog


    def test_list_all(self):
        called_args = self.min_args

        url_models = f'{self.api_url}/V1/models/'
        url_portfolios = f'{self.api_url}/V1/portfolios/'
        url_analyses = f'{self.api_url}/V1/analyses/'

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url_models, json=RETURN_MODELS)
                rsps.get(url_portfolios, json=RETURN_PORT)
                rsps.get(url_analyses, json=RETURN_ANALYSIS)
                self.manager.platform_list(**called_args)

            self.assertEqual(self._caplog.messages[2], MODELS_TABLE)           
            self.assertEqual(self._caplog.messages[4], PORT_TABLE)           
            self.assertEqual(self._caplog.messages[6], ANAL_TABLE)           


    #def test_list_models__success(self):
    #def test_list_models__logs_error(self):
    #def test_list_portfolios__success(self):
    #def test_list_portfolios__logs_error(self):
    #def test_list_analyses__success(self):
    #def test_list_analyses__logs_error(self):



#class TestPlatformRun(ComputationChecker):
#class TestPlatformRunInputs(ComputationChecker):
#class TestPlatformRunLosses(ComputationChecker):
#class TestPlatformDelete(ComputationChecker):
#class TestPlatformGet(ComputationChecker):
