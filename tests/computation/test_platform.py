import unittest
from unittest.mock import MagicMock, patch, Mock, call

import os

import logging

import oasislmf
from oasislmf.utils.exceptions import OasisException
from oasislmf.manager import OasisManager

from .data.common import *
from .data.platform_returns import *
from .test_computation import ComputationChecker

import responses
from responses.registries import OrderedRegistry
from responses.matchers import json_params_matcher

from hypothesis import given, settings
from hypothesis import strategies as st
# from hypothesis import provisional as pv
import string


class TestPlatformList(ComputationChecker):

    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()
        cls.default_args = cls.manager._params_platform_list()

    def add_connection_startup(self, responce_queue):
        responce_queue.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})

        responce_queue.post(
            url=f'{self.api_url}/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"})

    def setUp(self):
        self.tmp_files = self.create_tmp_files([a for a in self.default_args.keys() if 'json' in a])
        assert responses, 'responses package required to run'
        self.api_url = 'http://example.com/api'
        self.api_ver = 'v2'
        responses.start()
        self.min_args = {'server_url': self.api_url}

    def tearDown(self):
        super().tearDown()
        responses.stop()
        responses.reset()

    def test_list_all(self):
        called_args = self.min_args
        url_models = f'{self.api_url}/{self.api_ver}/models/'
        url_portfolios = f'{self.api_url}/{self.api_ver}/portfolios/'
        url_analyses = f'{self.api_url}/{self.api_ver}/analyses/'

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url_models, json=RETURN_MODELS)
                rsps.get(url_portfolios, json=RETURN_PORT)
                rsps.get(url_analyses, json=RETURN_ANALYSIS)
                self.manager.platform_list(**called_args)

            self.assertEqual(self._caplog.messages[3], MODELS_TABLE)
            self.assertEqual(self._caplog.messages[5], PORT_TABLE)
            self.assertEqual(self._caplog.messages[7], ANAL_TABLE)

    def test_list_models__success(self):
        called_args = self.combine_args([self.min_args, {'models': [1, 2]}])
        url_1 = f'{self.api_url}/{self.api_ver}/models/1/'
        url_2 = f'{self.api_url}/{self.api_ver}/models/2/'

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url_1, json=RETURN_MODELS[0])
                rsps.get(url_2, json=RETURN_MODELS[1])
                self.manager.platform_list(**called_args)
                self.assertIn('Model (id=1):', "\n".join(self._caplog.messages))
                self.assertIn('Model (id=2):', "\n".join(self._caplog.messages))

    def test_list_models__logs_error(self):
        called_args = self.combine_args([self.min_args, {'models': [1, 2]}])
        url_1 = f'{self.api_url}/{self.api_ver}/models/1/'
        url_2 = f'{self.api_url}/{self.api_ver}/models/2/'

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url_1, json=RETURN_MODELS[0])
                rsps.get(url_2, json={'error': 'model not found'}, status=404)
                self.manager.platform_list(**called_args)
                self.assertIn('Model (id=1):', "\n".join(self._caplog.messages))
                self.assertIn('{"error": "model not found"}', "\n".join(self._caplog.messages))

    def test_list_portfolios__success(self):
        called_args = self.combine_args([self.min_args, {'portfolios': [1, 2]}])
        url_1 = f'{self.api_url}/{self.api_ver}/portfolios/1/'
        url_2 = f'{self.api_url}/{self.api_ver}/portfolios/2/'

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url_1, json=RETURN_PORT[0])
                rsps.get(url_2, json=RETURN_PORT[1])
                self.manager.platform_list(**called_args)
                self.assertIn('Portfolio (id=1):', "\n".join(self._caplog.messages))
                self.assertIn('Portfolio (id=2):', "\n".join(self._caplog.messages))

    def test_list_portfolios__logs_error(self):
        called_args = self.combine_args([self.min_args, {'portfolios': [1, 2]}])
        url_1 = f'{self.api_url}/{self.api_ver}/portfolios/1/'
        url_2 = f'{self.api_url}/{self.api_ver}/portfolios/2/'

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url_1, json=RETURN_PORT[0])
                rsps.get(url_2, json={'error': 'portfolio not found'}, status=404)
                self.manager.platform_list(**called_args)
                self.assertIn('Portfolio (id=1):', "\n".join(self._caplog.messages))
                self.assertIn('{"error": "portfolio not found"}', "\n".join(self._caplog.messages))

    def test_list_analyses__success(self):
        called_args = self.combine_args([self.min_args, {'analyses': [4]}])
        url = f'{self.api_url}/{self.api_ver}/analyses/4/'

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url, json=RETURN_ANALYSIS[0])
                self.manager.platform_list(**called_args)
                self.assertIn('Analysis (id=4):', "\n".join(self._caplog.messages))

    def test_list_analyses__logs_error__and_model_success(self):
        called_args = self.combine_args([self.min_args, {'models': [1], 'analyses': [4]}])
        url_1 = f'{self.api_url}/{self.api_ver}/models/1/'
        url_2 = f'{self.api_url}/{self.api_ver}/analyses/4/'

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url_1, json=RETURN_MODELS[0])
                rsps.get(url_2, json={'error': 'analysis not found'}, status=404)
                self.manager.platform_list(**called_args)
                self.assertIn('Model (id=1):', "\n".join(self._caplog.messages))
                self.assertIn('{"error": "analysis not found"}', "\n".join(self._caplog.messages))


class TestPlatformRunInputs(ComputationChecker):
    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()
        cls.default_args = cls.manager._params_platform_run_inputs()

    def add_connection_startup(self, responce_queue=None):
        responce_queue.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})

        responce_queue.post(
            url=f'{self.api_url}/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"})

    def setUp(self):
        self.tmp_files = self.create_tmp_files(
            [a for a in self.default_args.keys() if 'csv' in a] +
            [a for a in self.default_args.keys() if 'json' in a]
        )
        assert responses, 'responses package required to run'
        self.api_url = 'http://localhost:8000'
        self.api_ver = 'v2'
        responses.start()

        self.write_json(self.tmp_files.get('analysis_settings_json'), {'test': 'run settings'})
        self.write_str(self.tmp_files.get('oed_location_csv'), "Sample location content")
        self.write_str(self.tmp_files.get('oed_accounts_csv'), "Sample accounts content")
        self.write_str(self.tmp_files.get('oed_info_csv'), "Sample info content")
        self.write_str(self.tmp_files.get('oed_scope_csv'), "Sample scope content")

    def tearDown(self):
        super().tearDown()
        responses.stop()
        responses.reset()

    def test_run_inputs__no_data_given__error_is_raised(self):
        responses.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})
        responses.post(
            url=f'{self.api_url}/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"})

        with self.assertRaises(OasisException) as context:
            self.manager.platform_run_inputs()
        self.assertEqual(str(context.exception),
                         'Error: At least one of the following inputs is required [portfolio_id, oed_location_csv, oed_accounts_csv]')

    @patch('builtins.input', side_effect=['y', 'AzureDiamond'])
    @patch('getpass.getpass', return_value='hunter2')
    def test_run_inputs__enter_password__unauthorized(self, mock_password, mock_input):
        responses.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})
        responses.post(
            url=f'{self.api_url}/access_token/',
            json={'error': 'unauthorized'},
            status=401)
        responses.post(
            url=f'{self.api_url}/access_token/',
            json={'error': 'unauthorized'},
            status=401)

        with self.assertRaises(OasisException) as context:
            self.manager.platform_run_inputs()
        self.assertIn(f'HTTPError: 401 Client Error: Unauthorized for url:', str(context.exception))
        mock_input.assert_has_calls([
            call('Use simple JWT [Y/n]: '),
            call('Username: ')
        ])
        self.assertEqual(mock_input.call_count, 2)
        mock_password.assert_called_once_with('Password: ')

    @patch('builtins.input', side_effect=['y', 'AzureDiamond'])
    @patch('getpass.getpass', return_value='hunter2')
    def test_run_inputs__auth_failed__error_is_raised(self, mock_password, mock_input):
        responses.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})
        responses.post(
            url=f'{self.api_url}/access_token/',
            json={'Error': 'server failed'},
            status=500)
        with self.assertRaises(OasisException) as context:
            self.manager.platform_run_inputs()
        self.assertEqual(str(context.exception),
                         f'Authentication Error, HTTPError: 500 Server Error: Internal Server Error for url: {self.api_url}/access_token/')

    @patch('builtins.input', side_effect=['y', 'AzureDiamond'])
    @patch('getpass.getpass', return_value='hunter2')
    def test_run_inputs__enter_password__authorized(self, mock_password, mock_input):
        responses.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})
        responses.post(
            url=f'{self.api_url}/access_token/',
            json={'error': 'unauthorized'},
            status=401)
        responses.post(
            url=f'{self.api_url}/access_token/',
            json={'error': 'unauthorized'},
            status=401)
        responses.post(
            url=f'{self.api_url}/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"},
            match=[json_params_matcher({"username": "AzureDiamond", "password": "hunter2"})]
        )
        with self.assertRaises(OasisException) as context:
            self.manager.platform_run_inputs()

        unauthorized_req = responses.calls[1].request
        unauthorized_rsp = responses.calls[1].response
        self.assertEqual(unauthorized_req.body, b'{"username": "admin", "password": "password"}')
        self.assertEqual(unauthorized_rsp.status_code, 401)

        unauthorized_req = responses.calls[3].request
        unauthorized_rsp = responses.calls[3].response
        self.assertEqual(unauthorized_req.body, b'{"client_id": "oasis-service", "client_secret": "serviceNotSoSecret"}')
        self.assertEqual(unauthorized_rsp.status_code, 401)

        authorized_req = responses.calls[5].request
        authorized_rsp = responses.calls[5].response
        self.assertEqual(authorized_req.body, b'{"username": "AzureDiamond", "password": "hunter2"}')
        self.assertEqual(authorized_rsp.status_code, 200)

    def test_run_inputs__load_json_credentials(self):
        responses.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})
        responses.post(
            url=f'{self.api_url}/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"},
            match=[json_params_matcher({"username": "AzureDiamond", "password": "hunter2"})]
        )

        json_credentials_file = self.tmp_files.get('server_login_json')
        self.write_json(json_credentials_file, {"username": "AzureDiamond", "password": "hunter2"})

        with self.assertRaises(OasisException) as context:
            self.manager.platform_run_inputs(server_login_json=json_credentials_file.name)

        authorized_req = responses.calls[1].request
        authorized_rsp = responses.calls[1].response
        self.assertEqual(authorized_req.body, b'{"username": "AzureDiamond", "password": "hunter2"}')
        self.assertEqual(authorized_rsp.status_code, 200)

    def test_run_inputs__load_json_credentials_oidc(self):
        responses.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})
        responses.post(
            url=f'{self.api_url}/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"},
            match=[json_params_matcher({"client_id": "serviceId", "client_secret": "serviceSecret"})]
        )

        json_credentials_file = self.tmp_files.get('server_login_json')
        self.write_json(json_credentials_file, {"client_id": "serviceId", "client_secret": "serviceSecret"})

        with self.assertRaises(OasisException) as context:
            self.manager.platform_run_inputs(server_login_json=json_credentials_file.name)

        authorized_req = responses.calls[1].request
        authorized_rsp = responses.calls[1].response
        self.assertEqual(authorized_req.body, b'{"client_id": "serviceId", "client_secret": "serviceSecret"}')
        self.assertEqual(authorized_rsp.status_code, 200)

    def test_run_inputs__load_json_credentials_invalid(self):
        json_credentials_file = self.tmp_files.get('server_login_json')
        self.write_json(json_credentials_file, {"invalid_credentials_1": "credentials_1", "invalid_credentials_2": "credentials_2"})

        with self.assertRaises(OasisException) as context:
            self.manager.platform_run_inputs(server_login_json=json_credentials_file.name)

        self.assertIn("Error: No valid credentials provided for platform", str(context.exception))
        self.assertIn("invalid_credentials_1", str(context.exception))
        self.assertIn("invalid_credentials_2", str(context.exception))

    def test_run_inputs__given_analysis_id(self):
        ID = 4
        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            rsps.get(url=f'{self.api_url}/{self.api_ver}/analyses/{ID}/', json={'id': ID, 'status': 'NEW'})
            rsps.post(url=f'{self.api_url}/{self.api_ver}/analyses/{ID}/generate_inputs/', json={'id': ID, 'status': 'READY'})
            self.manager.platform_run_inputs(analysis_id=ID)

    def test_run_inputs__given_analysis_id__server_error__exception_raised(self):
        ID = 4
        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            rsps.get(url=f'{self.api_url}/{self.api_ver}/analyses/{ID}/', json={'error': 'server failed'}, status=500)

            with self.assertRaises(OasisException) as context:
                self.manager.platform_run_inputs(analysis_id=ID)
            self.assertEqual(str(context.exception),
                             f'Error running analysis ({ID}) - 500 Server Error: Internal Server Error for url: {self.api_url}/{self.api_ver}/analyses/{ID}/')

    @patch('oasislmf.computation.run.platform.APIClient.cancel_generate', return_value=True)
    @patch('oasislmf.computation.run.platform.APIClient.cancel_analysis', return_value=True)
    def test_run_inputs__given_analysis_id__cancel_is_called(self, mock_cancel_analysis, mock_cancel_generate):
        ID = 4
        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            rsps.get(url=f'{self.api_url}/{self.api_ver}/analyses/{ID}/', json={'id': ID, 'status': 'RUN_STARTED'})
            rsps.post(url=f'{self.api_url}/{self.api_ver}/analyses/{ID}/generate_inputs/', json={'id': ID, 'status': 'READY'})
            self.manager.platform_run_inputs(analysis_id=ID)

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            rsps.get(url=f'{self.api_url}/{self.api_ver}/analyses/{ID}/', json={'id': ID, 'status': 'INPUTS_GENERATION_QUEUED'})
            rsps.post(url=f'{self.api_url}/{self.api_ver}/analyses/{ID}/generate_inputs/', json={'id': ID, 'status': 'READY'})
            self.manager.platform_run_inputs(analysis_id=ID)

        mock_cancel_analysis.assert_called_with(ID)
        mock_cancel_generate.assert_called_with(ID)

    @patch('oasislmf.computation.run.platform.APIClient.create_analysis', return_value={'id': 1})
    @patch('oasislmf.computation.run.platform.APIClient.run_generate', return_value=True)
    def test_inputs__given_portfolio_id__model_is_autoselected(self, mock_run_generate, mock_create_analysis):
        model_id = 3
        port_id = 2
        exposure_files = {f: self.tmp_files.get(f).name for f in self.default_args if 'csv' in f}
        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            rsps.get(url=f'{self.api_url}/{self.api_ver}/models/', json=[{'id': model_id}])
            rsps.get(url=f'{self.api_url}/{self.api_ver}/portfolios/', json=[{'id': port_id}])
            self.manager.platform_run_inputs(portfolio_id=port_id, **exposure_files)
            mock_run_generate.assert_called_once_with(1)
            mock_create_analysis.assert_called_once_with(portfolio_id=port_id, model_id=model_id, analysis_settings_fp=None)

    @patch('oasislmf.computation.run.platform.APIClient.create_analysis', return_value={'id': 1})
    @patch('oasislmf.computation.run.platform.APIClient.run_generate', return_value=True)
    @patch('builtins.input')
    def test_inputs__given_portfolio_id__model_is_selected(self, mock_input, mock_run_generate, mock_create_analysis):
        model_id = 2
        port_id = 4
        exposure_files = {f: self.tmp_files.get(f).name for f in self.default_args if 'csv' in f}
        setting_file = self.tmp_files.get('analysis_settings_json').name
        mock_input.return_value = model_id

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            rsps.get(url=f'{self.api_url}/{self.api_ver}/models/', json=RETURN_MODELS)
            rsps.get(url=f'{self.api_url}/{self.api_ver}/models/', json=RETURN_MODELS)
            rsps.get(url=f'{self.api_url}/{self.api_ver}/portfolios/', json=[{'id': port_id}])
            self.manager.platform_run_inputs(
                portfolio_id=port_id,
                analysis_settings_json=setting_file,
                **exposure_files
            )
            mock_run_generate.assert_called_once_with(1)
            mock_create_analysis.assert_called_once_with(portfolio_id=port_id, model_id=model_id, analysis_settings_fp=setting_file)

    @patch('oasislmf.computation.run.platform.APIClient.create_analysis', return_value={'id': 1})
    @patch('oasislmf.computation.run.platform.APIClient.run_generate', return_value=True)
    @patch('builtins.input', side_effect=KeyboardInterrupt('Test'))
    def test_inputs__given_portfolio_id__model_selection_is_cancelled(self, mock_input, mock_run_generate, mock_create_analysis):
        model_id = 2
        port_id = 4
        exposure_files = {f: self.tmp_files.get(f).name for f in self.default_args if 'csv' in f}
        setting_file = self.tmp_files.get('analysis_settings_json').name

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            rsps.get(url=f'{self.api_url}/{self.api_ver}/models/', json=RETURN_MODELS)
            rsps.get(url=f'{self.api_url}/{self.api_ver}/models/', json=RETURN_MODELS)
            with self.assertRaises(OasisException) as context:
                self.manager.platform_run_inputs(
                    portfolio_id=port_id,
                    analysis_settings_json=setting_file,
                    **exposure_files
                )
            mock_run_generate.assert_not_called()
            mock_create_analysis.assert_not_called()
            self.assertEqual(str(context.exception), ' Model selection cancelled')

    @patch('oasislmf.computation.run.platform.APIClient.create_analysis', return_value={'id': 1})
    @patch('oasislmf.computation.run.platform.APIClient.run_generate', return_value=True)
    @patch('builtins.input', side_effect=['q24dsa', '1.000', ValueError('bad value'), KeyboardInterrupt('break')])
    def test_inputs__given_portfolio_id__model_select_is_invalid(self, mock_input, mock_run_generate, mock_create_analysis):
        model_id = 2
        port_id = 4
        exposure_files = {f: self.tmp_files.get(f).name for f in self.default_args if 'csv' in f}
        setting_file = self.tmp_files.get('analysis_settings_json').name

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url=f'{self.api_url}/{self.api_ver}/models/', json=RETURN_MODELS)
                rsps.get(url=f'{self.api_url}/{self.api_ver}/models/', json=RETURN_MODELS)
                with self.assertRaises(OasisException) as context:
                    self.manager.platform_run_inputs(
                        portfolio_id=port_id,
                        analysis_settings_json=setting_file,
                        **exposure_files
                    )
                mock_run_generate.assert_not_called()
                mock_create_analysis.assert_not_called()
                self.assertEqual(str(context.exception), ' Model selection cancelled')

            expected_error_log = "not among the valid ids: ['1', '2'] - ctrl-c to exit"
            self.assertIn(expected_error_log, "\n".join(self._caplog.messages))
            self.assertIn(expected_error_log, "\n".join(self._caplog.messages))
            self.assertEqual(self._caplog.messages[6], 'Invalid Response: 1.000')

    @patch('oasislmf.computation.run.platform.APIClient.create_analysis', return_value={'id': 1})
    @patch('oasislmf.computation.run.platform.APIClient.run_generate', return_value=True)
    def test_inputs__given_portfolio_id__model_is_missing(self, mock_run_generate, mock_create_analysis):
        exposure_files = {f: self.tmp_files.get(f).name for f in self.default_args if 'csv' in f}
        setting_file = self.tmp_files.get('analysis_settings_json').name
        port_id = 9001

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url=f'{self.api_url}/{self.api_ver}/models/', json=[])
                with self.assertRaises(OasisException) as context:
                    self.manager.platform_run_inputs(
                        portfolio_id=port_id,
                        analysis_settings_json=setting_file,
                        **exposure_files
                    )
                mock_run_generate.assert_not_called()
                mock_create_analysis.assert_not_called()
                self.assertEqual(str(context.exception), f'No models found in API: {self.api_url}')

    @patch('oasislmf.computation.run.platform.APIClient.create_analysis', return_value={'id': 1})
    @patch('oasislmf.computation.run.platform.APIClient.run_generate', return_value=True)
    def test_inputs__given_portfolio_id__portfolio_is_missing(self, mock_run_generate, mock_create_analysis):
        model_id = 2
        port_id = 4
        exposure_files = {f: self.tmp_files.get(f).name for f in self.default_args if 'csv' in f}
        setting_file = self.tmp_files.get('analysis_settings_json').name

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url=f'{self.api_url}/{self.api_ver}/portfolios/', json=RETURN_PORT)
                with self.assertRaises(OasisException) as context:
                    self.manager.platform_run_inputs(
                        portfolio_id=port_id,
                        model_id=model_id,
                        analysis_settings_json=setting_file,
                        **exposure_files
                    )
                mock_run_generate.assert_not_called()
                mock_create_analysis.assert_not_called()
                self.assertEqual(str(context.exception), f'Portfolio "{port_id}" not found in API: {self.api_url}')

    @patch('oasislmf.computation.run.platform.APIClient.upload_inputs', return_value={'id': 4})
    @patch('oasislmf.computation.run.platform.APIClient.create_analysis', return_value={'id': 1})
    @patch('oasislmf.computation.run.platform.APIClient.run_generate', return_value=True)
    def test_inputs__given_model_id__portfolio_is_created(self, mock_run_generate, mock_create_analysis, mock_upload_inputs):
        model_id = 2
        port_id = 4
        analysis_id = 1
        exposure_files = {f: self.tmp_files.get(f).name for f in self.default_args if 'csv' in f}
        setting_file = self.tmp_files.get('analysis_settings_json').name

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)

            self.manager.platform_run_inputs(
                model_id=model_id,
                analysis_settings_json=setting_file,
                **exposure_files
            )
            mock_run_generate.assert_called_once_with(analysis_id)
            mock_create_analysis.assert_called_once_with(
                portfolio_id=port_id,
                model_id=model_id,
                analysis_settings_fp=setting_file,
            )
            mock_upload_inputs.assert_called_once_with(
                portfolio_id=None,
                location_fp=exposure_files['oed_location_csv'],
                accounts_fp=exposure_files['oed_accounts_csv'],
                ri_info_fp=exposure_files['oed_info_csv'],
                ri_scope_fp=exposure_files['oed_scope_csv']
            )


class TestPlatformRunLosses(ComputationChecker):
    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()
        cls.default_args = cls.manager._params_platform_run_losses()

    def setUp(self):
        self.tmp_dirs = self.create_tmp_dirs([a for a in self.default_args if 'dir' in a])
        self.tmp_files = self.create_tmp_files([a for a in self.default_args if 'json' in a])
        self.write_json(self.tmp_files.get('analysis_settings_json'), {'test': 'run settings'})
        self.api_url = 'http://localhost:8000'
        self.api_ver = 'v2'
        responses.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})
        responses.post(
            url=f'{self.api_url}/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"})
        responses.start()

    def tearDown(self):
        super().tearDown()
        responses.stop()
        responses.reset()

    @patch('oasislmf.computation.run.platform.APIClient.run_analysis', return_value={'id': 1})
    @patch('oasislmf.computation.run.platform.APIClient.download_output', return_value=True)
    def test_run_analysis__only_analysis_id(self, mock_download_output, mock_run_analysis):
        analysis_id = 4

        self.manager.platform_run_losses(analysis_id=analysis_id)
        mock_run_analysis.assert_called_once_with(analysis_id, None)
        mock_download_output.assert_called_once_with(analysis_id, './')

    @patch('oasislmf.computation.run.platform.APIClient.run_analysis', return_value={'id': 1})
    @patch('oasislmf.computation.run.platform.APIClient.download_output', return_value=True)
    def test_run_analysis__all_options_set(self, mock_download_output, mock_run_analysis):
        analysis_id = 4
        setting_file = self.tmp_files.get('analysis_settings_json').name
        output_dir = self.tmp_dirs.get('output_dir').name

        self.manager.platform_run_losses(
            analysis_id=analysis_id,
            analysis_settings_json=setting_file,
            output_dir=output_dir
        )
        mock_run_analysis.assert_called_once_with(analysis_id, setting_file)
        mock_download_output.assert_called_once_with(analysis_id, output_dir)


class TestPlatformRun(ComputationChecker):

    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()

        # Args
        cls.default_args = cls.manager._params_platform_run()
        cls.gen_files_args = cls.manager._params_platform_run_inputs()
        cls.gen_loss_args = cls.manager._params_platform_run_losses()

        # Tempfiles
        cls.api_url = 'http://localhost:8000'

    def setUp(self):
        self.tmp_dirs = self.create_tmp_dirs([a for a in self.default_args.keys() if 'dir' in a])
        self.tmp_files = self.create_tmp_files(
            [a for a in self.default_args.keys() if 'csv' in a] +
            [a for a in self.default_args.keys() if 'json' in a]
        )
        self.write_json(self.tmp_files.get('analysis_settings_json'), {'test': 'run settings'})
        self.api_url = 'http://localhost:8000'
        self.api_ver = 'v2'
        responses.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})
        responses.post(
            url=f'{self.api_url}/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"})
        responses.start()

    def tearDown(self):
        super().tearDown()
        responses.stop()
        responses.reset()

    def test_args__default_combine(self):
        expt_combined_args = self.combine_args([
            self.gen_files_args,
            self.gen_loss_args,
        ])
        self.assertEqual(expt_combined_args, self.default_args)

    @settings(deadline=None, max_examples=10)
    @given(
        server_version=st.text(alphabet=string.ascii_letters) | st.none(),
        # server_url=st.text(alphabet=string.ascii_letters),
        model_id=st.integers(min_value=1) | st.none(),
        portfolio_id=st.integers(min_value=1) | st.none(),
        analysis_id=st.integers(min_value=1) | st.none(),
        analysis_settings_json=st.booleans(),
        server_login_json=st.booleans(),
        oed_location_csv=st.booleans(),
        oed_accounts_csv=st.booleans(),
        oed_info_csv=st.booleans(),
        oed_scope_csv=st.booleans(),
        output_dir=st.booleans(),
    )
    def test_args__passed_correctly(self,
                                    server_login_json,
                                    # server_url,
                                    server_version,
                                    model_id,
                                    portfolio_id,
                                    analysis_id,
                                    analysis_settings_json,
                                    oed_location_csv,
                                    oed_accounts_csv,
                                    oed_info_csv,
                                    oed_scope_csv,
                                    output_dir):

        # Extract funcution kwargs into dict, and replace booleans with temp file paths
        call_args = {k: v for k, v in locals().items() if k in self.default_args}
        call_args['server_url'] = self.api_url
        for k, v in self.combine_args([self.tmp_files, self.tmp_dirs]).items():
            if call_args[k] is True:
                call_args[k] = v.name
                if k == 'server_login_json':
                    self.write_json(v, {"username": "dummy", "password": "dummy"})
            else:
                call_args[k] = None

        run_mock = Mock()
        # run_mock.run.side_effect = lambda *args, **kwargs: 23
        analysis_id_return = analysis_id if analysis_id else 42
        run_mock.run.side_effect = lambda *args, **kwargs: analysis_id_return
        plat_files_mock = Mock()
        plat_losses_mock = Mock()

        responses.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})
        responses.post(
            url=f'{self.api_url}/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"})

        with patch.object(oasislmf.computation.run.platform, 'PlatformRunInputs', plat_files_mock), \
                patch.object(oasislmf.computation.run.platform, 'PlatformRunLosses', plat_losses_mock):
            plat_files_mock.return_value = run_mock
            self.manager.platform_run(**call_args)

        plat_files_mock.assert_called_once_with(**call_args)
        call_args['analysis_id'] = analysis_id_return
        plat_losses_mock.assert_called_once_with(**call_args)


class TestPlatformDelete(ComputationChecker):
    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()

    def add_connection_startup(self, responce_queue):
        responce_queue.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})

        responce_queue.post(
            url=f'{self.api_url}/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"})

    def setUp(self):
        self.api_url = 'http://localhost:8000'
        self.api_ver = 'v2'
        self.min_args = {'server_url': self.api_url}

    def test_delete__no_input_given__exception_raised(self):
        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            with self.assertRaises(OasisException) as context:
                self.manager.platform_delete()
            self.assertIn('Select item(s) to delete, list of either:', str(context.exception))

    def test_delete__non_int_given__exception_raised(self):
        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            bad_data = ['FOOBAR', 3.14159265359]
            with self.assertRaises(OasisException) as context:
                self.manager.platform_delete(models=bad_data)

            self.assertEqual(str(context.exception),
                             f"Invalid input, 'models', must be a list of type Int, not {bad_data}")

    def test_delete__some_id_are_missing__log_issue_but_dont_fail(self):
        model_list = [1, 4, 6]

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            rsps.delete(url=f'{self.api_url}/{self.api_ver}/models/1/')
            rsps.delete(url=f'{self.api_url}/{self.api_ver}/models/4/', status=404)
            rsps.delete(url=f'{self.api_url}/{self.api_ver}/models/6/')

            with self._caplog.at_level(logging.INFO):
                self.manager.platform_delete(models=model_list)
                expected_err_log = f'Delete error models_id=4 - 404 Client Error: Not Found for url: {self.api_url}/{self.api_ver}/models/4/'
                self.assertEqual(self._caplog.messages[2], 'Deleted models_id=1')
                self.assertEqual(self._caplog.messages[3], expected_err_log)
                self.assertEqual(self._caplog.messages[4], 'Deleted models_id=6')

    @settings(deadline=None, max_examples=25)
    @given(
        models=st.lists(st.integers(min_value=1), min_size=1),
        portfolios=st.lists(st.integers(min_value=1), min_size=1),
        analyses=st.lists(st.integers(min_value=1), min_size=1),
    )
    def test_delete__data_is_ok__endpoints_are_called(self, models, portfolios, analyses):
        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)

            for id in models:
                rsps.delete(url=f'{self.api_url}/{self.api_ver}/models/{id}/')
            for id in portfolios:
                rsps.delete(url=f'{self.api_url}/{self.api_ver}/portfolios/{id}/')
            for id in analyses:
                rsps.delete(url=f'{self.api_url}/{self.api_ver}/analyses/{id}/')
            self.manager.platform_delete(models=models, portfolios=portfolios, analyses=analyses)


class TestPlatformGet(ComputationChecker):
    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()

    def add_connection_startup(self, responce_queue):
        responce_queue.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})

        responce_queue.post(
            url=f'{self.api_url}/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"})

    def setUp(self):
        self.api_url = 'http://localhost:8000'
        self.api_ver = 'v2'
        self.tmp_dirs = self.create_tmp_dirs(['output_dir'])
        self.min_args = {'server_url': self.api_url, 'output_dir': self.tmp_dirs['output_dir'].name}

    def test_get__no_input_given__exception_raised(self):
        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            with self.assertRaises(OasisException) as context:
                self.manager.platform_get()
            self.assertIn('Select file for download', str(context.exception))

    def test_get__some_id_are_missing__log_issue_but_dont_fail(self):
        analysis_list = [1, 4, 6]
        expected_content_1 = b'file 1 content'
        expected_content_6 = b'file 6 content'

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            rsps.get(url=f'{self.api_url}/{self.api_ver}/analyses/1/settings_file/', body=expected_content_1, content_type='application/json')
            rsps.get(url=f'{self.api_url}/{self.api_ver}/analyses/4/settings_file/', status=404)
            rsps.get(url=f'{self.api_url}/{self.api_ver}/analyses/6/settings_file/', body=expected_content_6, content_type='application/json')

            with self._caplog.at_level(logging.INFO):
                self.manager.platform_get(**self.min_args, analyses_settings_file=analysis_list)
                expected_err_log = f'Download failed: - 404 Client Error: Not Found for url: {self.api_url}/{self.api_ver}/analyses/4/settings_file/'
                self.assertEqual(self._caplog.messages[3], expected_err_log)

                filepath_1 = os.path.join(self.min_args['output_dir'], '1_analyses_settings_file.json')
                filepath_6 = os.path.join(self.min_args['output_dir'], '6_analyses_settings_file.json')

                self.assertTrue(os.path.isfile(filepath_1))
                self.assertTrue(os.path.isfile(filepath_6))
                self.assertEqual(self.read_file(filepath_1), expected_content_1)
                self.assertEqual(self.read_file(filepath_6), expected_content_6)

    def test_get__download_success__multiple_endpoints(self):
        model_id = [1]
        portfolio_id = [4]

        expected_content_loc = b'loc content'
        expected_content_acc = b'acc content'
        expected_content_settings = b'settings content'
        expected_content_version = b'version content'

        requested_files = {
            'portfolio_location_file': portfolio_id,
            'portfolio_accounts_file': portfolio_id,
            'model_settings': model_id,
            'model_versions': model_id,
        }

        with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps:
            self.add_connection_startup(rsps)
            rsps.get(url=f'{self.api_url}/{self.api_ver}/portfolios/4/location_file/', body=expected_content_loc, content_type='text/csv')
            rsps.get(url=f'{self.api_url}/{self.api_ver}/portfolios/4/accounts_file/', body=expected_content_acc, content_type='text/csv')
            rsps.get(url=f'{self.api_url}/{self.api_ver}/models/1/settings/', body=expected_content_settings, content_type='application/json')
            rsps.get(url=f'{self.api_url}/{self.api_ver}/models/1/versions/', body=expected_content_version, content_type='application/json')
            self.manager.platform_get(**self.min_args, **requested_files)

            downloaded_files = [
                (expected_content_loc, os.path.join(self.min_args['output_dir'], '4_portfolios_location_file.csv')),
                (expected_content_acc, os.path.join(self.min_args['output_dir'], '4_portfolios_accounts_file.csv')),
                (expected_content_settings, os.path.join(self.min_args['output_dir'], '1_models_settings.json')),
                (expected_content_version, os.path.join(self.min_args['output_dir'], '1_models_versions.json')),
            ]
            for expected_content, filepath in downloaded_files:
                self.assertTrue(os.path.isfile(filepath))
                self.assertEqual(self.read_file(filepath), expected_content)
