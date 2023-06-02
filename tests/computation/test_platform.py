import unittest
from unittest.mock import MagicMock, patch, Mock

import os

import logging

from oasislmf.utils.exceptions import OasisException
from oasislmf.manager import OasisManager

from .data.common import *
from .data.platform_returns import *
from .test_computation import ComputationChecker

import responses
from responses.registries import OrderedRegistry
from responses.matchers import json_params_matcher


class TestPlatformList(ComputationChecker):

    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()
        cls.default_args = cls.manager._params_platform_list()
        cls.tmp_files = cls.create_tmp_files([a for a in cls.default_args.keys() if 'json' in a])

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
        self.min_args = {'server_url': self.api_url}

    def tearDown(self):
        responses.stop()
        responses.reset()

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

    def test_list_models__success(self):
        called_args = self.combine_args([self.min_args, {'models': [1, 2]}])
        url_1 = f'{self.api_url}/V1/models/1/'
        url_2 = f'{self.api_url}/V1/models/2/'

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url_1, json=RETURN_MODELS[0])
                rsps.get(url_2, json=RETURN_MODELS[1])
                self.manager.platform_list(**called_args)
                self.assertIn('Model (id=1):', self._caplog.messages[1])
                self.assertIn('Model (id=2):', self._caplog.messages[2])

    def test_list_models__logs_error(self):
        called_args = self.combine_args([self.min_args, {'models': [1, 2]}])
        url_1 = f'{self.api_url}/V1/models/1/'
        url_2 = f'{self.api_url}/V1/models/2/'

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url_1, json=RETURN_MODELS[0])
                rsps.get(url_2, json={'error': 'model not found'}, status=404)
                self.manager.platform_list(**called_args)
                self.assertIn('Model (id=1):', self._caplog.messages[1])
                self.assertIn('{"error": "model not found"}', self._caplog.messages[2])

    def test_list_portfolios__success(self):
        called_args = self.combine_args([self.min_args, {'portfolios': [1, 2]}])
        url_1 = f'{self.api_url}/V1/portfolios/1/'
        url_2 = f'{self.api_url}/V1/portfolios/2/'

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url_1, json=RETURN_PORT[0])
                rsps.get(url_2, json=RETURN_PORT[1])
                self.manager.platform_list(**called_args)
                self.assertIn('Portfolio (id=1):', self._caplog.messages[1])
                self.assertIn('Portfolio (id=2):', self._caplog.messages[2])

    def test_list_portfolios__logs_error(self):
        called_args = self.combine_args([self.min_args, {'portfolios': [1, 2]}])
        url_1 = f'{self.api_url}/V1/portfolios/1/'
        url_2 = f'{self.api_url}/V1/portfolios/2/'

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url_1, json=RETURN_PORT[0])
                rsps.get(url_2, json={'error': 'portfolio not found'}, status=404)
                self.manager.platform_list(**called_args)
                self.assertIn('Portfolio (id=1):', self._caplog.messages[1])
                self.assertIn('{"error": "portfolio not found"}', self._caplog.messages[2])

    def test_list_analyses__success(self):
        called_args = self.combine_args([self.min_args, {'analyses': [4]}])
        url = f'{self.api_url}/V1/analyses/4/'

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url, json=RETURN_ANALYSIS[0])
                self.manager.platform_list(**called_args)
                self.assertIn('Analysis (id=4):', self._caplog.messages[1])

    def test_list_analyses__logs_error__and_model_success(self):
        called_args = self.combine_args([self.min_args, {'models': [1], 'analyses': [4]}])
        url_1 = f'{self.api_url}/V1/models/1/'
        url_2 = f'{self.api_url}/V1/analyses/4/'

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url_1, json=RETURN_MODELS[0])
                rsps.get(url_2, json={'error': 'analysis not found'}, status=404)
                self.manager.platform_list(**called_args)
                self.assertIn('Model (id=1):', self._caplog.messages[1])
                self.assertIn('{"error": "analysis not found"}', self._caplog.messages[2])


class TestPlatformRunInputs(ComputationChecker):
    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()
        cls.default_args = cls.manager._params_platform_run_inputs()
        cls.tmp_files = cls.create_tmp_files(
            [a for a in cls.default_args.keys() if 'csv' in a] +
            [a for a in cls.default_args.keys() if 'json' in a]
        )

    def add_connection_startup(self, responce_queue=None):
        responce_queue.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})

        responce_queue.post(
            url=f'{self.api_url}/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"})

    def setUp(self):
        assert responses, 'responses package required to run'
        self.api_url = 'http://localhost:8000'
        responses.start()

        self.write_json(self.tmp_files.get('analysis_settings_json'), {'test': 'run settings'})
        self.write_str(self.tmp_files.get('oed_location_csv'), "Sample location content")
        self.write_str(self.tmp_files.get('oed_accounts_csv'), "Sample accounts content")
        self.write_str(self.tmp_files.get('oed_info_csv'), "Sample info content")
        self.write_str(self.tmp_files.get('oed_scope_csv'), "Sample scope content")

    def tearDown(self):
        responses.stop()
        responses.reset()

    def test_run_inputs__no_data_given__error_is_raised(self):
        responses.get(
            url=f'http://localhost:8000/healthcheck/',
            json={"status": "OK"})
        responses.post(
            url=f'http://localhost:8000/access_token/',
            json={"access_token": "acc_tkn", "refresh_token": "ref_tkn"},
            headers={"authorization": "Bearer acc_tkn"})

        with self.assertRaises(OasisException) as context:
            self.manager.platform_run_inputs()
        self.assertEqual(str(context.exception), 'Error: Either select a "portfolio_id" or a location file is required.')

    @patch('builtins.input', side_effect=['AzureDiamond'])
    @patch('getpass.getpass', return_value='hunter2')
    def test_run_inputs__enter_password__unauthorized(self, mock_password, mock_username):
        responses.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})
        responses.post(
            url=f'{self.api_url}/access_token/',
            json={'error': 'unauthorized'},
            status=401)

        with self.assertRaises(OasisException) as context:
            self.manager.platform_run_inputs()
        self.assertIn(f'HTTPError: 401 Client Error: Unauthorized for url:', str(context.exception))
        mock_username.assert_called_once_with('Username: ')
        mock_password.assert_called_once_with('Password: ')

    @patch('builtins.input', side_effect=['AzureDiamond'])
    @patch('getpass.getpass', return_value='hunter2')
    def test_run_inputs__enter_password__authorized(self, mock_username, mock_password):
        responses.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})
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

        authorized_req = responses.calls[3].request
        authorized_rsp = responses.calls[3].response
        self.assertEqual(authorized_req.body, b'{"username": "AzureDiamond", "password": "hunter2"}')
        self.assertEqual(authorized_rsp.status_code, 200)

    def test_run_inputs__load_json_credentials(self):
        responses.get(
            url=f'{self.api_url}/healthcheck/',
            json={"status": "OK"})
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

        json_credentials_file = self.tmp_files.get('server_login_json')
        self.write_json(json_credentials_file, {"username": "AzureDiamond", "password": "hunter2"})

        with self.assertRaises(OasisException) as context:
            self.manager.platform_run_inputs(server_login_json=json_credentials_file.name)

        unauthorized_req = responses.calls[1].request
        unauthorized_rsp = responses.calls[1].response
        self.assertEqual(unauthorized_req.body, b'{"username": "admin", "password": "password"}')
        self.assertEqual(unauthorized_rsp.status_code, 401)

        authorized_req = responses.calls[3].request
        authorized_rsp = responses.calls[3].response
        self.assertEqual(authorized_req.body, b'{"username": "AzureDiamond", "password": "hunter2"}')
        self.assertEqual(authorized_rsp.status_code, 200)

    def test_run_inputs__given_analysis_id(self):
        ID = 4
        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            rsps.get(url=f'{self.api_url}/V1/analyses/{ID}/', json={'id': ID, 'status': 'NEW'})
            rsps.post(url=f'{self.api_url}/V1/analyses/{ID}/generate_inputs/', json={'id': ID, 'status': 'READY'})
            self.manager.platform_run_inputs(analysis_id=ID)

    @patch('oasislmf.computation.run.platform.APIClient.cancel_generate', return_value=True)
    @patch('oasislmf.computation.run.platform.APIClient.cancel_analysis', return_value=True)
    def test_run_inputs__given_analysis_id__cancel_is_called(self, mock_cancel_analysis, mock_cancel_generate):
        ID = 4
        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            rsps.get(url=f'{self.api_url}/V1/analyses/{ID}/', json={'id': ID, 'status': 'RUN_STARTED'})
            rsps.post(url=f'{self.api_url}/V1/analyses/{ID}/generate_inputs/', json={'id': ID, 'status': 'READY'})
            self.manager.platform_run_inputs(analysis_id=ID)

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            rsps.get(url=f'{self.api_url}/V1/analyses/{ID}/', json={'id': ID, 'status': 'INPUTS_GENERATION_QUEUED'})
            rsps.post(url=f'{self.api_url}/V1/analyses/{ID}/generate_inputs/', json={'id': ID, 'status': 'READY'})
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
            rsps.get(url=f'{self.api_url}/V1/models/', json=[{'id': model_id}])
            rsps.get(url=f'{self.api_url}/V1/portfolios/', json=[{'id': port_id}])
            self.manager.platform_run_inputs(portfolio_id=port_id, **exposure_files)
            mock_run_generate.assert_called_once_with(1)
            mock_create_analysis.assert_called_once_with(portfolio_id=port_id, model_id=model_id, analysis_settings_fp=None)

    @patch('oasislmf.computation.run.platform.APIClient.create_analysis', return_value={'id': 1})
    @patch('oasislmf.computation.run.platform.APIClient.run_generate', return_value=True)
    @patch('oasislmf.computation.run.platform.PlatformRunInputs.select_id')
    def test_inputs__given_portfolio_id__model_is_selected(self, mock_select, mock_run_generate, mock_create_analysis):
        model_id = 2
        port_id = 4
        exposure_files = {f: self.tmp_files.get(f).name for f in self.default_args if 'csv' in f}
        setting_file = self.tmp_files.get('analysis_settings_json').name
        mock_select.return_value = model_id

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            rsps.get(url=f'{self.api_url}/V1/models/', json=RETURN_MODELS)
            rsps.get(url=f'{self.api_url}/V1/models/', json=RETURN_MODELS)
            rsps.get(url=f'{self.api_url}/V1/portfolios/', json=[{'id': port_id}])
            self.manager.platform_run_inputs(
                portfolio_id=port_id,
                analysis_settings_json=setting_file,
                **exposure_files
            )
            mock_run_generate.assert_called_once_with(1)
            mock_create_analysis.assert_called_once_with(portfolio_id=port_id, model_id=model_id, analysis_settings_fp=setting_file)

    @patch('oasislmf.computation.run.platform.apiclient.create_analysis', return_value={'id': 1})
    @patch('oasislmf.computation.run.platform.apiclient.run_generate', return_value=True)
    @patch('builtins.input', side_effect=KeyboardInterrupt('Test'))
    def test_inputs__given_portfolio_id__model_selection_is_cancelled(self, mock_input, mock_run_generate, mock_create_analysis):
        model_id = 2
        port_id = 4
        exposure_files = {f: self.tmp_files.get(f).name for f in self.default_args if 'csv' in f}
        setting_file = self.tmp_files.get('analysis_settings_json').name

        with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
            self.add_connection_startup(rsps)
            rsps.get(url=f'{self.api_url}/V1/models/', json=RETURN_MODELS)
            rsps.get(url=f'{self.api_url}/V1/models/', json=RETURN_MODELS)
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
                rsps.get(url=f'{self.api_url}/V1/models/', json=RETURN_MODELS)
                rsps.get(url=f'{self.api_url}/V1/models/', json=RETURN_MODELS)
                with self.assertRaises(OasisException) as context:
                    self.manager.platform_run_inputs(
                        portfolio_id=port_id,
                        analysis_settings_json=setting_file,
                        **exposure_files
                    )
                mock_run_generate.assert_not_called()
                mock_create_analysis.assert_not_called()
                self.assertEqual(str(context.exception), ' Model selection cancelled')

            expected_error_log = "Not a valid id from: ['1', '2'] - ctrl-c to exit"
            self.assertEqual(self._caplog.messages[3], expected_error_log)
            self.assertEqual(self._caplog.messages[4], expected_error_log)
            self.assertEqual(self._caplog.messages[5], 'Invalid Response: 1.000')

    @patch('oasislmf.computation.run.platform.APIClient.create_analysis', return_value={'id': 1})
    @patch('oasislmf.computation.run.platform.APIClient.run_generate', return_value=True)
    def test_inputs__given_portfolio_id__model_is_missing(self, mock_run_generate, mock_create_analysis):
        exposure_files = {f: self.tmp_files.get(f).name for f in self.default_args if 'csv' in f}
        setting_file = self.tmp_files.get('analysis_settings_json').name
        port_id = 9001

        with self._caplog.at_level(logging.INFO):
            with responses.RequestsMock(assert_all_requests_are_fired=True, registry=OrderedRegistry) as rsps:
                self.add_connection_startup(rsps)
                rsps.get(url=f'{self.api_url}/V1/models/', json=[])
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
                rsps.get(url=f'{self.api_url}/V1/portfolios/', json=RETURN_PORT)
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


# class TestPlatformRunLosses(ComputationChecker):

# class TestPlatformRun(ComputationChecker):


# class TestPlatformDelete(ComputationChecker):
# class TestPlatformGet(ComputationChecker):
