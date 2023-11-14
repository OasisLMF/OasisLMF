import pathlib
import os
import logging
import responses

from unittest import mock
from unittest.mock import patch, Mock, ANY

from ods_tools.oed.common import OdsException
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.path import setcwd
from oasislmf.manager import OasisManager
from .data.common import *
from .test_computation import ComputationChecker


TEST_DIR = pathlib.Path(os.path.realpath(__file__)).parent.parent
LOOKUP_CONFIG = TEST_DIR.joinpath('model_preparation').joinpath('meta_data').joinpath('lookup_config.json')


class TestGenFiles(ComputationChecker):

    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()
        cls.default_args = cls.manager._params_generate_files()

    def setUp(self):
        # Tempfiles
        self.tmp_dirs = self.create_tmp_dirs([a for a in self.default_args.keys() if 'dir' in a])
        self.tmp_files = self.create_tmp_files(
            [a for a in self.default_args.keys() if 'csv' in a] +
            [a for a in self.default_args.keys() if 'path' in a] +
            [a for a in self.default_args.keys() if 'json' in a]
        )

        self.min_args = {
            'lookup_config_json': LOOKUP_CONFIG,
            'oed_location_csv': self.tmp_files['oed_location_csv'].name,
        }
        self.il_args = {
            **self.min_args,
            'oed_accounts_csv': self.tmp_files['oed_accounts_csv'].name,
        }
        self.ri_args = {
            **self.il_args,
            'oed_info_csv': self.tmp_files['oed_info_csv'].name,
            'oed_scope_csv': self.tmp_files['oed_scope_csv'].name,
        }

        self.write_json(self.tmp_files.get('lookup_complex_config_json'), MIN_RUN_SETTINGS)
        self.write_json(self.tmp_files.get('model_settings_json'), MIN_MODEL_SETTINGS)
        self.write_json(self.tmp_files.get('analysis_settings_json'), MIN_RUN_SETTINGS)
        self.write_str(self.tmp_files.get('oed_location_csv'), MIN_LOC)
        self.write_str(self.tmp_files.get('oed_accounts_csv'), MIN_ACC)
        self.write_str(self.tmp_files.get('oed_info_csv'), MIN_INF)
        self.write_str(self.tmp_files.get('oed_scope_csv'), MIN_SCP)
        self.write_str(self.tmp_files.get('keys_data_csv'), MIN_KEYS)
        self.write_str(self.tmp_files.get('keys_errors_csv'), MIN_KEYS_ERR)

    def test_files__no_input__exception_raised(self):
        with self.assertRaises(OasisException) as context:
            self.manager.generate_files()
        expected_err_msg = 'No pre-generated keys file provided, and no lookup assets provided to generate a keys file'
        self.assertIn(expected_err_msg, str(context.exception))

    @patch('oasislmf.computation.generate.files.GenerateFiles._get_output_dir')
    def test_files__check_return__min_args(self, mock_output_dir):
        with self.tmp_dir() as t_dir:
            expected_run_dir = os.path.join(t_dir, 'runs', 'files-TIMESTAMP')
            mock_output_dir.return_value = expected_run_dir
            file_gen_return = self.manager.generate_files(**self.min_args)

            expected_return = {
                'items': f'{expected_run_dir}/items.csv',
                'coverages': f'{expected_run_dir}/coverages.csv',
                'amplifications': f'{expected_run_dir}/amplifications.csv'
            }
            self.assertEqual(file_gen_return, expected_return)
            for _, filepath in expected_return.items():
                self.assertTrue(os.path.isfile(filepath))
                self.assertTrue(os.path.getsize(filepath) > 0)

    @patch('oasislmf.computation.generate.files.GenerateFiles._get_output_dir')
    def test_files__check_return__il_args(self, mock_output_dir):
        with self.tmp_dir() as t_dir:
            expected_run_dir = os.path.join(t_dir, 'runs', 'files-TIMESTAMP')
            mock_output_dir.return_value = expected_run_dir
            file_gen_return = self.manager.generate_files(**self.il_args)

            expected_return = {
                'items': f'{expected_run_dir}/items.csv',
                'coverages': f'{expected_run_dir}/coverages.csv',
                'amplifications': f'{expected_run_dir}/amplifications.csv',
                'fm_policytc': f'{expected_run_dir}/fm_policytc.csv',
                'fm_profile': f'{expected_run_dir}/fm_profile.csv',
                'fm_programme': f'{expected_run_dir}/fm_programme.csv',
                'fm_xref': f'{expected_run_dir}/fm_xref.csv'
            }
            self.assertEqual(file_gen_return, expected_return)
            for _, filepath in expected_return.items():
                self.assertTrue(os.path.isfile(filepath))
                self.assertTrue(os.path.getsize(filepath) > 0)

    def test_files__check_return__ri_args__given_output_dir(self):
        with self.tmp_dir() as t_dir:
            expected_run_dir = t_dir
            file_gen_return = self.manager.generate_files(**{**self.ri_args, 'oasis_files_dir': t_dir})
            expected_return = {
                'items': f'{expected_run_dir}/items.csv',
                'coverages': f'{expected_run_dir}/coverages.csv',
                'amplifications': f'{expected_run_dir}/amplifications.csv',
                'fm_policytc': f'{expected_run_dir}/fm_policytc.csv',
                'fm_profile': f'{expected_run_dir}/fm_profile.csv',
                'fm_programme': f'{expected_run_dir}/fm_programme.csv',
                'fm_xref': f'{expected_run_dir}/fm_xref.csv',
                'ri_layers': f'{expected_run_dir}/ri_layers.json',
                'RI_1': f'{expected_run_dir}/RI_1'
            }
            self.assertEqual(file_gen_return, expected_return)
            for _, filepath in expected_return.items():
                if filepath.endswith('RI_1'):
                    self.assertTrue(os.path.isdir(filepath))
                else:
                    self.assertTrue(os.path.isfile(filepath))
                    self.assertTrue(os.path.getsize(filepath) > 0)

    @patch('oasislmf.computation.generate.files.get_il_input_items')
    def test_files__fm_aggregation__str_to_int_called(self, mock_get_il_items):
        mock_get_il_items.return_value = FAKE_IL_ITEMS_RETURN
        with self.tmp_dir() as t_dir:
            with setcwd(t_dir):
                expected_fm_agg_profile = self.default_args['profile_fm_agg']
                celery_mangled_fm_agg = {str(key): val for key, val in expected_fm_agg_profile.items()}

                call_args = {**self.il_args, 'profile_fm_agg': celery_mangled_fm_agg}
                file_gen_return = self.manager.generate_files(**call_args)

                mock_get_il_items.assert_called_once()
                called_fm_agg = mock_get_il_items.call_args.kwargs['fm_aggregation_profile']
                self.assertEqual(called_fm_agg, expected_fm_agg_profile)

    @responses.activate
    @patch('oasislmf.computation.generate.files.get_il_input_items')
    def test_files__reporting_currency__is_set_valid(self, mock_get_il_items):
        currency_config = {
            "currency_conversion_type": "FxCurrencyRates",
            "datetime": "2018-10-10"
        }

        responses.get(url='https://theforexapi.com/api/2018-10-10?base=GBP&symbols=JPY&rtype=fpy', json={'rates': {"JPY": 180.4}})
        mock_get_il_items.return_value = FAKE_IL_ITEMS_RETURN
        currency_config_file = self.tmp_files.get('currency_conversion_json')
        self.write_json(currency_config_file, currency_config)
        CURRENCY = 'JPY'

        with self.tmp_dir() as t_dir:
            with setcwd(t_dir):
                call_args = {
                    **self.ri_args,
                    'reporting_currency': CURRENCY,
                    'currency_conversion_json': currency_config_file.name
                }
                self.manager.generate_files(**call_args)
                loc_df = mock_get_il_items.call_args.kwargs['exposure_df']
                acc_df = mock_get_il_items.call_args.kwargs['accounts_df']
                self.assertEqual(loc_df['LocCurrency'].unique().to_list(), [CURRENCY])
                self.assertEqual(acc_df['AccCurrency'].unique().to_list(), [CURRENCY])

    def test_files__model_settings_given__group_fields_are_missing__warning_logged(self):
        model_settings_file = self.tmp_files.get('model_settings_json').name
        with self.tmp_dir() as t_dir:
            with self._caplog.at_level(logging.WARN):
                call_args = {**self.ri_args, 'model_settings_json': model_settings_file}
                file_gen_return = self.manager.generate_files(**call_args)
                self.assertIn('WARNING: Failed to load "damage_group_fields"', self._caplog.messages[0])
                self.assertIn('WARNING: Failed to load "hazard_group_fields"', self._caplog.messages[1])

    def test_files__model_settings_given__group_fields_are_set(self):
        model_settings_file = self.tmp_files.get('model_settings_json')
        self.write_json(model_settings_file, GROUP_FIELDS_MODEL_SETTINGS)
        with self.tmp_dir() as t_dir:
            call_args = {**self.ri_args, 'model_settings_json': model_settings_file.name}
            file_gen_return = self.manager.generate_files(**call_args)

    def test_files__model_settings_given__old_group_fields_are_valid(self):
        model_settings_file = self.tmp_files.get('model_settings_json')
        self.write_json(model_settings_file, OLD_GROUP_FIELDS_MODEL_SETTINGS)
        with self.tmp_dir() as t_dir:
            call_args = {**self.ri_args, 'model_settings_json': model_settings_file.name}
            file_gen_return = self.manager.generate_files(**call_args)

    @patch('oasislmf.computation.generate.files.establish_correlations')
    def test_files__model_settings_given__analysis_settings_replace_correlations(self, establish_correlations):
        model_settings_file = self.tmp_files.get('model_settings_json')
        self.write_json(model_settings_file, CORRELATIONS_MODEL_SETTINGS)
        analysis_settings_file = self.tmp_files.get('analysis_settings_json')
        self.write_json(analysis_settings_file, MIN_RUN_CORRELATIONS_SETTINGS)

        with self.tmp_dir() as _:
            call_args = {
                **self.ri_args,
                'model_settings_json': model_settings_file.name,
                'analysis_settings_json': analysis_settings_file.name
            }
            self.manager.generate_files(**call_args)

            establish_correlations.assert_called_once()
            used_correlations = establish_correlations.call_args.kwargs['model_settings']
            self.assertEqual(used_correlations['correlation_settings'], MIN_RUN_CORRELATIONS_SETTINGS['correlation_settings'])
