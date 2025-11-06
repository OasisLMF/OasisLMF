import pathlib
import os
import logging
import re
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
            [a for a in self.default_args.keys() if 'json' in a] +
            ['oed_location_csv__2r']
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
        self.write_str(self.tmp_files.get('oed_location_csv__2r'), N2_LOC)
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

        responses.add(
            responses.GET,
            re.compile(r'https://(theforexapi|theratesapi)\.com/api/.*'),
            json={'rates': {"JPY": 180.4}},
            status=200
        )
        mock_get_il_items.return_value = FAKE_IL_ITEMS_RETURN
        currency_config_file = self.tmp_files.get('currency_conversion_json')
        self.write_json(currency_config_file, currency_config)
        CURRENCY = 'JPY'

        with self.tmp_dir() as t_dir:
            with setcwd(t_dir):
                call_args = {
                    **self.il_args,
                    'reporting_currency': CURRENCY,
                    'currency_conversion_json': currency_config_file.name
                }
                self.manager.generate_files(**call_args)
                loc_df = mock_get_il_items.call_args.kwargs['exposure_data'].location.dataframe
                acc_df = mock_get_il_items.call_args.kwargs['exposure_data'].account.dataframe
                self.assertEqual(loc_df['LocCurrency'].unique().to_list(), [CURRENCY])
                self.assertEqual(acc_df['AccCurrency'].unique().to_list(), [CURRENCY])

    def test_files__reporting_currency__is_set_invalid(self):
        CURRENCY = 'JPY'
        with self.tmp_dir() as t_dir:
            with self.assertRaises(OdsException) as context:
                call_args = {**self.ri_args, 'reporting_currency': CURRENCY}
                file_gen_return = self.manager.generate_files(**call_args)
            expected_err_msg = f'Currency Convertion needs to be specified in order to convert term to reporting currency {CURRENCY}'
            self.assertIn(expected_err_msg, str(context.exception))

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

    def test_files__keys_csv__is_given(self):

        keys_file = self.tmp_files.get('keys_data_csv').name
        keys_err_file = self.tmp_files.get('keys_errors_csv').name
        with self.tmp_dir() as t_dir:
            call_args = {**self.ri_args,
                         'oasis_files_dir': t_dir,
                         'keys_data_csv': keys_file,
                         'keys_errors_csv': keys_err_file}
            file_gen_return = self.manager.generate_files(**call_args)

    def test_files__keys_csv__missing_loc_id__error_is_raised(self):
        keys_file = self.tmp_files.get('keys_data_csv').name
        keys_err_file = self.tmp_files.get('keys_errors_csv').name
        loc_file = self.tmp_files.get('oed_location_csv__2r').name
        with self.tmp_dir() as t_dir:
            with self.assertRaises(OasisException) as context:
                call_args = {**self.ri_args,
                             'oed_location_csv': loc_file,
                             'keys_data_csv': keys_file,
                             'keys_errors_csv': keys_err_file}
                file_gen_return = self.manager.generate_files(**call_args)
        expected_err_msg = 'Lookup error: missing "loc_id" values from keys return: [2]'
        self.assertIn(expected_err_msg, str(context.exception))

    def test_files__error_file_not_given__missing_loc_id__error_is_raised(self):
        keys_file = self.tmp_files.get('keys_data_csv').name
        keys_err_file = self.tmp_files.get('keys_errors_csv').name
        loc_file = self.tmp_files.get('oed_location_csv__2r').name
        with self.tmp_dir() as t_dir:
            with self.assertRaises(OasisException) as context:
                call_args = {**self.ri_args,
                             'oed_location_csv': loc_file,
                             'keys_data_csv': keys_file}
                file_gen_return = self.manager.generate_files(**call_args)
        expected_err_msg = 'Lookup error: missing "loc_id" values from keys return: [2]'
        self.assertIn(expected_err_msg, str(context.exception))

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
            self.assertEqual(used_correlations['model_settings']['correlation_settings'],
                             MIN_RUN_CORRELATIONS_SETTINGS['model_settings']['correlation_settings'])

    @patch('oasislmf.computation.generate.files.GenerateFiles._get_output_dir')
    def test_files__given_legacy_correlation_settings__correlation_csv_file_created(self, mock_output_dir):
        LEGACY_CORRELATIONS_SETTINGS = {
            "version": "3",
            "model_settings": {},
            "correlation_settings": [
                {"peril_correlation_group": 1, "damage_correlation_value": "0.7", "hazard_correlation_value": "0.4"},
                {"peril_correlation_group": 2, "damage_correlation_value": "0.5", "hazard_correlation_value": "0.2"}
            ],
            "lookup_settings": {
                "supported_perils": [
                    {"id": "WSS", "desc": "Single Peril: Storm Surge", "peril_correlation_group": 1},
                    {"id": "WTC", "desc": "Single Peril: Tropical Cyclone", "peril_correlation_group": 2},
                    {"id": "WW1", "desc": "Group Peril: Windstorm with storm surge"},
                    {"id": "WW2", "desc": "Group Peril: Windstorm w/o storm surge"}
                ]
            },
            "model_default_samples": 10,
            "data_settings": {
                "damage_group_fields": ["PortNumber", "AccNumber", "LocNumber"],
                "hazard_group_fields": ["PortNumber", "AccNumber", "LocNumber"]
            }
        }
        model_settings_file = self.tmp_files.get('model_settings_json')
        self.write_json(model_settings_file, LEGACY_CORRELATIONS_SETTINGS)

        with self.tmp_dir() as t_dir:
            # Run
            expected_run_dir = os.path.join(t_dir, 'runs', 'files-TIMESTAMP')
            mock_output_dir.return_value = expected_run_dir
            call_args = {
                **self.ri_args,
                'model_settings_json': model_settings_file.name
            }
            file_gen_return = self.manager.generate_files(**call_args)

            # check correlations files exist
            correlations_csv_path = os.path.join(expected_run_dir, 'correlations.csv')
            correlations_bin_path = os.path.join(expected_run_dir, 'correlations.bin')
            self.assertTrue(os.path.exists(correlations_csv_path))
            self.assertTrue(os.path.exists(correlations_bin_path))

            # check correlations csv content
            correlations_csv_data = self.read_file(correlations_csv_path)
            self.assertEqual(EXPECTED_CORRELATION_CSV, correlations_csv_data)

    @patch('oasislmf.computation.generate.files.GenerateFiles._get_output_dir')
    def test_files__given_new_correlation_settings__correlation_csv_file_created(self, mock_output_dir):
        NEW_CORRELATIONS_SETTINGS = {
            "version": "3",
            "model_settings": {
                "correlation_settings": [
                    {"peril_correlation_group": 1, "damage_correlation_value": "0.7", "hazard_correlation_value": "0.4"},
                    {"peril_correlation_group": 2, "damage_correlation_value": "0.5", "hazard_correlation_value": "0.2"}
                ],
            },
            "lookup_settings": {
                "supported_perils": [
                    {"id": "WSS", "desc": "Single Peril: Storm Surge", "peril_correlation_group": 1},
                    {"id": "WTC", "desc": "Single Peril: Tropical Cyclone", "peril_correlation_group": 2},
                    {"id": "WW1", "desc": "Group Peril: Windstorm with storm surge"},
                    {"id": "WW2", "desc": "Group Peril: Windstorm w/o storm surge"}
                ]
            },
            "model_default_samples": 10,
            "data_settings": {
                "damage_group_fields": ["PortNumber", "AccNumber", "LocNumber"],
                "hazard_group_fields": ["PortNumber", "AccNumber", "LocNumber"]
            }
        }
        model_settings_file = self.tmp_files.get('model_settings_json')
        self.write_json(model_settings_file, NEW_CORRELATIONS_SETTINGS)

        with self.tmp_dir() as t_dir:
            # Run
            expected_run_dir = os.path.join(t_dir, 'runs', 'files-TIMESTAMP')
            mock_output_dir.return_value = expected_run_dir
            call_args = {
                **self.ri_args,
                'model_settings_json': model_settings_file.name
            }
            file_gen_return = self.manager.generate_files(**call_args)

            # check correlations files exist
            correlations_csv_path = os.path.join(expected_run_dir, 'correlations.csv')
            correlations_bin_path = os.path.join(expected_run_dir, 'correlations.bin')
            self.assertTrue(os.path.exists(correlations_csv_path))
            self.assertTrue(os.path.exists(correlations_bin_path))

            # check correlations csv content
            correlations_csv_data = self.read_file(correlations_csv_path)
            self.assertEqual(EXPECTED_CORRELATION_CSV, correlations_csv_data)


class TestGenDummyModelFiles(ComputationChecker):

    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()
        cls.default_args = cls.manager._params_generate_dummy_model_files()

    def setUp(self):
        self.required_args = [
            'num_vulnerabilities',
            'num_intensity_bins',
            'num_damage_bins',
            'num_events',
            'num_areaperils',
            'num_periods',
        ]
        self.min_args = {k: 10 for k in self.required_args}

    def test_gen_model__required_args_misisng__exception_raised(self):
        with self.assertRaises(OasisException) as context:
            self.manager.generate_dummy_model_files()
        expected_err_msg = f'parameter num_vulnerabilities is required'
        self.assertIn(expected_err_msg, str(context.exception))

    def test_gen_model__min_args(self):
        with self.tmp_dir() as t_dir:
            with setcwd(t_dir):
                self.manager.generate_dummy_model_files(**self.min_args)

    def test_gen_model__target_dir_set__files_created(self):
        with self.tmp_dir() as t_dir:
            call_args = {**self.min_args, 'target_dir': t_dir}
            self.manager.generate_dummy_model_files(**call_args)

            expected_files = [
                os.path.join(t_dir, 'input', 'events.bin'),
                os.path.join(t_dir, 'input', 'occurrence.bin'),
                os.path.join(t_dir, 'static', 'damage_bin_dict.bin'),
                os.path.join(t_dir, 'static', 'footprint.bin'),
                os.path.join(t_dir, 'static', 'footprint.idx'),
                os.path.join(t_dir, 'static', 'vulnerability.bin'),
            ]
            for filepath in expected_files:
                self.assertTrue(os.path.isfile(filepath))
                self.assertTrue(os.path.getsize(filepath) > 0)

    def test_gen_model__num_randoms_set__file_created(self):
        with self.tmp_dir() as t_dir:
            call_args = {**self.min_args, 'target_dir': t_dir, 'num_randoms': 10}
            self.manager.generate_dummy_model_files(**call_args)
            expected_file = os.path.join(t_dir, 'static', 'random.bin')
            self.assertTrue(os.path.isfile(expected_file))
            self.assertTrue(os.path.getsize(expected_file) > 0)

    def test_validate__vulnerability_sparseness__exception_raised(self):
        with self.tmp_dir() as t_dir:
            with self.assertRaises(OasisException) as context:
                call_args = {**self.min_args, 'target_dir': t_dir, 'vulnerability_sparseness': 2.0}
                self.manager.generate_dummy_model_files(**call_args)
        expected_err_msg = 'Invalid value for --vulnerability-sparseness'
        self.assertIn(expected_err_msg, str(context.exception))

    def test_validate__intensity_sparsenes__exception_raised(self):
        with self.tmp_dir() as t_dir:
            with self.assertRaises(OasisException) as context:
                call_args = {**self.min_args, 'target_dir': t_dir, 'intensity_sparseness': 2.0}
                self.manager.generate_dummy_model_files(**call_args)
        expected_err_msg = 'Invalid value for --intensity-sparseness'
        self.assertIn(expected_err_msg, str(context.exception))

    def test_validate__areaperils_per_event__exception_raised(self):
        with self.tmp_dir() as t_dir:
            with self.assertRaises(OasisException) as context:
                call_args = {**self.min_args, 'target_dir': t_dir, 'areaperils_per_event': 20}
                self.manager.generate_dummy_model_files(**call_args)
        expected_err_msg = 'Number of areaperils per event exceeds total number of areaperils'
        self.assertIn(expected_err_msg, str(context.exception))

    def test_validate__num_amplifications__exception_raised(self):
        with self.tmp_dir() as t_dir:
            with self.assertRaises(OasisException) as context:
                call_args = {**self.min_args, 'target_dir': t_dir, 'num_amplifications': -1}
                self.manager.generate_dummy_model_files(**call_args)
        expected_err_msg = 'Invalid value for --num-amplifications'
        self.assertIn(expected_err_msg, str(context.exception))

    def test_validate__max_and_min_pla_factors__exception_raised(self):
        with self.tmp_dir() as t_dir:
            with self.assertRaises(OasisException) as context:
                call_args = {**self.min_args, 'target_dir': t_dir, 'min_pla_factor': 2.0, 'max_pla_factor': 1.0}
                self.manager.generate_dummy_model_files(**call_args)
        expected_err_msg = 'Value for --max-pla-factor must be greater than that for --min-pla-factor'
        self.assertIn(expected_err_msg, str(context.exception))

    def test_validate__min_pla_factor__exception_raised(self):
        with self.tmp_dir() as t_dir:
            with self.assertRaises(OasisException) as context:
                call_args = {**self.min_args, 'target_dir': t_dir, 'min_pla_factor': -1.0}
                self.manager.generate_dummy_model_files(**call_args)
        expected_err_msg = 'Invalid value for --min-pla-factor'
        self.assertIn(expected_err_msg, str(context.exception))

    def test_validate__random_seed__exception_raised(self):
        with self.tmp_dir() as t_dir:
            with self.assertRaises(OasisException) as context:
                call_args = {**self.min_args, 'target_dir': t_dir, 'random_seed': -10}
                self.manager.generate_dummy_model_files(**call_args)
        expected_err_msg = 'Invalid random seed'
        self.assertIn(expected_err_msg, str(context.exception))


class TestGenDummyOasisFiles(ComputationChecker):

    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()
        cls.default_args = cls.manager._params_generate_dummy_oasis_files()

    def setUp(self):
        self.required_args = [
            'num_vulnerabilities',
            'num_intensity_bins',
            'num_damage_bins',
            'num_events',
            'num_areaperils',
            'num_periods',
            'num_amplifications',
        ]
        self.min_args = {k: 10 for k in self.required_args}
        self.min_args['min_pla_factor'] = 0.2
        self.min_args['max_pla_factor'] = 0.6
        self.min_args['num_locations'] = 10000
        self.min_args['coverages_per_location'] = 4
        self.min_args['num_layers'] = 5

    def test_gen_oasis_files__required_args_misisng__exception_raised(self):
        with self.assertRaises(OasisException) as context:
            self.manager.generate_dummy_oasis_files()
        expected_err_msg = 'parameter num_locations is required'
        self.assertIn(expected_err_msg, str(context.exception))

    def test_gen_oasis_files__min_args(self):
        with self.tmp_dir() as t_dir:
            with setcwd(t_dir):
                self.manager.generate_dummy_oasis_files(**self.min_args)

    def test_gen_oasis_files__target_dir_set__files_created(self):
        with self.tmp_dir() as t_dir:
            call_args = {**self.min_args, 'target_dir': t_dir}
            self.manager.generate_dummy_oasis_files(**call_args)
            expected_files = [
                os.path.join(t_dir, 'input', 'coverages.bin'),
                os.path.join(t_dir, 'input', 'events.bin'),
                os.path.join(t_dir, 'input', 'fm_policytc.bin'),
                os.path.join(t_dir, 'input', 'fm_profile.bin'),
                os.path.join(t_dir, 'input', 'fm_programme.bin'),
                os.path.join(t_dir, 'input', 'fmsummaryxref.bin'),
                os.path.join(t_dir, 'input', 'fm_xref.bin'),
                os.path.join(t_dir, 'input', 'gulsummaryxref.bin'),
                os.path.join(t_dir, 'input', 'items.bin'),
                os.path.join(t_dir, 'input', 'occurrence.bin'),
                os.path.join(t_dir, 'static', 'damage_bin_dict.bin'),
                os.path.join(t_dir, 'static', 'footprint.bin'),
                os.path.join(t_dir, 'static', 'footprint.idx'),
                os.path.join(t_dir, 'static', 'vulnerability.bin'),
            ]
            for filepath in expected_files:
                self.assertTrue(os.path.isfile(filepath))
                self.assertTrue(os.path.getsize(filepath) > 0)

    def test_validate__coverages_per_location__exception_raised(self):
        with self.tmp_dir() as t_dir:
            with self.assertRaises(OasisException) as context:
                call_args = {**self.min_args, 'target_dir': t_dir}
                call_args['coverages_per_location'] = 20
                self.manager.generate_dummy_oasis_files(**call_args)
        expected_err_msg = 'Number of supported coverage types is 1 to 4'
        self.assertIn(expected_err_msg, str(context.exception))
