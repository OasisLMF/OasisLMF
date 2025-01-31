import logging
import pathlib
import os

from unittest import mock
from unittest.mock import patch, Mock

from ods_tools.oed.common import OdsException
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.path import setcwd
from oasislmf.manager import OasisManager
from .data.common import *
from .test_computation import ComputationChecker


TEST_DIR = pathlib.Path(os.path.realpath(__file__)).parent.parent


class TestGenKeys(ComputationChecker):

    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()
        cls.default_args = cls.manager._params_generate_keys()

    def setUp(self):
        # Tempfiles
        self.tmp_dirs = self.create_tmp_dirs([a for a in self.default_args.keys() if 'dir' in a])
        self.tmp_files = self.create_tmp_files(
            [a for a in self.default_args.keys() if 'csv' in a] +
            [a for a in self.default_args.keys() if 'path' in a] +
            [a for a in self.default_args.keys() if 'json' in a]
        )

        self.min_args = {
            'lookup_module_path': FAKE_COMPLEX_LOOKUP_MODULE,
            'lookup_data_dir': self.tmp_dirs['lookup_data_dir'].name,
            'oed_location_csv': self.tmp_files['oed_location_csv'].name,
            'model_version_csv': self.tmp_files['model_version_csv'].name,

        }
        self.min_args_output_set = {
            **self.min_args,
            'keys_data_csv': self.tmp_files['keys_data_csv'].name,
            'keys_errors_csv': self.tmp_files['keys_errors_csv'].name,
        }
        self.write_str(self.tmp_files.get('oed_location_csv'), MIN_LOC)
        self.write_json(self.tmp_files.get('lookup_complex_config_json'), MIN_RUN_SETTINGS)
        self.write_str(self.tmp_files.get('model_version_csv'), "Supplier,FakeComplexModel,0.0.0.1")

    def test_keys__check_return(self):
        expected_return = (self.min_args_output_set['keys_data_csv'], 4, self.min_args_output_set['keys_errors_csv'], 0)
        keys_return = self.manager.generate_keys(**self.min_args_output_set)
        keys_csv_data = self.read_file(self.min_args_output_set['keys_data_csv'])
        error_csv_data = self.read_file(self.min_args_output_set['keys_errors_csv'])

        self.assertEqual(keys_csv_data, EXPECTED_KEYS_COMPLEX)
        self.assertEqual(error_csv_data, EXPECTED_ERROR_COMPLEX)
        self.assertEqual(keys_return, expected_return)

    def test__logger_gets_output(self):
        with self._caplog.at_level(logging.INFO):
            keys_return = self.manager.generate_keys(**self.min_args_output_set)
        self.assertIn('Log Message From FakeComplexModelKeysLookup', self._caplog.text)

    def test_keys__lookup_complex_config_json__is_valid(self):
        lookup_complex_config_file = self.tmp_files['lookup_complex_config_json']
        call_args = {
            **self.min_args_output_set,
            'lookup_complex_config_json': lookup_complex_config_file.name}
        keys_return = self.manager.generate_keys(**call_args)

    def test_keys__lookup_complex_config_json__is_invalid(self):
        lookup_complex_config_file = self.tmp_files['lookup_complex_config_json']
        self.write_json(lookup_complex_config_file, {})
        call_args = {
            **self.min_args_output_set,
            'lookup_complex_config_json': lookup_complex_config_file.name}
        with self.assertRaises(OdsException) as context:
            keys_return = self.manager.generate_keys(**call_args)
        expected_err_msg = f'JSON Validation error'
        self.assertIn(expected_err_msg, str(context.exception))

    def test_keys__missing_lookup__execption_raised(self):
        with self.assertRaises(OasisException) as context:
            keys_return = self.manager.generate_keys()

        expected_err_msg = 'No pre-generated keys file provided, and no lookup assets provided'
        self.assertIn(expected_err_msg, str(context.exception))

    def test_keys__no_output_files_given(self):
        with self.tmp_dir() as t_dir:
            with setcwd(t_dir):
                keys_fp, _, error_fp, _ = self.manager.generate_keys(**self.min_args)
                self.assertTrue(keys_fp.startswith(t_dir))
                self.assertTrue(keys_fp.endswith('keys.csv'))
                self.assertTrue(error_fp.startswith(t_dir))
                self.assertTrue(error_fp.endswith('keys-errors.csv'))

    @patch('oasislmf.computation.generate.keys.KeyServerFactory.create')
    @patch('oasislmf.computation.generate.keys.get_exposure_data')
    def test_args__passed_correctly(self, mock_get_exposure, mock_keys_factory):
        key_server_mock = Mock()
        key_server_mock.generate_key_files.return_value = (
            self.min_args_output_set['keys_data_csv'], 4,
            self.min_args_output_set['keys_errors_csv'], 2)

        mock_keys_factory.return_value = ('model_info', key_server_mock)
        exposure_data = Mock()
        mock_get_exposure.return_value = exposure_data

        call_args = self.combine_args([
            {k: v.name for k, v in self.tmp_dirs.items()},
            {k: v.name for k, v in self.tmp_files.items()},
            self.default_args])

        call_args['model_settings_json'] = FAKE_MODEL_SETTINGS_JSON
        self.manager.generate_keys(**call_args)

        mock_keys_factory.assert_called_once_with(
            lookup_config_fp=call_args['lookup_config_json'],
            model_keys_data_path=call_args['lookup_data_dir'],
            model_version_file_path=call_args['model_version_csv'],
            lookup_module_path=call_args['lookup_module_path'],
            complex_lookup_config_fp=call_args['lookup_complex_config_json'],
            user_data_dir=call_args['user_data_dir'],
            output_directory=os.path.dirname(call_args['keys_data_csv'])
        )

        key_server_mock.generate_key_files.assert_called_once_with(
            location_df=exposure_data.get_subject_at_risk_source().dataframe,
            successes_fp=call_args['keys_data_csv'],
            errors_fp=call_args['keys_errors_csv'],
            format=call_args['keys_format'],
            keys_success_msg=True,
            multiproc_enabled=call_args['lookup_multiprocessing'],
            multiproc_num_cores=call_args['lookup_num_processes'],
            multiproc_num_partitions=call_args['lookup_num_chunks'],
        )
