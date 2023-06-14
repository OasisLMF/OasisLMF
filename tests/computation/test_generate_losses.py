
import pathlib
import os
import logging
import shutil

from unittest import mock
from unittest.mock import patch, Mock, ANY

from hypothesis import given, settings
from hypothesis import strategies as st

from ods_tools.oed.common import OdsException
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.path import setcwd
from oasislmf.manager import OasisManager
from .data.common import *
from .test_computation import ComputationChecker


TEST_DIR = pathlib.Path(os.path.realpath(__file__)).parent.parent
LOOKUP_CONFIG = TEST_DIR.joinpath('model_preparation').joinpath('meta_data').joinpath('lookup_config.json')


class TestGenLosses(ComputationChecker):

    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()
        cls.default_args = cls.manager._params_generate_losses()
        cls.files_args = cls.manager._params_generate_files()

        # Create Tempfiles (losses)
        cls.tmp_dirs = cls.create_tmp_dirs([a for a in cls.default_args.keys() if 'dir' in a])
        cls.tmp_files = cls.create_tmp_files(
            [a for a in cls.default_args.keys() if 'csv' in a] +
            [a for a in cls.default_args.keys() if 'path' in a] +
            [a for a in cls.default_args.keys() if 'json' in a]
        )
        # Create Tempfiles (files)
        cls.tmp_oasis_files = cls.create_tmp_files(
            [a for a in cls.files_args.keys() if 'csv' in a] +
            [a for a in cls.files_args.keys() if 'path' in a] +
            [a for a in cls.files_args.keys() if 'json' in a]
        )
        # tmp dirs for holding oasis files
        cls.tmp_oasis_files_dirs = cls.create_tmp_dirs([
            'oasis_files_dir__gul',
            'oasis_files_dir__il',
            'oasis_files_dir__ri',
        ])
        # Write minimum data to gen files
        cls.write_json(cls.tmp_oasis_files.get('lookup_complex_config_json'), MIN_RUN_SETTINGS)
        cls.write_str(cls.tmp_oasis_files.get('oed_location_csv'), MIN_LOC)
        cls.write_str(cls.tmp_oasis_files.get('oed_accounts_csv'), MIN_ACC)
        cls.write_str(cls.tmp_oasis_files.get('oed_info_csv'), MIN_INF)
        cls.write_str(cls.tmp_oasis_files.get('oed_scope_csv'), MIN_SCP)
        cls.write_str(cls.tmp_oasis_files.get('keys_data_csv'), MIN_KEYS)
        cls.write_str(cls.tmp_oasis_files.get('keys_errors_csv'), MIN_KEYS_ERR)

        # Args for generating sample data
        cls.args_gen_files_gul = {
            'lookup_config_json': LOOKUP_CONFIG,
            'oed_location_csv': cls.tmp_oasis_files['oed_location_csv'].name,
            'oasis_files_dir': cls.tmp_oasis_files_dirs.get('oasis_files_dir__gul').name
        }
        cls.args_gen_files_il = {
            **cls.args_gen_files_gul,
            'oed_accounts_csv': cls.tmp_oasis_files['oed_accounts_csv'].name,
            'oasis_files_dir': cls.tmp_oasis_files_dirs.get('oasis_files_dir__il').name
        }
        cls.args_gen_files_ri = {
            **cls.args_gen_files_il,
            'oed_info_csv': cls.tmp_oasis_files['oed_info_csv'].name,
            'oed_scope_csv': cls.tmp_oasis_files['oed_scope_csv'].name,
            'oasis_files_dir': cls.tmp_oasis_files_dirs.get('oasis_files_dir__ri').name
        }

        # args for generating model data
        cls.required_args_model_data = [
            'num_vulnerabilities',
            'num_intensity_bins',
            'num_damage_bins',
            'num_events',
            'num_areaperils',
            'num_periods',
        ]
        cls.args_model_data = {k: 10 for k in cls.required_args_model_data}

        # populate model data dir
        cls.model_data_dir = cls.tmp_dirs.get('model_data_dir').name
        cls.manager.generate_dummy_model_files(**{
            **cls.args_model_data,
            'target_dir': cls.model_data_dir
        })
        for src_file in pathlib.Path(cls.model_data_dir).glob('*/*.*'):
            shutil.copy(src_file, cls.model_data_dir)
        shutil.rmtree(pathlib.Path(cls.model_data_dir).joinpath('static'))
        shutil.rmtree(pathlib.Path(cls.model_data_dir).joinpath('input'))

    def setUp(self):
        # set base args
        self.min_args = {
            'analysis_settings_json': self.tmp_files.get('analysis_settings_json').name,
            'oasis_files_dir': self.args_gen_files_gul['oasis_files_dir'],
            'model_data_dir': self.model_data_dir, 
            'model_run_dir': self.tmp_dirs.get('model_run_dir').name,
        }

    def test_losses__no_input__exception_raised(self):
        with self.assertRaises(OasisException) as context:
            self.manager.generate_losses()
        expected_err_msg = 'parameter oasis_files_dir is required'
        self.assertIn(expected_err_msg, str(context.exception))

    def test_losses__run_gul(self):
        self.write_json(self.tmp_files.get('analysis_settings_json'), MIN_RUN_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)
        self.manager.generate_losses(**self.min_args)

    def test_losses__run_il(self):
        self.manager.generate_files(**self.args_gen_files_il)
        run_settings = self.tmp_files.get('analysis_settings_json')
        self.write_json(run_settings, IL_RUN_SETTINGS)
        call_args = {
            **self.min_args,
            'oasis_files_dir': self.args_gen_files_il['oasis_files_dir'],
        }
        self.manager.generate_losses(**call_args)

    def test_losses__run_ri(self):
        self.manager.generate_files(**self.args_gen_files_ri)
        run_settings = self.tmp_files.get('analysis_settings_json')
        self.write_json(run_settings, RI_RUN_SETTINGS)
        call_args = {
            **self.min_args,
            'oasis_files_dir': self.args_gen_files_ri['oasis_files_dir'],
        }
        self.manager.generate_losses(**call_args)

    def test_losses__chucked_workflow(self):
        num_chunks = 5
        self.manager.generate_files(**self.args_gen_files_ri)
        run_settings = self.tmp_files.get('analysis_settings_json')
        self.write_json(run_settings, RI_RUN_SETTINGS)
        run_dir = self.tmp_dirs.get('model_run_dir').name
        call_args = {
            **self.min_args,
            'oasis_files_dir': self.args_gen_files_ri['oasis_files_dir'],
            'max_process_id': num_chunks,
        }
        run_settings_return = self.manager.generate_losses_dir(**call_args)
        main_work_dir = os.path.join(run_dir, 'work')

        for i in range(1, num_chunks + 1):
            chunk_args = {
                **call_args,
                'process_number': i
            }
            self.manager.generate_losses_partial(**chunk_args)
            chunk_bash_path = os.path.join(run_dir, f'{i}.run_analysis.sh')
            chunk_work_dir = os.path.join(run_dir, f'{i}.work')

            self.assertTrue(os.path.isfile(chunk_bash_path))
            self.assertTrue(len(os.listdir(chunk_work_dir)) > 0)

            # should probably be checked and called from inside tested func
            # called to merge chunk work into main work dir
            merge_dirs(chunk_work_dir, main_work_dir)

        self.manager.generate_losses_output(**call_args)

    @patch('oasislmf.execution.runner.run')
    def test_losses__supplier_model_ruuner(self, mock_run_func):
        self.write_json(self.tmp_files.get('analysis_settings_json'), MIN_RUN_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)
        call_args = {
            **self.min_args,
            'model_package_dir': FAKE_MODEL_RUNNER,
        }
        self.manager.generate_losses(**call_args)
        mock_run_func.assert_called_once()

    @patch('oasislmf.execution.runner.run')
    def test_losses__supplier_model_ruuner_old(self, mock_run_func):
        self.write_json(self.tmp_files.get('analysis_settings_json'), MIN_RUN_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)
        call_args = {
            **self.min_args,
            'model_package_dir': FAKE_MODEL_RUNNER__OLD,
        }
        self.manager.generate_losses(**call_args)
        mock_run_func.assert_called_once()

    @given(
        gul_alloc=st.sampled_from([None, 99]),
        il_alloc=st.sampled_from([None, 99]),
        ri_alloc=st.sampled_from([None, 99]),
        event_shuffle=st.sampled_from([None, 99]),
        gulpy_random_generator=st.sampled_from([None, 99])
    )
    @patch('oasislmf.execution.runner.run')
    def test_losses__ktools_alloc_set_invaild(self, mock_run_func, gul_alloc, il_alloc, ri_alloc, event_shuffle, gulpy_random_generator):
        if any([gul_alloc, il_alloc, ri_alloc, event_shuffle, gulpy_random_generator]):
            call_args = {
                **self.min_args,
                'ktools_alloc_rule_gul': gul_alloc,
                'ktools_alloc_rule_il': il_alloc,
                'ktools_alloc_rule_ri': ri_alloc,
                'ktools_event_shuffle': event_shuffle,
                'gulpy_random_generator': gulpy_random_generator
            }
            with self.assertRaises(OasisException) as context:
                self.manager.generate_losses(**call_args)
            self.assertIn('Not within valid ranges', str(context.exception))
            mock_run_func.assert_not_called()

    def test_losses__custom_gul_not_found__expection_raised(self):
        self.write_json(self.tmp_files.get('analysis_settings_json'), MIN_RUN_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)

        call_args = {
            **self.min_args,
            'model_custom_gulcalc': 'dude_wheres_my_gulcalc?',
        }
        with self.assertRaises(OasisException) as context:
            self.manager.generate_losses(**call_args)
        self.assertIn('Run error: Custom Gulcalc command', str(context.exception))

    def test_losses__invalid_events__expection_raised(self):
        self.write_json(self.tmp_files.get('analysis_settings_json'), INVALID_RUN_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)

        with self.assertRaises(OasisException) as context:
            self.manager.generate_losses(**self.min_args)
        self.assertIn('Could not find events data file:', str(context.exception))

    @patch('oasislmf.execution.bash.get_modelcmd')
    def test_losses__bash_error__expection_raised__all(self, mock_inject_bash_error):
        mock_inject_bash_error.return_value = 'exit 1'
        self.write_json(self.tmp_files.get('analysis_settings_json'), MIN_RUN_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)
        call_args = {
            **self.min_args,
            'verbose': True,
            'model_custom_gulcalc': 'gulmc',
        }
        with self._caplog.at_level(logging.INFO):
            with self.assertRaises(OasisException) as context:
                self.manager.generate_losses(**call_args)
            expected_error = 'Ktools run Error: non-zero exit code or error/warning messages detected'
            self.assertIn(expected_error, str(context.exception))
        self.assertIn('BASH_TRACE:', self._caplog.text)
        self.assertIn('KTOOLS_STDERR:', self._caplog.text)
        self.assertIn('GUL_STDERR', self._caplog.text)
        self.assertIn('STDOUT:', self._caplog.text)

    @patch('oasislmf.execution.bash.get_modelcmd')
    def test_losses__bash_error__expection_raised__single_chunk(self, mock_inject_bash_error):
        mock_inject_bash_error.return_value = 'exit 1'
        self.write_json(self.tmp_files.get('analysis_settings_json'), MIN_RUN_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)
        call_args = {
            **self.min_args,
            'verbose': True,
            'max_process_id': 250,
            'model_custom_gulcalc': 'gulmc',
            'process_number': 1
        }
        with self._caplog.at_level(logging.INFO):
            with self.assertRaises(OasisException) as context:
                self.manager.generate_losses_partial(**call_args)
            expected_error = 'Ktools run Error: non-zero exit code or error/warning messages detected'
            self.assertIn(expected_error, str(context.exception))
        self.assertIn('BASH_TRACE:', self._caplog.text)
        self.assertIn('KTOOLS_STDERR:', self._caplog.text)
        self.assertIn('GUL_STDERR', self._caplog.text)
        self.assertIn('STDOUT:', self._caplog.text)

    def test_losses__bash_error__expection_raised__outputs(self):
        self.manager.generate_files(**self.args_gen_files_ri)
        run_settings = self.tmp_files.get('analysis_settings_json')
        self.write_json(run_settings, RI_AAL_SETTINGS)
        run_dir = self.tmp_dirs.get('model_run_dir').name
        call_args = {
            **self.min_args,
            'oasis_files_dir': self.args_gen_files_ri['oasis_files_dir'],
            'verbose': True,
            'max_process_id': 1,
        }
        run_settings_return = self.manager.generate_losses_dir(**call_args)
        main_work_dir = os.path.join(run_dir, 'work')

        chunk_args = {
            **call_args,
            'process_number': 1
        }
        self.manager.generate_losses_partial(**chunk_args)
        with self.assertRaises(OasisException) as context:
            self.manager.generate_losses_output(**call_args)
        expected_error = 'Ktools run Error: non-zero exit code or error/warning messages detected'
        self.assertIn(expected_error, str(context.exception))



    # Patch return of parquet check func
    def test_losses__parquet_output__supported(self):
        self.write_json(self.tmp_files.get('analysis_settings_json'), PARQUET_GUL_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)

        call_args = {
            **self.min_args,
            'verbose': True,
        }    
        import ipdb; ipdb.set_trace()
        ret = self.manager.generate_losses(**call_args)

    def test_losses__parquet_output__unsupported(self):
        pass

    # Pass incorrect run settings vs generated files
    def test_losses__il_files_missing__expection_raised(self):
        self.write_json(self.tmp_files.get('analysis_settings_json'), RI_RUN_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)

        call_args = {
            **self.min_args,
            'check_missing_inputs': True,
        }
        with self.assertRaises(OasisException) as context:
            self.manager.generate_losses(**call_args)
        expected_error = "[\'IL\', \'RI\'] are enabled in the analysis_settings without the generated input files"
        self.assertIn(expected_error, str(context.exception))

    def test_losses__il_files_missing__output_skipped(self):
        self.write_json(self.tmp_files.get('analysis_settings_json'), RI_RUN_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)

        with self._caplog.at_level(logging.WARN):
            self.manager.generate_losses(**self.min_args)
        expected_warning = "[\'IL\', \'RI\'] are enabled in the analysis_settings without the generated input files"
        self.assertIn(expected_warning, self._caplog.text)

    # tests for default sample selection
    def test_losses__no_samples_set__expection_raised(self):
        pass
    def test_losses__samples_set__in_analysis_settings(self):
        pass
    def test_losses__samples_set__in_model_settings(self):
        pass

   # def test_losses__(self):
   #     pass
   # def test_losses__(self):
   #     pass
   # def test_losses__(self):
   #     pass
   # def test_losses__(self):
   #     pass
   # def test_losses__(self):
   #     pass
   # def test_losses__(self):
   #     pass
   # def test_losses__(self):
   #     pass
   # def test_losses__(self):
   #     pass
   # def test_losses__(self):
   #     pass
