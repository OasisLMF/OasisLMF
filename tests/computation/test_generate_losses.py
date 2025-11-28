
import pathlib
import os
import logging
import shutil

from unittest.mock import patch, Mock

from hypothesis import given
from hypothesis import strategies as st

from oasislmf.utils.exceptions import OasisException
from oasislmf.manager import OasisManager
from .data.common import (
    EXPECTED_SUMMARY_INFO_CSV, MIN_RUN_SETTINGS, MIN_LOC, MIN_ACC, MIN_INF, MIN_SCP, MIN_KEYS, MIN_KEYS_ERR, IL_RUN_SETTINGS, RI_RUN_SETTINGS,
    RI_ALL_OUTPUT_SETTINGS, ALL_EXPECTED_SCRIPT, FAKE_MODEL_RUNNER, FAKE_MODEL_RUNNER__OLD, INVALID_RUN_SETTINGS, RI_AAL_SETTINGS,
    PARQUET_GUL_SETTINGS, MIN_MODEL_SETTINGS, merge_dirs
)
from .test_computation import ComputationChecker
from oasislmf.computation.generate.losses import GenerateLosses

TEST_DIR = pathlib.Path(os.path.realpath(__file__)).parent.parent
LOOKUP_CONFIG = TEST_DIR.joinpath('model_preparation').joinpath('meta_data').joinpath('lookup_config.json')
ANALYSIS_SETTINGS = TEST_DIR.joinpath('model_preparation').joinpath('meta_data').joinpath('analysis_settings.json')

summary_types = ['summarycalc', 'summarypy']


class TestGenLosses(ComputationChecker):

    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()
        cls.default_args = cls.manager._params_generate_losses()
        cls.files_args = cls.manager._params_generate_files()

    def setUp(self):
        # Create Tempfiles (losses)
        self.tmp_dirs = self.create_tmp_dirs([a for a in self.default_args.keys() if 'dir' in a])
        self.tmp_files = self.create_tmp_files(
            [a for a in self.default_args.keys() if 'csv' in a] +
            [a for a in self.default_args.keys() if 'path' in a] +
            [a for a in self.default_args.keys() if 'json' in a]
        )
        # Create Tempfiles (files)
        self.tmp_oasis_files = self.create_tmp_files(
            [a for a in self.files_args.keys() if 'csv' in a] +
            [a for a in self.files_args.keys() if 'path' in a] +
            [a for a in self.files_args.keys() if 'json' in a]
        )
        # tmp dirs for holding oasis files
        self.tmp_oasis_files_dirs = self.create_tmp_dirs([
            'oasis_files_dir__gul',
            'oasis_files_dir__il',
            'oasis_files_dir__ri',
        ])
        # Write minimum data to gen files
        self.write_json(self.tmp_oasis_files.get('lookup_complex_config_json'), MIN_RUN_SETTINGS)
        self.write_str(self.tmp_oasis_files.get('oed_location_csv'), MIN_LOC)
        self.write_str(self.tmp_oasis_files.get('oed_accounts_csv'), MIN_ACC)
        self.write_str(self.tmp_oasis_files.get('oed_info_csv'), MIN_INF)
        self.write_str(self.tmp_oasis_files.get('oed_scope_csv'), MIN_SCP)
        self.write_str(self.tmp_oasis_files.get('keys_data_csv'), MIN_KEYS)
        self.write_str(self.tmp_oasis_files.get('keys_errors_csv'), MIN_KEYS_ERR)

        # Args for generating sample data
        self.args_gen_files_gul = {
            'lookup_config_json': LOOKUP_CONFIG,
            'oed_location_csv': self.tmp_oasis_files['oed_location_csv'].name,
            'oasis_files_dir': self.tmp_oasis_files_dirs.get('oasis_files_dir__gul').name
        }
        self.args_gen_files_il = {
            **self.args_gen_files_gul,
            'oed_accounts_csv': self.tmp_oasis_files['oed_accounts_csv'].name,
            'oasis_files_dir': self.tmp_oasis_files_dirs.get('oasis_files_dir__il').name
        }
        self.args_gen_files_ri = {
            **self.args_gen_files_il,
            'oed_info_csv': self.tmp_oasis_files['oed_info_csv'].name,
            'oed_scope_csv': self.tmp_oasis_files['oed_scope_csv'].name,
            'oasis_files_dir': self.tmp_oasis_files_dirs.get('oasis_files_dir__ri').name
        }

        # args for generating model data
        self.required_args_model_data = [
            'num_vulnerabilities',
            'num_intensity_bins',
            'num_damage_bins',
            'num_events',
            'num_areaperils',
            'num_periods',
            'num_amplifications'
        ]
        self.args_model_data = {k: 10 for k in self.required_args_model_data}

        # populate model data dir
        self.model_data_dir = self.tmp_dirs.get('model_data_dir').name
        self.manager.generate_dummy_model_files(**{
            **self.args_model_data,
            'target_dir': self.model_data_dir
        })
        for src_file in pathlib.Path(self.model_data_dir).glob('*/*.*'):
            shutil.copy(src_file, self.model_data_dir)
        shutil.rmtree(pathlib.Path(self.model_data_dir).joinpath('static'))
        shutil.rmtree(pathlib.Path(self.model_data_dir).joinpath('input'))

        self.min_args = {
            'analysis_settings_json': self.tmp_files.get('analysis_settings_json').name,
            'oasis_files_dir': self.args_gen_files_gul['oasis_files_dir'],
            'model_data_dir': self.model_data_dir,
            'model_run_dir': self.tmp_dirs.get('model_run_dir').name,
        }

    def test_losses__no_input__exception_raised(self):
        with (self.assertRaises(OasisException) as context,
              patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10000"})):
            # Patch for port ensures when tests run side by side port not in use by other test
            self.manager.generate_losses()
        expected_err_msg = 'parameter oasis_files_dir is required'
        self.assertIn(expected_err_msg, str(context.exception))

    def test_losses__run_gul(self):
        self.write_json(self.tmp_files.get('analysis_settings_json'), MIN_RUN_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)
        with patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10001"}):
            self.manager.generate_losses(**self.min_args)

    def test_losses__run_gul_with_pla(self):
        PLA_SETTINGS = MIN_RUN_SETTINGS.copy()
        PLA_SETTINGS['pla'] = True
        PLA_SETTINGS['pla_secondary_factor'] = 0.5

        self.write_json(self.tmp_files.get('analysis_settings_json'), PLA_SETTINGS)
        try:
            self.manager.generate_files(**self.args_gen_files_gul)
            with patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10002"}):
                self.manager.generate_losses(**self.min_args)
        except Exception:
            print(os.path.join(self.tmp_dirs['model_run_dir'].name, 'log', 'stderror.err'))
            print(open(os.path.join(self.tmp_dirs['model_run_dir'].name, 'log', 'stderror.err')).read())
            raise

    def test_losses__run_gul_with_oed(self):
        OED_SETTINGS = MIN_RUN_SETTINGS.copy()
        gul_summary = {
            **OED_SETTINGS['gul_summaries'][0],
            'oed_fields': ['LocNumber', 'AccNumber', 'PolNumber']
        }
        OED_SETTINGS['gul_summaries'] = [gul_summary]

        gen_args = {
            **self.args_gen_files_gul,
            'oed_accounts_csv': self.tmp_oasis_files['oed_accounts_csv'].name  # accounts file for PolNumber
        }
        self.write_json(self.tmp_files.get('analysis_settings_json'), OED_SETTINGS)
        self.manager.generate_files(**gen_args)
        with patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10003"}):
            self.manager.generate_losses(**self.min_args)

    def test_losses__summary_info(self):
        OED_SETTINGS = MIN_RUN_SETTINGS.copy()
        gul_summary = {
            **OED_SETTINGS['gul_summaries'][0],
            'oed_fields': ['LocNumber', 'AccNumber', 'PolNumber', 'AccCurrency']
        }
        gul_summary['eltcalc'] = False
        OED_SETTINGS['gul_summaries'] = [gul_summary]
        gen_args = {
            **self.args_gen_files_gul,
            'oed_accounts_csv': self.tmp_oasis_files['oed_accounts_csv'].name  # accounts file for PolNumber, AccCurrency
        }
        self.write_json(self.tmp_files.get('analysis_settings_json'), OED_SETTINGS)
        self.manager.generate_files(**gen_args)
        with patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10003"}):
            self.manager.generate_losses(**self.min_args)

        # Verify correctness of summary-info files
        summary_info_fpath = os.path.join(self.tmp_dirs.get('model_run_dir').name, 'output', 'gul_S1_summary-info.csv')
        summary_info_csv = self.read_file(summary_info_fpath)

        assert summary_info_csv == EXPECTED_SUMMARY_INFO_CSV

    def test_losses__run_il(self):
        self.manager.generate_files(**self.args_gen_files_il)
        run_settings = self.tmp_files.get('analysis_settings_json')
        self.write_json(run_settings, IL_RUN_SETTINGS)
        call_args = {
            **self.min_args,
            'oasis_files_dir': self.args_gen_files_il['oasis_files_dir'],
        }
        with patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10004"}):
            self.manager.generate_losses(**call_args)

    def test_losses__run_ri(self):
        self.manager.generate_files(**self.args_gen_files_ri)
        run_settings = self.tmp_files.get('analysis_settings_json')
        self.write_json(run_settings, RI_RUN_SETTINGS)
        call_args = {
            **self.min_args,
            'oasis_files_dir': self.args_gen_files_ri['oasis_files_dir'],
        }
        with patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10005"}):
            self.manager.generate_losses(**call_args)

    @patch('oasislmf.computation.hooks.post_analysis.PostAnalysis.run')
    def test_losses__run__post_analysis_is_called(self, mock_post_analysis):
        mock_post_analysis.__name__ = "run"
        mock_post_analysis.__globals__ = {'__name__': "oasislmf.computation.hooks.post_analysis.PostAnalysis"}
        self.write_json(self.tmp_files.get('analysis_settings_json'), MIN_RUN_SETTINGS)

        with self.tmp_dir() as tmp_module_dir:
            fake_module_path = pathlib.Path(tmp_module_dir, 'empty_module.py')
            fake_module_path.touch()
            call_args = {
                **self.min_args,
                'post_analysis_module': str(fake_module_path),
                'post_analysis_class_name': 'missing_class',
            }
            self.manager.generate_files(**self.args_gen_files_gul)
            self.manager.generate_oasis_losses(**call_args)
            self.assertTrue(mock_post_analysis.called)

    @patch('subprocess.check_output')
    def test_losses__run_ri__all_outputs__check_bash_script(self, sub_process_run):
        for summary_type in summary_types:
            with self.tmp_dir() as model_run_dir, self.subTest(summary_type):
                self.manager.generate_files(summarypy=summary_type == 'summarypy', **self.args_gen_files_ri)
                run_settings = self.tmp_files.get('analysis_settings_json')
                self.write_json(run_settings, RI_ALL_OUTPUT_SETTINGS)
                call_args = {
                    **self.min_args,
                    'ktools_num_processes': 2,
                    'ktools_fifo_relative': True,
                    'oasis_files_dir': self.args_gen_files_ri['oasis_files_dir'],
                    'model_run_dir': model_run_dir,
                    'summarypy': summary_type == 'summarypy'
                }
                with patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10006"}):
                    self.manager.generate_losses(**call_args)

                # Check bash script vs reference
                self.assertTrue(sub_process_run.called)
                bash_script_path = sub_process_run.call_args.args[0][1]
                result_script = self.read_file(bash_script_path).decode()
                expected_script = self.read_file(ALL_EXPECTED_SCRIPT.format(summary_type)).decode()
                self.assertEqual(expected_script, result_script)

    def test_losses__chucked_workflow(self):
        num_chunks = 5
        self.manager.generate_files(**self.args_gen_files_ri)
        run_settings = self.tmp_files.get('analysis_settings_json')
        self.write_json(run_settings, RI_RUN_SETTINGS)

        with self.tmp_dir() as model_run_dir:
            call_args = {
                **self.min_args,
                'oasis_files_dir': self.args_gen_files_ri['oasis_files_dir'],
                'max_process_id': num_chunks,
                'model_run_dir': model_run_dir,
            }
            self.manager.generate_losses_dir(**call_args)
            main_work_dir = os.path.join(model_run_dir, 'work')

            for i in range(1, num_chunks + 1):
                chunk_args = {
                    **call_args,
                    'process_number': i,
                    'analysis_pk': 1
                }
                with patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10007"}):
                    self.manager.generate_losses_partial(**chunk_args)
                chunk_bash_path = os.path.join(model_run_dir, f'{i}.run_analysis.sh')
                chunk_work_dir = os.path.join(model_run_dir, f'{i}.work')

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
        with patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10008"}):
            self.manager.generate_losses(**call_args)
        mock_run_func.assert_called_once()

    @patch('oasislmf.execution.runner.run')
    def test_losses__supplier_model_runner_old(self, mock_run_func):
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
    def test_losses__ktools_alloc_set_invalid(self, mock_run_func, gul_alloc, il_alloc, ri_alloc, event_shuffle, gulpy_random_generator):
        if any([gul_alloc, il_alloc, ri_alloc, event_shuffle, gulpy_random_generator]):
            call_args = {
                **self.min_args,
                'analysis_settings_json': ANALYSIS_SETTINGS,
                'ktools_alloc_rule_gul': gul_alloc,
                'ktools_alloc_rule_il': il_alloc,
                'ktools_alloc_rule_ri': ri_alloc,
                'ktools_event_shuffle': event_shuffle,
                'gulpy_random_generator': gulpy_random_generator
            }
            with (self.assertRaises(OasisException) as context,
                  patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10010"})):
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
        with self.tmp_dir() as model_run_dir:
            self.write_json(self.tmp_files.get('analysis_settings_json'), INVALID_RUN_SETTINGS)
            self.manager.generate_files(**self.args_gen_files_gul)
            with (self.assertRaises(OasisException) as context,
                  patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10012"})):
                call_args = {**self.min_args, 'model_run_dir': model_run_dir}
                self.manager.generate_losses(**call_args)
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
            with (self.assertRaises(OasisException) as context,
                  patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10013"})):
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
            with (self.assertRaises(OasisException) as context,
                  patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10014"})):
                self.manager.generate_losses_partial(**call_args)
            expected_error = 'Ktools run Error: non-zero exit code or error/warning messages detected'
            self.assertIn(expected_error, str(context.exception))
        self.assertIn('BASH_TRACE:', self._caplog.text)
        self.assertIn('KTOOLS_STDERR:', self._caplog.text)
        self.assertIn('GUL_STDERR', self._caplog.text)
        self.assertIn('STDOUT:', self._caplog.text)

    def test_losses__bash_error__exception_raised__outputs(self):
        self.manager.generate_files(**self.args_gen_files_ri)
        run_settings = self.tmp_files.get('analysis_settings_json')
        self.write_json(run_settings, RI_AAL_SETTINGS)
        run_dir = self.tmp_dirs.get('model_run_dir').name
        call_args = {
            **self.min_args,
            'oasis_files_dir': self.args_gen_files_ri['oasis_files_dir'],
            'verbose': True,
            'max_process_id': 1,
            'model_run_dir': run_dir
        }
        self.manager.generate_losses_dir(**call_args)

        chunk_args = {
            **call_args,
            'process_number': 1,
            'analysis_pk': 1
        }
        with patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10015"}):
            self.manager.generate_losses_partial(**chunk_args)
        with self.assertRaises(OasisException) as context:
            self.manager.generate_losses_output(**call_args)
        expected_error = 'Ktools run Error: non-zero exit code or error/warning messages detected'
        self.assertIn(expected_error, str(context.exception))

    @patch('oasislmf.execution.runner.run')
    @patch('oasislmf.computation.generate.losses.subprocess.run')
    def test_losses__parquet_output__supported(self, mock_subprocess, mock_runner):
        self.write_json(self.tmp_files.get('analysis_settings_json'), PARQUET_GUL_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)

        mock_check_parquet = Mock()
        mock_check_parquet.stderr.decode.return_value = 'Parquet output enabled'
        mock_subprocess.return_value = mock_check_parquet

        with patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10016"}):
            self.manager.generate_losses(**self.min_args)
        mock_runner.assert_called_once()

    @patch('oasislmf.execution.runner.run')
    @patch('oasislmf.computation.generate.losses.subprocess.run')
    def test_losses__parquet_output__unsupported(self, mock_subprocess, mock_runner):
        self.write_json(self.tmp_files.get('analysis_settings_json'), PARQUET_GUL_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)

        mock_check_parquet = Mock()
        mock_check_parquet.stderr.decode.return_value = 'Parquet output disabled'
        mock_subprocess.return_value = mock_check_parquet

        with (self.assertRaises(OasisException) as context,
              patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10017"})):
            self.manager.generate_losses(**self.min_args)
        expected_error = 'Parquet output format requested but not supported by ktools components.'
        self.assertIn(expected_error, str(context.exception))
        mock_runner.assert_not_called()

    def test_losses__il_files_missing__expection_raised(self):
        self.write_json(self.tmp_files.get('analysis_settings_json'), RI_RUN_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)

        call_args = {
            **self.min_args,
            'check_missing_inputs': True,
        }
        with (self.assertRaises(OasisException) as context,
              patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10018"})):
            self.manager.generate_losses(**call_args)
        expected_error = "[\'IL\', \'RI\', \'RL\'] are enabled in the analysis_settings without the generated input files"
        self.assertIn(expected_error, str(context.exception))

    def test_losses__il_files_missing__output_skipped(self):
        self.write_json(self.tmp_files.get('analysis_settings_json'), RI_RUN_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)

        with (self._caplog.at_level(logging.WARN),
              patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10019"})):
            self.manager.generate_losses(**self.min_args)
        expected_warning = "[\'IL\', \'RI\', \'RL\'] are enabled in the analysis_settings without the generated input files"
        self.assertIn(expected_warning, self._caplog.text)

    def test_losses__no_samples_set__expection_raised(self):
        NO_SAMPLES_SETTINGS = MIN_RUN_SETTINGS.copy()
        NO_SAMPLES_SETTINGS.pop('number_of_samples')

        self.write_json(self.tmp_files.get('analysis_settings_json'), NO_SAMPLES_SETTINGS)
        self.manager.generate_files(**self.args_gen_files_gul)
        with (self.assertRaises(OasisException) as context,
              patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10020"})):
            self.manager.generate_losses(**self.min_args)
        expected_error = "'number_of_samples' not set in analysis_settings and no model_settings.json file provided"
        self.assertIn(expected_error, str(context.exception))

    @patch('oasislmf.execution.runner.run')
    def test_losses__samples_set__in_model_settings(self, mock_runner):
        expected_samples = 250
        MODEL_SETTINGS = MIN_MODEL_SETTINGS.copy()
        MODEL_SETTINGS['model_default_samples'] = expected_samples

        NO_SAMPLES_SETTINGS = MIN_RUN_SETTINGS.copy()
        NO_SAMPLES_SETTINGS.pop('number_of_samples')

        self.write_json(self.tmp_files.get('model_settings_json'), MODEL_SETTINGS)
        self.write_json(self.tmp_files.get('analysis_settings_json'), NO_SAMPLES_SETTINGS)

        call_args = {**self.min_args, 'model_settings_json': self.tmp_files.get('model_settings_json').name}
        self.manager.generate_files(**self.args_gen_files_gul)
        with patch.dict(os.environ, {"OASIS_SOCKET_SERVER_PORT": "10021"}):
            self.manager.generate_losses(**call_args)

        called_settings = mock_runner.call_args.args[0]
        self.assertEqual(expected_samples, called_settings['number_of_samples'])
