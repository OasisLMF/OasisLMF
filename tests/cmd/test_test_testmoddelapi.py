import json
from collections import Counter
from tempfile import NamedTemporaryFile
from unittest import TestCase

import os

import shutil
import six
from hypothesis import given
from hypothesis.strategies import sampled_from, text, dictionaries, booleans, integers
from mock import Mock, ANY, patch
from backports.tempfile import mkdtemp, TemporaryDirectory
from pathlib2 import Path

from oasislmf.api_client.client import OasisAPIClient
from oasislmf.cmd import RootCmd
from oasislmf.cmd.test import TestModelApiCmd
from oasislmf.utils.exceptions import OasisException


class TestModelApiCmdLoadAnalysisSettingsJson(TestCase):
    def test_do_il_is_true___result_has_input_dict_and_do_il_is_true(self):
        conf = {
            'analysis_settings': {
                'il_output': True,
                'foo': 'bar',
            }
        }

        with NamedTemporaryFile('w') as f:
            json.dump(conf, f)
            f.flush()

            res_conf, do_il = TestModelApiCmd().load_analysis_settings_json(f.name)

            self.assertEqual(conf, res_conf)
            self.assertTrue(do_il)

    def test_do_il_is_false___result_has_input_dict_and_do_il_is_false(self):
        conf = {
            'analysis_settings': {
                'il_output': False,
                'foo': 'bar',
            }
        }

        with NamedTemporaryFile('w') as f:
            json.dump(conf, f)
            f.flush()

            res_conf, do_il = TestModelApiCmd().load_analysis_settings_json(f.name)

            self.assertEqual(conf, res_conf)
            self.assertFalse(do_il)

    @given(sampled_from(['true', 'TRUE', 'True']))
    def test_do_il_is_string_true___result_has_input_dict_and_do_il_is_true(self, do_il_in):
        conf = {
            'analysis_settings': {
                'il_output': do_il_in,
                'foo': 'bar',
            }
        }

        with NamedTemporaryFile('w') as f:
            json.dump(conf, f)
            f.flush()

            res_conf, do_il = TestModelApiCmd().load_analysis_settings_json(f.name)

            self.assertEqual(conf, res_conf)
            self.assertTrue(do_il)

    @given(sampled_from(['false', 'FALSE', 'False']))
    def test_do_il_is_string_false___result_has_input_dict_and_do_il_is_false(self, do_il_in):
        conf = {
            'analysis_settings': {
                'il_output': do_il_in,
                'foo': 'bar',
            }
        }

        with NamedTemporaryFile('w') as f:
            json.dump(conf, f)
            f.flush()

            res_conf, do_il = TestModelApiCmd().load_analysis_settings_json(f.name)

            self.assertEqual(conf, res_conf)
            self.assertFalse(do_il)


class TestModelApiCmdRunAnalysis(TestCase):
    @given(text(), text(), dictionaries(text(), text()), booleans(), integers(min_value=0, max_value=5), integers(min_value=0, max_value=5))
    def test_no_errors_are_raised___completed_is_incremented(self, input_dir, output_dir, settings, do_il, initial_complete, initial_failed):
        client = OasisAPIClient('http://localhost:8001')
        client.upload_inputs_from_directory = Mock(return_value='input_location')
        client.run_analysis_and_poll = Mock()

        counter = Counter({
            'completed': initial_complete,
            'failed': initial_failed,
        })

        TestModelApiCmd().run_analysis(client, input_dir, output_dir, settings, do_il, counter)

        self.assertEqual(initial_complete + 1, counter['completed'])
        self.assertEqual(initial_failed, counter['failed'])
        client.upload_inputs_from_directory.assert_called_once_with(input_dir, bin_directory=ANY, do_il=do_il, do_build=True)
        client.run_analysis_and_poll.assert_called_once_with(settings, 'input_location', output_dir)

    @given(text(), text(), dictionaries(text(), text()), booleans(), integers(min_value=0, max_value=5), integers(min_value=0, max_value=5))
    def test_uploading_raises_an_error___failed_counter_is_incremented(self, input_dir, output_dir, settings, do_il, initial_complete, initial_failed):
        client = OasisAPIClient('http://localhost:8001')
        client.upload_inputs_from_directory = Mock(side_effect=OasisException())
        client.run_analysis_and_poll = Mock()

        counter = Counter({
            'completed': initial_complete,
            'failed': initial_failed,
        })

        TestModelApiCmd().run_analysis(client, input_dir, output_dir, settings, do_il, counter)

        self.assertEqual(initial_complete, counter['completed'])
        self.assertEqual(initial_failed + 1, counter['failed'])

    @given(text(), text(), dictionaries(text(), text()), booleans(), integers(min_value=0, max_value=5), integers(min_value=0, max_value=5))
    def test_run_and_poll_raises_an_error___failed_counter_is_incremented(self, input_dir, output_dir, settings, do_il, initial_complete, initial_failed):
        client = OasisAPIClient('http://localhost:8001')
        client.upload_inputs_from_directory = Mock()
        client.run_analysis_and_poll = Mock(side_effect=OasisException())

        counter = Counter({
            'completed': initial_complete,
            'failed': initial_failed,
        })

        TestModelApiCmd().run_analysis(client, input_dir, output_dir, settings, do_il, counter)

        self.assertEqual(initial_complete, counter['completed'])
        self.assertEqual(initial_failed + 1, counter['failed'])


class TestModelApiCmdRun(TestCase):
    def setUp(self):
        self.directory = mkdtemp()
        self._orig_cwd = os.getcwd()
        self.create_working_directory(self.directory)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.directory)

    def create_working_directory(self, d):
        os.mkdir(os.path.join(d, 'input'))
        Path(os.path.join(d, 'analysis_settings.json')).touch()

        os.chdir(d)

    def get_command(self, api_server_url='http://localhost:8001', analysis_directory=None, extras=None):
        kwargs = {}

        if analysis_directory:
            kwargs['input-directory'] = os.path.join(analysis_directory, 'input')
            kwargs['output-directory'] = os.path.join(analysis_directory, 'output')
            kwargs['analysis-settings-file'] = os.path.join(analysis_directory, 'analysis_settings.json')

        kwargs.update(extras or {})
        kwargs_str = ' '.join('--{} {}'.format(k, v) for k, v in six.iteritems(kwargs))

        return RootCmd(argv='test model-api {} {}'.format(kwargs_str, api_server_url).split())

    def test_api_server_is_not_supplied___parsing_args_raises_an_error(self):
        cmd = self.get_command(api_server_url='')

        with self.assertRaises(SystemExit):
            cmd.parse_args()

    def test_input_directory_is_not_supplied___default_is_abs_path_to_input_in_cwd(self):
        cmd = self.get_command()

        args = cmd.parse_args()

        self.assertEqual(args.input_directory, os.path.abspath('input'))

    def test_input_directory_is_supplied___path_is_abs_path_to_supplied_dir(self):
        with TemporaryDirectory() as input_dir:
            cmd = self.get_command(extras={'input-directory': input_dir})

            args = cmd.parse_args()

            self.assertEqual(args.input_directory, os.path.abspath(input_dir))

    def test_output_directory_is_not_supplied___default_is_abs_path_to_output_in_cwd(self):
        cmd = self.get_command()

        args = cmd.parse_args()

        self.assertEqual(args.output_directory, os.path.abspath('output'))

    def test_output_directory_is_supplied___path_is_abs_path_to_supplied_dir(self):
        with TemporaryDirectory() as output_dir:
            cmd = self.get_command(extras={'output-directory': output_dir})

            args = cmd.parse_args()

            self.assertEqual(args.output_directory, os.path.abspath(output_dir))

    def test_analysis_settings_is_not_supplied___default_is_abs_path_to_analysis_settings_in_cwd(self):
        cmd = self.get_command()

        args = cmd.parse_args()

        self.assertEqual(args.analysis_settings_file, os.path.abspath('analysis_settings.json'))

    def test_analysis_settings_is_supplied___path_is_abs_path_to_supplied_file(self):
        with TemporaryDirectory() as settings_path:
            settings_path = os.path.join(settings_path, 'analysis_settings.json')
            Path(settings_path).touch()

            cmd = self.get_command(extras={'analysis-settings-file': settings_path})

            args = cmd.parse_args()

            self.assertEqual(args.analysis_settings_file, os.path.abspath(settings_path))

    def test_num_analyses_is_not_supplied___default_1(self):
        cmd = self.get_command()

        args = cmd.parse_args()

        self.assertEqual(1, args.num_analyses)

    def test_num_analyses_is_supplied___num_analyses_is_correct(self):
        cmd = self.get_command(extras={'num-analyses': 6})

        args = cmd.parse_args()

        self.assertEqual(6, args.num_analyses)

    def test_health_check_attempts_is_not_supplied___default_1(self):
        cmd = self.get_command()

        args = cmd.parse_args()

        self.assertEqual(1, args.health_check_attempts)

    def test_health_check_attempts_is_supplied___health_check_attempts_is_correct(self):
        cmd = self.get_command(extras={'health-check-attempts': 6})

        args = cmd.parse_args()

        self.assertEqual(6, args.health_check_attempts)

    def test_input_directory_does_not_exist___error_is_raised(self):
        shutil.rmtree(os.path.join(self.directory, 'input'))

        cmd = self.get_command(analysis_directory=self.directory)
        cmd._logger = Mock()

        res = cmd.run()

        self.assertEqual(1, res)
        cmd.logger.error.assert_called_once_with('Input directory does not exist: {}'.format(os.path.join(self.directory, 'input')))

    def test_analysis_settings_file_does_not_exist___error_is_raised(self):
        os.remove(os.path.join(self.directory, 'analysis_settings.json'))

        cmd = self.get_command(analysis_directory=self.directory)
        cmd._logger = Mock()

        res = cmd.run()

        self.assertEqual(1, res)
        cmd.logger.error.assert_called_once_with('Analysis settings does not exist: {}'.format(os.path.join(self.directory, 'analysis_settings.json')))

    @given(integers(min_value=1, max_value=5))
    def test_client_health_check_fails___error_is_written_to_the_log(self, health_check_attempts):
        with patch('oasislmf.api_client.client.OasisAPIClient.health_check', Mock(return_value=False)) as health_check_mock:
            cmd = self.get_command(analysis_directory=self.directory, extras={'health-check-attempts': health_check_attempts})
            cmd._logger = Mock()

            res = cmd.run()

            self.assertEqual(1, res)
            cmd.logger.error.assert_called_once_with('Health check failed for http://localhost:8001.')
            health_check_mock.assert_called_once_with(health_check_attempts)

    @given(integers(min_value=1, max_value=5), booleans(), dictionaries(text(), text()))
    def test_validation_is_successful___threadpool_is_started_for_each_analysis(self, num_analyses, do_il, settings):
        pool_mock_object = Mock()

        with patch('oasislmf.api_client.client.OasisAPIClient.health_check', Mock(return_value=True)), \
                patch('oasislmf.cmd.test.TestModelApiCmd.load_analysis_settings_json', Mock(return_value=(settings, do_il))), \
                patch('oasislmf.cmd.test.ThreadPool', Mock(return_value=pool_mock_object)) as pool_mock:
            cmd = self.get_command(analysis_directory=self.directory, extras={'num-analyses': num_analyses})
            cmd._logger = Mock()

            res = cmd.run()

            pool_mock.assert_called_once_with(processes=num_analyses)

            self.assertEqual(0, res)
            self.assertEqual(1, pool_mock_object.map.call_count)
            fn = pool_mock_object.map.call_args[0][0]
            args = list(pool_mock_object.map.call_args[0][1])
            self.assertEqual(fn.__name__, 'run_analysis')
            self.assertIsInstance(fn.__self__, TestModelApiCmd)
            self.assertEqual(args, [(ANY, cmd.args.input_directory, cmd.args.output_directory, settings, do_il, ANY)] * num_analyses)
