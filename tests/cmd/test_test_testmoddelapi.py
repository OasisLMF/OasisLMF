import json
from unittest import TestCase

import os

import shutil
import six
from hypothesis import given
from hypothesis.strategies import sampled_from
from mock import Mock
from backports.tempfile import mkdtemp, TemporaryDirectory
from pathlib2 import Path

from oasislmf.cmd import RootCmd
from oasislmf.cmd.test import TestModelApiCmd


class TestModelApiCmdBaseTest(TestCase):
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


class TestModelApiCmdLoadAnalysisSettingsJson(TestModelApiCmdBaseTest):
    def test_do_il_is_true___result_has_input_dict_and_do_il_is_true(self):
        conf = {
            'analysis_settings': {
                'il_output': True,
                'foo': 'bar',
            }
        }

        filename = os.path.join(self.directory, 'analysis_settings.json')
        with open(filename, 'w') as f:
            json.dump(conf, f)

        res_conf, do_il = TestModelApiCmd().load_analysis_settings_json(filename)

        self.assertEqual(conf, res_conf)
        self.assertTrue(do_il)

    def test_do_il_is_false___result_has_input_dict_and_do_il_is_false(self):
        conf = {
            'analysis_settings': {
                'il_output': False,
                'foo': 'bar',
            }
        }

        filename = os.path.join(self.directory, 'analysis_settings.json')
        with open(filename, 'w') as f:
            json.dump(conf, f)

        res_conf, do_il = TestModelApiCmd().load_analysis_settings_json(filename)

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

        filename = os.path.join(self.directory, 'analysis_settings.json')
        with open(filename, 'w') as f:
            json.dump(conf, f)

        res_conf, do_il = TestModelApiCmd().load_analysis_settings_json(filename)

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

        filename = os.path.join(self.directory, 'analysis_settings.json')
        with open(filename, 'w') as f:
            json.dump(conf, f)

        res_conf, do_il = TestModelApiCmd().load_analysis_settings_json(filename)

        self.assertEqual(conf, res_conf)
        self.assertFalse(do_il)


class TestModelApiCmdRun(TestModelApiCmdBaseTest):
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
