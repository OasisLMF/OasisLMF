from unittest import TestCase

import os
import six
from backports.tempfile import TemporaryDirectory
from mock import patch

from oasislmf.cmd import RootCmd


def get_command(target_dir=None, extras=None):
    kwargs_str = ' '.join('--{} {}'.format(k, v) for k, v in six.iteritems(extras or {}))

    return RootCmd(argv='bin check {} {}'.format(kwargs_str, target_dir or '').split())


class CheckCmdRun(TestCase):
    @patch('oasislmf.cmd.bin.check_inputs_directory')
    @patch('oasislmf.cmd.bin.check_conversion_tools')
    def test_target_is_not_supplied___cwd_is_checked(self, check_conv_tools, check_inputs_mock):
        cmd = get_command()

        res = cmd.run()

        self.assertEqual(0, res)
        check_inputs_mock.assert_called_once_with(os.path.abspath('.'), do_il=False, check_binaries=False)
        check_conv_tools.assert_called_once_with(do_il=False)

    @patch('oasislmf.cmd.bin.check_inputs_directory')
    @patch('oasislmf.cmd.bin.check_conversion_tools')
    def test_target_is_supplied___supplied_path_is_checked(self, check_conv_tools, check_inputs_mock):
        with TemporaryDirectory() as d:
            cmd = get_command(target_dir=d)

            res = cmd.run()

            self.assertEqual(0, res)
            check_inputs_mock.assert_called_once_with(d, do_il=False, check_binaries=False)
            check_conv_tools.assert_called_once_with(do_il=False)

    @patch('oasislmf.cmd.bin.check_inputs_directory')
    @patch('oasislmf.cmd.bin.check_conversion_tools')
    def test_do_il_is_true___il_input_files_are_checked(self, check_conv_tools, check_inputs_mock):
        with TemporaryDirectory() as d:
            cmd = get_command(target_dir=d, extras={'do-il': ''})

            res = cmd.run()

            self.assertEqual(0, res)
            check_inputs_mock.assert_called_once_with(d, do_il=True, check_binaries=False)
            check_conv_tools.assert_called_once_with(do_il=True)

    @patch('oasislmf.cmd.bin.check_inputs_directory')
    @patch('oasislmf.cmd.bin.check_conversion_tools')
    def test_check_binaries_is_true__existance_of_bin_files_are_checked(self, check_conv_tools, check_inputs_mock):
        with TemporaryDirectory() as d:
            cmd = get_command(target_dir=d, extras={'check-binaries': ''})

            res = cmd.run()

            self.assertEqual(0, res)
            check_inputs_mock.assert_called_once_with(d, do_il=False, check_binaries=True)
            check_conv_tools.assert_called_once_with(do_il=False)
