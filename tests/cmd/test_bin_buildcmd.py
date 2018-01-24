import uuid
from unittest import TestCase

import os
import six
from backports.tempfile import TemporaryDirectory
from mock import patch

from oasislmf.cmd import RootCmd
from oasislmf.utils.exceptions import OasisException


def get_command(src_dir=None, dst_dir=None, extras=None):
    kwargs_str = ' '.join('--{} {}'.format(k, v) for k, v in six.iteritems(extras or {}))

    return RootCmd(argv='bin build {} {} {}'.format(kwargs_str, src_dir or '', dst_dir or '').split())


class BuildCmdParseArgs(TestCase):
    def test_source_directory_is_not_supplied___default_is_current_directory(self):
        cmd = get_command()

        args = cmd.parse_args()

        self.assertEqual(args.source, os.path.abspath('.'))

    def test_source_directory_is_supplied___value_is_abs_path_to_the_supplied_direcory(self):
        with TemporaryDirectory() as d:
            cmd = get_command(src_dir=d)

            args = cmd.parse_args()

            self.assertEqual(args.source, os.path.abspath(d))

    def test_source_directory_doesnt_exist___error_is_raised(self):
        cmd = get_command(src_dir='non_existing_dir_{}'.format(uuid.uuid4().hex))

        with self.assertRaises(OasisException):
            cmd.parse_args()

    def test_destination_directory_is_not_supplied___default_is_none(self):
        cmd = get_command()

        args = cmd.parse_args()

        self.assertIsNone(args.destination)

    def test_destination_directory_is_supplied___value_is_abs_path_to_the_supplied_direcory(self):
        with TemporaryDirectory() as src, TemporaryDirectory() as dst:
            cmd = get_command(src_dir=src, dst_dir=dst)

            args = cmd.parse_args()

            self.assertEqual(args.destination, os.path.abspath(dst))

    def test_do_il_is_not_supplied___do_il_is_false(self):
        cmd = get_command()

        args = cmd.parse_args()

        self.assertFalse(args.do_il)

    def test_do_il_is_supplied___do_il_is_true(self):
        cmd = get_command(extras={'do-il': ''})

        args = cmd.parse_args()

        self.assertTrue(args.do_il)

    def test_build_tar_is_not_supplied___build_tar_il_is_false(self):
        cmd = get_command()

        args = cmd.parse_args()

        self.assertFalse(args.build_tar)

    def test_build_tar_is_supplied___build_tar_is_true(self):
        cmd = get_command(extras={'build-tar': ''})

        args = cmd.parse_args()

        self.assertTrue(args.build_tar)


class BuildCmdRun(TestCase):
    @patch('oasislmf.cmd.bin.check_conversion_tools')
    @patch('oasislmf.cmd.bin.check_inputs_directory')
    @patch('oasislmf.cmd.bin.create_binary_files')
    @patch('oasislmf.cmd.bin.create_binary_tar_file')
    def test_dst_dir_is_not_supplied___flow_is_correct_using_src_as_dst(self, create_tar_mock, create_bin_mock, check_ins_mock, check_conv_mock):
        with TemporaryDirectory() as src:
            cmd = get_command(src_dir=src, extras={'build-tar': ''})

            res = cmd.run()

            self.assertEqual(0, res)
            check_conv_mock.assert_called_once_with(do_il=False)
            check_ins_mock.assert_called_once_with(src, do_il=False, check_binaries=False)
            create_bin_mock.assert_called_once_with(src, src, do_il=False)
            create_tar_mock.assert_called_once_with(src)

    @patch('oasislmf.cmd.bin.check_conversion_tools')
    @patch('oasislmf.cmd.bin.check_inputs_directory')
    @patch('oasislmf.cmd.bin.create_binary_files')
    @patch('oasislmf.cmd.bin.create_binary_tar_file')
    def test_dst_dir_is_supplied___flow_is_correct_using_supplied_dir_as_dst(self, create_tar_mock, create_bin_mock, check_ins_mock, check_conv_mock):
        with TemporaryDirectory() as src, TemporaryDirectory() as dst:
            cmd = get_command(src_dir=src, dst_dir=dst, extras={'build-tar': ''})

            res = cmd.run()

            self.assertEqual(0, res)
            check_conv_mock.assert_called_once_with(do_il=False)
            check_ins_mock.assert_called_once_with(src, do_il=False, check_binaries=False)
            create_bin_mock.assert_called_once_with(src, dst, do_il=False)
            create_tar_mock.assert_called_once_with(dst)

    @patch('oasislmf.cmd.bin.check_conversion_tools')
    @patch('oasislmf.cmd.bin.check_inputs_directory')
    @patch('oasislmf.cmd.bin.create_binary_files')
    @patch('oasislmf.cmd.bin.create_binary_tar_file')
    def test_build_tar_is_false___create_tar_file_is_not_called(self, create_tar_mock, create_bin_mock, check_ins_mock, check_conv_mock):
        with TemporaryDirectory() as src:
            cmd = get_command(src_dir=src)

            res = cmd.run()

            self.assertEqual(0, res)
            check_conv_mock.assert_called_once_with(do_il=False)
            check_ins_mock.assert_called_once_with(src, do_il=False, check_binaries=False)
            create_bin_mock.assert_called_once_with(src, src, do_il=False)
            create_tar_mock.assert_not_called()

    @patch('oasislmf.cmd.bin.check_conversion_tools')
    @patch('oasislmf.cmd.bin.check_inputs_directory')
    @patch('oasislmf.cmd.bin.create_binary_files')
    @patch('oasislmf.cmd.bin.create_binary_tar_file')
    def test_do_il_is_true___il_files_are_generated(self, create_tar_mock, create_bin_mock, check_ins_mock, check_conv_mock):
        with TemporaryDirectory() as src:
            cmd = get_command(src_dir=src, extras={'do-il': ''})

            res = cmd.run()

            self.assertEqual(0, res)
            check_conv_mock.assert_called_once_with(do_il=True)
            check_ins_mock.assert_called_once_with(src, do_il=True, check_binaries=False)
            create_bin_mock.assert_called_once_with(src, src, do_il=True)
            create_tar_mock.assert_not_called()
