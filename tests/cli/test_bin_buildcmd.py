import os
import uuid

from future.utils import viewitems
from unittest import TestCase

import pytest

from backports.tempfile import TemporaryDirectory
from mock import patch

import oasislmf
from oasislmf.cli import RootCmd
from oasislmf.utils.exceptions import OasisException



def get_command(src_dir=None, dst_dir=None, extras=None):
    kwargs_str = ' '.join('--{} {}'.format(k, v) for k, v in viewitems(extras or {}))

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

    def test_il_is_not_supplied___il_is_false(self):
        cmd = get_command()

        args = cmd.parse_args()

        self.assertFalse(args.il)

    def test_il_is_supplied___il_is_true(self):
        cmd = get_command(extras={'il': ''})

        args = cmd.parse_args()

        self.assertTrue(args.il)

    def test_build_tar_is_not_supplied___build_tar_il_is_false(self):
        cmd = get_command()

        args = cmd.parse_args()

        self.assertFalse(args.build_tar)

    def test_build_tar_is_supplied___build_tar_is_true(self):
        cmd = get_command(extras={'build-tar': ''})

        args = cmd.parse_args()

        self.assertTrue(args.build_tar)


class BuildCmdRun(TestCase):

    @patch('oasislmf.model_execution.bin.check_conversion_tools')
    @patch('oasislmf.model_execution.bin.check_inputs_directory')
    @patch('oasislmf.model_execution.bin.csv_to_bin')
    @patch('oasislmf.model_execution.bin.create_binary_tar_file')
    def test_dst_dir_is_not_supplied___flow_is_correct_using_src_as_dst(self, check_conv_mock, check_ins_mock, csv_to_bin_mock, create_tar_mock):
        with TemporaryDirectory() as src:
            cmd = get_command(src_dir=src, extras={'build-tar': ''})

            res = cmd.run()

            self.assertEqual(1, res)
            check_conv_mock.assert_called_once_with(il=False)
            check_ins_mock.assert_called_once_with(src, il=False, check_binaries=False)
            csv_to_bin_mock.assert_called_once_with(src, src, il=False)
            create_tar_mock.assert_called_once_with(src)

    @patch('oasislmf.model_execution.bin.check_conversion_tools')
    @patch('oasislmf.model_execution.bin.check_inputs_directory')
    @patch('oasislmf.model_execution.bin.csv_to_bin')
    @patch('oasislmf.model_execution.bin.create_binary_tar_file')
    def test_dst_dir_is_supplied___flow_is_correct_using_supplied_dir_as_dst(self, check_conv_mock, check_ins_mock, csv_to_bin_mock, create_tar_mock):
        with TemporaryDirectory() as src, TemporaryDirectory() as dst:
            cmd = get_command(src_dir=src, dst_dir=dst, extras={'build-tar': ''})

            res = cmd.run()

            self.assertEqual(0, res)
            check_conv_mock.assert_called_once_with(il=False)
            check_ins_mock.assert_called_once_with(src, il=False, check_binaries=False)
            csv_to_bin_mock.assert_called_once_with(src, dst, il=False)
            create_tar_mock.assert_called_once_with(dst)

    @patch('oasislmf.model_execution.bin.check_conversion_tools')
    @patch('oasislmf.model_execution.bin.check_inputs_directory')
    @patch('oasislmf.model_execution.bin.csv_to_bin')
    @patch('oasislmf.model_execution.bin.create_binary_tar_file')
    def test_build_tar_is_false___create_tar_file_is_not_called(self, check_conv_mock, check_ins_mock, csv_to_bin_mock, create_tar_mock):
        with TemporaryDirectory() as src:
            cmd = get_command(src_dir=src)

            res = cmd.run()

            self.assertEqual(0, res)
            check_conv_mock.assert_called_once_with(il=False)
            check_ins_mock.assert_called_once_with(src, il=False, check_binaries=False)
            csv_to_bin_mock.assert_called_once_with(src, src, il=False)
            create_tar_mock.assert_not_called()

    @patch('oasislmf.model_execution.bin.check_conversion_tools')
    @patch('oasislmf.model_execution.bin.check_inputs_directory')
    @patch('oasislmf.model_execution.bin.csv_to_bin')
    @patch('oasislmf.model_execution.bin.create_binary_tar_file')
    def test_il_is_true___il_files_are_generated(self, check_conv_mock, check_ins_mock, csv_to_bin_mock, create_tar_mock):
        with TemporaryDirectory() as src:
            cmd = get_command(src_dir=src, extras={'il': ''})

            res = cmd.run()

            self.assertEqual(0, res)
            check_conv_mock.assert_called_once_with(il=True)
            check_ins_mock.assert_called_once_with(src, il=True, check_binaries=False)
            csv_to_bin_mock.assert_called_once_with(src, src, il=True)
            create_tar_mock.assert_not_called()
