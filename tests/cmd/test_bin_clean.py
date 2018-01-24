from unittest import TestCase

import os
import six
from backports.tempfile import TemporaryDirectory
from mock import patch

from oasislmf.cmd import RootCmd


def get_command(target_dir=None, extras=None):
    kwargs_str = ' '.join('--{} {}'.format(k, v) for k, v in six.iteritems(extras or {}))

    return RootCmd(argv='bin clean {} {}'.format(kwargs_str, target_dir or '').split())


class CleanCmdRun(TestCase):
    @patch('oasislmf.cmd.bin.cleanup_bin_directory')
    def test_src_is_not_supplied___cwd_is_cleaned(self, clean_mock):
        cmd = get_command()

        res = cmd.run()

        self.assertEqual(0, res)
        clean_mock.assert_called_once_with(os.path.abspath('.'))

    @patch('oasislmf.cmd.bin.cleanup_bin_directory')
    def test_src_is_supplied___supplied_path_is_cleaned(self, clean_mock):
        with TemporaryDirectory() as d:
            cmd = get_command(target_dir=d)

            res = cmd.run()

            self.assertEqual(0, res)
            clean_mock.assert_called_once_with(os.path.abspath(d))
