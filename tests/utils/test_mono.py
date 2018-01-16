from __future__ import print_function

from collections import OrderedDict
from unittest import TestCase

from mock import patch, Mock

from oasislmf.utils.mono import run_mono_executable


class RunMonoExecutable(TestCase):
    def test_executable_and_args_are_supplied___subprocess_is_created_with_correct_arguments(self):
        with patch('subprocess.call', Mock(return_value=0)) as call_mock:
            run_mono_executable('my_mono_executable', OrderedDict((('first', 'a'), ('second', 'b'))))

            call_mock.assert_called_once_with('mono my_mono_executable -first a -second b', shell=True)
