from __future__ import print_function

from collections import OrderedDict
from unittest import TestCase

from hypothesis import given
from hypothesis.strategies import integers
from mock import patch, Mock

from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.mono import run_mono_executable


class RunMonoExecutable(TestCase):
    def test_executable_and_args_are_supplied___subprocess_is_created_with_correct_arguments(self):
        with patch('subprocess.call', Mock(return_value=0)) as call_mock:
            run_mono_executable('my_mono_executable', OrderedDict((('first', 'a'), ('second', 'b'))))

            call_mock.assert_called_once_with('mono my_mono_executable -first a -second b', shell=True)

    @given(integers(min_value=-1, max_value=1))
    def test_call_executes_without_error___call_result_is_returned(self, expected):
        with patch('subprocess.call', Mock(return_value=expected)):
            result = run_mono_executable('my_mono_executable', OrderedDict((('first', 'a'), ('second', 'b'))))

            self.assertEqual(expected, result)

    def test_call_raises_an_os_error___oasis_exception_is_raised(self):
        def raising_function(*args, **kwargs):
            raise OSError()

        with patch('subprocess.call', Mock(side_effect=raising_function)), self.assertRaises(OasisException):
            run_mono_executable('my_mono_executable', OrderedDict((('first', 'a'), ('second', 'b'))))
