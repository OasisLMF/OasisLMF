from random import random
from unittest import TestCase

from mock import Mock, patch

from oasislmf.utils.log import oasis_log


def create_callable(result=None):
    def _callable(*args, **kwargs):
        if not hasattr(_callable, 'calls'):
            _callable.calls = []

        _callable.calls.append({'args': args, 'kwargs': kwargs})
        return result

    return _callable


class OasisLog(TestCase):
    def test_function_is_wrapped_without_arg_list___return_value_is_correct(self):
        expected = random()
        callable = create_callable(result=expected)

        wrapped = oasis_log(callable)
        res = wrapped()

        self.assertEqual(expected, res)

    def test_function_is_wrapped_with_arg_list___return_value_is_correct(self):
        expected = random()
        callable = create_callable(result=expected)

        wrapped = oasis_log()(callable)
        res = wrapped()

        self.assertEqual(expected, res)

    def test_function_is_wrapped_without_arg_list___callable_is_called_once_with_the_correct_args(self):
        callable = create_callable()

        wrapped = oasis_log(callable)
        wrapped('first', second='second')

        self.assertEqual(1, len(callable.calls))
        self.assertEqual(('first', ), callable.calls[0]['args'])
        self.assertEqual({'second': 'second'}, callable.calls[0]['kwargs'])

    def test_function_is_wrapped_with_arg_list___callable_is_called_once_with_the_correct_args(self):
        callable = create_callable()

        wrapped = oasis_log()(callable)
        wrapped('first', second='second')

        self.assertEqual(1, len(callable.calls))
        self.assertEqual(('first', ), callable.calls[0]['args'])
        self.assertEqual({'second': 'second'}, callable.calls[0]['kwargs'])

    def test_wrapped_funtion_is_called___debug_logging_is_called(self):
        logger_mock = Mock()

        with patch('oasislmf.utils.log.logging.getLogger', Mock(return_value=logger_mock)):
            callable = create_callable()

            wrapped = oasis_log(callable)
            wrapped('first', second='second')

            self.assertGreater(logger_mock.debug.call_count, 0)

    def test_wrapped_funtion_is_called___info_logging_is_called(self):
        logger_mock = Mock()

        with patch('oasislmf.utils.log.logging.getLogger', Mock(return_value=logger_mock)):
            callable = create_callable()

            wrapped = oasis_log(callable)
            wrapped('first', second='second')

            self.assertGreater(logger_mock.info.call_count, 0)