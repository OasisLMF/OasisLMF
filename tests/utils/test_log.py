import logging
from logging.handlers import RotatingFileHandler
from random import random
from unittest import TestCase

from hypothesis import given
from hypothesis.strategies import sampled_from
from mock import Mock, patch

from oasislmf.utils.log import oasis_log, read_log_config


class MockLogger(object):
    def __init__(self):
        self.debug = Mock()
        self.info = Mock()


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

    def test_wrapped_function_is_called___info_logging_is_called(self):
        logger_mock = MockLogger()

        with patch('oasislmf.utils.log.logging.getLogger', Mock(return_value=logger_mock)):
            callable = create_callable()

            wrapped = oasis_log(callable)
            wrapped('first', second='second')

            self.assertGreater(logger_mock.info.call_count, 0)

    def test_wrapped_function_is_called___args_and_kwargs_are_logged_to_debug_excluding_self(self):
        logger_mock = MockLogger()

        with patch('oasislmf.utils.log.logging.getLogger', Mock(return_value=logger_mock)):
            class FakeObj(object):
                @oasis_log
                def method(self, first, second=None):
                    pass

            FakeObj().method('a', second='b')

            logger_mock.debug.assert_any_call("    first == a")
            logger_mock.debug.assert_any_call("    second == b")

            for args in logger_mock.debug.call_args_list:
                if 'self' in args[0][0]:
                    self.fail('"self" was logged')


class ReadLogConfig(TestCase):
    @given(sampled_from([logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]))
    def test_log_config_is_loaded___logger_is_updated(self, level):
        with patch('logging.root', logging.RootLogger(logging.NOTSET)):
            read_log_config({
                'LOG_FILE': '/tmp/log_file.txt',
                'LOG_LEVEL': level,
                'LOG_MAX_SIZE_IN_BYTES': 100,
                'LOG_BACKUP_COUNT': 10,
            })

            logger = logging.getLogger()

            self.assertEqual(level, logger.level)

            self.assertEqual(1, len(logger.handlers))
            handler = logger.handlers[0]
            self.assertIsInstance(handler, RotatingFileHandler)
            self.assertEqual('/tmp/log_file.txt', handler.baseFilename)
            self.assertEqual(100, handler.maxBytes)
            self.assertEqual(10, handler.backupCount)
