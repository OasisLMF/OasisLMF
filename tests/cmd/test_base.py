import logging
from argparse import Namespace
from unittest import TestCase

from oasislmf.cmd.base import OasisBaseCommand


class BaseLogger(TestCase):
    def setUp(self):
        self._orig_root_logger = logging.root
        logging.root = logging.RootLogger(logging.WARNING)

    def tearDown(self):
        logging.root = self._orig_root_logger

    def test_logger_is_set___set_logger_is_returned(self):
        cmd = OasisBaseCommand()

        cmd._logger = 'my logger'

        self.assertEqual(cmd._logger, 'my logger')

    def test_args_are_not_set___logger_is_returned_and_logger_backing_field_is_not_set(self):
        cmd = OasisBaseCommand()

        self.assertEqual(cmd.logger, logging.getLogger())
        self.assertIsNone(cmd._logger)

    def test_verbose_is_false___log_level_is_info(self):
        cmd = OasisBaseCommand()
        cmd.args = Namespace(verbose=False)

        self.assertEqual(cmd.logger.level, logging.INFO)
        self.assertEqual(cmd.logger.handlers[0].formatter._fmt, '%(message)s')

    def test_verbose_is_true___log_level_is_debug(self):
        cmd = OasisBaseCommand()
        cmd.args = Namespace(verbose=True)

        self.assertEqual(cmd.logger.level, logging.DEBUG)
        self.assertEqual(cmd.logger.handlers[0].formatter._fmt, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
