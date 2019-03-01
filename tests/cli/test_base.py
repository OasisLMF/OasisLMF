# -*- coding: utf-8 -*-

import logging
from unittest import TestCase

from oasislmf.cli.base import OasisBaseCommand


class BaseLogger(TestCase):
    def setUp(self):
        self._orig_root_logger = logging.root
        logging.root = logging.RootLogger(logging.WARNING)

    def tearDown(self):
        logging.root = self._orig_root_logger

    def test_verbose_is_false___log_level_is_info(self):
        cmd = OasisBaseCommand(argv=[])
        cmd.parse_args()

        self.assertEqual(cmd.logger.level, logging.INFO)
        self.assertEqual(cmd.logger.handlers[0].formatter._fmt, '%(message)s')

    def test_verbose_is_true___log_level_is_debug(self):
        cmd = OasisBaseCommand(argv=['--verbose'])
        cmd.parse_args()

        self.assertEqual(cmd.logger.level, logging.DEBUG)
        self.assertEqual(cmd.logger.handlers[0].formatter._fmt, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
