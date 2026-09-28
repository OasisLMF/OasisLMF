import io
import json
import logging
import os
from argparse import Namespace
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from oasislmf.cli.command import OasisBaseCommand


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
        self.assertEqual(cmd.logger.handlers[0].formatter._fmt, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def test_verbose_is_true___log_level_is_debug(self):
        cmd = OasisBaseCommand(argv=['--verbose'])
        cmd.parse_args()

        self.assertEqual(cmd.logger.level, logging.DEBUG)
        self.assertEqual(cmd.logger.handlers[0].formatter._fmt, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class LoadConfigDict(TestCase):
    def _cmd_with_config(self, config_fp):
        cmd = OasisBaseCommand(argv=[])
        cmd.args = Namespace(config=config_fp)
        return cmd

    def test_no_config___empty_dict(self):
        self.assertEqual(self._cmd_with_config(None)._load_config_dict(), {})

    def test_valid_config___keys_lower_cased(self):
        with TemporaryDirectory() as d:
            config_fp = os.path.join(d, 'oasislmf.json')
            with open(config_fp, 'w') as f:
                json.dump({'LOGGING': {'level': 'DEBUG'}}, f)

            self.assertEqual(self._cmd_with_config(config_fp)._load_config_dict(), {'logging': {'level': 'DEBUG'}})

    def test_missing_config___warns_and_returns_empty_dict(self):
        with TemporaryDirectory() as d:
            config_fp = os.path.join(d, 'missing.json')
            with patch('sys.stderr', new_callable=io.StringIO) as stderr:
                self.assertEqual(self._cmd_with_config(config_fp)._load_config_dict(), {})

            self.assertIn(f'Warning: Config file not found: {config_fp}', stderr.getvalue())
