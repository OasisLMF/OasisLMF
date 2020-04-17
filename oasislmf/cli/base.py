__all__ = ["OasisBaseCommand"]

import logging
import os
import sys

from argparsetree import BaseCommand

from ..utils.path import PathCleaner


class OasisBaseCommand(BaseCommand):
    """
    The base command to inherit from for each command.

    2 additional arguments (``--verbose`` and ``--config``) are added to
    the parser so that they are available for all commands.
    """
    def __init__(self, *args, **kwargs):
        self._logger = None
        self.args = None
        self.log_verbose = False
        super(OasisBaseCommand, self).__init__(*args, **kwargs)

    def add_args(self, parser):
        """
        Adds arguments to the argument parser. This is used to modify
        which arguments are processed by the command.

        2 global parameters (``--verbose`` and ``--config``) are added
        so that they are available to all commands.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        parser.add_argument('-V', '--verbose', action='store_true', help='Use verbose logging.')
        parser.add_argument(
            '-C', '--config', required=False, type=PathCleaner('MDK config. JSON file', preexists=True),
            help='MDK config. JSON file', default='./oasislmf.json' if os.path.isfile('./oasislmf.json') else None
        )

    def parse_args(self):
        """
        Parses the command line arguments and sets them in ``self.args``

        :return: The arguments taken from the command line
        """
        try:
            self.args = super(OasisBaseCommand, self).parse_args()
            self.log_verbose = self.args.verbose
            self.setup_logger()
            return self.args
        except Exception:
            self.setup_logger()
            raise

    def setup_logger(self):
        """
        The logger to use for the command with the verbosity set
        """
        if not self._logger:
            if self.log_verbose:
                log_level = logging.DEBUG
                log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            else:
                log_level = logging.INFO
                log_format = '%(message)s'

            logging.basicConfig(stream=sys.stdout, level=log_level, format=log_format)
            self._logger = logging.getLogger()

    @property
    def logger(self):
        if self._logger:
            return self._logger

        return logging.getLogger()
