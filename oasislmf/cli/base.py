# -*- coding: utf-8 -*-
import io
import json
import logging
import os
import sys

from argparsetree import BaseCommand

from ..utils.exceptions import OasisException
from ..utils.path import PathCleaner


class InputValues(object):
    """
    Helper class for accessing the input values from either
    the command line or the configuration file.
    """
    def __init__(self, args):
        self.args = args

        self.config_fp = None
        self.config = {}

        try:
            self.config_fp = os.path.abspath(args.config)
        except (AttributeError, OSError, TypeError):
            pass
        else:
            self.config_dir = os.path.dirname(self.config_fp)
            try:
                with io.open(self.config_fp, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except FileNotFoundError:
                raise OasisException('MDK config. file path {} provided does not exist'.format(self.config_fp))

    def get(self, name, default=None, required=False, is_path=False):
        """
        Gets the names parameter from the command line arguments.

        If it is not set on the command line the configuration file
        is checked.

        If it is also not present in the configuration file then
        ``default`` is returned unless ``required`` is false in which
        case an ``OasisException`` is raised.

        :param name: The name of the parameter to lookup
        :type name: str

        :param default: The default value to return if the name is not
            found on the command line or in the configuration file.

        :param required: Flag whether the value is required, if so and
            the parameter is not found on the command line or in the
            configuration file an error is raised.
        :type required: bool

        :param is_path: Flag whether the value should be treated as a path,
            is so the value is processed as relative to the config file.
        :type is_path: bool

        :raise OasisException: If the value is not found and ``required``
            is True

        :return: The found value or the default
        """
        value = None

        cmd_value = getattr(self.args, name, None)
        config_value = self.config.get(name)

        if cmd_value is not None:
            value = cmd_value
            if is_path and not os.path.isabs(value):
                value = os.path.abspath(value)
        elif config_value is not None:
            value = self.config[name]
            if is_path and not os.path.isabs(value):
                value = os.path.join(self.config_dir, value)

        if required and value is None:
            raise OasisException(
                'Required argument {} could not be found in the command args or the MDK config. file'.format(name)
            )

        if value is None:
            value = default

        if is_path and value is not None and not os.path.isabs(value):
            value = os.path.abspath(value) if cmd_value else os.path.join(self.config_dir, value)

        return value


class OasisBaseCommand(BaseCommand):
    """
    The base command to inherit from for each command.

    2 additional arguments (``--verbose`` and ``--config``) are added to
    the parser so that they are available for all commands.
    """
    def __init__(self, *args, **kwargs):
        self._logger = None
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
            help='MDK config. JSON file', default=None
        )

    def parse_args(self):
        """
        Parses the command line arguments and sets them in ``self.args``

        :return: The arguments taken from the command line
        """
        try:
            self.args = super(OasisBaseCommand, self).parse_args()
            self.setup_logger(self.args.verbose)
            return self.args
        except Exception:
            self.setup_logger(False)
            raise

    def setup_logger(self, verbose):
        """
        The logger to use for the command with the verbosity set
        """
        if not self._logger:
            if verbose:
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
