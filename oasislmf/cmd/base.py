import json
import logging

import sys

import os
import io

from argparsetree import BaseCommand

from oasislmf.utils.exceptions import OasisException
from .cleaners import PathCleaner


class InputValues(object):
    """
    Helper class for accessing the input values from either
    the command line or the configuration file.
    """
    def __init__(self, args):
        self.args = args

        self.config = {}
        self.config_dir = os.path.dirname(args.config)
        if os.path.exists(args.config):
            with io.open(args.config, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

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
        if cmd_value is not None:
            value = cmd_value
        elif name in self.config:
            value = self.config[name]

        if required and value is None:
            raise OasisException(
                '{} could not be found in the command args or config file ({}) but is required'.format(name, self.args.config)
            )

        if value is None:
            value = default

        if is_path and value is not None:
            p = os.path.join(self.config_dir, value)
            value = os.path.abspath(p) if not os.path.isabs(value) else p

        return value


class OasisBaseCommand(BaseCommand):
    """
    The base command to inherit from for each command.

    2 additional arguments (``--verbose`` and ``--config``) are added to
    the parser so that they are available for all commands.
    """
    def __init__(self, *args, **kwargs):
        self._logger = None
        self.args = None
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
            '-C', '--config', type=PathCleaner('Config file', preexists=False),
            help='The oasislmf config to load', default='./oasislmf.json'
        )

    def parse_args(self):
        """
        Parses the command line arguments and sets them in ``self.args``

        :return: The arguments taken from the command line
        """
        self.args = super(OasisBaseCommand, self).parse_args()
        return self.args

    @property
    def logger(self):
        """
        The logger to use for the command with the verbosity set
        """
        if self._logger:
            return self._logger

        if self.args:
            self._logger = logging.getLogger()

            if self.args.verbose:
                log_level = logging.DEBUG
                log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            else:
                log_level = logging.INFO
                log_format = ' %(message)s'

            logging.basicConfig(stream=sys.stdout, level=log_level, format=log_format)
            return self._logger
        else:
            return logging.getLogger()
