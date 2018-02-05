import json
import logging

import sys

import os
from argparsetree import BaseCommand

from oasislmf.utils.exceptions import OasisException
from .cleaners import PathCleaner


class InputValues(object):
    def __init__(self, args):
        self.args = args

        self.config = {}
        if os.path.exists(args.config):
            with open(args.config) as f:
                self.config = json.load(f)

    def get(self, name, default=None, required=False):
        if name in self.args:
            return self.args[name]

        if name in self.config:
            return self.config[name]

        if required:
            raise OasisException(
                '{} could not be found in the command args or config file ({}) but is required'.format(name, self.args.config)
            )

        return default


class OasisBaseCommand(BaseCommand):
    def __init__(self, *args, **kwargs):
        self._logger = None
        self.args = None
        super(OasisBaseCommand, self).__init__(*args, **kwargs)

    def add_args(self, parser):
        parser.add_argument('-V', '--verbose', action='store_true', help='Use verbose logging.')
        parser.add_argument(
            '-C', '--config', type=PathCleaner('Config file', preexists=False),
            help='The oasislmf config to load', default='./oasislmf.json'
        )

    def parse_args(self):
        self.args = super(OasisBaseCommand, self).parse_args()
        return self.args

    @property
    def logger(self):
        if self._logger:
            return self._logger

        if self.args:
            self._logger = logging.getLogger()

            if self.args.verbose_logging:
                log_level = logging.DEBUG
                log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            else:
                log_level = logging.INFO
                log_format = ' %(message)s'

            logging.basicConfig(stream=sys.stdout, level=log_level, format=log_format)
            return self._logger
        else:
            return logging.getLogger()
