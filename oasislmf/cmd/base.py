import logging

import sys
from argparsetree import BaseCommand


class OasisBaseCommand(BaseCommand):
    def __init__(self, *args, **kwargs):
        self._logger = None
        self.args = None
        super(OasisBaseCommand, self).__init__(*args, **kwargs)

    def add_args(self, parser):
        parser.add_argument(
            '-v', '--verbose_logging', action='store_true',
            help='Use verbose logging.'
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
