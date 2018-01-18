import logging

import sys
from argparsetree import BaseCommand


class OasisBaseCommand(BaseCommand):
    def __init__(self, *args, **kwargs):
        super(OasisBaseCommand, self).__init__(*args, **kwargs)
        self.logger = None
        self.args = None

    def add_args(self, parser):
        parser.add_argument(
            '-v', '--verbose_logging', action='store_true',
            help='Use verbose logging.'
        )

    def parse_args(self):
        args = super().parse_args()
        self.set_logger(args['verbose_logging'])
        return args

    def set_logger(self, verbose):
        self.logger = logging.getLogger()

        if verbose:
            log_level = logging.DEBUG
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            log_level = logging.INFO
            log_format = ' %(message)s'

        logging.basicConfig(stream=sys.stdout, level=log_level, format=log_format)
