__all__ = [
    "OasisBaseCommand",
    "OasisComputationCommand",
]

import logging
import os
import sys

from argparsetree import BaseCommand

from ..utils.path import PathCleaner
from ..utils.inputs import InputValues, str2bool

from ..manager import OasisManager as om


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


class OasisComputationCommand(OasisBaseCommand):
    """
    Eventually, the Parent class for all Oasis Computation Command
    create the command line interface from parameter define in the associated computation step
    """

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super().add_args(parser)

        for param in om.computations_params[self.computation_name]:
            add_argument_kwargs = {key: param.get(key) for key in ['action', 'nargs', 'const', 'type', 'choices',
                                                                   'help', 'metavar', 'dest'] if param.get(key) is not None}
            # If 'Help' is not set then this is a function only paramter, skip
            if 'help' in add_argument_kwargs:
                arg_name = f"--{param['name'].replace('_', '-')}"
                if param.get('flag'):
                    parser.add_argument(param.get('flag'), arg_name, **add_argument_kwargs)
                else:
                    parser.add_argument(arg_name, **add_argument_kwargs)

    def action(self, args):
        """
        Generic method that call the correct manager method from the child class computation_name

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        _kwargs = {
            param['name']: inputs.get(param['name'], required=param.get('required'), is_path=param.get('is_path'), dtype=param.get('type')) for
            param in om.computations_params[self.computation_name]}

        # Override logger setup from kwargs
        if 'verbose' in _kwargs:
            self.logger.level = logging.DEBUG if str2bool(_kwargs.get('verbose')) else logging.INFO

        self.logger.info(f'\nStating oasislmf command - {self.computation_name}')
        manager_method = getattr(om(), om.computation_name_to_method(self.computation_name))
        manager_method(**_kwargs)
