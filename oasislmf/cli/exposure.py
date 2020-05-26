__all__ = [
    'ExposureCmd',
    'RunCmd'
]

import os

from argparse import RawDescriptionHelpFormatter
from pathlib2 import Path
from ..manager import OasisManager as om
from ..utils.defaults import (
    KTOOLS_ALLOC_IL_DEFAULT,
    KTOOLS_ALLOC_RI_DEFAULT
)
from ..utils.exceptions import OasisException
from ..utils.path import as_path

from .base import OasisBaseCommand
from .inputs import InputValues

import tempfile


class RunCmd(OasisBaseCommand):
    """
    Generates deterministic losses using the installed ktools framework given
    direct Oasis files (GUL + optionally IL and RI input files).

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    Run ``oasislmf config --help`` for more information.
    """
    formatter_class = RawDescriptionHelpFormatter

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(self.__class__, self).add_args(parser)

        parser.add_argument(
            '-s', '--src-dir', type=str, default=None, required=True,
            help='Source files directory - should contain the OED exposure files'
        )
        parser.add_argument(
            '-r', '--run-dir', type=str, default=None, required=False,
            help='Run directory - where files should be generated'
        )
        parser.add_argument(
            '-l', '--loss-factor', type=float, nargs='+',
            help='Loss factors to apply to TIVs - default is 1.0. Multiple factors can be specified.'
        )
        parser.add_argument(
            '-a', '--alloc-rule-il', type=int, default=KTOOLS_ALLOC_IL_DEFAULT,
            help='Ktools IL back allocation rule to apply - default is 2, i.e. prior level loss basis'
        )
        parser.add_argument(
            '-A', '--alloc-rule-ri', type=int, default=KTOOLS_ALLOC_RI_DEFAULT,
            help='Ktools RI back allocation rule to apply - default is 3, i.e. All level loss basis'
        )
        parser.add_argument(
            '-o', '--output-level', default='item',
            help='Level to output losses. Options are: item, loc, pol, acc or port.', type=str
        )
        parser.add_argument(
            '-f', '--output-file', default=None,
            help='Write the output to file.', type=str
        )

    def action(self, args):
        """
        Generates deterministic losses using the installed ktools framework given
        direct Oasis files (GUL + optionally IL and RI input files).

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        self.logger.debug('\nProcessing arguments')

        call_dir = os.getcwd()

        src_dir = as_path(inputs.get('src_dir', default=call_dir, is_path=True), 'Source files directory', is_dir=True, preexists=True)

        loss_factors = inputs.get(
            'loss_factor', default=[1.0], required=False
        )
        include_loss_factor = not (len(loss_factors) == 1) 

        net_ri = True

        il_alloc_rule = inputs.get('alloc_rule_il', default=KTOOLS_ALLOC_IL_DEFAULT, required=False)
        ri_alloc_rule = inputs.get('alloc_rule_ri', default=KTOOLS_ALLOC_RI_DEFAULT, required=False)

        # item, loc, pol, acc, port
        output_level = inputs.get('output_level', default="item", required=False)
        if output_level not in ['port', 'acc', 'loc', 'pol', 'item']:
            raise OasisException(
                'Invalid output level. Must be one of port, acc, loc, pol or item.'
            )

        output_file = as_path(inputs.get('output_file', required=False, is_path=True), 'Output file path', preexists=False)

        run_dir = as_path(inputs.get('run_dir', is_path=True), 'Run directory', is_dir=True, preexists=False)
        if run_dir is None:
            with tempfile.TemporaryDirectory() as tmpdirname:
                om().run_exposure(
                    src_dir, tmpdirname, loss_factors, net_ri,
                    il_alloc_rule, ri_alloc_rule, output_level, output_file,
                    print_summary=True, include_loss_factor=include_loss_factor)
        else:
            if not os.path.exists(run_dir):
                Path(run_dir).mkdir(parents=True, exist_ok=True)
            om().run_exposure(
                src_dir, run_dir, loss_factors, net_ri,
                il_alloc_rule, ri_alloc_rule, output_level, output_file,
                print_summary=True, include_loss_factor=include_loss_factor)


class ExposureCmd(OasisBaseCommand):
    """
    Exposure subcommands::

        * generate - and optionally, validate - deterministic losses (GUL, IL or RIL)
    """
    sub_commands = {
        'run': RunCmd
    }
