__all__ = [
    'ExposureCmd',
    'RunCmd',
    'ValidateCmd'
]

import os

from argparse import RawDescriptionHelpFormatter

from pathlib2 import Path

from ..manager import OasisManager as om

from ..utils.data import (
    print_dataframe,
)
from ..utils.path import (
    as_path,
)
from .base import (
    InputValues,
    OasisBaseCommand,
)


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
            '-o', '--output-dir', type=str, default=None, required=False, help='Output directory'
        )
        parser.add_argument(
            '-i', '--input-dir', type=str, default=None, required=True,
            help='Input directory - should contain the OED exposure file + optionally the accounts, and RI info. and scope files'
        )
        parser.add_argument(
            '-l', '--loss-factor', type=float, default=None,
            help='Loss factor to apply to TIVs.'
        )
        parser.add_argument(
            '-n', '--net-ri-losses', default=False, help='Net RI losses', action='store_true'
        )

    def action(self, args):
        """
        Generates deterministic losses using the installed ktools framework given
        direct Oasis files (GUL + optionally IL and RI input files).

        :param args: The arguments from the command line
        :type args: Namespace
        """
        self.logger.info('\nProcessing arguments')
        inputs = InputValues(args)

        call_dir = os.getcwd()

        input_dir = as_path(inputs.get('input_dir', default=call_dir, is_path=True), 'Input directory', is_dir=True, preexists=True)

        output_dir = as_path(inputs.get('output_dir', default=os.path.join(input_dir, 'output'), is_path=True), 'Output directory', is_dir=True, preexists=False)
        if not os.path.exists(output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        loss_factor = inputs.get('loss_factor', default=1.0, required=False)

        net_ri = inputs.get('net_ri_losses', default=False, required=False)

        il = ri = False
        try:
            [fn for fn in os.listdir(input_dir) if fn.lower().startswith('account')][0]
        except IndexError:
            pass
        else:
            il = True
            try:
                [fn for fn in os.listdir(input_dir) if 'reinsinfo' in fn.lower()][0]
            except IndexError:
                pass
            else:
                try:
                    [fn for fn in os.listdir(input_dir) if 'reinsscope' in fn.lower()][0]
                except IndexError:
                    pass
                else:
                    ri = True

        self.logger.info('\nRunning deterministic losses (GUL=True, IL={}, RI={})'.format(il, ri))
        guls, ils, rils = om().run_deterministic(
            input_dir,
            output_dir,
            loss_percentage_of_tiv=loss_factor,
            net_ri=net_ri
        )
        print_dataframe(guls, frame_header='Ground-up losses (loss_factor={})'.format(loss_factor), string_cols=guls.columns)
        if il:
            print_dataframe(ils, frame_header='Direct insured losses (loss_factor={})'.format(loss_factor), string_cols=ils.columns)
        if ri:
            print_dataframe(rils, frame_header='Reinsurance losses  (loss_factor={}; net={})'.format(loss_factor, net_ri), string_cols=rils.columns)


class ValidateCmd(OasisBaseCommand):
    pass


class ExposureCmd(OasisBaseCommand):
    """
    Exposure subcommands::

        * generate deterministic losses (GUL, IL or RI)
        * validate OED exposure (not yet implemented)
    """
    sub_commands = {
        'run': RunCmd,
        'validate': ValidateCmd
    }
