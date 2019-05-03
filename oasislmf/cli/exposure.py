__all__ = [
    'ExposureCmd',
    'RunCmd'
]

import os

from argparse import RawDescriptionHelpFormatter
from filecmp import cmp as compare_files
from itertools import chain

from pathlib2 import Path

from ..manager import OasisManager as om
from ..model_preparation import oed
from ..utils.data import (
    print_dataframe,
)
from ..utils.defaults import (
    KTOOLS_ALLOC_RULE,
    OASIS_FILES_PREFIXES,
)
from ..utils.diff import column_diff
from ..utils.exceptions import OasisException
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
            '-t', '--test-case-name', type=str, default=None, required=False, help='Test case name'
        )
        parser.add_argument(
            '-s', '--src-dir', type=str, default=None, required=True,
            help='Source files directory - should contain the OED exposure file + optionally the accounts, and RI info. and scope files'
        )
        parser.add_argument(
            '-r', '--run-dir', type=str, default=None, required=False, help='Run directory'
        )
        parser.add_argument(
            '-l', '--loss-factor', type=float, default=None,
            help='Loss factor to apply to TIVs.'
        )
        parser.add_argument(
            '-n', '--net-ri-losses', default=False, help='Net RI losses', action='store_true'
        )
        parser.add_argument(
            '-a', '--alloc-rule', type=int, default=KTOOLS_ALLOC_RULE, help='Alloc rule ID'
        )
        parser.add_argument(
            '-v', '--validate', default=False, help='Validate', action='store_true'
        )

    def action(self, args):
        """
        Generates deterministic losses using the installed ktools framework given
        direct Oasis files (GUL + optionally IL and RI input files).

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        test_case_name = inputs.get('test_case_name')

        self.logger.info('\nProcessing arguments for {}'.format(test_case_name) if test_case_name else '\nProcessing arguments')

        call_dir = os.getcwd()

        src_dir = as_path(inputs.get('src_dir', default=call_dir, is_path=True), 'Source files directory', is_dir=True, preexists=True)

        run_dir = as_path(inputs.get('run_dir', default=os.path.join(src_dir, 'run'), is_path=True), 'Run directory', is_dir=True, preexists=False)
        if not os.path.exists(run_dir):
            Path(run_dir).mkdir(parents=True, exist_ok=True)

        loss_factor = inputs.get('loss_factor', default=1.0, required=False)

        net_ri = inputs.get('net_ri_losses', default=False, required=False)

        alloc_rule = inputs.get('alloc_rule', default=KTOOLS_ALLOC_RULE, required=False)

        validate = inputs.get('validate', default=False, required=False)

        src_contents = [fn.lower() for fn in os.listdir(src_dir)]

        if 'location.csv' not in src_contents:
            raise OasisException(
                'No location/exposure file found in source directory - '
                'a file named `location.csv` is expected'
            )

        il = ril = False
        try:
            assert('account.csv' in src_contents)
        except AssertionError:
            pass
        else:
            il = True
            try:
                assert([fn for fn in src_contents if 'reinsinfo' in fn])
            except AssertionError:
                pass
            else:
                try:
                    assert([fn for fn in src_contents if 'reinsscope' in fn])
                except AssertionError:
                    pass
                else:
                    ril = True

        self.logger.info('\nRunning deterministic losses (GUL=True, IL={}, RIL={})\n'.format(il, ril))
        guls, ils, rils = om().run_deterministic(
            src_dir,
            run_dir=run_dir,
            loss_percentage_of_tiv=loss_factor,
            net_ri=net_ri,
            alloc_rule=alloc_rule
        )
        guls.to_csv(path_or_buf=os.path.join(run_dir, 'guls.csv'), index=False, encoding='utf-8')
        print_dataframe(guls, frame_header='Ground-up losses (loss_factor={})'.format(loss_factor), string_cols=guls.columns)

        if il:
            ils.to_csv(path_or_buf=os.path.join(run_dir, 'ils.csv'), index=False, encoding='utf-8')
            print_dataframe(ils, frame_header='Direct insured losses (loss_factor={})'.format(loss_factor), string_cols=ils.columns)

        if ril:
            rils.to_csv(path_or_buf=os.path.join(run_dir, 'rils.csv'), index=False, encoding='utf-8')
            print_dataframe(rils, frame_header='Reinsurance losses  (loss_factor={}; net={})'.format(loss_factor, net_ri), string_cols=rils.columns)

        # Do not validate if the loss factor < 1 - this is because the
        # expected data files for validation are based on a loss factor
        # of 1.0
        if loss_factor < 1:
            validate = False

        if validate:
            expected_data_dir = os.path.join(src_dir, 'expected')
            if not os.path.exists(expected_data_dir):
                raise OasisException(
                    'No subfolder named `expected` found in the input directory - '
                    'this subfolder should contain the expected set of GUL + IL '
                    'input files, optionally the RI input files, and the expected '
                    'set of GUL, IL and optionally the RI loss files'
                )

            files = [
                '{}.csv'.format(fn)
                for ft, fn in chain(OASIS_FILES_PREFIXES['gul'].items(), OASIS_FILES_PREFIXES['il'].items())
            ]
            files += ['guls.csv']
            if il:
                files += ['ils.csv']
            if ril:
                files += ['rils.csv']

            status = 'PASS'
            for f in files:
                generated = os.path.join(run_dir, f)
                expected = os.path.join(expected_data_dir, f)
                self.logger.info('\nComparing generated {} vs expected {}'.format(generated, expected))
                try:
                    assert(compare_files(generated, expected) is True)
                except AssertionError:
                    status = 'FAIL'
                    self.logger.info('\n{}'.format(column_diff(generated, expected)))
                    self.logger.info('\tFAIL')
                else:
                    self.logger.info('\n\tPASS')

            self.logger.info(
                '\n{} validation complete: {}'.format(test_case_name, status) if test_case_name
                else 'Validation complete: {}'.format(status)
            )


class ExposureCmd(OasisBaseCommand):
    """
    Exposure subcommands::

        * generate - and optionally, validate - deterministic losses (GUL, IL or RIL)
    """
    sub_commands = {
        'run': RunCmd
    }
