__all__ = [
    'ExposureCmd',
    'RunCmd'
]

import os
import sys

from argparse import RawDescriptionHelpFormatter
from filecmp import cmp as compare_files
from itertools import chain

import pandas as pd

from pathlib2 import Path

from ..manager import OasisManager as om
from ..utils.data import (
    get_dataframe,
    print_dataframe,
)
from ..utils.defaults import (
    KTOOLS_ALLOC_IL_DEFAULT,
    KTOOLS_ALLOC_RI_DEFAULT,
    OASIS_FILES_PREFIXES,
)
from ..utils.diff import column_diff
from ..utils.exceptions import OasisException
from ..utils.path import as_path
from ..utils.profiles import get_oed_hierarchy

from .base import OasisBaseCommand
from .inputs import InputValues


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
            '-r', '--run-dir', type=str, default=None, required=False, help='Run directory - where files should be generated'
        )
        parser.add_argument(
            '-l', '--loss-factor', type=float, default=None,
            help='Loss factor to apply to TIVs - default is 1.0.'
        )
        parser.add_argument(
            '-a', '--alloc-rule-il', type=int, default=KTOOLS_ALLOC_IL_DEFAULT, help='Ktools IL back allocation rule to apply - default is 2, i.e. prior level loss basis'
        )
        parser.add_argument(
            '-A', '--alloc-rule-ri', type=int, default=KTOOLS_ALLOC_RI_DEFAULT, help='Ktools RI back allocation rule to apply - default is 3, i.e. All level loss basis'
        )
        parser.add_argument(
            '-v', '--validate', default=False, help='Validate input files and loss tables - default is False', action='store_true'
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

        test_case_name = inputs.get('test_case_name')

        self.logger.info('\nProcessing arguments for {}'.format(test_case_name) if test_case_name else '\nProcessing arguments')

        call_dir = os.getcwd()

        src_dir = as_path(inputs.get('src_dir', default=call_dir, is_path=True), 'Source files directory', is_dir=True, preexists=True)

        run_dir = as_path(inputs.get('run_dir', default=os.path.join(src_dir, 'run'), is_path=True), 'Run directory', is_dir=True, preexists=False)
        if not os.path.exists(run_dir):
            Path(run_dir).mkdir(parents=True, exist_ok=True)

        loss_factor = inputs.get('loss_factor', default=1.0, required=False)

        net_ri = True

        il_alloc_rule = inputs.get('alloc_rule_il', default=KTOOLS_ALLOC_IL_DEFAULT, required=False)
        ri_alloc_rule = inputs.get('alloc_rule_ri', default=KTOOLS_ALLOC_RI_DEFAULT, required=False)

        validate = inputs.get('validate', default=False, required=False)

        # item, loc, pol, acc, port
        output_level = inputs.get('output_level', default="item", required=False)
        if output_level not in ['port', 'acc', 'loc', 'pol', 'item']:
            raise OasisException(
                'Invalid output level. Must be one of port, acc, loc, pol or item.'
            )

        output_file = as_path(inputs.get('output_file', required=False, is_path=True), 'Output file path', preexists=False)

        src_contents = [fn.lower() for fn in os.listdir(src_dir)]

        if 'location.csv' not in src_contents:
            raise OasisException(
                'No location/exposure file found in source directory - '
                'a file named `location.csv` is expected'
            )

        il = ril = False
        il = ('account.csv' in src_contents)
        ril = il and ('ri_info.csv' in src_contents) and ('ri_scope.csv' in src_contents)

        self.logger.info('\nRunning deterministic losses (GUL=True, IL={}, RIL={})\n'.format(il, ril))
        guls_df, ils_df, rils_df = om().run_deterministic(
            src_dir,
            run_dir=run_dir,
            loss_percentage_of_tiv=loss_factor,
            net_ri=net_ri,
            il_alloc_rule=il_alloc_rule,
            ri_alloc_rule=ri_alloc_rule
        )

        # Read in the summary map
        summaries_df = get_dataframe(src_fp=os.path.join(run_dir, 'fm_summary_map.csv'))

        guls_df.to_csv(path_or_buf=os.path.join(run_dir, 'guls.csv'), index=False, encoding='utf-8')
        guls_df.rename(columns={'loss': 'loss_gul'}, inplace=True)

        total_gul = guls_df.loss_gul.sum()

        guls_df = guls_df.merge(
            right=summaries_df,
            left_on=["item_id"],
            right_on=["agg_id"]
        )

        if il:
            ils_df.to_csv(path_or_buf=os.path.join(run_dir, 'ils.csv'), index=False, encoding='utf-8')
            ils_df.rename(columns={'loss': 'loss_il'}, inplace=True)
            all_losses_df = guls_df.merge(
                how='left',
                right=ils_df,
                on=["event_id", "output_id"],
                suffixes=["_gul", "_il"]
            )
        if ril:
            rils_df.to_csv(path_or_buf=os.path.join(run_dir, 'rils.csv'), index=False, encoding='utf-8')
            rils_df.rename(columns={'loss': 'loss_ri'}, inplace=True)
            all_losses_df = all_losses_df.merge(
                how='left',
                right=rils_df,
                on=["event_id", "output_id"]
            )

        oed_hierarchy = get_oed_hierarchy()
        portfolio_num = oed_hierarchy['portnum']['ProfileElementName'].lower()
        acc_num = oed_hierarchy['accnum']['ProfileElementName'].lower()
        loc_num = oed_hierarchy['locnum']['ProfileElementName'].lower()
        policy_num = oed_hierarchy['polnum']['ProfileElementName'].lower()

        if output_level == 'port':
            summary_cols = [portfolio_num]
        elif output_level == 'acc':
            summary_cols = [portfolio_num, acc_num]
        elif output_level == 'pol':
            summary_cols = [portfolio_num, acc_num, policy_num]
        elif output_level == 'loc':
            summary_cols = [portfolio_num, acc_num, loc_num]
        elif output_level == 'item':
            summary_cols = ['output_id', portfolio_num, acc_num, loc_num, policy_num, 'coverage_type_id']

        guls_df = guls_df.loc[:, summary_cols + ['loss_gul']]

        if not il and not ril:
            all_losses_df = guls_df.loc[:, summary_cols + ['loss_gul']]
            all_losses_df.drop_duplicates(keep=False, inplace=True)
            header = 'Losses (loss factor={}; total gul={:,.00f})'.format(loss_factor, total_gul)
        elif not ril:
            total_il = ils_df.loss_il.sum()
            all_losses_df = all_losses_df.loc[:, summary_cols + ['loss_gul', 'loss_il']]
            summary_gul_df = pd.DataFrame({'loss_gul': guls_df.groupby(summary_cols)['loss_gul'].sum()}).reset_index()
            summary_il_df = pd.DataFrame({'loss_il': all_losses_df.groupby(summary_cols)['loss_il'].sum()}).reset_index()
            all_losses_df = summary_gul_df.merge(how='left', right=summary_il_df, on=summary_cols)
            header = 'Losses (loss factor={}; total gul={:,.00f}; total il={:,.00f})'.format(
                loss_factor,
                total_gul,
                total_il
            )
        else:
            total_il = ils_df.loss_il.sum()
            total_ri_net = rils_df.loss_ri.sum()
            total_ri_ceded = total_il - total_ri_net
            all_losses_df = all_losses_df.loc[:, summary_cols + ['loss_gul', 'loss_il', 'loss_ri']]
            summary_gul_df = pd.DataFrame({'loss_gul': guls_df.groupby(summary_cols)['loss_gul'].sum()}).reset_index()
            summary_il_df = pd.DataFrame({'loss_il': all_losses_df.groupby(summary_cols)['loss_il'].sum()}).reset_index()
            summary_ri_df = pd.DataFrame({'loss_ri': all_losses_df.groupby(summary_cols)['loss_ri'].sum()}).reset_index()
            all_losses_df = summary_gul_df.merge(how='left', right=summary_il_df, on=summary_cols)
            all_losses_df = all_losses_df.merge(how='left', right=summary_ri_df, on=summary_cols)
            header = 'Losses (loss factor={}; total gul={:,.00f}; total il={:,.00f}; total ri ceded={:,.00f})'.format(
                loss_factor, total_gul, total_il, total_ri_ceded)

        print_dataframe(all_losses_df, frame_header=header, string_cols=all_losses_df.columns)
        if output_file:
            all_losses_df.to_csv(output_file, index=False, encoding='utf-8')

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

            files = ['keys.csv', 'loc_summary.csv']
            files += [
                '{}.csv'.format(fn)
                for ft, fn in chain(OASIS_FILES_PREFIXES['gul'].items(), OASIS_FILES_PREFIXES['il'].items())
            ]
            files += ['gul_summary_map.csv', 'guls.csv']
            if il:
                files += ['fm_summary_map.csv', 'ils.csv']
            if ril:
                files += ['rils.csv']

            status = 'PASS'
            for f in files:
                generated = os.path.join(run_dir, f)
                expected = os.path.join(expected_data_dir, f)

                if not os.path.exists(expected):
                    continue

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

            sys.exit(0 if status == 'PASS' else -1)


class ExposureCmd(OasisBaseCommand):
    """
    Exposure subcommands::

        * generate - and optionally, validate - deterministic losses (GUL, IL or RIL)
    """
    sub_commands = {
        'run': RunCmd
    }
