__all__ = [
    'RunExposure',
    'RunFmTest',
]

import tempfile
import os
import csv
import shutil

from itertools import chain
import pandas as pd

from ..base import ComputationStep
from ..generate.keys import GenerateKeysDeterministic
from ..generate.files import GenerateFiles
from ..generate.losses import GenerateLossesDeterministic


from ...preparation.il_inputs import get_oed_hierarchy
from ...utils.exceptions import OasisException

from ...utils.data import (
    get_dataframe,
    print_dataframe,
)
from ...utils.inputs import str2bool
from ...utils.coverages import SUPPORTED_COVERAGE_TYPES
from ...utils.defaults import (
    KTOOLS_ALLOC_IL_DEFAULT,
    KTOOLS_ALLOC_RI_DEFAULT,
    KTOOLS_ALLOC_FM_MAX,
    OASIS_FILES_PREFIXES,
)


class RunExposure(ComputationStep):
    """
    Generates insured losses from preexisting Oasis files with specified
    loss factors (loss % of TIV).
    """
    step_params = [
        {'name': 'src_dir',              'flag':'-s', 'is_path': True, 'pre_exist': True, 'help': ''},
        {'name': 'run_dir',              'flag':'-r', 'is_path': True, 'pre_exist': False, 'help': ''},
        {'name': 'output_file',          'flag':'-f', 'is_path': True, 'pre_exist': False, 'help': '', 'type': str},
        {'name': 'loss_factor',          'flag':'-l', 'type' :float, 'nargs':'+', 'help': '', 'default': [1.0]},
        {'name': 'ktools_alloc_rule_il', 'flag':'-a', 'default': KTOOLS_ALLOC_IL_DEFAULT,  'type':int, 'help': 'Set the fmcalc allocation rule used in direct insured loss'},
        {'name': 'ktools_alloc_rule_ri', 'flag':'-A', 'default': KTOOLS_ALLOC_RI_DEFAULT,  'type':int, 'help': 'Set the fmcalc allocation rule used in reinsurance'},
        {'name': 'output_level',         'flag':'-o', 'help': 'Keys files output format', 'choices':['item', 'loc', 'pol', 'acc', 'port'], 'default': 'item'},
        {'name': 'num_subperils',        'flag':'-p', 'default': 1,  'type':int,          'help': 'Set the number of subperils returned by deterministic key generator'},
        {'name': 'coverage_types',       'type' :int, 'nargs':'+', 'default': list(v['id'] for v in SUPPORTED_COVERAGE_TYPES.values()), 'help': 'Select List of supported coverage_types [1, .. ,4]'},
        {'name': 'fmpy',                 'default': True, 'type': str2bool, 'const':True, 'nargs':'?', 'help': 'use fmcalc python version instead of c++ version'},
        {'name': 'fmpy_low_memory',      'default': False, 'type': str2bool, 'const':True, 'nargs':'?', 'help': 'use memory map instead of RAM to store loss array (may decrease performance but reduce RAM usage drastically)'},
        {'name': 'fmpy_sort_output', 'default': False, 'type': str2bool, 'const': True, 'nargs': '?', 'help': 'order fmpy output by item_id'},
        {'name': 'stream_type',          'flag':'-t', 'default': 2,  'type':int,  'help': 'Set the IL input stream type, 2 = default loss stream, 1 = deprecated cov/item stream'},
        {'name': 'net_ri', 'default': True},
        {'name': 'include_loss_factor', 'default': True},
        {'name': 'print_summary', 'default': True},
    ]

    def _check_alloc_rules(self):
        alloc_ranges = {
            'ktools_alloc_rule_il': KTOOLS_ALLOC_FM_MAX,
            'ktools_alloc_rule_ri': KTOOLS_ALLOC_FM_MAX}
        for rule in alloc_ranges:
            alloc_val = getattr(self, rule)
            if (alloc_val < 0) or (alloc_val > alloc_ranges[rule]):
                raise OasisException(f'Error: {rule}={alloc_val} - Not withing valid range [0..{alloc_ranges[rule]}]')

    def run(self):
        tmp_dir = None
        src_dir = self.src_dir if self.src_dir else os.getcwd()

        if self.run_dir:
            run_dir = self.run_dir
        else:
            tmp_dir = tempfile.TemporaryDirectory()
            run_dir = tmp_dir.name

        include_loss_factor = not (len(self.loss_factor) == 1)
        src_contents = [fn.lower() for fn in os.listdir(src_dir)]

        self._check_alloc_rules()


        if 'location.csv' not in src_contents:
            raise OasisException(
                'No location/exposure file found in source directory - '
                'a file named `location.csv` is expected'
            )

        il = ril = False
        il = ('account.csv' in src_contents)
        ril = il and ('ri_info.csv' in src_contents) and ('ri_scope.csv' in src_contents)

        self.logger.debug('\nRunning deterministic losses (GUL=True, IL={}, RIL={})\n'.format(il, ril))

        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        contents = [fn.lower() for fn in os.listdir(src_dir)]
        location_fp = [os.path.join(src_dir, fn) for fn in contents if fn == 'location.csv'][0]
        accounts_fp = [os.path.join(src_dir, fn) for fn in contents if fn == 'account.csv'][0]
        ri_info_fp = ri_scope_fp = None
        try:
            ri_info_fp = [os.path.join(src_dir, fn) for fn in contents if fn == 'ri_info.csv'][0]
        except IndexError:
            pass
        else:
            try:
                ri_scope_fp = [os.path.join(src_dir, fn) for fn in contents if fn == 'ri_scope.csv'][0]
            except IndexError:
                ri_info_fp = None

        # 1. Create Deterministic keys file
        keys_fp = os.path.join(run_dir, 'keys.csv')
        GenerateKeysDeterministic(
            oed_location_csv=location_fp,
            keys_data_csv=keys_fp,
            num_subperils=self.num_subperils,
            supported_oed_coverage_types=self.coverage_types,
        ).run()

        # 2. Start Oasis files generation
        GenerateFiles(
           oasis_files_dir=run_dir,
            oed_location_csv=location_fp,
            oed_accounts_csv=accounts_fp,
            oed_info_csv=ri_info_fp,
            oed_scope_csv=ri_scope_fp,
            keys_data_csv=keys_fp,
        ).run()

        # 3. Run Deterministic Losses
        losses = GenerateLossesDeterministic(
            oasis_files_dir=run_dir,
            output_dir=os.path.join(run_dir, 'output'),
            include_loss_factor=include_loss_factor,
            loss_factor=self.loss_factor,
            #num_subperils=self.num_subperils,
            net_ri=self.net_ri,
            ktools_alloc_rule_il=self.ktools_alloc_rule_il,
            ktools_alloc_rule_ri=self.ktools_alloc_rule_ri,
            fmpy=self.fmpy,
            fmpy_low_memory=self.fmpy_low_memory,
            fmpy_sort_output=self.fmpy_sort_output,
            il_stream_type=self.stream_type,
        ).run()

        guls_df = losses['gul']
        ils_df = losses['il']
        rils_df = losses['ri']

        # Read in the summary map
        summaries_df = get_dataframe(src_fp=os.path.join(run_dir, 'fm_summary_map.csv'))

        guls_df.to_csv(path_or_buf=os.path.join(run_dir, 'guls.csv'), index=False, encoding='utf-8')
        guls_df.rename(columns={'loss': 'loss_gul'}, inplace=True)

        guls_df = guls_df.merge(
            right=summaries_df,
            left_on=["item_id"],
            right_on=["agg_id"]
        )

        if include_loss_factor:
            join_cols = ["event_id", "output_id", "loss_factor_idx"]
        else:
            join_cols = ["event_id", "output_id"]

        if il:
            ils_df.to_csv(path_or_buf=os.path.join(run_dir, 'ils.csv'), index=False, encoding='utf-8')
            ils_df.rename(columns={'loss': 'loss_il'}, inplace=True)
            all_losses_df = guls_df.merge(
                how='left',
                right=ils_df,
                on=join_cols,
                suffixes=["_gul", "_il"]
            )
        if ril:
            rils_df.to_csv(path_or_buf=os.path.join(run_dir, 'rils.csv'), index=False, encoding='utf-8')
            rils_df.rename(columns={'loss': 'loss_ri'}, inplace=True)
            all_losses_df = all_losses_df.merge(
                how='left',
                right=rils_df,
                on=join_cols
            )

        oed_hierarchy = get_oed_hierarchy()
        portfolio_num = oed_hierarchy['portnum']['ProfileElementName'].lower()
        acc_num = oed_hierarchy['accnum']['ProfileElementName'].lower()
        loc_num = oed_hierarchy['locnum']['ProfileElementName'].lower()
        policy_num = oed_hierarchy['polnum']['ProfileElementName'].lower()

        if self.output_level == 'port':
            summary_cols = [portfolio_num]
        elif self.output_level == 'acc':
            summary_cols = [portfolio_num, acc_num]
        elif self.output_level == 'pol':
            summary_cols = [portfolio_num, acc_num, policy_num]
        elif self.output_level == 'loc':
            summary_cols = [portfolio_num, acc_num, loc_num]
        elif self.output_level == 'item':
            summary_cols = [
                'output_id', portfolio_num, acc_num, loc_num, policy_num,
                'coverage_type_id']

        if include_loss_factor:
            group_by_cols = summary_cols + ['loss_factor_idx']
        else:
            group_by_cols = summary_cols
        guls_df = guls_df.loc[:, group_by_cols + ['loss_gul']]

        if not il and not ril:
            all_loss_cols = group_by_cols + ['loss_gul']
            all_losses_df = guls_df.loc[:, all_loss_cols]
            all_losses_df.drop_duplicates(keep=False, inplace=True)
        elif not ril:
            all_loss_cols = group_by_cols + ['loss_gul', 'loss_il']
            all_losses_df = all_losses_df.loc[:, all_loss_cols]
            summary_gul_df = pd.DataFrame(
                {'loss_gul': guls_df.groupby(group_by_cols)['loss_gul'].sum()}).reset_index()
            summary_il_df = pd.DataFrame(
                {'loss_il': all_losses_df.groupby(group_by_cols)['loss_il'].sum()}).reset_index()
            all_losses_df = summary_gul_df.merge(how='left', right=summary_il_df, on=group_by_cols)
        else:
            all_loss_cols = group_by_cols + ['loss_gul', 'loss_il', 'loss_ri']
            all_losses_df = all_losses_df.loc[:, all_loss_cols]
            summary_gul_df = pd.DataFrame(
                {'loss_gul': guls_df.groupby(group_by_cols)['loss_gul'].sum()}).reset_index()
            summary_il_df = pd.DataFrame(
                {'loss_il': all_losses_df.groupby(group_by_cols)['loss_il'].sum()}).reset_index()
            summary_ri_df = pd.DataFrame(
                {'loss_ri': all_losses_df.groupby(group_by_cols)['loss_ri'].sum()}).reset_index()
            all_losses_df = summary_gul_df.merge(how='left', right=summary_il_df, on=group_by_cols)
            all_losses_df = all_losses_df.merge(how='left', right=summary_ri_df, on=group_by_cols)

        for i in range(len(self.loss_factor)):

            if include_loss_factor:
                total_gul = guls_df[guls_df.loss_factor_idx == i].loss_gul.sum()
            else:
                total_gul = guls_df.loss_gul.sum()

            if not il and not ril:
                all_loss_cols = all_loss_cols + ['loss_gul']
                all_losses_df = guls_df.loc[:, all_loss_cols]
                all_losses_df.drop_duplicates(keep=False, inplace=True)
                header = \
                    'Losses (loss factor={:.2%}; total gul={:,.00f})'.format(
                        self.loss_factor[i],
                        total_gul)
            elif not ril:
                if include_loss_factor:
                    total_il = ils_df[ils_df.loss_factor_idx == i].loss_il.sum()
                else:
                    total_il = ils_df.loss_il.sum()

                header = \
                    'Losses (loss factor={:.2%}; total gul={:,.00f}; total il={:,.00f})'.format(
                        self.loss_factor[i],
                        total_gul, total_il)
            else:
                if include_loss_factor:
                    total_il = ils_df[ils_df.loss_factor_idx == i].loss_il.sum()
                    total_ri_net = rils_df[rils_df.loss_factor_idx == i].loss_ri.sum()
                else:
                    total_il = ils_df.loss_il.sum()
                    total_ri_net = rils_df.loss_ri.sum()
                total_ri_ceded = total_il - total_ri_net
                header = \
                    'Losses (loss factor={:.2%}; total gul={:,.00f}; total il={:,.00f}; total ri ceded={:,.00f})'.format(
                        self.loss_factor[i],
                        total_gul, total_il, total_ri_ceded)

            # Convert output cols to strings for formatting
            for c in group_by_cols:
                all_losses_df[c] = all_losses_df[c].apply(str)

            if self.print_summary:
                cols_to_print = all_loss_cols.copy()
                if False:
                    cols_to_print.remove('loss_factor_idx')
                if include_loss_factor:
                    print_dataframe(
                        all_losses_df[all_losses_df.loss_factor_idx == str(i)],
                        frame_header=header,
                        cols=cols_to_print)
                else:
                    print_dataframe(
                        all_losses_df,
                        frame_header=header,
                        cols=cols_to_print)

        if self.output_file:
            all_losses_df.to_csv(self.output_file, index=False, encoding='utf-8')

        if tmp_dir:
            tmp_dir.cleanup()

        return (il, ril)


class RunFmTest(ComputationStep):
    """
    Runs an FM test case and validates generated
    losses against expected losses.

    only use 'update_expected' for debugging
    it replaces the expected file with generated
    """

    step_params = [
        {'name': 'test_case_name', 'flag': '-c', 'type': str, 'help': 'Test case name - runs a specific test in the test directory. Otherwise run all tests.'},
        {'name': 'test_case_dir', 'flag': '-t', 'help': 'Test directory - should contain test directories containing OED files and expected results'},
        {'name': 'list_tests', 'flag': '-l', 'action': 'store_true', 'help': 'List the valid test cases in the test directory rather than running'},
        {'name': 'run_dir', 'flag': '-r', 'help': 'Run directory - where files should be generated. If not sst, no files will be saved.'},
        {'name': 'test_tolerance', 'type' :float, 'help': 'Relative tolerance between expected values and results, default is "1e-4" or 0.0001', 'default': 1e-4},
        {'name': 'fmpy', 'default': True, 'type': str2bool, 'const': True, 'nargs': '?', 'help': 'use fmcalc python version instead of c++ version'},
        {'name': 'fmpy_low_memory', 'default': False, 'type': str2bool, 'const': True, 'nargs': '?', 'help': 'use memory map instead of RAM to store loss array (may decrease performance but reduce RAM usage drastically)'},
        {'name': 'fmpy_sort_output', 'default': False, 'type': str2bool, 'const': True, 'nargs': '?', 'help': 'order fmpy output by item_id'},
        {'name': 'update_expected', 'default': False},
    ]

    def search_test_cases(self):
        case_names = []
        for test_case in os.listdir(path=self.test_case_dir):
            if os.path.exists(
                os.path.join(self.test_case_dir, test_case, "expected")
            ):
                case_names.append(test_case)
        case_names.sort()
        return case_names, len(case_names)

    def run(self):

        # Setup and search test case dir
        if not self.test_case_dir:
            self.test_case_dir = os.getcwd()
        case_names, case_num = self.search_test_cases()

        # If enabled, list found test cases and exit
        if self.list_tests:
            for name in case_names:
                self.logger.info(name)
            exit(0)

        # Check selected test case exisits
        if self.test_case_name:
            if self.test_case_name not in case_names:
                raise OasisException(f'Error: case "{self.test_case_name}" not found in "{self.test_case_dir}"')

            return self.execute_test_case(self.test_case_name)

        # If test_case not selected run all cases
        self.logger.info(f"Running: All tests in '{self.test_case_dir}'")
        failed_tests = []
        exit_status = 0
        for case in case_names:
            test_result = self.execute_test_case(case)

            if not test_result:
                failed_tests.append(case)
                exit_status = 1

        if len(failed_tests) == 0:
            self.logger.info("All tests passed")
        else:
            self.logger.info("{} test failed: ".format(len(failed_tests)))
            [self.logger.info(n) for n in failed_tests]
        exit(exit_status)

    def execute_test_case(self, test_case):
        if self.run_dir:
            tmp_dir = None
            run_dir = self.run_dir
        else:
            tmp_dir = tempfile.TemporaryDirectory()
            run_dir = tmp_dir.name

        test_dir = os.path.join(self.test_case_dir, test_case)
        output_level = 'loc'
        loss_factor_fp = os.path.join(test_dir, 'loss_factors.csv')
        loss_factor = []
        include_loss_factor = False

        if os.path.exists(loss_factor_fp):
            loss_factor = []
            include_loss_factor = True
            try:
                with open(loss_factor_fp, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        loss_factor.append(
                            float(row['loss_factor']))
            except Exception as e:
                raise OasisException(f"Failed to read {loss_factor_fp}", e)
        else:
            loss_factor.append(1.0)

        output_file = os.path.join(run_dir, 'loc_summary.csv')
        (il, ril) = RunExposure(
            src_dir=test_dir,
            run_dir=run_dir,
            loss_factor=loss_factor,
            output_level=output_level,
            output_file=output_file,
            include_loss_factor=include_loss_factor,
            fmpy=self.fmpy,
            fmpy_low_memory=self.fmpy_low_memory,
            fmpy_sort_output=self.fmpy_sort_output
        ).run()

        expected_data_dir = os.path.join(test_dir, 'expected')
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

        test_result = True
        for f in files:
            generated = os.path.join(run_dir, f)
            expected = os.path.join(expected_data_dir, f)

            if not os.path.exists(expected):
                continue

            try:
                pd.testing.assert_frame_equal(
                    pd.read_csv(expected),
                    pd.read_csv(generated),
                    check_exact=False,
                    rtol=self.test_tolerance
                )
            except AssertionError:
                if self.update_expected:
                    shutil.copyfile(generated, expected)
                else:
                    print("Expected:")
                    with open(expected) as f:
                        self.logger.info(f.read())
                    print("Generated:")
                    with open(generated) as f:
                        self.logger.info(f.read())
                    raise OasisException(
                        f'\n FAIL: generated {generated} vs expected {expected}'
                    )
                test_result = False
        if tmp_dir:
            tmp_dir.cleanup()
        return test_result
