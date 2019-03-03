# -*- coding: utf-8 -*-

import os
import subprocess32 as subprocess
import time
import unittest

from backports.tempfile import TemporaryDirectory
from collections import OrderedDict

import pandas as pd
import numpy as np

from pandas.util.testing import assert_frame_equal

from parameterized import parameterized

from oasislmf.model_preparation import (
    oed,
    reinsurance_layer,
)
from oasislmf.model_execution import bin
from oasislmf.utils.data import get_dataframe, set_col_dtypes
from .direct_layer import DirectLayer

cwd = os.path.dirname(os.path.realpath(__file__))
expected_output_dir = os.path.join(cwd, 'expected', 'calc')

input_dir = os.path.join(cwd, 'examples')
test_examples = [
    'single_loc_level_fac',
    'single_pol_level_fac',
    'single_acc_level_fac',
    'single_lgr_level_fac',
    'multiple_facs_same_inuring_level',
    'single_loc_level_PR_all_risks',
    'single_lgr_level_PR_all_risks',
    'single_pol_level_PR_all_risks',
    'single_acc_level_PR_all_risks',
    'single_loc_level_SS_all_risks_loc',
    'single_loc_level_SS_all_risks_loc_pol',
    'single_pol_level_SS_all_risks',
    'single_acc_level_SS_all_risks',
    'single_QS_one_risk_with_%_ceded_and_%_placed_and_risk_limit_and_occ_limit',
    'single_QS_all_acc_with_%_ceded_and_%_placed_and_risk_limit_and_occ_limit',
    'single_loc_level_SS_with_%_ceded_and_%_placed',
    'single_pol_level_SS_with_%_ceded_and_%_placed',
    'single_QS_all_risks_with_%_ceded_and_%_placed',
    'single_QS_with_account_level_risk_limits',
    'single_cxl',
    'single_cxl_0_occ_limit_treat_as_unlimited',
    'single_cxl_empty_occ_limit_treat_as_unlimited',
    'single_QS_with_policy_level_risk_limits',
    'single_QS_with_location_level_risk_limits',
    'single_cxl_one_account',
    'single_QS_one_account_with_%_ceded_and_%_placed',
    'multiple_SS_same_inuring_level',
    'single_loc_level_SS_with_%_ceded_and_%_placed_and_occ_limit',
    'multiple_cxl_at_different_inuring_levels',
    'multiple_qs_all_risks_multiple_portfolios',
    'multiple_qs_single_portfolio',
    'multiple_QS_at_different_inuring_levels',
    'multiple_QS_at_same_inuring_level',
    'single_QS',
    'single_QS_no_ReinsLayerNumber_no_TreatyShare',
    'single_loc_level_PR_loc_filter',
    'single_loc_level_PR_pol_and_loc_filter',
    'single_loc_level_PR_pol_filter',
    'single_loc_level_PR_acc_filter',
    'simple_CXL_port_acc_pol_filter',
    'simple_CXL_port_acc_pol_loc_filter',
    'simple_CXL_port_acc_loc_filter',
    'simple_CXL_port_filter',
    'simple_CXL_port_acc_filter'
    ]

fm_examples = [
    'fm24',
    'fm27']

# error_examples = [
#     'single_loc_level_fac_with_pol_scope_error',
#     'single_loc_level_fac_with_acc_scope_error',
#     'single_loc_level_fac_with_lgr_scope_error',
#     'single_pol_level_fac_with_acc_scope_error',
#     'single_pol_level_fac_with_loc_scope_error',
#     'single_pol_level_fac_with_lgr_scope_error',
#     'single_acc_level_fac_with_pol_scope_error',
#     'single_acc_level_fac_with_loc_scope_error',
#     'single_acc_level_fac_with_lgr_scope_error',
#     'single_lgr_level_fac_with_acc_scope_error',
#     'single_lgr_level_fac_with_loc_scope_error',
#     'single_lgr_level_fac_with_pol_scope_error',                
#     'single_loc_level_pr_with_pol_scope_error',
#     'single_loc_level_pr_with_acc_scope_error',
#     'single_loc_level_pr_with_lgr_scope_error',
#     'single_pol_level_pr_with_acc_scope_error',
#     'single_pol_level_pr_with_loc_scope_error',
#     'single_pol_level_pr_with_lgr_scope_error',
#     'single_acc_level_pr_with_pol_scope_error',
#     'single_acc_level_pr_with_loc_scope_error',
#     'single_acc_level_pr_with_lgr_scope_error',
#     'single_lgr_level_pr_with_acc_scope_error',
#     'single_lgr_level_pr_with_loc_scope_error',
#     'single_lgr_level_pr_with_pol_scope_error',                
# ]

test_cases = []
for case in test_examples + fm_examples:
#for case in ['simple_CXL_port_acc_filter']:
    test_cases.append((
        case,
        os.path.join(input_dir, case),
        os.path.join(expected_output_dir, case)
    ))

class TestReinsurance(unittest.TestCase):

    def _run_fm(
            self,
            input_name,
            output_name,
            xref_descriptions,
            allocation=oed.ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID):

        command = "fmcalc -p {0} -n -a {2} < {1}.bin | tee {0}.bin | fmtocsv > {0}.csv".format(
            output_name, input_name, allocation)
        print(command)
        proc = subprocess.Popen(command, shell=True)
        proc.wait()
        if proc.returncode != 0:
            raise Exception("Failed to run fm")
        losses_df = pd.read_csv("{}.csv".format(output_name))
        inputs_df = pd.read_csv("{}.csv".format(input_name))

        losses_df.drop(losses_df[losses_df.sidx != 1].index, inplace=True)
        inputs_df.drop(inputs_df[inputs_df.sidx != 1].index, inplace=True)
        losses_df = pd.merge(
            inputs_df,
            losses_df, left_on='output_id', right_on='output_id',
            suffixes=('_pre', '_net'))

        losses_df = pd.merge(
            xref_descriptions,
            losses_df, left_on='xref_id', right_on='output_id')

        del losses_df['event_id_pre']
        del losses_df['sidx_pre']
        del losses_df['event_id_net']
        del losses_df['sidx_net']
        del losses_df['output_id']
        del losses_df['xref_id']
        return losses_df

    def _run_test(
            self,
            account_df, location_df, ri_info_df, ri_scope_df,
            loss_factor,
            do_reinsurance):
        """
        Run the direct and reinsurance layers through the Oasis FM.
        Returns an array of net loss data frames, the first for the direct layers
        and then one per inuring layer.
        """
        t_start = time.time()

        net_losses = OrderedDict()

        initial_dir = os.getcwd()
        try:

            with TemporaryDirectory() as run_dir:

                os.chdir(run_dir)

                direct_layer = DirectLayer(account_df, location_df)
                direct_layer.generate_oasis_structures()
                direct_layer.write_oasis_files()
                losses_df = direct_layer.apply_fm(
                    loss_percentage_of_tiv=loss_factor, net=False)
                net_losses['Direct'] = losses_df

                oed_validator = oed.OedValidator()
                if do_reinsurance:
                    (is_valid, error_msgs) = oed_validator.validate(ri_info_df, ri_scope_df)
                    if not is_valid:
                        print(error_msgs)
                        exit(1)

                ri_layers = reinsurance_layer.generate_files_for_reinsurance(
                    items=direct_layer.items,
                    coverages=direct_layer.coverages,
                    fm_xrefs=direct_layer.fm_xrefs,
                    xref_descriptions=direct_layer.xref_descriptions,
                    gulsummaryxref=pd.DataFrame(),
                    fmsummaryxref=pd.DataFrame(),
                    ri_info_df=ri_info_df,
                    ri_scope_df=ri_scope_df,
                    direct_oasis_files_dir='',
                )

                for idx in ri_layers:
                    '''
                    {'inuring_priority': 1, 'risk_level': 'LOC', 'directory': 'run/RI_1'}
                    {'inuring_priority': 1, 'risk_level': 'ACC', 'directory': 'run/RI_2'}
                    {'inuring_priority': 2, 'risk_level': 'LOC', 'directory': 'run/RI_3'}
                    {'inuring_priority': 3, 'risk_level': 'LOC', 'directory': 'run/RI_4'}
    
                    '''
                    if idx < 2:
                        input_name = "ils"
                    else:
                        input_name = ri_layers[idx - 1]['directory']
                    bin.csv_to_bin(ri_layers[idx]['directory'],
                                            ri_layers[idx]['directory'],
                                            il=True)

                    reinsurance_layer_losses_df = self._run_fm(
                        input_name,
                        ri_layers[idx]['directory'],
                        direct_layer.xref_descriptions)
                    output_name = "Inuring_priority:{} - Risk_level:{}".format(ri_layers[idx]['inuring_priority'],
                                                                               ri_layers[idx]['risk_level'])
                    net_losses[output_name] = reinsurance_layer_losses_df

                    return net_losses

        finally:
            os.chdir(initial_dir)
            t_end = time.time()
            print("Exec time: {}".format(t_end - t_start))


    def _load_acc_and_loc_dfs(self, oed_dir):

        # Account file
        oed_account_file = os.path.join(oed_dir, "account.csv")
        if not os.path.exists(oed_account_file):
            print("Path does not exist: {}".format(oed_account_file))
            exit(1)
        account_df = pd.read_csv(oed_account_file)

        # Location file
        oed_location_file = os.path.join(oed_dir, "location.csv")
        if not os.path.exists(oed_location_file):
            print("Path does not exist: {}".format(oed_location_file))
            exit(1)
        location_df = pd.read_csv(oed_location_file)

        return account_df, location_df

    @parameterized.expand(test_cases)
    def test_fmcalc(self, case, case_dir, expected_dir):

        print("Test case: {}".format(case))

        loss_factor = 1.0

        (
            account_df,
            location_df
        ) = self._load_acc_and_loc_dfs(case_dir)

        (
            ri_info_df,
            ri_scope_df, 
            do_reinsurance
        ) = oed.load_oed_dfs(case_dir)

        net_losses = self._run_test(
            account_df, location_df, ri_info_df, ri_scope_df,
            loss_factor,
            do_reinsurance,
        )

        for key in net_losses.keys():
            expected_file = os.path.join(
                expected_dir, 
                "{}.csv".format(key.replace(' ', '_'))
            )    

            dtypes = {
                "portfolio_number": "str",
                "policy_number": "str",
                "account_number": "str",
                "location_number": "str",
                "location_group": "str",
                "cedant_name": "str",
                "producer_name": "str",
                "lob": "str",
                "country_code": "str",
                "reins_tag": "str",
                "coverage_type_id": "str",
                "peril_id": "str",
                "tiv": "float",
                "loss_gul": "float",
                "loss_il": "float",
                "loss_net": "float"
            }

            expected_df = get_dataframe(expected_file, index_col=False)

            found_df = net_losses[key]
            found_df.to_csv("{}.csv".format(key.replace(' ', '_')))

            expected_df = expected_df.replace(np.nan, '', regex=True)
            found_df = found_df.replace(np.nan, '', regex=True)

            set_col_dtypes(expected_df, dtypes)
            set_col_dtypes(found_df, dtypes)
            
            expected_df.to_csv("/tmp/expected.csv", index=False)

            print(found_df.dtypes)
            print(expected_df.dtypes)

            assert_frame_equal(found_df, expected_df)
