import unittest
import os
import time
import subprocess

import pandas as pd

from parameterized import parameterized
from pandas.util.testing import assert_frame_equal
from oasislmf.exposures import oed
from oasislmf.model_execution import bin
from .direct_layer import DirectLayer
from oasislmf.exposures import reinsurance_layer
from collections import OrderedDict
from backports.tempfile import TemporaryDirectory

cwd = os.path.dirname(os.path.realpath(__file__))
expected_output_dir = os.path.join(cwd, 'expected', 'calc')

input_dir = os.path.join(cwd, 'examples')
test_examples = ['loc_SS',
                 'acc_SS',
                 'placed_acc_1_limit_QS',
                 'placed_acc_limit_QS',
                 'placed_loc_SS',
                 'placed_pol_SS',
                 'acc_limit_QS',
                 'simple_CAT_XL',
                 'pol_limit_QS',
                 'simple_pol_FAC',
                 'placed_acc_QS',
                 'acc_1_CAT_XL',
                 'placed_acc_1_QS',
                 'loc_limit_QS',
                 'multiple_SS',
                 'placed_loc_limit_SS',
                 'multiple_CAT_XL',
                 'multiple_portfolio',
                 'pol_SS',
                 'multiple_QS_2',
                 'multiple_FAC',
                 'simple_acc_FAC',
                 'simple_QS',
                 'multiple_QS_1',
                 'acc_PR',
                 'loc_1_2_PR',
                 'loc_PR',
                 'pol_PR',
                 'simple_loc_FAC']


fm_examples = ['fm24',
               'fm27']

test_cases = []
for case in test_examples + fm_examples:
#for case in ['placed_pol_SS']:
    test_cases.append((
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
                    bin.create_binary_files(ri_layers[idx]['directory'],
                                            ri_layers[idx]['directory'],
                                            do_il=True)

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
    def test_fmcalc(self, case_dir, expected_dir):
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

            expected_df = pd.read_csv(expected_file)
            found_df = net_losses[key]
            found_df.to_csv("{}.csv".format(key.replace(' ', '_')))

            assert_frame_equal(found_df, expected_df)
