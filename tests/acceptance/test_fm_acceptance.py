# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open as io_open
from builtins import str

from future import standard_library
standard_library.install_aliases()

import copy
import io
import itertools
import json
import os
import shutil
import string

from collections import OrderedDict
from future.utils import viewvalues
from unittest import TestCase

import pandas as pd
import pytest

from backports.tempfile import TemporaryDirectory
from hypothesis import (
    given,
    HealthCheck,
    reproduce_failure,
    settings,
)
from hypothesis.strategies import (
    dictionaries,
    integers,
    floats,
    just,
    lists,
    text,
    tuples,
)
from tabulate import tabulate
from tempfile import NamedTemporaryFile

from oasislmf.manager import OasisManager as om
from oasislmf.model_preparation.gul_inputs import (
    get_gul_input_items,
    write_gul_input_files,
)
from oasislmf.model_preparation.il_inputs import (
    get_il_input_items,
    unified_fm_profile_by_level_and_term_group,
    write_il_input_files,
)
from oasislmf.utils.defaults import (
    get_default_exposure_profile,
    get_default_accounts_profile,
    get_default_fm_aggregation_profile,
)
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.metadata import (
    COVERAGE_TYPES,
    OASIS_KEYS_STATUS,
    PERILS,
)
from ..data import (
    source_accounts,
    source_exposure,
    keys,
    write_source_files,
    write_keys_files,
)


class FmAcceptanceTests(TestCase):

    def setUp(self):
        self.manager = om()
        self.exposure_profile = self.manager.exposure_profile
        self.accounts_profile = self.manager.accounts_profile
        self.unified_profile = unified_fm_profile_by_level_and_term_group(profiles=[self.exposure_profile, self.accounts_profile])
        self.fm_aggregation_profile = self.manager.fm_aggregation_profile

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=1)
    @given(
        exposure=source_exposure(
            from_account_nums=just('1'),
            from_portfolio_nums=just('1'),
            from_location_perils=just('WTC;WEC;BFR;OO1'),
            from_location_perils_covered=just('WTC;WEC;BFR;OO1'),
            from_country_codes=just('US'),
            from_area_codes=just('CA'),
            from_building_tivs=just(1000000),
            from_building_deductibles=just(50000),
            from_building_limits=just(900000),
            from_other_tivs=just(100000),
            from_other_deductibles=just(5000),
            from_other_limits=just(90000),
            from_contents_tivs=just(50000),
            from_contents_deductibles=just(2500),
            from_contents_limits=just(45000),
            from_bi_tivs=just(20000),
            from_bi_deductibles=just(0),
            from_bi_limits=just(18000),
            from_sitepd_deductibles=just(0),
            from_sitepd_limits=just(0),
            from_siteall_deductibles=just(0),
            from_siteall_limits=just(0),
            size=1
        ),
        accounts=source_accounts(
            from_account_nums=just('1'),
            from_policy_nums=just('1'),
            from_portfolio_nums=just('1'),
            from_policy_perils=just('WTC;WEC;BFR;OO1'),
            from_condall_deductibles=just(0),
            from_condall_limits=just(0),
            from_policyall_deductibles=just(0),
            from_policyall_min_deductibles=just(0),
            from_policyall_max_deductibles=just(0),
            from_policylayer_deductibles=just(0),
            from_policylayer_limits=just(0),
            from_policylayer_shares=just(1),
            size=1
        ),
        keys=keys(
            from_peril_ids=just(1),
            from_coverage_type_ids=just(COVERAGE_TYPES['buildings']['id']),
            from_area_peril_ids=just(1),
            from_vulnerability_ids=just(1),
            from_statuses=just('success'),
            from_messages=just('success'),
            size=4
        )
    )
    def test_fm3(self, exposure, accounts, keys):
        keys[1]['locnumber'] = keys[2]['locnumber'] = keys[3]['locnumber'] = keys[0]['locnumber']
        keys[1]['coverage_type'] = COVERAGE_TYPES['other']['id']
        keys[2]['coverage_type'] = COVERAGE_TYPES['contents']['id']
        keys[3]['coverage_type'] = COVERAGE_TYPES['bi']['id']

        with NamedTemporaryFile('w') as ef, NamedTemporaryFile('w') as af, NamedTemporaryFile('w') as kf, TemporaryDirectory() as oasis_dir:
            write_source_files(exposure, ef.name, accounts, af.name)
            write_keys_files(keys, kf.name)

            gul_input_files, gul_inputs, exposure_df  = write_gul_input_files(
                ef.name, kf.name, oasis_dir,
                exposure_profile=self.exposure_profile
            )
    
            for p in viewvalues(gul_input_files): 
                if not p.endswith("complex_items.csv"):
                    self.assertTrue(os.path.exists(p))

            guls = pd.merge(
                pd.merge(pd.read_csv(gul_input_files['items']), pd.read_csv(gul_input_files['coverages']), left_on='coverage_id', right_on='coverage_id'),
                pd.read_csv(gul_input_files['gulsummaryxref']),
                left_on='coverage_id',
                right_on='coverage_id'
            )

            self.assertEqual(len(guls), 4)

            loc_groups = [(loc_id, loc_group) for loc_id, loc_group in guls.groupby('group_id')]
            self.assertEqual(len(loc_groups), 1)

            loc1_id, loc1_items = loc_groups[0]
            self.assertEqual(loc1_id, 1)
            self.assertEqual(len(loc1_items), 4)
            self.assertEqual(loc1_items['item_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(loc1_items['coverage_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(set(loc1_items['areaperil_id'].values), {1})
            self.assertEqual(loc1_items['vulnerability_id'].values.tolist(), [1,1,1,1])
            self.assertEqual(set(loc1_items['group_id'].values), {1})
            tivs = [exposure[0][t] for t in ['buildingtiv','othertiv','contentstiv','bitiv']]
            self.assertEqual(loc1_items['tiv'].values.tolist(), tivs)

            il_input_files, il_inputs, accounts_df = write_il_input_files(
                exposure_df, gul_inputs, oasis_dir,
                accounts_fp=af.name, exposure_profile=self.exposure_profile,
                accounts_profile=self.accounts_profile,
                fm_aggregation_profile=self.fm_aggregation_profile
            )

            for p in viewvalues(il_input_files): 
                if not p.endswith("complex_items.csv"):
                    self.assertTrue(os.path.exists(p))

            fm_programme_df = pd.read_csv(il_input_files['fm_programme'])
            level_groups = [group for _, group in fm_programme_df.groupby(['level_id'])]
            self.assertEqual(len(level_groups), 2)
            level1_group = level_groups[0]
            self.assertEqual(len(level1_group), 4)
            self.assertEqual(level1_group['from_agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(level1_group['to_agg_id'].values.tolist(), [1,2,3,4])
            level2_group = level_groups[1]
            self.assertEqual(len(level2_group), 4)
            self.assertEqual(level2_group['from_agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(level2_group['to_agg_id'].values.tolist(), [1,1,1,1])

            fm_profile_df = pd.read_csv(il_input_files['fm_profile'])
            self.assertEqual(len(fm_profile_df), 5)
            self.assertEqual(fm_profile_df['policytc_id'].values.tolist(), [1,2,3,4,5])
            self.assertEqual(fm_profile_df['calcrule_id'].values.tolist(), [1,1,1,14,2])
            self.assertEqual(fm_profile_df['deductible1'].values.tolist(), [50000,5000,2500,0,0])
            self.assertEqual(fm_profile_df['deductible2'].values.tolist(), [0,0,0,0,0])
            self.assertEqual(fm_profile_df['deductible3'].values.tolist(), [0,0,0,0,0])
            self.assertEqual(fm_profile_df['attachment1'].values.tolist(), [0,0,0,0,0])
            self.assertEqual(fm_profile_df['limit1'].values.tolist(), [900000,90000,45000,18000,9999999999])
            self.assertEqual(fm_profile_df['share1'].values.tolist(), [0,0,0,0,1])
            self.assertEqual(fm_profile_df['share2'].values.tolist(), [0,0,0,0,0])
            self.assertEqual(fm_profile_df['share3'].values.tolist(), [0,0,0,0,0])

            fm_policytc_df = pd.read_csv(il_input_files['fm_policytc'])
            level_groups = [group for _, group in fm_policytc_df.groupby(['level_id'])]
            self.assertEqual(len(level_groups), 2)
            level1_group = level_groups[0]
            self.assertEqual(len(level1_group), 4)
            self.assertEqual(level1_group['layer_id'].values.tolist(), [1,1,1,1])
            self.assertEqual(level1_group['agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(level1_group['policytc_id'].values.tolist(), [1,2,3,4])
            level2_group = level_groups[1]
            self.assertEqual(len(level2_group), 1)
            self.assertEqual(level2_group['layer_id'].values.tolist(), [1])
            self.assertEqual(level2_group['agg_id'].values.tolist(), [1])
            self.assertEqual(level2_group['policytc_id'].values.tolist(), [5])

            fm_xref_df = pd.read_csv(il_input_files['fm_xref'])
            self.assertEqual(len(fm_xref_df), 4)
            self.assertEqual(fm_xref_df['output'].values.tolist(), [1,2,3,4])
            self.assertEqual(fm_xref_df['agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(fm_xref_df['layer_id'].values.tolist(), [1,1,1,1])

            fmsummaryxref_df = pd.read_csv(il_input_files['fmsummaryxref'])
            self.assertEqual(len(fmsummaryxref_df), 4)
            self.assertEqual(fmsummaryxref_df['output'].values.tolist(), [1,2,3,4])
            self.assertEqual(fmsummaryxref_df['summary_id'].values.tolist(), [1,1,1,1])
            self.assertEqual(fmsummaryxref_df['summaryset_id'].values.tolist(), [1,1,1,1])

            expected_direct_losses = pd.DataFrame(
                columns=['event_id', 'output_id', 'loss'],
                data=[
                    (1,1,900000.0),
                    (1,2,90000.0),
                    (1,3,45000.0),
                    (1,4,18000.0)
                ]
            )
            bins_dir = os.path.join(oasis_dir, 'bin')
            os.mkdir(bins_dir)
            actual_direct_losses = self.manager.generate_deterministic_losses(oasis_dir, output_dir=bins_dir)['il']

            losses_ok = actual_direct_losses.equals(expected_direct_losses)
            self.assertTrue(losses_ok)
            if losses_ok:
                actual_direct_losses['event_id'] = actual_direct_losses['event_id'].astype(object)
                actual_direct_losses['output_id'] = actual_direct_losses['output_id'].astype(object)
                print('Correct losses generated for FM3:\n{}'.format(tabulate(actual_direct_losses, headers='keys', tablefmt='psql', floatfmt=".2f")))

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=1)
    @given(
        exposure=source_exposure(
            from_account_nums=just('1'),
            from_portfolio_nums=just('1'),
            from_location_perils=just('WTC;WEC;BFR;OO1'),
            from_location_perils_covered=just('WTC;WEC;BFR;OO1'),
            from_country_codes=just('US'),
            from_area_codes=just('CA'),
            from_building_tivs=just(1000000),
            from_building_deductibles=just(0),
            from_building_limits=just(0),
            from_other_tivs=just(100000),
            from_other_deductibles=just(0),
            from_other_limits=just(0),
            from_contents_tivs=just(50000),
            from_contents_deductibles=just(0),
            from_contents_limits=just(0),
            from_bi_tivs=just(20000),
            from_bi_deductibles=just(2000),
            from_bi_limits=just(18000),
            from_sitepd_deductibles=just(1000),
            from_sitepd_limits=just(1000000),
            from_siteall_deductibles=just(0),
            from_siteall_limits=just(0),
            size=1
        ),
        accounts=source_accounts(
            from_account_nums=just('1'),
            from_policy_nums=just('1'),
            from_portfolio_nums=just('1'),
            from_policy_perils=just('WTC;WEC;BFR;OO1'),
            from_condall_deductibles=just(0),
            from_condall_limits=just(0),
            from_policyall_deductibles=just(0),
            from_policyall_min_deductibles=just(0),
            from_policyall_max_deductibles=just(0),
            from_policylayer_deductibles=just(0),
            from_policylayer_limits=just(0),
            from_policylayer_shares=just(1),
            size=1
        ),
        keys=keys(
            from_peril_ids=just(1),
            from_coverage_type_ids=just(COVERAGE_TYPES['buildings']['id']),
            from_area_peril_ids=just(1),
            from_vulnerability_ids=just(1),
            from_statuses=just('success'),
            from_messages=just('success'),
            size=4
        )
    )
    def test_fm4(self, exposure, accounts, keys):
        keys[1]['locnumber'] = keys[2]['locnumber'] = keys[3]['locnumber'] = keys[0]['locnumber']
        keys[1]['coverage_type'] = COVERAGE_TYPES['other']['id']
        keys[2]['coverage_type'] = COVERAGE_TYPES['contents']['id']
        keys[3]['coverage_type'] = COVERAGE_TYPES['bi']['id']

        with NamedTemporaryFile('w') as ef, NamedTemporaryFile('w') as af, NamedTemporaryFile('w') as kf, TemporaryDirectory() as oasis_dir:
            write_source_files(exposure, ef.name, accounts, af.name)
            write_keys_files(keys, kf.name)

            gul_input_files, gul_inputs, exposure_df = write_gul_input_files(ef.name, kf.name, oasis_dir, exposure_profile=self.exposure_profile)

            for p in viewvalues(gul_input_files): 
                if not p.endswith("complex_items.csv"):
                    self.assertTrue(os.path.exists(p))

            guls = pd.merge(
                pd.merge(pd.read_csv(gul_input_files['items']), pd.read_csv(gul_input_files['coverages']), left_on='coverage_id', right_on='coverage_id'),
                pd.read_csv(gul_input_files['gulsummaryxref']),
                left_on='coverage_id',
                right_on='coverage_id'
            )

            self.assertEqual(len(guls), 4)

            loc_groups = [(loc_id, loc_group) for loc_id, loc_group in guls.groupby('group_id')]
            self.assertEqual(len(loc_groups), 1)

            loc1_id, loc1_items = loc_groups[0]
            self.assertEqual(loc1_id, 1)
            self.assertEqual(len(loc1_items), 4)
            self.assertEqual(loc1_items['item_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(loc1_items['coverage_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(set(loc1_items['areaperil_id'].values), {1})
            self.assertEqual(loc1_items['vulnerability_id'].values.tolist(), [1,1,1,1])
            self.assertEqual(set(loc1_items['group_id'].values), {1})
            tivs = [exposure[0][t] for t in ['buildingtiv','othertiv','contentstiv','bitiv']]
            self.assertEqual(loc1_items['tiv'].values.tolist(), tivs)

            il_input_files, il_inputs, accounts_df = write_il_input_files(
                exposure_df, gul_inputs, oasis_dir,
                accounts_fp=af.name, exposure_profile=self.exposure_profile,
                accounts_profile=self.accounts_profile,
                fm_aggregation_profile=self.fm_aggregation_profile
            )

            for p in viewvalues(il_input_files): 
                if not p.endswith("complex_items.csv"):
                    self.assertTrue(os.path.exists(p))

            fm_programme_df = pd.read_csv(il_input_files['fm_programme'])
            level_groups = [group for _, group in fm_programme_df.groupby(['level_id'])]
            self.assertEqual(len(level_groups), 3)
            level1_group = level_groups[0]
            self.assertEqual(len(level1_group), 4)
            self.assertEqual(level1_group['from_agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(level1_group['to_agg_id'].values.tolist(), [1,2,3,4])
            level2_group = level_groups[1]
            self.assertEqual(len(level2_group), 4)
            self.assertEqual(level2_group['from_agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(level2_group['to_agg_id'].values.tolist(), [1,1,1,2])
            level3_group = level_groups[2]
            self.assertEqual(len(level3_group), 2)
            self.assertEqual(level3_group['from_agg_id'].values.tolist(), [1,2])
            self.assertEqual(level3_group['to_agg_id'].values.tolist(), [1,1])

            fm_profile_df = pd.read_csv(il_input_files['fm_profile'])
            self.assertEqual(len(fm_profile_df), 4)
            self.assertEqual(fm_profile_df['policytc_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(fm_profile_df['calcrule_id'].values.tolist(), [12,1,1,2])
            self.assertEqual(fm_profile_df['deductible1'].values.tolist(), [0,2000,1000,0])
            self.assertEqual(fm_profile_df['deductible2'].values.tolist(), [0,0,0,0])
            self.assertEqual(fm_profile_df['deductible3'].values.tolist(), [0,0,0,0])
            self.assertEqual(fm_profile_df['attachment1'].values.tolist(), [0,0,0,0])
            self.assertEqual(fm_profile_df['limit1'].values.tolist(), [0,18000,1000000,9999999999])
            self.assertEqual(fm_profile_df['share1'].values.tolist(), [0,0,0,1])
            self.assertEqual(fm_profile_df['share2'].values.tolist(), [0,0,0,0])
            self.assertEqual(fm_profile_df['share3'].values.tolist(), [0,0,0,0])

            fm_policytc_df = pd.read_csv(il_input_files['fm_policytc'])
            level_groups = [group for _, group in fm_policytc_df.groupby(['level_id'])]
            self.assertEqual(len(level_groups), 3)
            level1_group = level_groups[0]
            self.assertEqual(len(level1_group), 4)
            self.assertEqual(level1_group['layer_id'].values.tolist(), [1,1,1,1])
            self.assertEqual(level1_group['agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(level1_group['policytc_id'].values.tolist(), [1,1,1,2])
            level2_group = level_groups[1]
            self.assertEqual(len(level2_group), 2)
            self.assertEqual(level2_group['layer_id'].values.tolist(), [1,1])
            self.assertEqual(level2_group['agg_id'].values.tolist(), [1,2])
            self.assertEqual(level2_group['policytc_id'].values.tolist(), [3,1])
            level3_group = level_groups[2]
            self.assertEqual(len(level3_group), 1)
            self.assertEqual(level3_group['layer_id'].values.tolist(), [1])
            self.assertEqual(level3_group['agg_id'].values.tolist(), [1])
            self.assertEqual(level3_group['policytc_id'].values.tolist(), [4])

            fm_xref_df = pd.read_csv(il_input_files['fm_xref'])
            self.assertEqual(len(fm_xref_df), 4)
            self.assertEqual(fm_xref_df['output'].values.tolist(), [1,2,3,4])
            self.assertEqual(fm_xref_df['agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(fm_xref_df['layer_id'].values.tolist(), [1,1,1,1])

            fmsummaryxref_df = pd.read_csv(il_input_files['fmsummaryxref'])
            self.assertEqual(len(fmsummaryxref_df), 4)
            self.assertEqual(fmsummaryxref_df['output'].values.tolist(), [1,2,3,4])
            self.assertEqual(fmsummaryxref_df['summary_id'].values.tolist(), [1,1,1,1])
            self.assertEqual(fmsummaryxref_df['summaryset_id'].values.tolist(), [1,1,1,1])

            expected_direct_losses = pd.DataFrame(
                columns=['event_id', 'output_id', 'loss'],
                data=[
                    (1,1,869565.25),
                    (1,2,86956.52),
                    (1,3,43478.26),
                    (1,4,18000.00)
                ]
            )
            bins_dir = os.path.join(oasis_dir, 'bin')
            os.mkdir(bins_dir)
            actual_direct_losses = self.manager.generate_deterministic_losses(oasis_dir, output_dir=bins_dir)['il']
            losses_ok = actual_direct_losses.equals(expected_direct_losses)
            self.assertTrue(losses_ok)
            if losses_ok:
                actual_direct_losses['event_id'] = actual_direct_losses['event_id'].astype(object)
                actual_direct_losses['output_id'] = actual_direct_losses['output_id'].astype(object)
                print('Correct losses generated for FM4:\n{}'.format(tabulate(actual_direct_losses, headers='keys', tablefmt='psql', floatfmt=".2f")))

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=1)
    @given(
        exposure=source_exposure(
            from_account_nums=just('1'),
            from_portfolio_nums=just('1'),
            from_location_perils=just('WTC;WEC;BFR;OO1'),
            from_location_perils_covered=just('WTC;WEC;BFR;OO1'),
            from_country_codes=just('US'),
            from_area_codes=just('CA'),
            from_building_tivs=just(1000000),
            from_building_deductibles=just(0),
            from_building_limits=just(0),
            from_other_tivs=just(100000),
            from_other_deductibles=just(0),
            from_other_limits=just(0),
            from_contents_tivs=just(50000),
            from_contents_deductibles=just(0),
            from_contents_limits=just(0),
            from_bi_tivs=just(20000),
            from_bi_deductibles=just(0),
            from_bi_limits=just(0),
            from_sitepd_deductibles=just(0),
            from_sitepd_limits=just(0),
            from_siteall_deductibles=just(1000),
            from_siteall_limits=just(1000000),
            size=1
        ),
        accounts=source_accounts(
            from_account_nums=just('1'),
            from_policy_nums=just('1'),
            from_portfolio_nums=just('1'),
            from_policy_perils=just('WTC;WEC;BFR;OO1'),
            from_condall_deductibles=just(0),
            from_condall_limits=just(0),
            from_policyall_deductibles=just(0),
            from_policyall_min_deductibles=just(0),
            from_policyall_max_deductibles=just(0),
            from_policylayer_deductibles=just(0),
            from_policylayer_limits=just(0),
            from_policylayer_shares=just(1),
            size=1
        ),
        keys=keys(
            from_peril_ids=just(1),
            from_coverage_type_ids=just(COVERAGE_TYPES['buildings']['id']),
            from_area_peril_ids=just(1),
            from_vulnerability_ids=just(1),
            from_statuses=just('success'),
            from_messages=just('success'),
            size=4
        )
    )
    def test_fm5(self, exposure, accounts, keys):
        keys[1]['locnumber'] = keys[2]['locnumber'] = keys[3]['locnumber'] = keys[0]['locnumber']
        keys[1]['coverage_type'] = COVERAGE_TYPES['other']['id']
        keys[2]['coverage_type'] = COVERAGE_TYPES['contents']['id']
        keys[3]['coverage_type'] = COVERAGE_TYPES['bi']['id']

        with NamedTemporaryFile('w') as ef, NamedTemporaryFile('w') as af, NamedTemporaryFile('w') as kf, TemporaryDirectory() as oasis_dir:
            write_source_files(exposure, ef.name, accounts, af.name)
            write_keys_files(keys, kf.name)

            gul_input_files, gul_inputs, exposure_df = write_gul_input_files(ef.name, kf.name, oasis_dir, exposure_profile=self.exposure_profile)

            for p in viewvalues(gul_input_files): 
                if not p.endswith("complex_items.csv"):
                    self.assertTrue(os.path.exists(p))

            guls = pd.merge(
                pd.merge(pd.read_csv(gul_input_files['items']), pd.read_csv(gul_input_files['coverages']), left_on='coverage_id', right_on='coverage_id'),
                pd.read_csv(gul_input_files['gulsummaryxref']),
                left_on='coverage_id',
                right_on='coverage_id'
            )

            self.assertEqual(len(guls), 4)

            loc_groups = [(loc_id, loc_group) for loc_id, loc_group in guls.groupby('group_id')]
            self.assertEqual(len(loc_groups), 1)

            loc1_id, loc1_items = loc_groups[0]
            self.assertEqual(loc1_id, 1)
            self.assertEqual(len(loc1_items), 4)
            self.assertEqual(loc1_items['item_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(loc1_items['coverage_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(set(loc1_items['areaperil_id'].values), {1})
            self.assertEqual(loc1_items['vulnerability_id'].values.tolist(), [1,1,1,1])
            self.assertEqual(set(loc1_items['group_id'].values), {1})
            tivs = [exposure[0][t] for t in ['buildingtiv','othertiv','contentstiv','bitiv']]
            self.assertEqual(loc1_items['tiv'].values.tolist(), tivs)

            il_input_files, il_inputs, accounts_df = write_il_input_files(
                exposure_df, gul_inputs, oasis_dir,
                accounts_fp=af.name, exposure_profile=self.exposure_profile,
                accounts_profile=self.accounts_profile,
                fm_aggregation_profile=self.fm_aggregation_profile
            )

            for p in viewvalues(il_input_files): 
                if not p.endswith("complex_items.csv"):
                    self.assertTrue(os.path.exists(p))

            fm_programme_df = pd.read_csv(il_input_files['fm_programme'])
            level_groups = [group for _, group in fm_programme_df.groupby(['level_id'])]
            self.assertEqual(len(level_groups), 3)
            level1_group = level_groups[0]
            self.assertEqual(len(level1_group), 4)
            self.assertEqual(level1_group['from_agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(level1_group['to_agg_id'].values.tolist(), [1,2,3,4])
            level2_group = level_groups[1]
            self.assertEqual(len(level2_group), 4)
            self.assertEqual(level2_group['from_agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(level2_group['to_agg_id'].values.tolist(), [1,1,1,1])
            level3_group = level_groups[2]
            self.assertEqual(len(level3_group), 1)
            self.assertEqual(level3_group['from_agg_id'].values.tolist(), [1])
            self.assertEqual(level3_group['to_agg_id'].values.tolist(), [1])

            fm_profile_df = pd.read_csv(il_input_files['fm_profile'])
            self.assertEqual(len(fm_profile_df), 3)
            self.assertEqual(fm_profile_df['policytc_id'].values.tolist(), [1,2,3])
            self.assertEqual(fm_profile_df['calcrule_id'].values.tolist(), [12,1,2])
            self.assertEqual(fm_profile_df['deductible1'].values.tolist(), [0,1000,0])
            self.assertEqual(fm_profile_df['deductible2'].values.tolist(), [0,0,0])
            self.assertEqual(fm_profile_df['deductible3'].values.tolist(), [0,0,0])
            self.assertEqual(fm_profile_df['attachment1'].values.tolist(), [0,0,0])
            self.assertEqual(fm_profile_df['limit1'].values.tolist(), [0,1000000,9999999999])
            self.assertEqual(fm_profile_df['share1'].values.tolist(), [0,0,1])
            self.assertEqual(fm_profile_df['share2'].values.tolist(), [0,0,0])
            self.assertEqual(fm_profile_df['share3'].values.tolist(), [0,0,0])

            fm_policytc_df = pd.read_csv(il_input_files['fm_policytc'])
            level_groups = [group for _, group in fm_policytc_df.groupby(['level_id'])]
            self.assertEqual(len(level_groups), 3)
            level1_group = level_groups[0]
            self.assertEqual(len(level1_group), 4)
            self.assertEqual(level1_group['layer_id'].values.tolist(), [1,1,1,1])
            self.assertEqual(level1_group['agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(level1_group['policytc_id'].values.tolist(), [1,1,1,1])
            level2_group = level_groups[1]
            self.assertEqual(len(level2_group), 1)
            self.assertEqual(level2_group['layer_id'].values.tolist(), [1])
            self.assertEqual(level2_group['agg_id'].values.tolist(), [1])
            self.assertEqual(level2_group['policytc_id'].values.tolist(), [2])
            level3_group = level_groups[2]
            self.assertEqual(len(level3_group), 1)
            self.assertEqual(level3_group['layer_id'].values.tolist(), [1])
            self.assertEqual(level3_group['agg_id'].values.tolist(), [1])
            self.assertEqual(level3_group['policytc_id'].values.tolist(), [3])

            fm_xref_df = pd.read_csv(il_input_files['fm_xref'])
            self.assertEqual(len(fm_xref_df), 4)
            self.assertEqual(fm_xref_df['output'].values.tolist(), [1,2,3,4])
            self.assertEqual(fm_xref_df['agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(fm_xref_df['layer_id'].values.tolist(), [1,1,1,1])

            fmsummaryxref_df = pd.read_csv(il_input_files['fmsummaryxref'])
            self.assertEqual(len(fmsummaryxref_df), 4)
            self.assertEqual(fmsummaryxref_df['output'].values.tolist(), [1,2,3,4])
            self.assertEqual(fmsummaryxref_df['summary_id'].values.tolist(), [1,1,1,1])
            self.assertEqual(fmsummaryxref_df['summaryset_id'].values.tolist(), [1,1,1,1])

            expected_direct_losses = pd.DataFrame(
                columns=['event_id', 'output_id', 'loss'],
                data=[
                    (1,1,854700.88),
                    (1,2,85470.09),
                    (1,3,42735.04),
                    (1,4,17094.02)
                ]
            )
            bins_dir = os.path.join(oasis_dir, 'bin')
            os.mkdir(bins_dir)
            actual_direct_losses = self.manager.generate_deterministic_losses(oasis_dir, output_dir=bins_dir)['il']
            losses_ok = actual_direct_losses.equals(expected_direct_losses)
            self.assertTrue(losses_ok)
            if losses_ok:
                actual_direct_losses['event_id'] = actual_direct_losses['event_id'].astype(object)
                actual_direct_losses['output_id'] = actual_direct_losses['output_id'].astype(object)
                print('Correct losses generated for FM5:\n{}'.format(tabulate(actual_direct_losses, headers='keys', tablefmt='psql', floatfmt=".2f")))

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=1)
    @given(
        exposure=source_exposure(
            from_account_nums=just('1'),
            from_portfolio_nums=just('1'),
            from_location_perils=just('WTC;WEC;BFR;OO1'),
            from_location_perils_covered=just('WTC;WEC;BFR;OO1'),
            from_country_codes=just('US'),
            from_area_codes=just('CA'),
            from_building_tivs=just(1000000),
            from_building_deductibles=just(0),
            from_building_limits=just(0),
            from_other_tivs=just(100000),
            from_other_deductibles=just(0),
            from_other_limits=just(0),
            from_contents_tivs=just(50000),
            from_contents_deductibles=just(0),
            from_contents_limits=just(0),
            from_bi_tivs=just(20000),
            from_bi_deductibles=just(0),
            from_bi_limits=just(0),
            from_sitepd_deductibles=just(0),
            from_sitepd_limits=just(0),
            from_siteall_deductibles=just(0),
            from_siteall_limits=just(0),
            size=2
        ),
        accounts=source_accounts(
            from_account_nums=just('1'),
            from_policy_nums=just('1'),
            from_portfolio_nums=just('1'),
            from_policy_perils=just('WTC;WEC;BFR;OO1'),
            from_condall_deductibles=just(0),
            from_condall_limits=just(0),
            from_policyall_deductibles=just(50000),
            from_policyall_min_deductibles=just(0),
            from_policyall_max_deductibles=just(0),
            from_policylayer_deductibles=just(0),
            from_policylayer_limits=just(2500000),
            from_policylayer_shares=just(1),
            size=1
        ),
        keys=keys(
            from_peril_ids=just(1),
            from_coverage_type_ids=just(COVERAGE_TYPES['buildings']['id']),
            from_area_peril_ids=just(1),
            from_vulnerability_ids=just(1),
            from_statuses=just('success'),
            from_messages=just('success'),
            size=8
        )
    )
    def test_fm6(self, exposure, accounts, keys):
        exposure[1]['buildingtiv'] = 1700000
        exposure[1]['othertiv'] = 30000
        exposure[1]['contentstiv'] = 1000000
        exposure[1]['bitiv'] = 50000

        keys[1]['locnumber'] = keys[2]['locnumber'] = keys[3]['locnumber'] = keys[0]['locnumber']
        keys[4]['locnumber'] = keys[5]['locnumber'] = keys[6]['locnumber'] = keys[7]['locnumber'] = '2'

        keys[4]['coverage_type'] = COVERAGE_TYPES['buildings']['id']
        keys[1]['coverage_type'] = keys[5]['coverage_type'] = COVERAGE_TYPES['other']['id']
        keys[2]['coverage_type'] = keys[6]['coverage_type'] = COVERAGE_TYPES['contents']['id']
        keys[3]['coverage_type'] = keys[7]['coverage_type'] = COVERAGE_TYPES['bi']['id']

        keys[4]['area_peril_id'] = keys[5]['area_peril_id'] = keys[6]['area_peril_id'] = keys[7]['area_peril_id'] = 2

        keys[4]['vulnerability_id'] = keys[5]['vulnerability_id'] = keys[6]['vulnerability_id'] = keys[7]['vulnerability_id'] = 2

        with NamedTemporaryFile('w') as ef, NamedTemporaryFile('w') as af, NamedTemporaryFile('w') as kf, TemporaryDirectory() as oasis_dir:
            write_source_files(exposure, ef.name, accounts, af.name)
            write_keys_files(keys, kf.name)

            gul_input_files, gul_inputs, exposure_df = write_gul_input_files(ef.name, kf.name, oasis_dir, exposure_profile=self.exposure_profile)

            for p in viewvalues(gul_input_files): 
                if not p.endswith("complex_items.csv"):
                    self.assertTrue(os.path.exists(p))

            guls = pd.merge(
                pd.merge(pd.read_csv(gul_input_files['items']), pd.read_csv(gul_input_files['coverages']), left_on='coverage_id', right_on='coverage_id'),
                pd.read_csv(gul_input_files['gulsummaryxref']),
                left_on='coverage_id',
                right_on='coverage_id'
            )

            self.assertEqual(len(guls), 8)

            loc_groups = [(loc_id, loc_group) for loc_id, loc_group in guls.groupby('group_id')]
            self.assertEqual(len(loc_groups), 2)

            loc1_id, loc1_items = loc_groups[0]
            self.assertEqual(loc1_id, 1)
            self.assertEqual(len(loc1_items), 4)
            self.assertEqual(loc1_items['item_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(loc1_items['coverage_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(set(loc1_items['areaperil_id'].values), {1})
            self.assertEqual(set(loc1_items['vulnerability_id'].values), {1})
            self.assertEqual(set(loc1_items['group_id'].values), {1})
            tivs = [exposure[0][t] for t in ['buildingtiv','othertiv','contentstiv','bitiv']]
            self.assertEqual(loc1_items['tiv'].values.tolist(), tivs)

            loc2_id, loc2_items = loc_groups[1]
            self.assertEqual(loc2_id, 2)
            self.assertEqual(len(loc2_items), 4)
            self.assertEqual(loc2_id, 2)
            self.assertEqual(len(loc2_items), 4)
            self.assertEqual(loc2_items['item_id'].values.tolist(), [5,6,7,8])
            self.assertEqual(loc2_items['coverage_id'].values.tolist(), [5,6,7,8])
            self.assertEqual(set(loc2_items['areaperil_id'].values), {2})
            self.assertEqual(set(loc2_items['vulnerability_id'].values), {2})
            self.assertEqual(set(loc2_items['group_id'].values), {2})
            tivs = [exposure[1][t] for t in ['buildingtiv','othertiv','contentstiv','bitiv']]
            self.assertEqual(loc2_items['tiv'].values.tolist(), tivs)

            il_input_files, il_inputs, accounts_df = write_il_input_files(
                exposure_df, gul_inputs, oasis_dir,
                accounts_fp=af.name, exposure_profile=self.exposure_profile,
                accounts_profile=self.accounts_profile,
                fm_aggregation_profile=self.fm_aggregation_profile
            )

            for p in viewvalues(il_input_files): 
                if not p.endswith("complex_items.csv"):
                    self.assertTrue(os.path.exists(p))

            fm_programme_df = pd.read_csv(il_input_files['fm_programme'])
            level_groups = [group for _, group in fm_programme_df.groupby(['level_id'])]
            self.assertEqual(len(level_groups), 3)
            level1_group = level_groups[0]
            self.assertEqual(len(level1_group), 8)
            self.assertEqual(level1_group['from_agg_id'].values.tolist(), [1,2,3,4,5,6,7,8])
            self.assertEqual(level1_group['to_agg_id'].values.tolist(), [1,2,3,4,5,6,7,8])
            level2_group = level_groups[1]
            self.assertEqual(len(level2_group), 8)
            self.assertEqual(level2_group['from_agg_id'].values.tolist(), [1,2,3,4,5,6,7,8])
            self.assertEqual(level2_group['to_agg_id'].values.tolist(), [1,1,1,1,1,1,1,1])
            level3_group = level_groups[2]
            self.assertEqual(len(level3_group), 1)
            self.assertEqual(level3_group['from_agg_id'].values.tolist(), [1])
            self.assertEqual(level3_group['to_agg_id'].values.tolist(), [1])

            fm_profile_df = pd.read_csv(il_input_files['fm_profile'])
            self.assertEqual(len(fm_profile_df), 3)
            self.assertEqual(fm_profile_df['policytc_id'].values.tolist(), [1,2,3])
            self.assertEqual(fm_profile_df['calcrule_id'].values.tolist(), [12,12,2])
            self.assertEqual(fm_profile_df['deductible1'].values.tolist(), [0,50000,0])
            self.assertEqual(fm_profile_df['deductible2'].values.tolist(), [0,0,0])
            self.assertEqual(fm_profile_df['deductible3'].values.tolist(), [0,0,0])
            self.assertEqual(fm_profile_df['attachment1'].values.tolist(), [0,0,0])
            self.assertEqual(fm_profile_df['limit1'].values.tolist(), [0,0,2500000])
            self.assertEqual(fm_profile_df['share1'].values.tolist(), [0,0,1])
            self.assertEqual(fm_profile_df['share2'].values.tolist(), [0,0,0])
            self.assertEqual(fm_profile_df['share3'].values.tolist(), [0,0,0])

            fm_policytc_df = pd.read_csv(il_input_files['fm_policytc'])
            level_groups = [group for _, group in fm_policytc_df.groupby(['level_id'])]
            self.assertEqual(len(level_groups), 3)
            level1_group = level_groups[0]
            self.assertEqual(len(level1_group), 8)
            self.assertEqual(level1_group['layer_id'].values.tolist(), [1,1,1,1,1,1,1,1])
            self.assertEqual(level1_group['agg_id'].values.tolist(), [1,2,3,4,5,6,7,8])
            self.assertEqual(level1_group['policytc_id'].values.tolist(), [1,1,1,1,1,1,1,1])
            level2_group = level_groups[1]
            self.assertEqual(len(level2_group), 1)
            self.assertEqual(level2_group['layer_id'].values.tolist(), [1])
            self.assertEqual(level2_group['agg_id'].values.tolist(), [1])
            self.assertEqual(level2_group['policytc_id'].values.tolist(), [2])
            level3_group = level_groups[2]
            self.assertEqual(len(level3_group), 1)
            self.assertEqual(level3_group['layer_id'].values.tolist(), [1])
            self.assertEqual(level3_group['agg_id'].values.tolist(), [1])
            self.assertEqual(level3_group['policytc_id'].values.tolist(), [3])

            fm_xref_df = pd.read_csv(il_input_files['fm_xref'])
            self.assertEqual(len(fm_xref_df), 8)
            self.assertEqual(fm_xref_df['output'].values.tolist(), [1,2,3,4,5,6,7,8])
            self.assertEqual(fm_xref_df['agg_id'].values.tolist(), [1,2,3,4,5,6,7,8])
            self.assertEqual(fm_xref_df['layer_id'].values.tolist(), [1,1,1,1,1,1,1,1])

            fmsummaryxref_df = pd.read_csv(il_input_files['fmsummaryxref'])
            self.assertEqual(len(fmsummaryxref_df), 8)
            self.assertEqual(fmsummaryxref_df['output'].values.tolist(), [1,2,3,4,5,6,7,8])
            self.assertEqual(fmsummaryxref_df['summary_id'].values.tolist(), [1,1,1,1,1,1,1,1])
            self.assertEqual(fmsummaryxref_df['summaryset_id'].values.tolist(), [1,1,1,1,1,1,1,1])

            expected_direct_losses = pd.DataFrame(
                columns=['event_id', 'output_id', 'loss'],
                data=[
                    (1,1,632911.38),
                    (1,2,63291.14),
                    (1,3,31645.57),
                    (1,4,12658.23),
                    (1,5,1075949.38),
                    (1,6,18987.34),
                    (1,7,632911.38),
                    (1,8,31645.57)
                ]
            )
            bins_dir = os.path.join(oasis_dir, 'bin')
            os.mkdir(bins_dir)
            actual_direct_losses = self.manager.generate_deterministic_losses(oasis_dir, output_dir=bins_dir)['il']
            losses_ok = actual_direct_losses.equals(expected_direct_losses)
            self.assertTrue(losses_ok)
            if losses_ok:
                actual_direct_losses['event_id'] = actual_direct_losses['event_id'].astype(object)
                actual_direct_losses['output_id'] = actual_direct_losses['output_id'].astype(object)
                print('Correct losses generated for FM6:\n{}'.format(tabulate(actual_direct_losses, headers='keys', tablefmt='psql', floatfmt=".2f")))

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=1)
    @given(
        exposure=source_exposure(
            from_account_nums=just('1'),
            from_portfolio_nums=just('1'),
            from_location_perils=just('WTC;WEC;BFR;OO1'),
            from_location_perils_covered=just('WTC;WEC;BFR;OO1'),
            from_country_codes=just('US'),
            from_area_codes=just('CA'),
            from_building_tivs=just(1000000),
            from_building_deductibles=just(10000),
            from_building_limits=just(0),
            from_other_tivs=just(100000),
            from_other_deductibles=just(5000),
            from_other_limits=just(0),
            from_contents_tivs=just(50000),
            from_contents_deductibles=just(5000),
            from_contents_limits=just(0),
            from_bi_tivs=just(20000),
            from_bi_deductibles=just(0),
            from_bi_limits=just(0),
            from_sitepd_deductibles=just(0),
            from_sitepd_limits=just(0),
            from_siteall_deductibles=just(0),
            from_siteall_limits=just(0),
            size=2
        ),
        accounts=source_accounts(
            from_account_nums=just('1'),
            from_policy_nums=just('1'),
            from_portfolio_nums=just('1'),
            from_policy_perils=just('WTC;WEC;BFR;OO1'),
            from_condall_deductibles=just(0),
            from_condall_limits=just(0),
            from_policyall_deductibles=just(50000),
            from_policyall_min_deductibles=just(0),
            from_policyall_max_deductibles=just(0),
            from_policylayer_deductibles=just(0),
            from_policylayer_limits=just(2500000),
            from_policylayer_shares=just(1),
            size=1
        ),
        keys=keys(
            from_peril_ids=just(1),
            from_coverage_type_ids=just(COVERAGE_TYPES['buildings']['id']),
            from_area_peril_ids=just(1),
            from_vulnerability_ids=just(1),
            from_statuses=just('success'),
            from_messages=just('success'),
            size=8
        )
    )
    def test_fm7(self, exposure, accounts, keys):
        exposure[1]['buildingtiv'] = 1700000
        exposure[1]['othertiv'] = 30000
        exposure[1]['contentstiv'] = 1000000
        exposure[1]['bitiv'] = 50000

        keys[1]['locnumber'] = keys[2]['locnumber'] = keys[3]['locnumber'] = keys[0]['locnumber']
        keys[4]['locnumber'] = keys[5]['locnumber'] = keys[6]['locnumber'] = keys[7]['locnumber'] = '2'

        keys[4]['coverage_type'] = COVERAGE_TYPES['buildings']['id']
        keys[1]['coverage_type'] = keys[5]['coverage_type'] = COVERAGE_TYPES['other']['id']
        keys[2]['coverage_type'] = keys[6]['coverage_type'] = COVERAGE_TYPES['contents']['id']
        keys[3]['coverage_type'] = keys[7]['coverage_type'] = COVERAGE_TYPES['bi']['id']

        keys[4]['area_peril_id'] = keys[5]['area_peril_id'] = keys[6]['area_peril_id'] = keys[7]['area_peril_id'] = 2

        keys[4]['vulnerability_id'] = keys[5]['vulnerability_id'] = keys[6]['vulnerability_id'] = keys[7]['vulnerability_id'] = 2

        with NamedTemporaryFile('w') as ef, NamedTemporaryFile('w') as af, NamedTemporaryFile('w') as kf, TemporaryDirectory() as oasis_dir:
            write_source_files(exposure, ef.name, accounts, af.name)
            write_keys_files(keys, kf.name)

            gul_input_files, gul_inputs, exposure_df = write_gul_input_files(ef.name, kf.name, oasis_dir, exposure_profile=self.exposure_profile)

            for p in viewvalues(gul_input_files): 
                if not p.endswith("complex_items.csv"):
                    self.assertTrue(os.path.exists(p))

            guls = pd.merge(
                pd.merge(pd.read_csv(gul_input_files['items']), pd.read_csv(gul_input_files['coverages']), left_on='coverage_id', right_on='coverage_id'),
                pd.read_csv(gul_input_files['gulsummaryxref']),
                left_on='coverage_id',
                right_on='coverage_id'
            )

            self.assertEqual(len(guls), 8)

            loc_groups = [(loc_id, loc_group) for loc_id, loc_group in guls.groupby('group_id')]
            self.assertEqual(len(loc_groups), 2)

            loc1_id, loc1_items = loc_groups[0]
            self.assertEqual(loc1_id, 1)
            self.assertEqual(len(loc1_items), 4)
            self.assertEqual(loc1_items['item_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(loc1_items['coverage_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(set(loc1_items['areaperil_id'].values), {1})
            self.assertEqual(set(loc1_items['vulnerability_id'].values), {1})
            self.assertEqual(set(loc1_items['group_id'].values), {1})
            tivs = [exposure[0][t] for t in ['buildingtiv','othertiv','contentstiv','bitiv']]
            self.assertEqual(loc1_items['tiv'].values.tolist(), tivs)

            loc2_id, loc2_items = loc_groups[1]
            self.assertEqual(loc2_id, 2)
            self.assertEqual(len(loc2_items), 4)
            self.assertEqual(loc2_id, 2)
            self.assertEqual(len(loc2_items), 4)
            self.assertEqual(loc2_items['item_id'].values.tolist(), [5,6,7,8])
            self.assertEqual(loc2_items['coverage_id'].values.tolist(), [5,6,7,8])
            self.assertEqual(set(loc2_items['areaperil_id'].values), {2})
            self.assertEqual(set(loc2_items['vulnerability_id'].values), {2})
            self.assertEqual(set(loc2_items['group_id'].values), {2})
            tivs = [exposure[1][t] for t in ['buildingtiv','othertiv','contentstiv','bitiv']]
            self.assertEqual(loc2_items['tiv'].values.tolist(), tivs)

            il_input_files, il_inputs, accounts_df = write_il_input_files(
                exposure_df, gul_inputs, oasis_dir,
                accounts_fp=af.name, exposure_profile=self.exposure_profile,
                accounts_profile=self.accounts_profile,
                fm_aggregation_profile=self.fm_aggregation_profile
            )

            for p in viewvalues(il_input_files): 
                if not p.endswith("complex_items.csv"):
                    self.assertTrue(os.path.exists(p))

            fm_programme_df = pd.read_csv(il_input_files['fm_programme'])
            level_groups = [group for _, group in fm_programme_df.groupby(['level_id'])]
            self.assertEqual(len(level_groups), 3)
            level1_group = level_groups[0]
            self.assertEqual(len(level1_group), 8)
            self.assertEqual(level1_group['from_agg_id'].values.tolist(), [1,2,3,4,5,6,7,8])
            self.assertEqual(level1_group['to_agg_id'].values.tolist(), [1,2,3,4,5,6,7,8])
            level2_group = level_groups[1]
            self.assertEqual(len(level2_group), 8)
            self.assertEqual(level2_group['from_agg_id'].values.tolist(), [1,2,3,4,5,6,7,8])
            self.assertEqual(level2_group['to_agg_id'].values.tolist(), [1,1,1,1,1,1,1,1])
            level3_group = level_groups[2]
            self.assertEqual(len(level3_group), 1)
            self.assertEqual(level3_group['from_agg_id'].values.tolist(), [1])
            self.assertEqual(level3_group['to_agg_id'].values.tolist(), [1])

            fm_profile_df = pd.read_csv(il_input_files['fm_profile'])
            self.assertEqual(len(fm_profile_df), 5)
            self.assertEqual(fm_profile_df['policytc_id'].values.tolist(), [1,2,3,4,5])
            self.assertEqual(fm_profile_df['calcrule_id'].values.tolist(), [12,12,12,12,2])
            self.assertEqual(fm_profile_df['deductible1'].values.tolist(), [10000,5000,0,50000,0])
            self.assertEqual(fm_profile_df['deductible2'].values.tolist(), [0,0,0,0,0])
            self.assertEqual(fm_profile_df['deductible3'].values.tolist(), [0,0,0,0,0])
            self.assertEqual(fm_profile_df['attachment1'].values.tolist(), [0,0,0,0,0])
            self.assertEqual(fm_profile_df['limit1'].values.tolist(), [0,0,0,0,2500000])
            self.assertEqual(fm_profile_df['share1'].values.tolist(), [0,0,0,0,1])
            self.assertEqual(fm_profile_df['share2'].values.tolist(), [0,0,0,0,0])
            self.assertEqual(fm_profile_df['share3'].values.tolist(), [0,0,0,0,0])

            fm_policytc_df = pd.read_csv(il_input_files['fm_policytc'])
            level_groups = [group for _, group in fm_policytc_df.groupby(['level_id'])]
            self.assertEqual(len(level_groups), 3)
            level1_group = level_groups[0]
            self.assertEqual(len(level1_group), 8)
            self.assertEqual(level1_group['layer_id'].values.tolist(), [1,1,1,1,1,1,1,1])
            self.assertEqual(level1_group['agg_id'].values.tolist(), [1,2,3,4,5,6,7,8])
            self.assertEqual(level1_group['policytc_id'].values.tolist(), [1,2,2,3,1,2,2,3])
            level2_group = level_groups[1]
            self.assertEqual(len(level2_group), 1)
            self.assertEqual(level2_group['layer_id'].values.tolist(), [1])
            self.assertEqual(level2_group['agg_id'].values.tolist(), [1])
            self.assertEqual(level2_group['policytc_id'].values.tolist(), [4])
            level3_group = level_groups[2]
            self.assertEqual(len(level3_group), 1)
            self.assertEqual(level3_group['layer_id'].values.tolist(), [1])
            self.assertEqual(level3_group['agg_id'].values.tolist(), [1])
            self.assertEqual(level3_group['policytc_id'].values.tolist(), [5])

            fm_xref_df = pd.read_csv(il_input_files['fm_xref'])
            self.assertEqual(len(fm_xref_df), 8)
            self.assertEqual(fm_xref_df['output'].values.tolist(), [1,2,3,4,5,6,7,8])
            self.assertEqual(fm_xref_df['agg_id'].values.tolist(), [1,2,3,4,5,6,7,8])
            self.assertEqual(fm_xref_df['layer_id'].values.tolist(), [1,1,1,1,1,1,1,1])

            fmsummaryxref_df = pd.read_csv(il_input_files['fmsummaryxref'])
            self.assertEqual(len(fmsummaryxref_df), 8)
            self.assertEqual(fmsummaryxref_df['output'].values.tolist(), [1,2,3,4,5,6,7,8])
            self.assertEqual(fmsummaryxref_df['summary_id'].values.tolist(), [1,1,1,1,1,1,1,1])
            self.assertEqual(fmsummaryxref_df['summaryset_id'].values.tolist(), [1,1,1,1,1,1,1,1])

            expected_direct_losses = pd.DataFrame(
                columns=['event_id', 'output_id', 'loss'],
                data=[
                    (1,1,632992.31),
                    (1,2,60741.68),
                    (1,3,28772.38),
                    (1,4,12787.72),
                    (1,5,1080562.62),
                    (1,6,15984.65),
                    (1,7,636189.31),
                    (1,8,31969.31)
                ]
            )
            bins_dir = os.path.join(oasis_dir, 'bin')
            os.mkdir(bins_dir)
            actual_direct_losses = self.manager.generate_deterministic_losses(oasis_dir, output_dir=bins_dir)['il']
            losses_ok = actual_direct_losses.equals(expected_direct_losses)
            self.assertTrue(losses_ok)
            if losses_ok:
                actual_direct_losses['event_id'] = actual_direct_losses['event_id'].astype(object)
                actual_direct_losses['output_id'] = actual_direct_losses['output_id'].astype(object)
                print('Correct losses generated for FM7:\n{}'.format(tabulate(actual_direct_losses, headers='keys', tablefmt='psql', floatfmt=".2f")))

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=1)
    @given(
        exposure=source_exposure(
            from_account_nums=just('1'),
            from_portfolio_nums=just('1'),
            from_location_perils=just('WTC;WEC;BFR;OO1'),
            from_location_perils_covered=just('WTC;WEC;BFR;OO1'),
            from_country_codes=just('US'),
            from_area_codes=just('CA'),
            from_building_tivs=just(1000000),
            from_building_deductibles=just(10000),
            from_building_limits=just(0),
            from_other_tivs=just(0),
            from_other_deductibles=just(0),
            from_other_limits=just(0),
            from_contents_tivs=just(0),
            from_contents_deductibles=just(0),
            from_contents_limits=just(0),
            from_bi_tivs=just(0),
            from_bi_deductibles=just(0),
            from_bi_limits=just(0),
            from_sitepd_deductibles=just(0),
            from_sitepd_limits=just(0),
            from_siteall_deductibles=just(0),
            from_siteall_limits=just(0),
            from_cond_numbers=just(0),
            size=6
        ),
        accounts=source_accounts(
            from_account_nums=just('1'),
            from_policy_nums=just('1'),
            from_portfolio_nums=just('1'),
            from_policy_perils=just('QQ1;WW1'),
            from_condall_deductibles=just(0),
            from_condall_limits=just(0),
            from_cond_numbers=just(0),
            from_policyall_deductibles=just(50000),
            from_policyall_min_deductibles=just(0),
            from_policyall_max_deductibles=just(0),
            from_policylayer_deductibles=just(0),
            from_policylayer_limits=just(1500000),
            from_policylayer_shares=just(0.1),
            size=2
        ),
        keys=keys(
            from_peril_ids=just('QQ1;WW1'),
            from_coverage_type_ids=just(COVERAGE_TYPES['buildings']['id']),
            from_area_peril_ids=just(1),
            from_vulnerability_ids=just(1),
            from_statuses=just('success'),
            from_messages=just('success'),
            size=6
        )
    )
    def test_fm40(self, exposure, accounts, keys):
        exposure[1]['buildingtiv'] = 1000000
        exposure[2]['buildingtiv'] = 1000000
        exposure[3]['buildingtiv'] = 2000000
        exposure[4]['buildingtiv'] = 2000000
        exposure[5]['buildingtiv'] = 2000000

        exposure[1]['locded1building'] = 0.01
        exposure[2]['locded1building'] = 0.05
        exposure[3]['locded1building'] = 15000
        exposure[4]['locded1building'] = 10000
        exposure[5]['locded1building'] = 0.1

        accounts[1]['accnumber'] = '1'
        accounts[1]['polnumber'] = '2'
        accounts[1]['layerparticipation'] = 0.5
        accounts[1]['layerlimit'] = 3500000
        accounts[1]['layerattachment'] = 1500000

        with NamedTemporaryFile('w') as ef, NamedTemporaryFile('w') as af, NamedTemporaryFile('w') as kf, TemporaryDirectory() as oasis_dir:
            write_source_files(exposure, ef.name, accounts, af.name)
            write_keys_files(keys, kf.name)

            gul_input_files, gul_inputs, exposure_df = write_gul_input_files(ef.name, kf.name, oasis_dir, exposure_profile=self.exposure_profile)

            for p in viewvalues(gul_input_files): 
                if not p.endswith("complex_items.csv"):
                    self.assertTrue(os.path.exists(p))

            guls = pd.merge(
                pd.merge(pd.read_csv(gul_input_files['items']), pd.read_csv(gul_input_files['coverages']), left_on='coverage_id', right_on='coverage_id'),
                pd.read_csv(gul_input_files['gulsummaryxref']),
                left_on='coverage_id',
                right_on='coverage_id'
            )

            self.assertEqual(len(guls), 6)

            self.assertEqual(guls['item_id'].values.tolist(), [1,2,3,4,5,6])
            self.assertEqual(guls['coverage_id'].values.tolist(), [1,2,3,4,5,6])
            self.assertEqual(guls['areaperil_id'].values.tolist(), [1,1,1,1,1,1])
            self.assertEqual(guls['vulnerability_id'].values.tolist(), [1,1,1,1,1,1])
            self.assertEqual(guls['group_id'].values.tolist(), [1,2,3,4,5,6])
            self.assertEqual(guls['tiv'].values.tolist(), [1000000,1000000,1000000,2000000,2000000,2000000])

            il_input_files, il_inputs, accounts_df = write_il_input_files(
                exposure_df, gul_inputs, oasis_dir,
                accounts_fp=af.name, exposure_profile=self.exposure_profile,
                accounts_profile=self.accounts_profile,
                fm_aggregation_profile=self.fm_aggregation_profile
            )

            for p in viewvalues(il_input_files): 
                if not p.endswith("complex_items.csv"):
                    self.assertTrue(os.path.exists(p))

            fm_programme_df = pd.read_csv(il_input_files['fm_programme'])
            level_groups = [group for _, group in fm_programme_df.groupby(['level_id'])]
            self.assertEqual(len(level_groups), 3)
            level1_group = level_groups[0]
            self.assertEqual(len(level1_group), 6)
            self.assertEqual(level1_group['from_agg_id'].values.tolist(), [1,2,3,4,5,6])
            self.assertEqual(level1_group['to_agg_id'].values.tolist(), [1,2,3,4,5,6])
            level2_group = level_groups[1]
            self.assertEqual(len(level2_group), 6)
            self.assertEqual(level2_group['from_agg_id'].values.tolist(), [1,2,3,4,5,6])
            self.assertEqual(level2_group['to_agg_id'].values.tolist(), [1,1,1,1,1,1])
            level3_group = level_groups[2]
            self.assertEqual(len(level3_group), 1)
            self.assertEqual(level3_group['from_agg_id'].values.tolist(), [1])
            self.assertEqual(level3_group['to_agg_id'].values.tolist(), [1])

            fm_profile_df = pd.read_csv(il_input_files['fm_profile'])
            self.assertEqual(len(fm_profile_df), 6)
            self.assertEqual(fm_profile_df['policytc_id'].values.tolist(), [1,2,3,4,5,6])
            self.assertEqual(fm_profile_df['calcrule_id'].values.tolist(), [12,12,12,12,2,2])
            self.assertEqual(fm_profile_df['deductible1'].values.tolist(), [10000,50000,15000,200000,0,1500000])
            self.assertEqual(fm_profile_df['deductible2'].values.tolist(), [0,0,0,0,0,0])
            self.assertEqual(fm_profile_df['deductible3'].values.tolist(), [0,0,0,0,0,0])
            self.assertEqual(fm_profile_df['attachment1'].values.tolist(), [0,0,0,0,0,1500000])
            self.assertEqual(fm_profile_df['limit1'].values.tolist(), [0,0,0,0,1500000,3500000])
            self.assertEqual(fm_profile_df['share1'].values.tolist(), [0,0,0,0,0.1,0.5])
            self.assertEqual(fm_profile_df['share2'].values.tolist(), [0,0,0,0,0,0])
            self.assertEqual(fm_profile_df['share3'].values.tolist(), [0,0,0,0,0,0])

            fm_policytc_df = pd.read_csv(il_input_files['fm_policytc'])
            level_groups = [group for _, group in fm_policytc_df.groupby(['level_id'])]
            self.assertEqual(len(level_groups), 3)
            level1_group = level_groups[0]
            self.assertEqual(len(level1_group), 6)
            self.assertEqual(level1_group['layer_id'].values.tolist(), [1,1,1,1,1,1])
            self.assertEqual(level1_group['agg_id'].values.tolist(), [1,2,3,4,5,6])
            self.assertEqual(level1_group['policytc_id'].values.tolist(), [1,1,2,3,1,4])
            level2_group = level_groups[1]
            self.assertEqual(len(level2_group), 1)
            self.assertEqual(level2_group['layer_id'].values.tolist(), [1])
            self.assertEqual(level2_group['agg_id'].values.tolist(), [1])
            self.assertEqual(level2_group['policytc_id'].values.tolist(), [2])
            level3_group = level_groups[2]
            self.assertEqual(len(level3_group), 2)
            self.assertEqual(level3_group['layer_id'].values.tolist(), [1,2])
            self.assertEqual(level3_group['agg_id'].values.tolist(), [1,1])
            self.assertEqual(level3_group['policytc_id'].values.tolist(), [5,6])

            fm_xref_df = pd.read_csv(il_input_files['fm_xref']).sort_values(['layer_id'])

            layer_groups = [group for _, group in fm_xref_df.groupby(['layer_id'])]
            self.assertEqual(len(layer_groups), 2)

            layer1_group = layer_groups[0]
            self.assertEqual(len(layer1_group), 6)
            self.assertEqual(layer1_group['agg_id'].values.tolist(), [1,2,3,4,5,6])

            layer2_group = layer_groups[1]
            self.assertEqual(len(layer2_group), 6)
            self.assertEqual(layer2_group['agg_id'].values.tolist(), [1,2,3,4,5,6])

            fmsummaryxref_df = pd.read_csv(il_input_files['fmsummaryxref'])
            self.assertEqual(len(fmsummaryxref_df), 12)
            self.assertEqual(fmsummaryxref_df['output'].values.tolist(), [1,2,3,4,5,6,7,8,9,10,11,12])
            self.assertEqual(fmsummaryxref_df['summary_id'].values.tolist(), [1,1,1,1,1,1,1,1,1,1,1,1])
            self.assertEqual(fmsummaryxref_df['summaryset_id'].values.tolist(), [1,1,1,1,1,1,1,1,1,1,1,1])

            expected_direct_losses = pd.DataFrame(
                columns=['event_id', 'output_id', 'loss'],
                data=[
                    (1,1,17059.16),
                    (1,2,199023.55),
                    (1,3,17059.16),
                    (1,4,199023.55),
                    (1,5,16369.90),
                    (1,6,190982.20),
                    (1,7,34204.48),
                    (1,8,399052.25),
                    (1,9,34290.64),
                    (1,10,400057.44),
                    (1,11,31016.66),
                    (1,12,361861.00)
                ]
            )
            bins_dir = os.path.join(oasis_dir, 'bin')
            os.mkdir(bins_dir)
            actual_direct_losses = self.manager.generate_deterministic_losses(oasis_dir, output_dir=bins_dir)['il']
            losses_ok = actual_direct_losses.equals(expected_direct_losses)
            self.assertTrue(losses_ok)
            if losses_ok:
                actual_direct_losses['event_id'] = actual_direct_losses['event_id'].astype(object)
                actual_direct_losses['output_id'] = actual_direct_losses['output_id'].astype(object)
                print('Correct losses generated for FM40:\n{}'.format(tabulate(actual_direct_losses, headers='keys', tablefmt='psql', floatfmt=".2f")))
