from __future__ import unicode_literals

import copy
import io
import itertools
import json
import os
import shutil
import string

from collections import OrderedDict
from unittest import TestCase

import pandas as pd
import pytest
import six

from backports.tempfile import TemporaryDirectory
from tempfile import NamedTemporaryFile

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

from oasislmf.exposures.manager import OasisExposuresManager

from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.fm import (
    unified_canonical_fm_profile_by_level_and_term_group,
)
from oasislmf.utils.metadata import (
    OASIS_COVERAGE_TYPES,
    OASIS_FM_LEVELS,
    OASIS_KEYS_STATUS,
    OASIS_PERILS,
    OED_COVERAGE_TYPES,
    OED_PERILS,
)

from ..models.fakes import fake_model

from ..data import (
    canonical_oed_accounts_data,
    canonical_oed_accounts_profile,
    canonical_oed_exposures_data,
    canonical_oed_exposures_profile,
    keys_data,
    oed_fm_agg_profile,
    write_canonical_files,
    write_canonical_oed_files,
    write_keys_files,
)



class FmAcceptanceTests(TestCase):

    def setUp(self):
        self.canexp_profile = copy.deepcopy(canonical_oed_exposures_profile)
        self.canacc_profile = copy.deepcopy(canonical_oed_accounts_profile)
        self.unified_can_profile = unified_canonical_fm_profile_by_level_and_term_group(profiles=[self.canexp_profile, self.canacc_profile])
        self.fm_agg_map = copy.deepcopy(oed_fm_agg_profile)
        self.manager = OasisExposuresManager()

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_oed_exposures_data(
            from_account_nums=just(1),
            from_location_perils=just('WTC;WEC;BFR;OO1'),
            from_country_codes=just('US'),
            from_area_codes=just('CA'),
            from_buildings_tivs=just(1000000),
            from_buildings_deductibles=just(50000),
            from_buildings_limits=just(900000),
            from_other_tivs=just(100000),
            from_other_deductibles=just(5000),
            from_other_limits=just(90000),
            from_contents_tivs=just(50000),
            from_contents_deductibles=just(2500),
            from_contents_limits=just(45000),
            from_bi_tivs=just(20000),
            from_bi_deductibles=just(0),
            from_bi_limits=just(18000),
            from_combined_deductibles=just(0),
            from_combined_limits=just(0),
            from_site_deductibles=just(0),
            from_site_limits=just(0),
            size=1
        ),
        accounts=canonical_oed_accounts_data(
            from_account_nums=just(1),
            from_portfolio_nums=just(1),
            from_policy_nums=just(1),
            from_policy_perils=just('WTC;WEC;BFR;OO1'),
            from_sublimit_deductibles=just(0),
            from_sublimit_limits=just(0),
            from_account_deductibles=just(0),
            from_account_min_deductibles=just(0),
            from_account_max_deductibles=just(0),
            from_layer_deductibles=just(0),
            from_layer_limits=just(0),
            from_layer_shares=just(1),
            size=1
        ),
        keys=keys_data(
            from_peril_ids=just(1),
            from_coverage_type_ids=just(OED_COVERAGE_TYPES['buildings']['id']),
            from_area_peril_ids=just(1),
            from_vulnerability_ids=just(1),
            from_statuses=just('success'),
            from_messages=just('success'),
            size=4
        )
    )
    def test_fm3(self, exposures, accounts, keys):
        keys[1]['id'] = keys[2]['id'] = keys[3]['id'] = 1
        keys[1]['coverage_type'] = OED_COVERAGE_TYPES['other']['id']
        keys[2]['coverage_type'] = OED_COVERAGE_TYPES['contents']['id']
        keys[3]['coverage_type'] = OED_COVERAGE_TYPES['bi']['id']
        keys[1]['vulnerability_id'] = 2
        keys[2]['vulnerability_id'] = 3
        keys[3]['vulnerability_id'] = 4

        with NamedTemporaryFile('w') as ef, NamedTemporaryFile('w') as af, NamedTemporaryFile('w') as kf, TemporaryDirectory() as outdir:
            write_canonical_oed_files(exposures, ef.name, accounts, af.name)
            write_keys_files(keys, kf.name)

            gul_items_df, canexp_df = self.manager.load_gul_items(self.canexp_profile, ef.name, kf.name)

            model = fake_model(resources={
                'canonical_exposures_df': canexp_df,
                'gul_items_df': gul_items_df,
                'canonical_exposures_profile': self.canexp_profile,
                'canonical_accounts_profile': self.canacc_profile,
                'fm_agg_profile': self.fm_agg_map
            })
            omr = model.resources
            ofp = omr['oasis_files_pipeline']

            ofp.keys_file_path = kf.name
            ofp.canonical_exposures_file_path = ef.name

            ofp.items_file_path = os.path.join(outdir, 'items.csv')
            ofp.coverages_file_path = os.path.join(outdir, 'coverages.csv')
            ofp.gulsummaryxref_file_path = os.path.join(outdir, 'gulsummaryxref.csv')

            gul_files = self.manager.write_gul_files(oasis_model=model)

            self.assertTrue(all(os.path.exists(p) for p in six.itervalues(gul_files)))

            guls = pd.merge(
                pd.merge(pd.read_csv(gul_files['items']), pd.read_csv(gul_files['coverages']), left_on='coverage_id', right_on='coverage_id'),
                pd.read_csv(gul_files['gulsummaryxref']),
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
            self.assertEqual(loc1_items['vulnerability_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(set(loc1_items['group_id'].values), {1})
            tivs = [exposures[0][t] for t in ['buildingtiv','othertiv','contentstiv','bitiv']]
            self.assertEqual(loc1_items['tiv'].values.tolist(), tivs)

            ofp.canonical_accounts_file_path = af.name
            ofp.fm_policytc_file_path = os.path.join(outdir, 'fm_policytc.csv')
            ofp.fm_profile_file_path = os.path.join(outdir, 'fm_profile.csv')
            ofp.fm_programme_file_path = os.path.join(outdir, 'fm_programme.csv')
            ofp.fm_xref_file_path = os.path.join(outdir, 'fm_xref.csv')
            ofp.fmsummaryxref_file_path = os.path.join(outdir, 'fmsummaryxref.csv')

            fm_files = self.manager.write_fm_files(oasis_model=model)

            self.assertTrue(all(os.path.exists(p) for p in six.itervalues(fm_files)))

            fm_programme_df = pd.read_csv(fm_files['fm_programme'])
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

            fm_profile_df = pd.read_csv(fm_files['fm_profile'])
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

            fm_policytc_df = pd.read_csv(fm_files['fm_policytc'])
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

            fm_xref_df = pd.read_csv(fm_files['fm_xref'])
            self.assertEqual(len(fm_xref_df), 4)
            self.assertEqual(fm_xref_df['output'].values.tolist(), [1,2,3,4])
            self.assertEqual(fm_xref_df['agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(fm_xref_df['layer_id'].values.tolist(), [1,1,1,1])

            fmsummaryxref_df = pd.read_csv(fm_files['fmsummaryxref'])
            self.assertEqual(len(fmsummaryxref_df), 4)
            self.assertEqual(fmsummaryxref_df['output'].values.tolist(), [1,2,3,4])
            self.assertEqual(fmsummaryxref_df['summary_id'].values.tolist(), [1,1,1,1])
            self.assertEqual(fmsummaryxref_df['summaryset_id'].values.tolist(), [1,1,1,1])

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_oed_exposures_data(
            from_account_nums=just(1),
            from_location_perils=just('WTC;WEC;BFR;OO1'),
            from_country_codes=just('US'),
            from_area_codes=just('CA'),
            from_buildings_tivs=just(1000000),
            from_buildings_deductibles=just(0),
            from_buildings_limits=just(0),
            from_other_tivs=just(100000),
            from_other_deductibles=just(0),
            from_other_limits=just(0),
            from_contents_tivs=just(50000),
            from_contents_deductibles=just(0),
            from_contents_limits=just(0),
            from_bi_tivs=just(20000),
            from_bi_deductibles=just(2000),
            from_bi_limits=just(18000),
            from_combined_deductibles=just(1000),
            from_combined_limits=just(1000000),
            from_site_deductibles=just(0),
            from_site_limits=just(0),
            size=1
        ),
        accounts=canonical_oed_accounts_data(
            from_account_nums=just(1),
            from_portfolio_nums=just(1),
            from_policy_nums=just(1),
            from_policy_perils=just('WTC;WEC;BFR;OO1'),
            from_sublimit_deductibles=just(0),
            from_sublimit_limits=just(0),
            from_account_deductibles=just(0),
            from_account_min_deductibles=just(0),
            from_account_max_deductibles=just(0),
            from_layer_deductibles=just(0),
            from_layer_limits=just(0),
            from_layer_shares=just(1),
            size=1
        ),
        keys=keys_data(
            from_peril_ids=just(1),
            from_coverage_type_ids=just(OED_COVERAGE_TYPES['buildings']['id']),
            from_area_peril_ids=just(1),
            from_vulnerability_ids=just(1),
            from_statuses=just('success'),
            from_messages=just('success'),
            size=4
        )
    )
    def test_fm4(self, exposures, accounts, keys):
        keys[1]['id'] = keys[2]['id'] = keys[3]['id'] = 1
        keys[1]['coverage_type'] = OED_COVERAGE_TYPES['other']['id']
        keys[2]['coverage_type'] = OED_COVERAGE_TYPES['contents']['id']
        keys[3]['coverage_type'] = OED_COVERAGE_TYPES['bi']['id']
        keys[1]['vulnerability_id'] = 2
        keys[2]['vulnerability_id'] = 3
        keys[3]['vulnerability_id'] = 4

        with NamedTemporaryFile('w') as ef, NamedTemporaryFile('w') as af, NamedTemporaryFile('w') as kf, TemporaryDirectory() as outdir:
            write_canonical_oed_files(exposures, ef.name, accounts, af.name)
            write_keys_files(keys, kf.name)

            gul_items_df, canexp_df = self.manager.load_gul_items(self.canexp_profile, ef.name, kf.name)

            model = fake_model(resources={
                'canonical_exposures_df': canexp_df,
                'gul_items_df': gul_items_df,
                'canonical_exposures_profile': self.canexp_profile,
                'canonical_accounts_profile': self.canacc_profile,
                'fm_agg_profile': self.fm_agg_map
            })
            omr = model.resources
            ofp = omr['oasis_files_pipeline']

            ofp.keys_file_path = kf.name
            ofp.canonical_exposures_file_path = ef.name

            ofp.items_file_path = os.path.join(outdir, 'items.csv')
            ofp.coverages_file_path = os.path.join(outdir, 'coverages.csv')
            ofp.gulsummaryxref_file_path = os.path.join(outdir, 'gulsummaryxref.csv')

            gul_files = self.manager.write_gul_files(oasis_model=model)

            self.assertTrue(all(os.path.exists(p) for p in six.itervalues(gul_files)))

            guls = pd.merge(
                pd.merge(pd.read_csv(gul_files['items']), pd.read_csv(gul_files['coverages']), left_on='coverage_id', right_on='coverage_id'),
                pd.read_csv(gul_files['gulsummaryxref']),
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
            self.assertEqual(loc1_items['vulnerability_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(set(loc1_items['group_id'].values), {1})
            tivs = [exposures[0][t] for t in ['buildingtiv','othertiv','contentstiv','bitiv']]
            self.assertEqual(loc1_items['tiv'].values.tolist(), tivs)

            ofp.canonical_accounts_file_path = af.name
            ofp.fm_policytc_file_path = os.path.join(outdir, 'fm_policytc.csv')
            ofp.fm_profile_file_path = os.path.join(outdir, 'fm_profile.csv')
            ofp.fm_programme_file_path = os.path.join(outdir, 'fm_programme.csv')
            ofp.fm_xref_file_path = os.path.join(outdir, 'fm_xref.csv')
            ofp.fmsummaryxref_file_path = os.path.join(outdir, 'fmsummaryxref.csv')

            fm_files = self.manager.write_fm_files(oasis_model=model)

            self.assertTrue(all(os.path.exists(p) for p in six.itervalues(fm_files)))

            fm_programme_df = pd.read_csv(fm_files['fm_programme'])
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

            fm_profile_df = pd.read_csv(fm_files['fm_profile'])
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

            fm_policytc_df = pd.read_csv(fm_files['fm_policytc'])
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

            fm_xref_df = pd.read_csv(fm_files['fm_xref'])
            self.assertEqual(len(fm_xref_df), 4)
            self.assertEqual(fm_xref_df['output'].values.tolist(), [1,2,3,4])
            self.assertEqual(fm_xref_df['agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(fm_xref_df['layer_id'].values.tolist(), [1,1,1,1])

            fmsummaryxref_df = pd.read_csv(fm_files['fmsummaryxref'])
            self.assertEqual(len(fmsummaryxref_df), 4)
            self.assertEqual(fmsummaryxref_df['output'].values.tolist(), [1,2,3,4])
            self.assertEqual(fmsummaryxref_df['summary_id'].values.tolist(), [1,1,1,1])
            self.assertEqual(fmsummaryxref_df['summaryset_id'].values.tolist(), [1,1,1,1])

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_oed_exposures_data(
            from_account_nums=just(1),
            from_location_perils=just('WTC;WEC;BFR;OO1'),
            from_country_codes=just('US'),
            from_area_codes=just('CA'),
            from_buildings_tivs=just(1000000),
            from_buildings_deductibles=just(0),
            from_buildings_limits=just(0),
            from_other_tivs=just(100000),
            from_other_deductibles=just(0),
            from_other_limits=just(0),
            from_contents_tivs=just(50000),
            from_contents_deductibles=just(0),
            from_contents_limits=just(0),
            from_bi_tivs=just(20000),
            from_bi_deductibles=just(0),
            from_bi_limits=just(0),
            from_combined_deductibles=just(0),
            from_combined_limits=just(0),
            from_site_deductibles=just(1000),
            from_site_limits=just(1000000),
            size=1
        ),
        accounts=canonical_oed_accounts_data(
            from_account_nums=just(1),
            from_portfolio_nums=just(1),
            from_policy_nums=just(1),
            from_policy_perils=just('WTC;WEC;BFR;OO1'),
            from_sublimit_deductibles=just(0),
            from_sublimit_limits=just(0),
            from_account_deductibles=just(0),
            from_account_min_deductibles=just(0),
            from_account_max_deductibles=just(0),
            from_layer_deductibles=just(0),
            from_layer_limits=just(0),
            from_layer_shares=just(1),
            size=1
        ),
        keys=keys_data(
            from_peril_ids=just(1),
            from_coverage_type_ids=just(OED_COVERAGE_TYPES['buildings']['id']),
            from_area_peril_ids=just(1),
            from_vulnerability_ids=just(1),
            from_statuses=just('success'),
            from_messages=just('success'),
            size=4
        )
    )
    def test_fm5(self, exposures, accounts, keys):
        keys[1]['id'] = keys[2]['id'] = keys[3]['id'] = 1
        keys[1]['coverage_type'] = OED_COVERAGE_TYPES['other']['id']
        keys[2]['coverage_type'] = OED_COVERAGE_TYPES['contents']['id']
        keys[3]['coverage_type'] = OED_COVERAGE_TYPES['bi']['id']
        keys[1]['vulnerability_id'] = 2
        keys[2]['vulnerability_id'] = 3
        keys[3]['vulnerability_id'] = 4

        with NamedTemporaryFile('w') as ef, NamedTemporaryFile('w') as af, NamedTemporaryFile('w') as kf, TemporaryDirectory() as outdir:
            write_canonical_oed_files(exposures, ef.name, accounts, af.name)
            write_keys_files(keys, kf.name)

            gul_items_df, canexp_df = self.manager.load_gul_items(self.canexp_profile, ef.name, kf.name)

            model = fake_model(resources={
                'canonical_exposures_df': canexp_df,
                'gul_items_df': gul_items_df,
                'canonical_exposures_profile': self.canexp_profile,
                'canonical_accounts_profile': self.canacc_profile,
                'fm_agg_profile': self.fm_agg_map
            })
            omr = model.resources
            ofp = omr['oasis_files_pipeline']

            ofp.keys_file_path = kf.name
            ofp.canonical_exposures_file_path = ef.name

            ofp.items_file_path = os.path.join(outdir, 'items.csv')
            ofp.coverages_file_path = os.path.join(outdir, 'coverages.csv')
            ofp.gulsummaryxref_file_path = os.path.join(outdir, 'gulsummaryxref.csv')

            gul_files = self.manager.write_gul_files(oasis_model=model)

            self.assertTrue(all(os.path.exists(p) for p in six.itervalues(gul_files)))

            guls = pd.merge(
                pd.merge(pd.read_csv(gul_files['items']), pd.read_csv(gul_files['coverages']), left_on='coverage_id', right_on='coverage_id'),
                pd.read_csv(gul_files['gulsummaryxref']),
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
            self.assertEqual(loc1_items['vulnerability_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(set(loc1_items['group_id'].values), {1})
            tivs = [exposures[0][t] for t in ['buildingtiv','othertiv','contentstiv','bitiv']]
            self.assertEqual(loc1_items['tiv'].values.tolist(), tivs)

            ofp.canonical_accounts_file_path = af.name
            ofp.fm_policytc_file_path = os.path.join(outdir, 'fm_policytc.csv')
            ofp.fm_profile_file_path = os.path.join(outdir, 'fm_profile.csv')
            ofp.fm_programme_file_path = os.path.join(outdir, 'fm_programme.csv')
            ofp.fm_xref_file_path = os.path.join(outdir, 'fm_xref.csv')
            ofp.fmsummaryxref_file_path = os.path.join(outdir, 'fmsummaryxref.csv')

            fm_files = self.manager.write_fm_files(oasis_model=model)

            fm_programme_df = pd.read_csv(fm_files['fm_programme'])
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

            fm_profile_df = pd.read_csv(fm_files['fm_profile'])
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

            fm_policytc_df = pd.read_csv(fm_files['fm_policytc'])
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
            self.assertEqual(level2_group['policytc_id'].values.tolist(), [3])
            level3_group = level_groups[2]
            self.assertEqual(len(level3_group), 1)
            self.assertEqual(level3_group['layer_id'].values.tolist(), [1])
            self.assertEqual(level3_group['agg_id'].values.tolist(), [1])
            self.assertEqual(level3_group['policytc_id'].values.tolist(), [2])

            fm_xref_df = pd.read_csv(fm_files['fm_xref'])
            self.assertEqual(len(fm_xref_df), 4)
            self.assertEqual(fm_xref_df['output'].values.tolist(), [1,2,3,4])
            self.assertEqual(fm_xref_df['agg_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(fm_xref_df['layer_id'].values.tolist(), [1,1,1,1])

            fmsummaryxref_df = pd.read_csv(fm_files['fmsummaryxref'])
            self.assertEqual(len(fmsummaryxref_df), 4)
            self.assertEqual(fmsummaryxref_df['output'].values.tolist(), [1,2,3,4])
            self.assertEqual(fmsummaryxref_df['summary_id'].values.tolist(), [1,1,1,1])
            self.assertEqual(fmsummaryxref_df['summaryset_id'].values.tolist(), [1,1,1,1])

            self.assertTrue(all(os.path.exists(p) for p in six.itervalues(fm_files)))

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_oed_exposures_data(
            from_account_nums=just(1),
            from_location_perils=just('WTC;WEC;BFR;OO1'),
            from_country_codes=just('US'),
            from_area_codes=just('CA'),
            from_buildings_tivs=just(1000000),
            from_buildings_deductibles=just(50000),
            from_buildings_limits=just(900000),
            from_other_tivs=just(100000),
            from_other_deductibles=just(5000),
            from_other_limits=just(90000),
            from_contents_tivs=just(50000),
            from_contents_deductibles=just(2500),
            from_contents_limits=just(45000),
            from_bi_tivs=just(20000),
            from_bi_deductibles=just(0),
            from_bi_limits=just(18000),
            from_combined_deductibles=just(0),
            from_combined_limits=just(0),
            from_site_deductibles=just(0),
            from_site_limits=just(0),
            size=2
        ),
        accounts=canonical_oed_accounts_data(
            from_account_nums=just(1),
            from_portfolio_nums=just(1),
            from_policy_nums=just(1),
            from_policy_perils=just('WTC;WEC;BFR;OO1'),
            from_sublimit_deductibles=just(0),
            from_sublimit_limits=just(0),
            from_account_deductibles=just(50000),
            from_account_min_deductibles=just(0),
            from_account_max_deductibles=just(0),
            from_layer_deductibles=just(0),
            from_layer_limits=just(2500000),
            from_layer_shares=just(1),
            size=1
        ),
        keys=keys_data(
            from_peril_ids=just(1),
            from_coverage_type_ids=just(OED_COVERAGE_TYPES['buildings']['id']),
            from_area_peril_ids=just(1),
            from_vulnerability_ids=just(1),
            from_statuses=just('success'),
            from_messages=just('success'),
            size=8
        )
    )
    def test_fm6(self, exposures, accounts, keys):
        exposures[1]['buildingtiv'] = 1700000
        exposures[1]['othertiv'] = 30000
        exposures[1]['contentstiv'] = 1000000
        exposures[1]['bitiv'] = 50000

        keys[1]['id'] = keys[2]['id'] = keys[3]['id'] = 1
        keys[4]['id'] = keys[5]['id'] = keys[6]['id'] = keys[7]['id'] = 2

        keys[4]['coverage_type'] = OED_COVERAGE_TYPES['buildings']['id']
        keys[1]['coverage_type'] = keys[5]['coverage_type'] = OED_COVERAGE_TYPES['other']['id']
        keys[2]['coverage_type'] = keys[6]['coverage_type'] = OED_COVERAGE_TYPES['contents']['id']
        keys[3]['coverage_type'] = keys[7]['coverage_type'] = OED_COVERAGE_TYPES['bi']['id']

        keys[4]['area_peril_id'] = keys[5]['area_peril_id'] = keys[6]['area_peril_id'] = keys[7]['area_peril_id'] = 2

        keys[4]['vulnerability_id'] = 1
        keys[1]['vulnerability_id'] = keys[5]['vulnerability_id'] = 2
        keys[2]['vulnerability_id'] = keys[6]['vulnerability_id'] = 3
        keys[3]['vulnerability_id'] = keys[7]['vulnerability_id'] = 4

        with NamedTemporaryFile('w') as ef, NamedTemporaryFile('w') as af, NamedTemporaryFile('w') as kf, TemporaryDirectory() as outdir:
            write_canonical_oed_files(exposures, ef.name, accounts, af.name)
            write_keys_files(keys, kf.name)

            gul_items_df, canexp_df = self.manager.load_gul_items(self.canexp_profile, ef.name, kf.name)

            model = fake_model(resources={
                'canonical_exposures_df': canexp_df,
                'gul_items_df': gul_items_df,
                'canonical_exposures_profile': self.canexp_profile,
                'canonical_accounts_profile': self.canacc_profile,
                'fm_agg_profile': self.fm_agg_map
            })
            omr = model.resources
            ofp = omr['oasis_files_pipeline']

            ofp.keys_file_path = kf.name
            ofp.canonical_exposures_file_path = ef.name

            ofp.items_file_path = os.path.join(outdir, 'items.csv')
            ofp.coverages_file_path = os.path.join(outdir, 'coverages.csv')
            ofp.gulsummaryxref_file_path = os.path.join(outdir, 'gulsummaryxref.csv')

            gul_files = self.manager.write_gul_files(oasis_model=model)

            self.assertTrue(all(os.path.exists(p) for p in six.itervalues(gul_files)))

            guls = pd.merge(
                pd.merge(pd.read_csv(gul_files['items']), pd.read_csv(gul_files['coverages']), left_on='coverage_id', right_on='coverage_id'),
                pd.read_csv(gul_files['gulsummaryxref']),
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
            self.assertEqual(loc1_items['vulnerability_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(set(loc1_items['group_id'].values), {1})
            tivs = [exposures[0][t] for t in ['buildingtiv','othertiv','contentstiv','bitiv']]
            self.assertEqual(loc1_items['tiv'].values.tolist(), tivs)

            loc2_id, loc2_items = loc_groups[1]
            self.assertEqual(loc2_id, 2)
            self.assertEqual(len(loc2_items), 4)
            self.assertEqual(loc2_id, 2)
            self.assertEqual(len(loc2_items), 4)
            self.assertEqual(loc2_items['item_id'].values.tolist(), [5,6,7,8])
            self.assertEqual(loc2_items['coverage_id'].values.tolist(), [5,6,7,8])
            self.assertEqual(set(loc2_items['areaperil_id'].values), {2})
            self.assertEqual(loc2_items['vulnerability_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(set(loc2_items['group_id'].values), {2})
            tivs = [exposures[1][t] for t in ['buildingtiv','othertiv','contentstiv','bitiv']]
            self.assertEqual(loc2_items['tiv'].values.tolist(), tivs)

            ofp.canonical_accounts_file_path = af.name
            ofp.fm_policytc_file_path = os.path.join(outdir, 'fm_policytc.csv')
            ofp.fm_profile_file_path = os.path.join(outdir, 'fm_profile.csv')
            ofp.fm_programme_file_path = os.path.join(outdir, 'fm_programme.csv')
            ofp.fm_xref_file_path = os.path.join(outdir, 'fm_xref.csv')
            ofp.fmsummaryxref_file_path = os.path.join(outdir, 'fmsummaryxref.csv')

            fm_files = self.manager.write_fm_files(oasis_model=model)

            self.assertTrue(all(os.path.exists(p) for p in six.itervalues(fm_files)))

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_oed_exposures_data(
            from_account_nums=just(1),
            from_location_perils=just('WTC;WEC;BFR;OO1'),
            from_country_codes=just('US'),
            from_area_codes=just('CA'),
            from_buildings_tivs=just(1000000),
            from_buildings_deductibles=just(10000),
            from_buildings_limits=just(0),
            from_other_tivs=just(100000),
            from_other_deductibles=just(5000),
            from_other_limits=just(0),
            from_contents_tivs=just(50000),
            from_contents_deductibles=just(5000),
            from_contents_limits=just(0),
            from_bi_tivs=just(20000),
            from_bi_deductibles=just(0),
            from_bi_limits=just(0),
            from_combined_deductibles=just(0),
            from_combined_limits=just(0),
            from_site_deductibles=just(0),
            from_site_limits=just(0),
            size=2
        ),
        accounts=canonical_oed_accounts_data(
            from_account_nums=just(1),
            from_portfolio_nums=just(1),
            from_policy_nums=just(1),
            from_policy_perils=just('WTC;WEC;BFR;OO1'),
            from_sublimit_deductibles=just(0),
            from_sublimit_limits=just(0),
            from_account_deductibles=just(50000),
            from_account_min_deductibles=just(0),
            from_account_max_deductibles=just(0),
            from_layer_deductibles=just(0),
            from_layer_limits=just(2500000),
            from_layer_shares=just(1),
            size=1
        ),
        keys=keys_data(
            from_peril_ids=just(1),
            from_coverage_type_ids=just(OED_COVERAGE_TYPES['buildings']['id']),
            from_area_peril_ids=just(1),
            from_vulnerability_ids=just(1),
            from_statuses=just('success'),
            from_messages=just('success'),
            size=8
        )
    )
    def test_fm7(self, exposures, accounts, keys):
        exposures[1]['buildingtiv'] = 1700000
        exposures[1]['othertiv'] = 30000
        exposures[1]['contentstiv'] = 1000000
        exposures[1]['bitiv'] = 50000

        keys[1]['id'] = keys[2]['id'] = keys[3]['id'] = 1
        keys[4]['id'] = keys[5]['id'] = keys[6]['id'] = keys[7]['id'] = 2

        keys[4]['coverage_type'] = OED_COVERAGE_TYPES['buildings']['id']
        keys[1]['coverage_type'] = keys[5]['coverage_type'] = OED_COVERAGE_TYPES['other']['id']
        keys[2]['coverage_type'] = keys[6]['coverage_type'] = OED_COVERAGE_TYPES['contents']['id']
        keys[3]['coverage_type'] = keys[7]['coverage_type'] = OED_COVERAGE_TYPES['bi']['id']

        keys[4]['area_peril_id'] = keys[5]['area_peril_id'] = keys[6]['area_peril_id'] = keys[7]['area_peril_id'] = 2

        keys[4]['vulnerability_id'] = 1
        keys[1]['vulnerability_id'] = keys[5]['vulnerability_id'] = 2
        keys[2]['vulnerability_id'] = keys[6]['vulnerability_id'] = 3
        keys[3]['vulnerability_id'] = keys[7]['vulnerability_id'] = 4

        with NamedTemporaryFile('w') as ef, NamedTemporaryFile('w') as af, NamedTemporaryFile('w') as kf, TemporaryDirectory() as outdir:
            write_canonical_oed_files(exposures, ef.name, accounts, af.name)
            write_keys_files(keys, kf.name)

            gul_items_df, canexp_df = self.manager.load_gul_items(self.canexp_profile, ef.name, kf.name)

            model = fake_model(resources={
                'canonical_exposures_df': canexp_df,
                'gul_items_df': gul_items_df,
                'canonical_exposures_profile': self.canexp_profile,
                'canonical_accounts_profile': self.canacc_profile,
                'fm_agg_profile': self.fm_agg_map
            })
            omr = model.resources
            ofp = omr['oasis_files_pipeline']

            ofp.keys_file_path = kf.name
            ofp.canonical_exposures_file_path = ef.name

            ofp.items_file_path = os.path.join(outdir, 'items.csv')
            ofp.coverages_file_path = os.path.join(outdir, 'coverages.csv')
            ofp.gulsummaryxref_file_path = os.path.join(outdir, 'gulsummaryxref.csv')

            gul_files = self.manager.write_gul_files(oasis_model=model)

            self.assertTrue(all(os.path.exists(p) for p in six.itervalues(gul_files)))

            guls = pd.merge(
                pd.merge(pd.read_csv(gul_files['items']), pd.read_csv(gul_files['coverages']), left_on='coverage_id', right_on='coverage_id'),
                pd.read_csv(gul_files['gulsummaryxref']),
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
            self.assertEqual(loc1_items['vulnerability_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(set(loc1_items['group_id'].values), {1})
            tivs = [exposures[0][t] for t in ['buildingtiv','othertiv','contentstiv','bitiv']]
            self.assertEqual(loc1_items['tiv'].values.tolist(), tivs)

            loc2_id, loc2_items = loc_groups[1]
            self.assertEqual(loc2_id, 2)
            self.assertEqual(len(loc2_items), 4)
            self.assertEqual(loc2_id, 2)
            self.assertEqual(len(loc2_items), 4)
            self.assertEqual(loc2_items['item_id'].values.tolist(), [5,6,7,8])
            self.assertEqual(loc2_items['coverage_id'].values.tolist(), [5,6,7,8])
            self.assertEqual(set(loc2_items['areaperil_id'].values), {2})
            self.assertEqual(loc2_items['vulnerability_id'].values.tolist(), [1,2,3,4])
            self.assertEqual(set(loc2_items['group_id'].values), {2})
            tivs = [exposures[1][t] for t in ['buildingtiv','othertiv','contentstiv','bitiv']]
            self.assertEqual(loc2_items['tiv'].values.tolist(), tivs)

            ofp.canonical_accounts_file_path = af.name
            ofp.fm_policytc_file_path = os.path.join(outdir, 'fm_policytc.csv')
            ofp.fm_profile_file_path = os.path.join(outdir, 'fm_profile.csv')
            ofp.fm_programme_file_path = os.path.join(outdir, 'fm_programme.csv')
            ofp.fm_xref_file_path = os.path.join(outdir, 'fm_xref.csv')
            ofp.fmsummaryxref_file_path = os.path.join(outdir, 'fmsummaryxref.csv')

            fm_files = self.manager.write_fm_files(oasis_model=model)

            self.assertTrue(all(os.path.exists(p) for p in six.itervalues(fm_files)))
