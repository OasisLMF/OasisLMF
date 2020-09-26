#import json
import os
#import random
import string

#from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from hypothesis import (
    given,
#    HealthCheck,
    settings,
)
import hypothesis.strategies as st
from hypothesis.strategies import (
    just,
    integers,
    one_of,
    sampled_from,
)

#from oasislmf.manager import OasisManager as om
from oasislmf.preparation.summaries import write_exposure_summary
from oasislmf.preparation.summaries import get_exposure_summary
from oasislmf.preparation.gul_inputs import get_gul_input_items
from oasislmf.utils.coverages import SUPPORTED_COVERAGE_TYPES
#from oasislmf.utils.data import get_location_df
#from oasislmf.utils.peril import PERILS, PERIL_GROUPS
#from oasislmf.utils.profiles import get_oed_hierarchy
#from oasislmf.utils.status import OASIS_KEYS_STATUS
from oasislmf.utils.defaults import get_default_exposure_profile

from tests.data import (
    keys,
    source_exposure,
    min_source_exposure,
    write_source_files,
    write_keys_files,
)



# https://towardsdatascience.com/automating-unit-tests-in-python-with-hypothesis-d53affdc1eba
class TestSummaries(TestCase):

    @given(st.data())
    @settings(max_examples=10, deadline=None)
    def test_single_peril__totals_correct(self, data):

        # Shared Values between Loc / keys
        loc_size = data.draw(integers(10, 20))
        perils = 'WTC'

        # Create Mock keys_df
        keys_df = pd.DataFrame.from_dict(data.draw(keys(
            size=loc_size,
            from_peril_ids=just(perils),
            from_area_peril_ids=just(1),
            from_vulnerability_ids=just(1),
            from_messages=just('str'),
        )))

        # Create Mock location_df
        loc_df = pd.DataFrame.from_dict(data.draw(min_source_exposure(
            size=loc_size,
            from_location_perils_covered=just(perils),
            from_location_perils=just(perils),
            from_building_tivs=integers(1000, 1000000),
            from_other_tivs=integers(100, 100000),
            from_contents_tivs=integers(50, 50000),
            from_bi_tivs=integers(20, 20000),
        )))
        loc_df['loc_id'] = loc_df.index

        # Run exposure_summary
        exp_summary = get_exposure_summary(
            exposure_df=loc_df,
            keys_df=keys_df,
        )

        # Run Gul Proccessing
        gul_inputs = get_gul_input_items(loc_df, keys_df)
        gul_inputs = gul_inputs[gul_inputs['status'].isin(['success', 'notatrisk'])]

        print(exp_summary['total'])

        # Fetch expected TIVS
        tiv_portfolio = loc_df[['buildingtiv', 'othertiv', 'bitiv', 'contentstiv']].sum(1).sum(0)
        tiv_modelled = gul_inputs['tiv'].sum()
        tiv_not_modelled = tiv_portfolio - tiv_modelled

        # Check TIV values
        self.assertEqual(tiv_portfolio, exp_summary['total']['portfolio']['tiv'])
        self.assertEqual(tiv_modelled, exp_summary['total']['modelled']['tiv'])
        self.assertEqual(tiv_not_modelled, exp_summary['total']['not-modelled']['tiv'])

        # Check number of locs
        self.assertEqual(len(loc_df), exp_summary['total']['portfolio']['number_of_locations'])
        self.assertEqual(len(gul_inputs), exp_summary['total']['modelled']['number_of_locations'])

        ## Cleaner way to do this?
        moddeld_locs = gul_inputs[gul_inputs['status'] == 'success']
        non_cov_locs = len(loc_df) - min(
            len(moddeld_locs[moddeld_locs.coverage_type_id == 1]),
            len(moddeld_locs[moddeld_locs.coverage_type_id == 2]),
            len(moddeld_locs[moddeld_locs.coverage_type_id == 3]),
            len(moddeld_locs[moddeld_locs.coverage_type_id == 4]))
        self.assertEqual(non_cov_locs, exp_summary['total']['not-modelled']['number_of_locations'])


    @given(st.data())
    @settings(max_examples=10, deadline=None)
    def test_multi_perils__perils_all_supported(self, data):

        # Shared Values between Loc / keys
        loc_size = data.draw(integers(10, 20))
        perils = data.draw(st.lists(
            st.text(alphabet=(string.ascii_letters + string.digits), min_size=2, max_size=6),
            min_size=2,
            max_size=6,
            unique=True
        ))

        # Create Mock keys_df
        keys_df = pd.DataFrame.from_dict(data.draw(keys(
            size=loc_size,
            from_peril_ids=st.sampled_from(perils),
            from_area_peril_ids=just(1),
            from_vulnerability_ids=just(1),
            from_messages=just('str'),
        )))
        perils_returned = keys_df.peril_id.unique().tolist()

        # Create Mock location_df
        perils_covered = ';'.join(perils)
        loc_df = pd.DataFrame.from_dict(data.draw(min_source_exposure(
            size=loc_size,
            from_location_perils_covered=just(perils_covered),
            from_location_perils=just(perils_covered),
            #from_location_perils_covered=st.sampled_from(perils),
            #from_location_perils=st.sampled_from(perils),
            from_building_tivs=st.one_of(st.floats(1.0, 1000.0), st.integers(0,1000)),
            from_other_tivs=st.one_of(st.floats(0.0, 1000.0), st.integers(0,1000)),
            from_contents_tivs=st.one_of(st.floats(0.0, 1000.0), st.integers(0,1000)),
            from_bi_tivs=st.one_of(st.floats(0.0, 1000.0), st.integers(0,1000)),
        )))
        loc_df['loc_id'] = loc_df.index

        # Run exposure_summary
        exp_summary = get_exposure_summary(
            exposure_df=loc_df,
            keys_df=keys_df,
        )

        # Run Gul Proccessing
        gul_inputs = get_gul_input_items(loc_df, keys_df)

        # Check each returned peril
        loc_rename_cols = {
            'bitiv': 'bi',
            'buildingtiv': 'buildings',
            'contentstiv': 'contents',
            'othertiv': 'other'
        }

        for peril in perils_returned:
            peril_summary = exp_summary[peril]
            
            # Check the 'All' section
            supported_tivs = loc_df[['buildingtiv', 'othertiv', 'bitiv', 'contentstiv']].sum(0).rename(loc_rename_cols)
            self.assertAlmostEqual(supported_tivs.sum(0), peril_summary['all']['tiv'])

            for cov_type in ['buildings', 'other', 'bi', 'contents']:
                self.assertAlmostEqual(supported_tivs[cov_type], peril_summary['all']['tiv_by_coverage'][cov_type])

            # Check each lookup status 
            peril_expected = gul_inputs[gul_inputs.peril_id == peril]  
            valid_status_list = ['success', 'fail', 'nomatch', 'fail_ap', 'fail_v', 'notatrisk']
            for status in valid_status_list:
                peril_status = peril_expected[peril_expected.status == status]
                self.assertAlmostEqual(peril_status.tiv.sum(), peril_summary[status]['tiv'])
                self.assertEqual(len(peril_status), peril_summary[status]['number_of_locations'])

                for cov_type in ['buildings', 'other', 'bi', 'contents']:
                    cov_type_id = SUPPORTED_COVERAGE_TYPES[cov_type]['id']
                    cov_type_tiv = peril_status[peril_status.coverage_type_id == cov_type_id].tiv.sum()
                    self.assertAlmostEqual(cov_type_tiv, peril_summary[status]['tiv_by_coverage'][cov_type])
            
            # Check 'noreturn' status 
            tiv_returned = sum([s[1]['tiv'] for s in peril_summary.items() if s[0] in valid_status_list])
            self.assertAlmostEqual(peril_summary['all']['tiv'] - tiv_returned, peril_summary['noreturn']['tiv'])    

            for cov_type in ['buildings', 'other', 'bi', 'contents']:
                cov_tiv_returned = sum(
                    [s[1]['tiv_by_coverage'][cov_type] for s in peril_summary.items() if s[0] in valid_status_list])
            self.assertAlmostEqual(peril_summary['all']['tiv_by_coverage'][cov_type] - cov_tiv_returned, peril_summary['noreturn']['tiv_by_coverage'][cov_type])    


    @given(st.data())
    @settings(max_examples=20, deadline=None)
    def test_multi_perils__perils_partialaly_supported(self, data):
        pass

    @given(st.data())
    @settings(max_examples=20, deadline=None)
    def test_peril_not_covered__is_zero(self, data):
        pass

    def test_summary_file_written(self):
        pass



