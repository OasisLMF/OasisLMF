import json
import os
import random

from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from hypothesis import (
    given,
    HealthCheck,
    settings,
)
import hypothesis.strategies as st
from hypothesis.strategies import (
    just,
    integers,
    sampled_from,
)

from oasislmf.manager import OasisManager as om
from oasislmf.model_preparation.summaries import write_exposure_summary
from oasislmf.model_preparation.summaries import get_exposure_summary
from oasislmf.model_preparation.gul_inputs import get_gul_input_items
from oasislmf.utils.coverages import SUPPORTED_COVERAGE_TYPES
from oasislmf.utils.data import get_location_df
from oasislmf.utils.peril import PERILS, PERIL_GROUPS
from oasislmf.utils.profiles import get_oed_hierarchy
from oasislmf.utils.status import OASIS_KEYS_STATUS
from oasislmf.utils.defaults import get_default_exposure_profile

from tests.data import (
    keys,
    source_exposure,
    write_source_files,
    write_keys_files,
)



# https://towardsdatascience.com/automating-unit-tests-in-python-with-hypothesis-d53affdc1eba
class TestSummaries(TestCase):

    @given(st.data())
    @settings(max_examples=5, deadline=None)
    def test_totals_match_single_peril(self, data):
    
        # Shared Values between Loc / keys
        loc_size = data.draw(integers(10, 20)) 
        perils = 'WTC'

        # Create Mock keys_df 
        keys_df = pd.DataFrame.from_dict(data.draw(keys(
            size=loc_size,
            from_peril_ids=just(perils),
        )))
        print(keys_df)
        keys_success = keys_df[keys_df['status'].isin(['success'])] 
        keys_failed = keys_df[~keys_df['status'].isin(['success'])]

        # Create Mock location_df 
        loc_df = pd.DataFrame.from_dict(data.draw(source_exposure(
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
            exposure_profile=get_default_exposure_profile(),
            keys_success_df=keys_success,
            keys_errors_df=keys_failed,
        ) 

        # Run Gul Proccessing 
        gul_inputs = get_gul_input_items(loc_df, keys_df)
        gul_inputs = gul_inputs[gul_inputs['status'].isin(['success'])]

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
        #self.assertEqual(len(loc_df) - len(gul_inputs), exp_summary['total']['not-modelled']['number_of_locations'])



    def test_summary_file_written(self):
        pass

