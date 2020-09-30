import json
import os
import string

from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from hypothesis import (
    given,
    settings,
)
import hypothesis.strategies as st
from hypothesis.strategies import (
    just,
    integers,
)

from oasislmf.preparation.summaries import write_exposure_summary
from oasislmf.preparation.summaries import get_exposure_summary
from oasislmf.preparation.gul_inputs import get_gul_input_items
from oasislmf.utils.coverages import SUPPORTED_COVERAGE_TYPES
from oasislmf.utils.defaults import get_default_exposure_profile
from oasislmf.utils.data import get_location_df, get_ids
from oasislmf.utils.status import OASIS_KEYS_STATUS_MODELLED

from tests.data import (
    keys,
    min_source_exposure,
    write_source_files,
    write_keys_files,
)


# https://towardsdatascience.com/automating-unit-tests-in-python-with-hypothesis-d53affdc1eba
class TestSummaries(TestCase):

    def assertDictAlmostEqual(self, d1, d2, msg=None, places=7):
        # check if both inputs are dicts
        self.assertIsInstance(d1, dict, 'First argument is not a dictionary')
        self.assertIsInstance(d2, dict, 'Second argument is not a dictionary')

        # check if both inputs have the same keys
        self.assertEqual(d1.keys(), d2.keys())

        # check each key
        for key, value in d1.items():
            if isinstance(value, dict):
                self.assertDictAlmostEqual(d1[key], d2[key], msg=msg)
            else:
                self.assertAlmostEqual(d1[key], d2[key], places=places, msg=msg)

    def assertSummaryIsValid(self, loc_df, gul_inputs, exp_summary, perils_expected=None):
        cov_types = ['buildings', 'other', 'bi', 'contents']
        lookup_status = ['success', 'fail', 'nomatch', 'fail_ap', 'fail_v', 'notatrisk']
        loc_rename_cols = {
            'bitiv': 'bi',
            'buildingtiv': 'buildings',
            'contentstiv': 'contents',
            'othertiv': 'other'
        }

        # Check each returned peril
        for peril in perils_expected:
            peril_summary = exp_summary[peril]

            # Check the 'All' section
            supported_tivs = loc_df[['buildingtiv', 'othertiv', 'bitiv', 'contentstiv']].sum(0).rename(loc_rename_cols)
            self.assertAlmostEqual(supported_tivs.sum(0), peril_summary['all']['tiv'])

            for cov in cov_types:
                self.assertAlmostEqual(supported_tivs[cov], peril_summary['all']['tiv_by_coverage'][cov])

            # Check each lookup status
            peril_expected = gul_inputs[gul_inputs.peril_id == peril]
            for status in lookup_status:
                peril_status = peril_expected[peril_expected.status == status]
                self.assertAlmostEqual(peril_status.tiv.sum(), peril_summary[status]['tiv'])
                self.assertEqual(len(peril_status.loc_id.unique()), peril_summary[status]['number_of_locations'])

                for cov in cov_types:
                    cov_type_id = SUPPORTED_COVERAGE_TYPES[cov]['id']
                    cov_type_tiv = peril_status[peril_status.coverage_type_id == cov_type_id].tiv.sum()
                    self.assertAlmostEqual(cov_type_tiv, peril_summary[status]['tiv_by_coverage'][cov])

            # Check 'noreturn' status
            tiv_returned = sum([s[1]['tiv'] for s in peril_summary.items() if s[0] in lookup_status])
            self.assertAlmostEqual(peril_summary['all']['tiv'] - tiv_returned, peril_summary['noreturn']['tiv'])

            for cov in cov_types:
                cov_tiv_returned = sum(
                    [s[1]['tiv_by_coverage'][cov] for s in peril_summary.items() if s[0] in lookup_status])
            self.assertAlmostEqual(peril_summary['all']['tiv_by_coverage'][cov] - cov_tiv_returned, peril_summary['noreturn']['tiv_by_coverage'][cov])

    @given(st.data())
    @settings(max_examples=20, deadline=None)
    def test_single_peril__totals_correct(self, data):

        # Shared Values between Loc / keys
        loc_size = data.draw(integers(10, 20))
        supported_cov = data.draw(st.lists(integers(1,4), unique=True, min_size=1, max_size=4))
        perils = 'WTC'

        # Create Mock keys_df
        keys_data = list()
        for i in supported_cov:
            keys_data += data.draw(keys(
                size=loc_size,
                from_peril_ids=just(perils),
                from_coverage_type_ids=just(i),
                from_area_peril_ids=just(1),
                from_vulnerability_ids=just(1),
                from_messages=just('str')))
        keys_df = pd.DataFrame.from_dict(keys_data)

        # Create Mock location_df
        loc_df = pd.DataFrame.from_dict(data.draw(min_source_exposure(
            size=loc_size,
            from_location_perils_covered=just(perils),
            from_location_perils=just(perils),
            from_building_tivs=integers(1000, 1000000),
            from_other_tivs=integers(100, 100000),
            from_contents_tivs=integers(50, 50000),
            from_bi_tivs=integers(20, 20000))))
        loc_df['loc_id'] = get_ids(loc_df, ['portnumber', 'accnumber', 'locnumber'])

        # Run exposure_summary
        exp_summary = get_exposure_summary(
            exposure_df=loc_df,
            keys_df=keys_df,
        )

        # Run Gul Proccessing
        gul_inputs = get_gul_input_items(loc_df, keys_df)
        gul_inputs = gul_inputs[gul_inputs['status'].isin(OASIS_KEYS_STATUS_MODELLED)]

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
        self.assertEqual(len(gul_inputs.loc_id.unique()), exp_summary['total']['modelled']['number_of_locations'])

        # Check number of not-modelled
        # WARNING: current assumption is that all cov types must be covered to be modelled 
        moddeled = 0
        moddeld_loc_ids = gul_inputs[gul_inputs['status'] == 'success'].loc_id.unique()
        for loc_id in moddeld_loc_ids:
            if len(gul_inputs[gul_inputs.loc_id == loc_id].coverage_type_id.unique()) == 4:
                moddeled+=1
        self.assertEqual(len(loc_df) - moddeled, exp_summary['total']['not-modelled']['number_of_locations'])

    @given(st.data())
    @settings(max_examples=10, deadline=None)
    def test_multi_perils__single_covarage(self, data):
        loc_size = data.draw(integers(10, 20))
        supported_cov = data.draw(integers(1,4))
        perils = data.draw(st.lists(
            st.text(alphabet=(string.ascii_letters + string.digits), min_size=2, max_size=6),
            min_size=2,
            max_size=6,
            unique=True
        ))

        # Create Mock keys_df
        keys_data = list()
        for p in perils:
            keys_data += data.draw(keys(
                size=loc_size,
                from_peril_ids=just(p),
                from_coverage_type_ids=just(supported_cov),
                from_area_peril_ids=just(1),
                from_vulnerability_ids=just(1),
                from_messages=just('str')))

        keys_df = pd.DataFrame.from_dict(keys_data)
        perils_returned = keys_df.peril_id.unique().tolist()

        # Create Mock location_df
        perils_covered = ';'.join(perils)
        loc_df = pd.DataFrame.from_dict(data.draw(min_source_exposure(
            size=loc_size,
            from_location_perils_covered=just(perils_covered),
            from_location_perils=just(perils_covered),
            from_building_tivs=st.one_of(st.floats(1.0, 1000.0), st.integers(1, 1000)),
            from_other_tivs=st.one_of(st.floats(1.0, 1000.0), st.integers(1, 1000)),
            from_contents_tivs=st.one_of(st.floats(1.0, 1000.0), st.integers(1, 1000)),
            from_bi_tivs=st.one_of(st.floats(1.0, 1000.0), st.integers(1, 1000)))))
        loc_df['loc_id'] = get_ids(loc_df, ['portnumber', 'accnumber', 'locnumber'])

        # Run Summary output check
        self.assertSummaryIsValid(
            loc_df,
            get_gul_input_items(loc_df, keys_df),
            get_exposure_summary(exposure_df=loc_df, keys_df=keys_df),
            perils_returned
        )

    @given(st.data())
    @settings(max_examples=20, deadline=None)
    def test_multi_perils__multi_covarage(self, data):
        loc_size = data.draw(integers(10, 20))
        supported_cov = data.draw(st.lists(integers(1,4), unique=True, min_size=1, max_size=4))
        perils = data.draw(st.lists(
            st.text(alphabet=(string.ascii_letters + string.digits), min_size=2, max_size=6),
            min_size=2,
            max_size=6,
            unique=True))

        # Create Mock keys_df
        keys_data = list()
        for c in supported_cov:
            for p in perils:
                keys_data += data.draw(keys(
                    size=loc_size,
                    from_peril_ids=just(p),
                    from_coverage_type_ids=just(c),
                    from_area_peril_ids=just(1),
                    from_vulnerability_ids=just(1),
                    from_messages=just('str')))

        keys_df = pd.DataFrame.from_dict(keys_data)
        perils_returned = keys_df.peril_id.unique().tolist()

        # Create Mock location_df
        loc_df = pd.DataFrame.from_dict(data.draw(min_source_exposure(
            size=loc_size,
            from_location_perils_covered=st.sampled_from(perils),
            from_location_perils=st.sampled_from(perils),
            from_building_tivs=st.one_of(st.floats(1.0, 1000.0), st.integers(1, 1000)),
            from_other_tivs=st.one_of(st.floats(1.0, 1000.0), st.integers(1, 1000)),
            from_contents_tivs=st.one_of(st.floats(1.0, 1000.0), st.integers(1, 1000)),
            from_bi_tivs=st.one_of(st.floats(1.0, 1000.0), st.integers(1, 1000)))))
        loc_df['loc_id'] = get_ids(loc_df, ['portnumber', 'accnumber', 'locnumber'])

        # Run Summary output check
        exp_summary = get_exposure_summary(exposure_df=loc_df, keys_df=keys_df)
        gul_inputs = get_gul_input_items(loc_df, keys_df)
        self.assertSummaryIsValid(
            loc_df,
            gul_inputs,
            exp_summary,
            perils_returned)

    @given(st.data())
    @settings(max_examples=10, deadline=None)
    def test_summary_file_written(self, data):
        loc_size = data.draw(integers(10, 20))

        # Create Mock keys_data
        keys_data = data.draw(keys(
            size=loc_size,
            from_area_peril_ids=just(1),
            from_vulnerability_ids=just(1),
            from_messages=just('str')))

        # Create Mock location_data
        loc_data = data.draw(min_source_exposure(
            size=loc_size,
            from_building_tivs=st.one_of(st.floats(1.0, 1000.0), st.integers(1, 1000)),
            from_other_tivs=st.one_of(st.floats(1.0, 1000.0), st.integers(1, 1000)),
            from_contents_tivs=st.one_of(st.floats(1.0, 1000.0), st.integers(1, 1000)),
            from_bi_tivs=st.one_of(st.floats(1.0, 1000.0), st.integers(1, 1000))))

        # Prepare arguments for write_exposure_summary
        with TemporaryDirectory() as tmp_dir:

            keys_fp = os.path.join(tmp_dir, 'keys.csv')
            successes = [k for k in keys_data if k['status'] in ['success']]
            keys_errors_fp = os.path.join(tmp_dir, 'keys_errors.csv')
            nonsuccesses = [k for k in keys_data if k['status'] not in ['success']]

            write_keys_files(
                keys=successes,
                keys_file_path=keys_fp,
                keys_errors=nonsuccesses,
                keys_errors_file_path=keys_errors_fp
            )

            location_fp = os.path.join(tmp_dir, 'location.csv')
            write_source_files(exposure=loc_data, exposure_fp=location_fp)
            location_df = get_location_df(location_fp)

            exposure_summary_fp = write_exposure_summary(
                tmp_dir,
                location_df,
                keys_fp,
                keys_errors_fp,
                get_default_exposure_profile())
            self.assertTrue(os.path.isfile(exposure_summary_fp))

            with open(exposure_summary_fp) as f:
                data = json.load(f)
                loc_df = pd.DataFrame.from_dict(loc_data)
                loc_df['loc_id'] = get_ids(loc_df, ['portnumber', 'accnumber', 'locnumber'])

                keys_df = pd.DataFrame.from_dict(keys_data)
                exp_summary = get_exposure_summary(
                    loc_df,
                    keys_df)
                self.assertDictAlmostEqual(data, exp_summary)
