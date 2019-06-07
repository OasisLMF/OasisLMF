import io
import os
import json
import pandas as pd
import random
import string
from unittest import TestCase
from tempfile import TemporaryDirectory
from hypothesis import (
    given,
    settings,
)
from hypothesis.strategies import (
    just,
    sampled_from,
    integers,
)

from oasislmf.manager import OasisManager as om
from oasislmf.model_preparation.summaries import write_exposure_summary
from oasislmf.model_preparation.gul_inputs import get_gul_input_items
from oasislmf.utils.coverages import SUPPORTED_COVERAGE_TYPES
from oasislmf.utils.status import OASIS_KEYS_STATUS
from oasislmf.utils.profiles import get_oed_hierarchy
from oasislmf.utils.peril import PERILS

from tests.data import (
    keys,
    source_exposure,
    write_source_files,
    write_keys_files,
)


class TestSummaries(TestCase):

    # To remove
    @staticmethod
    def write_csv_file(keys_dict, path):
        pd.DataFrame(keys_dict).to_csv(path, index=False, encoding='utf-8')


    @staticmethod
    def get_keys_dict(peril_ls, nlocations, successful=True, start_loc=1):
        """
        Compile and return dictionaries used to create keys and keys errors file.

        peril_ls type: list
        peril_ls desc: list of perils keys from PERILS dict

        nlocations type: int
        nlocations desc: number of unique locations to include

        successful type: bool
        successful desc: True to create keys dict, False to create keys errors dit

        start_loc type: int
        start_loc desc: first location id
        """

        def get_random_string(string_length):
            """
            Create and return random string of upper and lower case letters.

            string_length type: int
            string_length desc: length of string to be created.
            """
            letters = string.ascii_letters
            return ''.join(random.choice(letters) for _ in range(string_length))


        peril_ids = [PERILS[peril]['id'] for peril in peril_ls]
        coverage_type_ids = [SUPPORTED_COVERAGE_TYPES['buildings']['id'],
                             SUPPORTED_COVERAGE_TYPES['contents']['id']]
        entries = nlocations * len(peril_ids) * len(coverage_type_ids)

        keys_dict = {
            'locnumber': [(i // 6) + start_loc for i in range(entries)],
            'peril_id': [y for _ in range(nlocations) for x in peril_ids for y in [x, x]],
            'coverage_type': [coverage for _ in range(nlocations) for _ in range(len(peril_ids)) for coverage in coverage_type_ids],
            'area_peril_id': [random.randint(1, 10) for _ in range(entries)],
            'vulnerability_id': [random.randint(1, 10) for _ in range(entries)],
            'status': [OASIS_KEYS_STATUS['success']['id'] if successful else (OASIS_KEYS_STATUS['fail']['id'] if i < entries / 2 else OASIS_KEYS_STATUS['nomatch']['id']) for i in range(entries)],
            'message': [get_random_string(string_length=random.randint(1,10)) for _ in range(entries)]
        }

        return pd.DataFrame(keys_dict).to_dict('records')


    def setUp(self):

        self.peril_ls = [
            'extra tropical cyclone',
            'earthquake',
            'flash flood'
        ]
        self.successes_nloc = 2
        self.nonsuccesses_nloc = 2
        self.keys_successes = self.get_keys_dict(
            peril_ls=self.peril_ls,
            nlocations=self.successes_nloc
        )
        self.keys_nonsuccesses = self.get_keys_dict(
            peril_ls=self.peril_ls,
            nlocations=self.nonsuccesses_nloc,
            successful=False,
            start_loc=self.successes_nloc+1
        )


    @settings(deadline=None, max_examples=10)
    @given(
        exposure = source_exposure(
            from_account_ids=just('1'),
            from_portfolio_ids=just('1'),
            from_location_perils=just('WTC;WEC;BFR;OO1'),
            from_location_perils_covered=just('WTC;WEC;BFR;OO1'),
            from_country_codes=just('US'),
            from_area_codes=just('CA'),
            from_building_tivs=integers(1000,1000000),
            from_building_deductibles=just(0),
            from_building_min_deductibles=just(0),
            from_building_max_deductibles=just(0),
            from_building_limits=just(0),
            from_other_tivs=integers(100,100000),
            from_other_deductibles=just(0),
            from_other_min_deductibles=just(0),
            from_other_max_deductibles=just(0),
            from_other_limits=just(0),
            from_contents_tivs=integers(50,50000),
            from_contents_deductibles=just(0),
            from_contents_min_deductibles=just(0),
            from_contents_max_deductibles=just(0),
            from_contents_limits=just(0),
            from_bi_tivs=integers(20,20000),
            from_bi_deductibles=just(0),
            from_bi_min_deductibles=just(0),
            from_bi_max_deductibles=just(0),
            from_bi_limits=just(0),
            from_sitepd_deductibles=just(0),
            from_sitepd_min_deductibles=just(0),
            from_sitepd_max_deductibles=just(0),
            from_sitepd_limits=just(0),
            from_siteall_deductibles=just(0),
            from_siteall_min_deductibles=just(0),
            from_siteall_max_deductibles=just(0),
            from_siteall_limits=just(0),
            size=4
        )
    )
    def test_write_exposure_summary(self, exposure):
        """
        Test write_exposure_summary method.
        """

        with TemporaryDirectory() as d:

            # Prepare arguments for write_exposure_summary
            target_dir = os.path.join(d, 'inputs')
            os.mkdir(target_dir)

            keys_fp = os.path.join(d, 'keys.csv')
            keys_errors_fp = os.path.join(d, 'keys_errors.csv')
            write_keys_files(
                keys=self.keys_successes,
                keys_file_path=keys_fp,
                keys_errors=self.keys_nonsuccesses,
                keys_errors_file_path=keys_errors_fp
            )

            exposure_fp = os.path.join(d, 'exposure.csv')
            write_source_files(exposure=exposure, exposure_fp=exposure_fp)

            self.manager = om()
            exposure_profile = self.manager.exposure_profile

            gul_inputs_df, exposure_df = get_gul_input_items(
                exposure_fp, keys_fp, exposure_profile
            )

            oed_hierarchy = get_oed_hierarchy(exposure_profile=exposure_profile)

            # Execute method
            write_exposure_summary(
                target_dir,
                gul_inputs_df,
                exposure_df,
                exposure_fp,
                keys_errors_fp,
                exposure_profile,
                oed_hierarchy
            )

            # Get output file for testing
            output_filename = target_dir + "/exposure_summary_report.json"
            with open(output_filename) as f:
                data = json.load(f)

            # Test integrity of output file
            # Loop over all modelled perils
            for peril in self.peril_ls:
                
                # Test modelled peril is in output file
                self.assertIn(peril, data.keys())

                tiv_per_peril = 0
                tiv_per_coverage = {}
                total_nlocations = 0

                # Loop over all keys statuses
                for status in OASIS_KEYS_STATUS.values():
                    status_id = status['id']
                    tiv_per_status = 0

                    # Loop over all supported coverage types
                    for coverage_type in SUPPORTED_COVERAGE_TYPES.keys():
                        coverage_tiv = data[peril][status_id]['tiv_by_coverage'][coverage_type]
                        tiv_per_status += coverage_tiv
                        if coverage_type in tiv_per_coverage.keys():
                            tiv_per_coverage[coverage_type] += coverage_tiv
                        else:
                            tiv_per_coverage[coverage_type] = coverage_tiv

                    # Test sum of TIV by coverage per status
                    self.assertEqual(
                        tiv_per_status,
                        data[peril][status_id]['tiv']
                    )

                    tiv_per_peril += tiv_per_status
                    total_nlocations += data[peril][status_id]['number_of_locations']

                # Test sum of TIV by status per peril
                self.assertEqual(tiv_per_peril, data[peril]['all']['tiv'])

                # Loop over all supported coverage types
                for coverage_type in SUPPORTED_COVERAGE_TYPES.keys():

                    # Test sum of TIV by coverage and status per peril
                    self.assertEqual(
                        tiv_per_coverage[coverage_type],
                        data[peril]['all']['tiv_by_coverage'][coverage_type]
                    )

                # Test sum of number of locations per status
                self.assertEqual(
                    total_nlocations,
                    data[peril]['all']['number_of_locations']
                )
