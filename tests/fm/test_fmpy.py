import datetime
import os.path
import sys
import tempfile
from unittest import TestCase

import pytest

from oasislmf.manager import OasisManager
from oasislmf.utils.defaults import DISAGGREGATION_ITEMS, DISAGGREGATION_SAMPLES


@pytest.fixture(scope="class")
def update_expected(request):
    # set a class attribute on the invoking context
    request.cls.update_expected = request.config.getoption('--update-expected')


@pytest.fixture(scope="class")
def fm_keep_output(request):
    # set a class attribute on the invoking context
    request.cls.fm_keep_output = request.config.getoption('--fm-keep-output')


@pytest.mark.usefixtures("update_expected")
@pytest.mark.usefixtures("fm_keep_output")
class FmAcceptanceTests(TestCase):

    def setUp(self):
        self.test_cases_fp = os.path.join(sys.path[0], 'validation')

    def run_test(self, test_case, model_perils_covered, fmpy=False, expected_dir="expected",
                 disaggregation=DISAGGREGATION_ITEMS, compare_files=None):
        with tempfile.TemporaryDirectory() as tmp_run_dir:

            run_dir = tmp_run_dir
            if self.fm_keep_output:
                utcnow = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
                run_dir = os.path.join(
                    self.test_cases_fp, 'runs', 'test-fmpy-p{}-{}-{}'.format('_'.join(model_perils_covered), test_case, utcnow)
                )
                print(f'Generating Output in: {run_dir}')

            result = OasisManager().run_fm_test(
                test_case_dir=self.test_cases_fp,
                test_case_name=test_case,
                run_dir=run_dir,
                update_expected=self.update_expected,
                model_perils_covered=model_perils_covered,
                fmpy=fmpy,
                fmpy_sort_output=True,
                test_tolerance=0.001,
                expected_output_dir=expected_dir,
                disaggregation=disaggregation,
                compare_files=compare_files,
            )
        self.assertTrue(result)

    def test_building_packing(self):
        self.run_test('building_packing', ['BBF'], fmpy=True)

    def test_building_packing_samples_matches_items(self):
        """Packing a location's buildings into the sample dimension must not change the loss.

        ``items`` and ``samples`` are two encodings of the same four-building risk, so the unit's
        ``expected/`` -- generated under ``items`` -- is the reference for both. This runs the same
        case packed and compares against it.

        Only ``loc_summary.csv`` is compared: the per-item files say where the buildings live,
        which is exactly what the two modes disagree about by design.
        """
        self.run_test('building_packing', ['BBF'], fmpy=True,
                      disaggregation=DISAGGREGATION_SAMPLES,
                      compare_files=['loc_summary.csv'])

    def test_insurance_conditions(self):
        self.run_test('insurance_conditions', model_perils_covered=['WTC'], fmpy=True)

    def test_insurance_and_step(self):
        self.run_test('insurance_and_step', model_perils_covered=['WTC'], fmpy=True)

    def test_insurance_account(self):
        self.run_test('insurance_account', model_perils_covered=['WTC'], fmpy=True)

    def test_reinsurance1(self):
        self.run_test('reinsurance1', model_perils_covered=['WTC'], fmpy=True)

    def test_reinsurance2(self):
        self.run_test('reinsurance2', model_perils_covered=['WTC'], fmpy=True)

    def test_reinsurance3(self):
        self.run_test('reinsurance3', model_perils_covered=['WTC'], fmpy=True)

    def test_reinsurance4(self):
        self.run_test('reinsurance4', model_perils_covered=['WTC'], fmpy=True)

    def test_insurance_bi(self):
        self.run_test('insurance_bi', fmpy=True, model_perils_covered=['WTC'])

    def test_issues(self):
        self.run_test('issues', model_perils_covered=['WTC'], fmpy=True)

    def test_issue_1816(self):
        self.run_test('issue_1816', model_perils_covered=['WTC'], fmpy=True)

    def test_issue_layerparticipation(self):
        self.run_test('issue_1953_layerparticipation', model_perils_covered=['WTC'], fmpy=True)

    def test_insurance_policy_coverage(self):
        self.run_test('insurance_policy_coverage', model_perils_covered=['WTC'], fmpy=True)

    def test_insurance_conditions_2_subperils(self):
        self.run_test('insurance_conditions', fmpy=True, model_perils_covered=['WW2'], expected_dir="expected_subperils")

    def test_insurance_and_step_2_subperils(self):
        self.run_test('insurance_and_step', fmpy=True, model_perils_covered=['WW2'], expected_dir="expected_subperils")

    def test_reinsurance1_2_subperils(self):
        self.run_test('reinsurance1', fmpy=True, model_perils_covered=['WW2'], expected_dir="expected_subperils")

    def test_reinsurance2_2_subperils(self):
        self.run_test('reinsurance2', fmpy=True, model_perils_covered=['WW2'], expected_dir="expected_subperils")

    def test_issues_2_subperils(self):
        self.run_test('issues', fmpy=True, model_perils_covered=['WW2'], expected_dir="expected_subperils")

    def test_insurance_policy_coverage_2_subperils(self):
        self.run_test('insurance_policy_coverage', fmpy=True, model_perils_covered=['WW2'], expected_dir="expected_subperils")

    def test_insurance_bi_2_subperils(self):
        self.run_test('insurance_bi', fmpy=True, model_perils_covered=['WW2'], expected_dir="expected_subperils")

    def test_cyber(self):
        self.run_test('cyber', model_perils_covered=['CSB'], fmpy=True)

    def test_cyber_2_subperils(self):
        self.run_test('cyber', fmpy=True, model_perils_covered=['CC1'], expected_dir="expected_subperils")
