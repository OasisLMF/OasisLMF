import datetime
import os.path
import sys
import tempfile
from unittest import TestCase

import pytest

from oasislmf.manager import OasisManager


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

    def run_test(self, test_case, fmpy=False, subperils=1, expected_dir="expected"):
        with tempfile.TemporaryDirectory() as tmp_run_dir:

            run_dir = tmp_run_dir
            if self.fm_keep_output:
                utcnow = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
                run_dir = os.path.join(
                    self.test_cases_fp, 'runs', 'test-fmpy-p{}-{}-{}'.format(subperils, test_case, utcnow)
                )
                print(f'Generating Output in: {run_dir}')

            result = OasisManager().run_fm_test(
                test_case_dir=self.test_cases_fp,
                test_case_name=test_case,
                run_dir=run_dir,
                update_expected=self.update_expected,
                fmpy=fmpy,
                fmpy_sort_output=True,
                num_subperils=subperils,
                test_tolerance=0.001,
                expected_output_dir=expected_dir,
            )
        self.assertTrue(result)

    def test_insurance_conditions(self):
        self.run_test('insurance_conditions', fmpy=True)

    def test_insurance_and_step(self):
        self.run_test('insurance_and_step', fmpy=True)

    def test_reinsurance1(self):
        self.run_test('reinsurance1', fmpy=True)

    def test_reinsurance2(self):
        self.run_test('reinsurance2', fmpy=True)

    def test_issues(self):
        self.run_test('issues', fmpy=True)

    def test_insurance_policy_coverage(self):
        self.run_test('insurance_policy_coverage', fmpy=True)

    def test_insurance_conditions_2_subperils(self):
        self.run_test('insurance_conditions', fmpy=True, subperils=2, expected_dir="expected_subperils")

    def test_insurance_and_step_2_subperils(self):
        self.run_test('insurance_and_step', fmpy=True, subperils=2, expected_dir="expected_subperils")

    def test_reinsurance1_2_subperils(self):
        self.run_test('reinsurance1', fmpy=True, subperils=2, expected_dir="expected_subperils")

    def test_reinsurance2_2_subperils(self):
        self.run_test('reinsurance2', fmpy=True, subperils=2, expected_dir="expected_subperils")

    def test_issues_2_subperils(self):
        self.run_test('issues', fmpy=True, subperils=2, expected_dir="expected_subperils")

    def test_insurance_policy_coverage_2_subperils(self):
        self.run_test('insurance_policy_coverage', fmpy=True, subperils=2, expected_dir="expected_subperils")
