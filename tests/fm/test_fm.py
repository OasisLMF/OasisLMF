import os.path
import sys
import tempfile
import shutil

from oasislmf.manager import OasisManager
from oasislmf.utils.data import get_utctimestamp
from unittest import TestCase

import pytest

class FmAcceptanceTests(TestCase):

    def setUp(self):
        self.test_cases_fp = os.path.join(sys.path[0], 'validation')
        self.update_expected = False
        self.keep_output = True

    def _store_output(self, test_case, tmp_run_dir):
        if self.keep_output:
            utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
            output_dir = os.path.join(
                self.test_cases_fp, 'runs', 'test-fm-{}-{}'.format(test_case,utcnow)
            )
            shutil.copytree(tmp_run_dir, output_dir)
            print(f'Generated Output stored in: {output_dir}')

    def run_test(self, test_case, fmpy=False, subperils=1):
        with tempfile.TemporaryDirectory() as tmp_run_dir:
            result = OasisManager().run_fm_test(
                test_case_dir=self.test_cases_fp,
                test_case_name=test_case,
                run_dir=tmp_run_dir,
                update_expected=self.update_expected,
                fmpy=fmpy,
                num_subperils=subperils,
                test_tolerance=0.001
            )
            self._store_output(test_case, tmp_run_dir)

        self.assertTrue(result)


    def test_insurance(self):
        self.run_test('insurance')

    def test_insurance_step(self):
        self.run_test('insurance_step')

    def test_reinsurance1(self):
        self.run_test('reinsurance1')

    def test_reinsurance2(self):
        self.run_test('reinsurance2')

    # multiperil tests 
    def test_insurance_2_subperils(self):
        self.run_test('insurance', subperils=2)

    def test_insurance_step_2_subperils(self):
        self.run_test('insurance_step', subperils=2)

    def test_reinsurance1_2_subperils(self):
        self.run_test('reinsurance1', subperils=2)

    def test_reinsurance2_2_subperils(self):
        self.run_test('reinsurance2', subperils=2)
    