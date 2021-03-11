import os.path
import sys
import tempfile
import shutil
import datetime

from oasislmf.manager import OasisManager
from unittest import TestCase

import pytest

class FmAcceptanceTests(TestCase):

    def setUp(self):
        self.test_cases_fp = os.path.join(sys.path[0], 'validation')
        self.update_expected = False
        self.keep_output = True

    def _store_output(self, test_case, tmp_run_dir):
        if self.keep_output:
            utcnow = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
            output_dir = os.path.join(
                self.test_cases_fp, 'runs', 'test-fmpy-{}-{}'.format(test_case,utcnow)
            )
            shutil.copytree(tmp_run_dir, output_dir)
            print(f'Generated Output stored in: {output_dir}')

    def run_test(self, test_case, fmpy=False):
        with tempfile.TemporaryDirectory() as tmp_run_dir:
            result = OasisManager().run_fm_test(
                test_case_dir=self.test_cases_fp,
                test_case_name=test_case,
                run_dir=tmp_run_dir,
                update_expected=self.update_expected,
                fmpy=fmpy,
                fmpy_sort_output=True,
                test_tolerance=0.0001 # one location has -0.0001% error in insurance_fm
            )
            self._store_output(test_case, tmp_run_dir)

        self.assertTrue(result)

    def test_insurance(self):
        self.run_test('insurance',fmpy=True)

    def test_insurance_step(self):
        self.run_test('insurance_step',fmpy=True)

    def test_reinsurance1(self):
        self.run_test('reinsurance1',fmpy=True)

    def test_reinsurance2(self):
        self.run_test('reinsurance2',fmpy=True)



