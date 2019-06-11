import os.path
import subprocess
import sys

from unittest import TestCase

import pytest


class FmAcceptanceTests(TestCase):

    def setUp(self):
        self.test_cases_fp = os.path.join(sys.path[0], 'validation', 'examples')

    def run_test(self, test_case):
        cmd_str = 'oasislmf exposure run -s {} --validate'.format(os.path.join(self.test_cases_fp, test_case))
        failed = False
        try:
            subprocess.run(cmd_str.split(), check=True)
        except subprocess.CalledProcessError as e:
            failed = (e.returncode != 0)
        self.assertTrue(failed is False)

    def test_fm3(self):
        self.run_test('fm3')

    def test_fm4(self):
        self.run_test('fm4')

    def test_fm5(self):
        self.run_test('fm5')

    def test_fm6(self):
        self.run_test('fm6')

    def test_fm7(self):
        self.run_test('fm7')

    def test_fm8(self):
        self.run_test('fm8')

    def test_fm9(self):
        self.run_test('fm9')

    @pytest.mark.skip(reason='Needs fixing')
    def test_fm12(self):
        self.run_test('fm12')

    def test_fm40(self):
        self.run_test('fm40')

    def test_fm41(self):
        self.run_test('fm41')

    def test_CX29_03(self):
        self.run_test('CX29_03')
