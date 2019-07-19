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

    def test_CX29_05(self):
        self.run_test('CX29_05')

    def test_CX29_08(self):
        self.run_test('CX29_08')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_CX29_95(self):
        self.run_test('CX29_95')

    def test_CX30_03(self):
        self.run_test('CX30_03')

    def test_CX30_05(self):
        self.run_test('CX30_05')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_CX30_97(self):
        self.run_test('CX30_97')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_CX30_100(self):
        self.run_test('CX30_100')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_CX31_102(self):
        self.run_test('CX31_102')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_CX31_104(self):
        self.run_test('CX31_104')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_CX32_107(self):
        self.run_test('CX32_107')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_CX32_108(self):
        self.run_test('CX32_108')

    def test_FA37_02(self):
        self.run_test('FA37_02')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_FA37_125(self):
        self.run_test('FA37_125')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_FA38_129(self):
        self.run_test('FA38_129')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_FA39_133(self):
        self.run_test('FA39_133')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_FA40_134(self):
        self.run_test('FA40_134')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_FA41_138(self):
        self.run_test('FA41_138')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_FA42_142(self):
        self.run_test('FA42_142')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_FA43_143(self):
        self.run_test('FA43_143')

    @pytest.mark.skip(reason='Needs fixing')
    def test_FA44_147(self):
        self.run_test('FA44_147')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR13_42(self):
        self.run_test('PR13_42')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR13_44(self):
        self.run_test('PR13_44')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR14_47(self):
        self.run_test('PR14_47')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR15_48(self):
        self.run_test('PR15_48')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR16_52(self):
        self.run_test('PR16_52')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR17_56(self):
        self.run_test('PR17_56')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR17_57(self):
        self.run_test('PR17_57')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR18_58(self):
        self.run_test('PR18_58')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR19_62(self):
        self.run_test('PR19_62')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR20_66(self):
        self.run_test('PR20_66')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR21_67(self):
        self.run_test('PR21_67')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR22_72(self):
        self.run_test('PR22_72')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR23_76(self):
        self.run_test('PR23_76')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR24_77(self):
        self.run_test('PR24_77')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR25_81(self):
        self.run_test('PR25_81')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR25_83(self):
        self.run_test('PR25_83')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR26_86(self):
        self.run_test('PR26_86')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR27_87(self):
        self.run_test('PR27_87')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_PR28_91(self):
        self.run_test('PR28_91')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_QS05_15(self):
        self.run_test('QS05_15')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_QS05_16(self):
        self.run_test('QS05_16')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_QS06_17(self):
        self.run_test('QS06_17')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_QS07_21(self):
        self.run_test('QS07_21')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_QS08_26(self):
        self.run_test('QS08_26')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_QS09_27(self):
        self.run_test('QS09_27')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_QS09_30(self):
        self.run_test('QS09_30')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_QS10_32(self):
        self.run_test('QS10_32')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_QS11_36(self):
        self.run_test('QS11_36')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_QS11_37(self):
        self.run_test('QS11_37')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_QS12_38(self):
        self.run_test('QS12_38')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_SS01_2(self):
        self.run_test('SS01_2')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_SS02_6(self):
        self.run_test('SS02_6')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_SS03_7(self):
        self.run_test('SS03_7')

    #@pytest.mark.skip(reason='Needs fixing')
    def test_SS04_11(self):
        self.run_test('SS04_11')

    @pytest.mark.skip(reason='Needs fixing')
    def test_ri_multiple_cxl_at_different_inuring_levels(self):
        self.run_test('ri_multiple_cxl_at_different_inuring_levels')

    @pytest.mark.skip(reason='Needs fixing')
    def test_ri_multiple_cxl_at_same_inuring_level(self):
        self.run_test('ri_multiple_cxl_at_same_inuring_level')

    @pytest.mark.skip(reason='Needs fixing')
    def test_ri_single_QS(self):
        self.run_test('ri_single_QS')

    @pytest.mark.skip(reason='Needs fixing')
    def test_ri_single_QS_with_location_level_risk_limits(self):
        self.run_test('ri_single_QS_with_location_level_risk_limits')

    @pytest.mark.skip(reason='Needs fixing')
    def test_ri_single_cxl(self):
        self.run_test('ri_single_cxl')

    @pytest.mark.skip(reason='Needs fixing')
    def test_ri_single_loc_level_PR_all_risks(self):
        self.run_test('ri_single_loc_level_PR_all_risks')

    @pytest.mark.skip(reason='Needs fixing')
    def test_ri_single_loc_level_PR_loc_filter(self):
        self.run_test('ri_single_loc_level_PR_loc_filter')

    @pytest.mark.skip(reason='Needs fixing')
    def test_ri_single_loc_level_fac(self):
        self.run_test('ri_single_loc_level_fac')

    @pytest.mark.skip(reason='Needs fixing')
    def test_ri_single_pol_level_fac(self):
        self.run_test('ri_single_pol_level_fac')
