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
                self.test_cases_fp, 'runs', 'test-{}-{}'.format(test_case,utcnow)
            )
            shutil.copytree(tmp_run_dir, output_dir)
            print(f'Generated Output stored in: {output_dir}')

    def run_test(self, test_case):
        with tempfile.TemporaryDirectory() as tmp_run_dir:
            result = OasisManager().run_fm_test(
                test_case_dir=self.test_cases_fp,
                test_case_name=test_case,
                run_dir=tmp_run_dir,
                update_expected=self.update_expected)
            self._store_output(test_case, tmp_run_dir)

        self.assertTrue(result)

    def test_insurance(self):
        self.run_test('insurance')

    def test_insurance_step(self):
        self.run_test('insurance_step')

    def test_reinsurance1(self):
        self.run_test('reinsurance1')

    # def test_fm3(self):
    #     self.run_test('fm3')

    # def test_fm4(self):
    #     self.run_test('fm4')

    # def test_fm5(self):
    #     self.run_test('fm5')

    # def test_fm6(self):
    #     self.run_test('fm6')

    # def test_fm7(self):
    #     self.run_test('fm7')

    # def test_fm8(self):
    #     self.run_test('fm8')

    # def test_fm9(self):
    #     self.run_test('fm9')

    # def test_fm12(self):
    #     self.run_test('fm12')

    # def test_fm40(self):
    #     self.run_test('fm40')

    # def test_fm41(self):
    #     self.run_test('fm41')

    # def test_fm54(self):
    #     self.run_test('fm54')

    # def test_fm55(self):
    #     self.run_test('fm55')

    # def test_fm57(self):
    #     self.run_test('fm57')

    # def test_fm58(self):
    #     self.run_test('fm58')

    # def test_Q1_all(self):
    #     self.run_test('Q1_all')

    # def test_Q2_all(self):
    #     self.run_test('Q2_all')

    # def test_Q3_all(self):
    #     self.run_test('Q3_all')

    # def test_Q4_all(self):
    #     self.run_test('Q4_all')


# Tests replaced by reinsurance1
    # def test_CX29_03(self):
    #     self.run_test('CX29_03')

    # def test_CX29_05(self):
    #     self.run_test('CX29_05')

    # def test_CX29_08(self):
    #     self.run_test('CX29_08')

    # def test_CX29_95(self):
    #     self.run_test('CX29_95')

    # def test_CX30_03(self):
    #     self.run_test('CX30_03')

    # def test_CX30_05(self):
    #     self.run_test('CX30_05')

    # def test_CX30_97(self):
    #     self.run_test('CX30_97')

    # def test_CX30_100(self):
    #     self.run_test('CX30_100')

    # def test_CX31_102(self):
    #     self.run_test('CX31_102')

    # def test_CX31_104(self):
    #     self.run_test('CX31_104')

    # def test_CX32_107(self):
    #     self.run_test('CX32_107')

    # def test_CX32_108(self):
    #     self.run_test('CX32_108')

    # def test_FA37_02(self):
    #     self.run_test('FA37_02')

    # def test_FA37_125(self):
    #     self.run_test('FA37_125')

    # def test_FA38_129(self):
    #     self.run_test('FA38_129')

    # def test_FA39_133(self):
    #     self.run_test('FA39_133')

    # def test_FA40_134(self):
    #     self.run_test('FA40_134')

    # def test_FA41_138(self):
    #     self.run_test('FA41_138')

    # def test_FA42_142(self):
    #     self.run_test('FA42_142')

    # def test_FA43_143(self):
    #     self.run_test('FA43_143')

    # def test_FA44_147(self):
    #     self.run_test('FA44_147')

    # def test_FA45_149(self):
    #     self.run_test('FA45_149')

    # def test_PR13_42(self):
    #     self.run_test('PR13_42')

    # def test_PR13_44(self):
    #     self.run_test('PR13_44')

    # def test_PR14_47(self):
    #     self.run_test('PR14_47')

    # def test_PR15_48(self):
    #     self.run_test('PR15_48')

    # def test_PR16_52(self):
    #     self.run_test('PR16_52')

    # def test_PR17_56(self):
    #     self.run_test('PR17_56')

    # def test_PR17_57(self):
    #     self.run_test('PR17_57')

    # def test_PR18_58(self):
    #     self.run_test('PR18_58')

    # def test_PR19_62(self):
    #     self.run_test('PR19_62')

    # def test_PR20_66(self):
    #     self.run_test('PR20_66')

    # def test_PR21_67(self):
    #     self.run_test('PR21_67')

    # def test_PR22_72(self):
    #     self.run_test('PR22_72')

    # def test_PR23_76(self):
    #     self.run_test('PR23_76')

    # def test_PR24_77(self):
    #     self.run_test('PR24_77')

    # def test_PR25_81(self):
    #     self.run_test('PR25_81')

    # def test_PR25_83(self):
    #     self.run_test('PR25_83')

    # def test_PR26_86(self):
    #     self.run_test('PR26_86')

    # def test_PR27_87(self):
    #     self.run_test('PR27_87')

    # def test_PR28_91(self):
    #     self.run_test('PR28_91')

    # def test_QS05_15(self):
    #     self.run_test('QS05_15')

    # def test_QS05_16(self):
    #     self.run_test('QS05_16')

    # def test_QS06_17(self):
    #     self.run_test('QS06_17')

    # def test_QS07_21(self):
    #     self.run_test('QS07_21')

    # def test_QS08_26(self):
    #     self.run_test('QS08_26')

    # def test_QS09_27(self):
    #     self.run_test('QS09_27')

    # def test_QS09_30(self):
    #     self.run_test('QS09_30')

    # def test_QS10_32(self):
    #     self.run_test('QS10_32')

    # def test_QS11_36(self):
    #     self.run_test('QS11_36')

    # def test_QS11_37(self):
    #     self.run_test('QS11_37')

    # def test_QS12_38(self):
    #     self.run_test('QS12_38')

    # def test_SS01_2(self):
    #     self.run_test('SS01_2')

    # def test_SS02_6(self):
    #     self.run_test('SS02_6')

    # def test_SS03_7(self):
    #     self.run_test('SS03_7')

    # def test_SS04_11(self):
    #     self.run_test('SS04_11')

    # def test_CX33_109(self):
    #     self.run_test('CX33_109')

    # def test_CX33_112(self):
    #     self.run_test('CX33_112')

    # def test_CX34_114(self):
    #     self.run_test('CX34_114')

    # def test_CX34_116(self):
    #     self.run_test('CX34_116')

    # def test_CX35_119(self):
    #     self.run_test('CX35_119')

    # def test_CX35_120(self):
    #     self.run_test('CX35_120')

    # def test_CX36_121(self):
    #     self.run_test('CX36_121')

    # def test_CX36_124(self):
    #     self.run_test('CX36_124')

#Tests to deprecate
    # @pytest.mark.skip(reason='Needs fixing')
    # def test_ri_multiple_cxl_at_different_inuring_levels(self):
    #     self.run_test('ri_multiple_cxl_at_different_inuring_levels')

    # @pytest.mark.skip(reason='Needs fixing')
    # def test_ri_multiple_cxl_at_same_inuring_level(self):
    #     self.run_test('ri_multiple_cxl_at_same_inuring_level')

    # @pytest.mark.skip(reason='Needs fixing')
    # def test_ri_single_QS(self):
    #     self.run_test('ri_single_QS')

    # @pytest.mark.skip(reason='Needs fixing')
    # def test_ri_single_QS_with_location_level_risk_limits(self):
    #     self.run_test('ri_single_QS_with_location_level_risk_limits')

    # @pytest.mark.skip(reason='Needs fixing')
    # def test_ri_single_cxl(self):
    #     self.run_test('ri_single_cxl')

    # @pytest.mark.skip(reason='Needs fixing')
    # def test_ri_single_loc_level_PR_all_risks(self):
    #     self.run_test('ri_single_loc_level_PR_all_risks')

    # @pytest.mark.skip(reason='Needs fixing')
    # def test_ri_single_loc_level_fac(self):
    #     self.run_test('ri_single_loc_level_fac')

    # @pytest.mark.skip(reason='Needs fixing')
    # def test_ri_single_pol_level_fac(self):
    #     self.run_test('ri_single_pol_level_fac')


# Todo reinsurance2
#     def test_xx45_L(self):
#         self.run_test('xx45_L')

#     def test_xx45_A(self):
#         self.run_test('xx45_A')

#     def test_xx45_P(self):
#         self.run_test('xx45_P')

#      def test_xx32(self):
#         self.run_test('xx32')

#     def test_xx38(self):
#         self.run_test('xx38')

#     def test_xx39(self):
#         self.run_test('xx39')

#     def test_xx42(self):
#         self.run_test('xx42')

#     def test_xx43(self):
#         self.run_test('xx43')

#     def test_xx47(self):
#         self.run_test('xx47')

#     def test_xx53(self):
#         self.run_test('xx53')

#     def test_QS46_1(self):
#         self.run_test('QS46_1')

#     def test_FA45_150(self):
#         self.run_test('FA45_150')

#     def test_FA45_151(self):
#         self.run_test('FA45_151')

#     def test_FA45_152(self):
#         self.run_test('FA45_152')

#     def test_xxAcc1(self):
#         self.run_test('xxAcc1')

#     def test_xxAcc2(self):
#         self.run_test('xxAcc2')

#     def test_xxAcc3(self):
#         self.run_test('xxAcc3')

#     def test_xxAcc4(self):
#         self.run_test('xxAcc4')

# #To do 
#     def test_fm24(self):
#         self.run_test('fm24')

#    def test_fm27(self):
#         self.run_test('fm27')


