import os
import unittest

from oasislmf.model_preparation import (
    oed
)

cwd = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(cwd, 'examples')


class TestReinsurance(unittest.TestCase):

    def test_validate_oed_direct_inly(self):

        case_dir = os.path.join(input_dir, "direct_only")

        (
            ri_info_df,
            ri_scope_df,
            do_reinsurance
        ) = oed.load_oed_dfs(
            os.path.join(case_dir, 'ri_info.csv'),
            os.path.join(case_dir, 'ri_scope.csv'))

        self.assertFalse(do_reinsurance)

    def test_validate_oed_single_cxl(self):

        case_dir = os.path.join(input_dir, "single_cxl")

        (
            ri_info_df,
            ri_scope_df,
            do_reinsurance
        ) = oed.load_oed_dfs(
            os.path.join(case_dir, 'ri_info.csv'),
            os.path.join(case_dir, 'ri_scope.csv'))

        self.assertTrue(do_reinsurance)
