"""
This file tests gul/core.py functionality
"""
from unittest import main, TestCase
import numpy as np

from oasislmf.pytools.gul.core import split_tiv_classic


class TestGulpyCore(TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_split_tiv_classic(self) -> None:
        Ngulitems = 100
        tiv = 1e3

        # test 1: sum(gulitems) < tiv, no split should be executed
        gulitems_orig = np.ones(Ngulitems) * tiv / Ngulitems - 1e-3
        gulitems = gulitems_orig.copy()
        split_tiv_classic(gulitems, tiv)
        np.testing.assert_array_equal(gulitems_orig, gulitems)

        # test 2: sum(gulitems) > tiv, split should be executed
        gulitems_orig = np.ones(Ngulitems) * tiv / Ngulitems + 1e-3
        gulitems = gulitems_orig.copy()
        split_tiv_classic(gulitems, tiv)
        np.testing.assert_array_almost_equal(gulitems_orig * tiv / np.sum(gulitems_orig), gulitems, decimal=1e-14)

    def test_split_tiv_fractional(self) -> None:
        # WIP
        pass


if __name__ == "__main__":
    main()
