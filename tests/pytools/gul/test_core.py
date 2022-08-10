"""
This file tests gul/core.py functionality
"""
from unittest import main, TestCase
import numpy as np

from oasislmf.pytools.gul.core import split_tiv_classic, split_tiv_multiplicative


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

    def test_split_tiv_multiplicative(self) -> None:
        Ngulitems = 10
        tiv = 1e3

        # test 1: all gulitems are zeros, should remain all zeros
        gulitems_orig = np.zeros(Ngulitems)
        gulitems = gulitems_orig.copy()
        split_tiv_multiplicative(gulitems, tiv)
        np.testing.assert_array_equal(np.zeros(Ngulitems), gulitems)

        # test 2: sum(gulitems) < tiv, gulitems losses should remain unchanged
        gulitems_orig = np.array([40., 60., 100., 20., 120., 188., 300., 80., 30., 50.])
        assert np.sum(gulitems_orig) < tiv
        gulitems = gulitems_orig.copy()
        split_tiv_multiplicative(gulitems, tiv)
        multiplicative_loss = tiv * (1. - np.prod(1. - gulitems_orig / tiv))
        expected_gulitems = gulitems_orig * (multiplicative_loss / np.sum(gulitems_orig))
        np.testing.assert_array_almost_equal(expected_gulitems, gulitems, decimal=1e-14)


if __name__ == "__main__":
    main()
