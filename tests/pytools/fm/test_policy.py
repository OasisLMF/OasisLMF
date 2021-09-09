import numpy as np
from numpy.testing import assert_array_almost_equal
from oasislmf.pytools.fm.common import fm_profile_dtype
from oasislmf.pytools.fm.policy import calc, UnknownCalcrule


def test_calcrule_1():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 1, 15, 0, 0, 10, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 0., 5., 15., 25., 30., 30.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_2():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 2, 15, 0, 0, 10, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 0., 0., 2.5, 7.5, 12.5, 15.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_3():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 3, 15, 0, 0, 10, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 0., 20., 30., 30., 30., 30.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_5():
    # ded + limit > 1
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 5, 0.25, 0, 0, 10, 0.8, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 7.5, 15., 22.5, 30., 37.5, 45.])

    # ded + limit < 1
    assert_array_almost_equal(loss_out, loss_expected)

    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 5, 0.25, 0, 0, 10, 0.5, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 5., 10., 15., 20., 25., 30.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_12():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 12, 15, 0, 0, 10, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 0., 5., 15., 25., 35., 45.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_14():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 14, 15, 0, 0, 10, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 10., 20., 30., 30., 30., 30.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_15():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])

    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 15, 15, 0, 0, 10, 0.6, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 0., 5., 15., 24., 30., 36.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_16():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 16, 1/4, 0, 0, 10, 0.6, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 7.5, 15., 22.5, 30., 37.5, 45.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_17():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])

    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 17, 1/4, 0, 0, 10, 25, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 0, 2.5, 6.25, 10., 12.5, 12.5])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_20():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 20, 25, 0, 0, 10, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 10., 20., 0., 0., 0., 0.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_22():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])

    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 22, 0, 0, 0, 0, 40, 1/2, 3/4, 4/5)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 3., 6., 9., 12., 15., 18.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_23():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])

    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 23, 0, 0, 0, 0, 40, 1/2, 3/4, 4/5)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 6., 12., 18., 24., 24., 24.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_24():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 24, 0, 0, 0, 5, 20, 1/2, 3/4, 4/5)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 0., 3., 6., 9., 12., 12.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_25():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])

    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 25, 0, 0, 0, 0, 0, 1/2, 3/4, 4/5)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 3., 6., 9., 12., 15., 18.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_33():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])

    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 33, 1/4, 0, 0, 10, 30, 0, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 7.5, 15., 22.5, 30., 30., 30.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_34():
    """as there is shares, deductible won't be use later on so no need to compute it"""
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])

    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 34, 15, 0, 0, 10, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, None)

    loss_expected = np.array([0., 0., 0., 2.5, 7.5, 12.5, 17.5])

    assert_array_almost_equal(loss_out, loss_expected)
