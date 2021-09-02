import numpy as np
from numpy.testing import assert_array_almost_equal
from oasislmf.pytools.fm.common import fm_profile_dtype
from oasislmf.pytools.fm.policy import calc, UnknownCalcrule


def test_calcrule_1():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 1, 15, 0, 0, 10, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 0., 5., 15., 25., 30., 30.])
    deductible_expected = np.array([5., 15., 20., 20., 20., 20., 20.])
    over_limit_expected = np.array([5., 5., 5., 5., 5., 10., 20.])
    under_limit_expected = np.array([5., 15., 20., 15.,  5., 0., 0.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_2():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 2, 15, 0, 0, 10, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 0., 0., 2.5, 7.5, 12.5, 15.])
    deductible_expected = np.array([5., 5., 5., 5., 5., 5., 5])
    over_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5])
    under_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_3():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 3, 15, 0, 0, 10, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 0., 20., 30., 30., 30., 30.])
    deductible_expected = np.array([5., 15., 5., 5., 5., 5., 5.])
    over_limit_expected = np.array([5., 5., 5., 5., 15., 25., 35.])
    under_limit_expected = np.array([5., 15., 5., 0., 0., 0., 0.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_5():
    # ded + limit > 1
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 5, 0.25, 0, 0, 10, 0.8, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 7.5, 15., 22.5, 30., 37.5, 45.])
    deductible_expected = np.array([5., 7.5, 10., 12.5, 15., 17.5, 20.])
    over_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    under_limit_expected = np.array([0., 0.5, 1., 1.5, 2., 2.5, 3.])

    # ded + limit < 1
    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)

    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 5, 0.25, 0, 0, 10, 0.5, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 5., 10., 15., 20., 25., 30.])
    deductible_expected = np.array([5., 7.5, 10., 12.5, 15., 17.5, 20.])
    over_limit_expected = np.array([5., 7.5, 10., 12.5, 15., 17.5, 20.])
    under_limit_expected = np.array([0., 0., 0., 0., 0., 0., 0.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_7():
    loss_in = np.array([20., 20., 20., 20., 20., 20., 1., 20, 60])
    deductible = np.array([0., 0., 0., 30., 30., 30., 16., 10, 10])
    over_limit = np.array([0., 3., 10., 10., 10., 0., 0., 10, 10])
    under_limit = np.array([0., 10., 10., 0., 5., 15., 0., 10, 10])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 7, 5, 10, 20, 0, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([10., 13., 20., 20., 25., 30., 0., 15., 30])
    deductible_expected = np.array([10., 10., 10., 30., 25., 20., 17., 15., 15])
    over_limit_expected = np.array([0., 0., 5., 20., 15., 0., 0., 10., 35])
    under_limit_expected = np.array([5., 12., 10., 0., 0., 0., 1., 15., 0.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_8():
    loss_in = np.array([20., 20., 20., 20., 20., 20., 1., 20, 60])
    deductible = np.array([0., 0., 0., 30., 30., 30., 16., 10, 10])
    over_limit = np.array([0., 3., 10., 10., 10., 0., 0., 10, 10])
    under_limit = np.array([0., 10., 10., 0., 5., 15., 0., 10, 10])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 8, 5, 10, 20, 0, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([10., 13., 20., 15., 15., 15., 0., 15., 30])
    deductible_expected = np.array([10., 10., 10., 35., 35., 35., 17., 15., 15])
    over_limit_expected = np.array([0., 0., 5., 10., 10., 0., 0., 10., 35])
    under_limit_expected = np.array([5., 12., 10., 5., 10., 15., 1., 15., 0.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_10():
    loss_in = np.array([20., 20., 20., 20., 20., 20., 1., 20, 60])
    deductible = np.array([0., 0., 0., 30., 30., 30., 16., 10, 10])
    over_limit = np.array([0., 3., 10., 10., 10., 0., 0., 10, 10])
    under_limit = np.array([0., 10., 10., 0., 5., 15., 0., 10, 10])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 10, 5, 10, 20, 0, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([15., 15., 15., 20., 25., 30., 0., 15., 55])
    deductible_expected = np.array([5., 5., 5., 30., 25., 20., 17., 15., 15])
    over_limit_expected = np.array([0., 3., 10., 20., 15., 0., 0., 10., 10])
    under_limit_expected = np.array([5., 15., 15., 0., 0., 5., 1., 15., 15.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_11():
    loss_in = np.array([20., 20., 20., 20., 20., 20., 1., 20, 60])
    deductible = np.array([0., 0., 0., 30., 30., 30., 16., 10, 10])
    over_limit = np.array([0., 3., 10., 10., 10., 0., 0., 10, 10])
    under_limit = np.array([0., 10., 10., 0., 5., 15., 0., 10, 10])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 11, 5, 10, 20, 0, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([10., 13., 20., 15., 15., 15., 0., 15., 55])
    deductible_expected = np.array([10., 10., 10., 35., 35., 35., 17., 15., 15])
    over_limit_expected = np.array([0., 0., 5., 10., 10., 0., 0., 10., 10])
    under_limit_expected = np.array([5., 12., 10., 5., 10., 20., 1., 15., 15.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_12():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 12, 15, 0, 0, 10, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 0., 5., 15., 25., 35., 45.])
    deductible_expected = np.array([5., 15., 20., 20., 20., 20., 20.])
    over_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    under_limit_expected = np.array([5., 15., 20., 20., 20., 20., 20.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_13():
    loss_in = np.array([20., 20., 20., 20., 20., 20., 1., 20, 60])
    deductible = np.array([0., 0., 0., 30., 30., 30., 16., 10, 10])
    over_limit = np.array([0., 3., 10., 10., 10., 0., 0., 10, 10])
    under_limit = np.array([0., 10., 10., 0., 5., 15., 0., 10, 10])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 13, 5, 10, 20, 0, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([10., 13., 20., 20., 25., 30., 0., 15., 55])
    deductible_expected = np.array([10., 10., 10., 30., 25., 20., 17., 15., 15])
    over_limit_expected = np.array([0., 0., 5., 20., 15., 0., 0., 10., 10])
    under_limit_expected = np.array([5., 12., 10., 0., 0., 5., 1., 15., 15.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_14():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 14, 15, 0, 0, 10, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 10., 20., 30., 30., 30., 30.])
    deductible_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    over_limit_expected = np.array([5., 5., 5., 5., 15., 25., 35.])
    under_limit_expected = np.array([5., 5., 5., 0., 0., 0., 0.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_15():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 15, 15, 0, 0, 10, 0.6, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 0., 5., 15., 24., 30., 36.])
    deductible_expected = np.array([5., 15., 20., 20., 20., 20., 20.])
    over_limit_expected = np.array([5., 5., 5., 5., 6., 10., 14.])
    under_limit_expected = np.array([5., 15., 17.5, 7.5,  0., 0., 0.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected, decimal=4)


def test_calcrule_16():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 16, 1/4, 0, 0, 10, 0.6, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 7.5, 15., 22.5, 30., 37.5, 45.])
    deductible_expected = np.array([5., 7.5, 10., 12.5, 15., 17.5, 20.])
    over_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    under_limit_expected = np.array([5., 7.5, 10., 12.5, 15., 17.5, 20.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected, decimal=4)


def test_calcrule_17():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 17, 1/4, 0, 0, 10, 25, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 0, 2.5, 6.25, 10., 12.5, 12.5])
    deductible_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    over_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    under_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_19():
    loss_in = np.array([20., 20., 20., 20., 20., 20., 1., 20, 60])
    deductible = np.array([0., 0., 0., 30., 30., 30., 16., 10, 10])
    over_limit = np.array([0., 3., 10., 10., 10., 0., 0., 10, 10])
    under_limit = np.array([0., 10., 10., 0., 5., 15., 0., 10, 10])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 19, 1/4, 10, 20, 0, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([10., 13., 20., 20., 25., 30., 0.75, 15., 50])
    deductible_expected = np.array([10., 10., 10., 30., 25., 20., 16.25, 15., 20])
    over_limit_expected = np.array([0., 0., 0.25, 20., 15., 0., 0., 10., 10])
    under_limit_expected = np.array([9.75, 16.75, 10., 0., 0., 5., 0.25, 15., 20.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_20():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 20, 25, 0, 0, 10, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 10., 20., 0., 0., 0., 0.])

    assert_array_almost_equal(loss_out, loss_expected)


def test_calcrule_22():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 22, 0, 0, 0, 0, 40, 1/2, 3/4, 4/5)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 3., 6., 9., 12., 15., 18.])
    deductible_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    over_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    under_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_23():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 23, 0, 0, 0, 0, 40, 1/2, 3/4, 4/5)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 6., 12., 18., 24., 24., 24.])
    deductible_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    over_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    under_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_24():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 24, 0, 0, 0, 5, 20, 1/2, 3/4, 4/5)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 0., 3., 6., 9., 12., 12.])
    deductible_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    over_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    under_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_25():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 25, 0, 0, 0, 0, 0, 1/2, 3/4, 4/5)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 3., 6., 9., 12., 15., 18.])
    deductible_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    over_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    under_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_26():
    loss_in = np.array([20., 20., 20., 20., 20., 20., 1., 20, 60])
    deductible = np.array([0., 0., 0., 30., 30., 30., 16., 10, 10])
    over_limit = np.array([0., 3., 10., 10., 10., 0., 0., 10, 10])
    under_limit = np.array([0., 10., 10., 0., 5., 15., 0., 10, 10])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 26, 1/4, 10, 20, 0, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([10., 13., 20., 20., 25., 30., 0.75, 15., 30])
    deductible_expected = np.array([10., 10., 10., 30., 25., 20., 16.25, 15., 20])
    over_limit_expected = np.array([0., 0., 5., 20., 15., 0., 0., 10., 30])
    under_limit_expected = np.array([5., 12., 10., 0., 0., 0., 0.25, 15., 0.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_33():
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 33, 1/4, 0, 0, 10, 30, 0, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 7.5, 15., 22.5, 30., 30., 30.])
    deductible_expected = np.array([5., 7.5, 10., 12.5, 15., 17.5, 20.])
    over_limit_expected = np.array([5., 5., 5., 5., 5., 12.5, 20.])
    under_limit_expected = np.array([5., 7.5, 10., 7.5, 0., 0., 0.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected, decimal=4)


def test_calcrule_34():
    """as there is shares, deductible won't be use later on so no need to compute it"""
    loss_in = np.array([0., 10., 20., 30., 40., 50., 60.])
    deductible = np.ones_like(loss_in) * 5
    over_limit = np.ones_like(loss_in) * 5
    under_limit = np.ones_like(loss_in) * 5
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 34, 15, 0, 0, 10, 30, 0.5, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([0., 0., 0., 2.5, 7.5, 12.5, 17.5])
    deductible_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    over_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])
    under_limit_expected = np.array([5., 5., 5., 5., 5., 5., 5.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_35():
    loss_in = np.array([20., 20., 20., 20., 20., 20., 1., 20, 60])
    deductible = np.array([0., 0., 0., 30., 30., 30., 16., 10, 10])
    over_limit = np.array([0., 3., 10., 10., 10., 0., 0., 10, 10])
    under_limit = np.array([0., 10., 10., 0., 5., 15., 0., 10, 10])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 35, 1/4, 10, 20, 0, 4/5, 0, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([10., 13., 16., 16., 16., 16., 0.75, 15., 48])
    deductible_expected = np.array([10., 10., 10., 30., 25., 20., 16.25, 15., 20])
    over_limit_expected = np.array([0., 0., 9., 24., 24., 14., 0., 10., 12])
    under_limit_expected = np.array([5., 3., 0., 0., 0., 0., 0.05, 1., 0.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)


def test_calcrule_36():
    loss_in = np.array([20., 20., 20., 20., 20., 20., 1., 20, 60])
    deductible = np.array([0., 0., 0., 30., 30., 30., 16., 10, 10])
    over_limit = np.array([0., 3., 10., 10., 10., 0., 0., 10, 10])
    under_limit = np.array([0., 10., 10., 0., 5., 15., 0., 10, 10])
    loss_out = np.empty_like(loss_in)
    policy = np.array([(0, 36, 5, 10, 20, 0, 4/5, 0, 0, 0)], dtype=fm_profile_dtype)[0]
    calc(policy, loss_out, loss_in, deductible, over_limit, under_limit, None)

    loss_expected = np.array([10., 13., 16., 16., 16., 16., 0., 15., 48])
    deductible_expected = np.array([10., 10., 10., 30., 25., 20., 17, 15., 15])
    over_limit_expected = np.array([0., 0., 9., 24., 24., 14., 0., 10., 17])
    under_limit_expected = np.array([5., 3., 0., 0., 0., 0., 0.8, 1., 0.])

    assert_array_almost_equal(loss_out, loss_expected)
    assert_array_almost_equal(deductible, deductible_expected)
    assert_array_almost_equal(over_limit, over_limit_expected)
    assert_array_almost_equal(under_limit, under_limit_expected)
