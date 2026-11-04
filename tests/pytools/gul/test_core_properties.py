"""Property-based tests of the ground up loss kernels in gul/core.py."""
import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from oasislmf.pytools.common.data import oasis_float
from oasislmf.pytools.gul.core import compute_mean_loss, get_gul, split_tiv_classic, split_tiv_multiplicative

RTOL = 1e-5

unit_floats = st.floats(0, 1, width=32)
scalings = st.floats(1, 1e7)
tivs = st.floats(1, 1e7, width=32)
property_settings = settings(max_examples=300, deadline=None)


@st.composite
def damage_bins(draw):
    """Draw one damage bin: (bin_from, bin_to, bin_mean, prob_from, prob_to) with the mean inside the bin.

    A bin is either a point or at least 1e-6 wide; narrower bins square to zero in float32.
    """
    bin_from, bin_to = sorted([draw(unit_floats), draw(unit_floats)])
    assume(bin_to == bin_from or bin_to - bin_from >= 1e-6)
    bin_mean = bin_from + draw(unit_floats) * (bin_to - bin_from)
    prob_from, prob_to = sorted([draw(unit_floats), draw(unit_floats)])
    assume(prob_to > prob_from)
    return tuple(np.float32(v) for v in (bin_from, bin_to, bin_mean, prob_from, prob_to))


@st.composite
def damage_distributions(draw):
    """Draw a damage distribution: increasing bin edges in [0, 1], a mean inside each bin and a cdf."""
    n_bins = draw(st.integers(1, 20))
    edges = np.sort(draw(arrays(np.float64, n_bins + 1, elements=st.floats(0, 1))))
    position_in_bin = draw(arrays(np.float64, n_bins, elements=st.floats(0, 1)))
    pdf = draw(arrays(np.float64, n_bins, elements=st.floats(0, 1)))
    assume(pdf.sum() > 0)
    bin_mean = (edges[:-1] + position_in_bin * (edges[1:] - edges[:-1])).astype(oasis_float)
    prob_to = np.minimum(np.cumsum(pdf / pdf.sum()), 1).astype(oasis_float)
    return prob_to, bin_mean, oasis_float.type(edges[-1])


@property_settings
@given(damage_bin=damage_bins(), rval_positions=st.tuples(st.floats(0, 1), st.floats(0, 1)), bin_scaling=scalings)
def test_get_gul_stays_in_the_bin_and_grows_with_the_random_value(damage_bin, rval_positions, bin_scaling):
    bin_from, bin_to, bin_mean, prob_from, prob_to = damage_bin
    low, high = sorted(rval_positions)
    rval_low = np.float64(prob_from) + low * (np.float64(prob_to) - np.float64(prob_from))
    rval_high = np.float64(prob_from) + high * (np.float64(prob_to) - np.float64(prob_from))

    gul_low = get_gul(bin_from, bin_to, bin_mean, prob_from, prob_to, rval_low, bin_scaling)
    gul_high = get_gul(bin_from, bin_to, bin_mean, prob_from, prob_to, rval_high, bin_scaling)

    tolerance = RTOL * bin_scaling
    for gul in (gul_low, gul_high):
        assert not np.isnan(gul)
        assert bin_from * bin_scaling - tolerance <= gul <= bin_to * bin_scaling + tolerance
    assert gul_low <= gul_high + tolerance


@property_settings
@given(distribution=damage_distributions(), bin_scaling=scalings)
def test_compute_mean_loss_is_bounded_by_the_damage_range(distribution, bin_scaling):
    prob_to, bin_mean, max_damage_bin_to = distribution

    gul_mean, std_dev, chance_of_loss, max_loss = compute_mean_loss(
        bin_scaling, prob_to, bin_mean, prob_to.shape[0], max_damage_bin_to)

    tolerance = RTOL * bin_scaling
    assert max_loss == np.float64(max_damage_bin_to) * bin_scaling
    assert -tolerance <= gul_mean <= max_loss + tolerance
    # a distribution on [0, max_loss] has a standard deviation of at most max_loss / 2
    assert 0 <= std_dev <= max_loss / 2 + tolerance
    assert 0 <= chance_of_loss <= 1


@property_settings
@given(gulitems=arrays(oasis_float, st.integers(1, 20), elements=st.floats(0, 1e7, width=32)), tiv=tivs)
def test_split_tiv_classic_caps_the_total_at_the_tiv_and_keeps_proportions(gulitems, tiv):
    original = gulitems.astype(np.float64)
    split_tiv_classic(gulitems, tiv)

    if original.sum() <= tiv:
        np.testing.assert_array_equal(gulitems, original.astype(oasis_float))
    else:
        assert gulitems.sum() <= tiv * (1 + RTOL)
        np.testing.assert_allclose(gulitems, original * tiv / original.sum(), rtol=RTOL, atol=RTOL * tiv)


@property_settings
@given(damage_ratios=arrays(np.float64, st.integers(1, 20), elements=st.floats(0, 1)), tiv=tivs)
def test_split_tiv_multiplicative_total_is_between_the_largest_loss_and_the_sum(damage_ratios, tiv):
    gulitems = (damage_ratios * tiv).astype(oasis_float)
    original = gulitems.astype(np.float64)
    split_tiv_multiplicative(gulitems, tiv)

    total = gulitems.astype(np.float64).sum()
    tolerance = RTOL * tiv
    assert total <= tiv + tolerance
    assert original.max() - tolerance <= total <= original.sum() + tolerance
    if original.sum() > 0:
        np.testing.assert_allclose(gulitems, original * total / original.sum(), rtol=RTOL, atol=tolerance)
