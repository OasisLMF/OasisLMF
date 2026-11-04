"""Property-based tests of the FM calcrules in policy.py and policy_extras.py.

The losses are float32 like the fmpy loss arrays, so the comparisons use float32-scale tolerances.
The step calcrules (28, 281, 32, 37, 38) are not covered here.
"""
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from oasislmf.pytools.common.data import fm_profile_dtype, oasis_float
from oasislmf.pytools.fm import policy, policy_extras

SHARED_CALCRULES = [1, 2, 3, 5, 12, 14, 15, 16, 17, 20, 22, 23, 24, 25, 33, 34, 39, 40, 41,
                    100, 101, 103, 105, 112, 114, 115, 116, 120, 133, 200]
MIN_MAX_DEDUCTIBLE_CALCRULES = [7, 8, 10, 11, 13, 19, 26, 35, 36, 107, 108, 110, 111, 113, 119]
REVERSE_FRANCHISE_CALCRULES = {20, 120}

# calcrules whose policy_extras version accounts for every unit of loss as either loss,
# deductible or over limit; the share and reinsurance rules do not track deductibles
CONSERVING_CALCRULES = [1, 3, 5, 12, 14, 15, 16, 33, 39, 100, 7, 8, 10, 11, 13, 19, 26, 35, 36]
LOWER_LEVEL_CALCRULES = [1, 3, 5, 12, 14, 15, 16, 33, 39, 100]
MIN_DEDUCTIBLE_CALCRULES = [7, 8, 11, 13, 19, 36]
MONETARY_LIMIT_CALCRULES = [1, 3, 14, 33, 40, 41]

FRACTION_DEDUCTIBLE = {5, 16, 17, 19, 33, 105, 116, 119, 133}
FRACTION_LIMIT = {5, 15, 105, 115}

MAX_AMOUNT = 1e5
RTOL = 1e-5
ATOL = 0.05

amounts = st.floats(0, MAX_AMOUNT, width=32)
fractions = st.floats(0, 1, width=32)
losses = arrays(oasis_float, st.integers(1, 20), elements=st.floats(0, 3 * MAX_AMOUNT, width=32))
property_settings = settings(max_examples=200, deadline=None)


@st.composite
def profiles(draw, calcrule_ids, full_limit=False):
    """Draw an fm profile with valid terms for one of `calcrule_ids`.

    Percentage terms are fractions, the min deductible (deductible2) is at most the max
    deductible (deductible3), and a %-of-loss limit is below 100% unless `full_limit` is set.
    """
    calcrule_id = draw(st.sampled_from(calcrule_ids))
    profile = np.zeros(1, dtype=fm_profile_dtype)[0]
    profile['calcrule_id'] = calcrule_id
    profile['deductible1'] = draw(fractions if calcrule_id in FRACTION_DEDUCTIBLE else amounts)
    min_deductible, max_deductible = sorted([draw(amounts), draw(amounts)])
    profile['deductible2'] = min_deductible
    profile['deductible3'] = max_deductible
    profile['attachment1'] = draw(amounts)
    if full_limit:
        profile['limit1'] = 1.
    elif calcrule_id in FRACTION_LIMIT:
        profile['limit1'] = draw(st.floats(0, 1, width=32, exclude_max=True))
    else:
        profile['limit1'] = draw(amounts)
    profile['share1'] = draw(fractions)
    profile['share2'] = draw(fractions)
    profile['share3'] = draw(fractions)
    return profile


def run_extras(profile, loss_in, deductible=None, over_limit=None, under_limit=None):
    """Run policy_extras.calc, starting from an empty deductible / over / under limit state by default."""
    deductible = np.zeros_like(loss_in) if deductible is None else deductible
    over_limit = np.zeros_like(loss_in) if over_limit is None else over_limit
    under_limit = np.zeros_like(loss_in) if under_limit is None else under_limit
    loss_out = np.empty_like(loss_in)
    policy_extras.calc(profile, loss_out, loss_in, deductible, over_limit, under_limit, None)
    return loss_out, deductible, over_limit, under_limit


@property_settings
@given(profile=profiles(SHARED_CALCRULES), loss_in=losses)
def test_policy_and_policy_extras_agree_on_the_loss(profile, loss_in):
    """policy_extras re-implements the policy calcrules; both must produce the same loss."""
    loss_out = np.empty_like(loss_in)
    policy.calc(profile, loss_out, loss_in, None)
    extras_loss_out, _, _, _ = run_extras(profile, loss_in)

    np.testing.assert_allclose(extras_loss_out, loss_out, rtol=RTOL, atol=ATOL)


@property_settings
@given(profile=profiles(SHARED_CALCRULES + MIN_MAX_DEDUCTIBLE_CALCRULES), loss_in=losses)
def test_loss_is_between_zero_and_the_incoming_loss(profile, loss_in):
    loss_out, _, _, _ = run_extras(profile, loss_in)

    assert np.all(loss_out >= -ATOL)
    assert np.all(loss_out <= loss_in * (1 + RTOL) + ATOL)


@property_settings
@given(profile=profiles([c for c in SHARED_CALCRULES + MIN_MAX_DEDUCTIBLE_CALCRULES if c not in REVERSE_FRANCHISE_CALCRULES]),
       loss_in=losses)
def test_loss_does_not_decrease_as_the_incoming_loss_grows(profile, loss_in):
    loss_in = np.sort(loss_in)
    loss_out, _, _, _ = run_extras(profile, loss_in)

    assert np.all(np.diff(loss_out) >= -(RTOL * np.abs(loss_out).max() + ATOL))


@property_settings
@given(profile=profiles(MONETARY_LIMIT_CALCRULES), loss_in=losses)
def test_loss_does_not_exceed_the_limit(profile, loss_in):
    loss_out, _, _, _ = run_extras(profile, loss_in)

    assert np.all(loss_out <= profile['limit1'] * (1 + RTOL) + ATOL)


@property_settings
@given(profile=profiles(CONSERVING_CALCRULES), loss_in=losses)
def test_loss_deductible_and_over_limit_add_up_to_the_incoming_loss(profile, loss_in):
    loss_out, deductible, over_limit, _ = run_extras(profile, loss_in)

    np.testing.assert_allclose(loss_out + deductible + over_limit, loss_in, rtol=RTOL, atol=ATOL)
    assert np.all(deductible >= -ATOL)
    assert np.all(over_limit >= -ATOL)


@property_settings
@given(lower_profile=profiles(LOWER_LEVEL_CALCRULES), profile=profiles(MIN_DEDUCTIBLE_CALCRULES), gul=losses)
def test_min_deductible_conserves_the_loss_of_the_level_below(lower_profile, profile, gul):
    """A min deductible applied on top of a lower level must keep every unit of ground up loss
    accounted for as loss, deductible or over limit, and none of them may go negative.
    See https://github.com/OasisLMF/OasisLMF/issues/2246.
    """
    lower_loss, deductible, over_limit, under_limit = run_extras(lower_profile, gul)
    loss_out, deductible, over_limit, _ = run_extras(profile, lower_loss, deductible, over_limit, under_limit)

    np.testing.assert_allclose(loss_out + deductible + over_limit, gul, rtol=RTOL, atol=ATOL)
    assert np.all(loss_out >= -ATOL)
    assert np.all(deductible >= -ATOL)
    assert np.all(over_limit >= -ATOL)


@property_settings
@given(profile=profiles([15, 115], full_limit=True), loss_in=losses)
def test_a_full_percentage_of_loss_limit_is_accepted(profile, loss_in):
    """A %-of-loss limit of 100% caps nothing, so calcrule 15 must behave like a plain deductible.
    See https://github.com/OasisLMF/OasisLMF/issues/2247.
    """
    loss_out = np.empty_like(loss_in)
    policy.calc(profile, loss_out, loss_in, None)
    extras_loss_out, _, _, _ = run_extras(profile, loss_in)

    expected = np.maximum(loss_in - profile['deductible1'], 0)
    if profile['calcrule_id'] == 115:
        expected = expected * profile['share1']
    np.testing.assert_allclose(loss_out, expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(extras_loss_out, expected, rtol=RTOL, atol=ATOL)
