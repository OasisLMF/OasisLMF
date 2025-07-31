"""
This file contains the core mathematical functions used in gulpy.

"""
from math import sqrt  # faster than numpy.sqrt

import numpy as np
from numba import njit

from oasislmf.pytools.gul.common import NUM_IDX, MAX_LOSS_IDX, MEAN_IDX, TIV_IDX


@njit(cache=True, fastmath=False, error_model="numpy")
def get_gul(bin_from, bin_to, bin_mean, prob_from, prob_to, rval, bin_scaling):
    """Compute the ground-up loss using linear or quadratic interpolaiton if necessary.

    Args:
        bin_from (oasis_float): bin minimum damage.
        bin_to (oasis_float): bin maximum damage.
        bin_mean (oasis_float): bin mean damage (`interpolation` column in damagebins file).
        prob_from (oasis_float): bin minimum probability
        prob_to (oasis_float): bin maximum probability
        rval (float64): the random cdf value.
        bin_scaling (oasis_float): scaling on the bins.

    Returns:
        float64: the computed ground-up loss
    """
    bin_width = bin_to - bin_from

    # point-like bin
    if bin_width == 0.:
        gul = bin_scaling * bin_to

        return gul

    bin_height = prob_to - prob_from
    rval_bin_offset = rval - prob_from

    # linear interpolation
    x = np.float64((bin_mean - bin_from) / bin_width)
    if np.abs(x - 0.5) <= 5e-6:
        # this condition requires 1 less operation
        gul = bin_scaling * (bin_from + rval_bin_offset * bin_width / bin_height)

        return gul

    # quadratic interpolation
    aa = 3. * bin_height / bin_width**2 * (2. * x - 1.)
    bb = 2. * bin_height / bin_width * (2. - 3. * x)
    cc = - rval_bin_offset

    gul = bin_scaling * (bin_from + (sqrt(bb**2. - 4. * aa * cc) - bb) / (2. * aa))

    return gul


@njit(cache=True, fastmath=True)
def setmaxloss_i(losses, sidx):
    loss_max = 0.
    max_loss_count = 0

    # find maximum losses and count occurrences
    for j in range(losses.shape[1]):
        if losses[sidx, j] > loss_max:
            loss_max = losses[sidx, j]
            max_loss_count = 1
        elif losses[sidx, j] == loss_max:
            max_loss_count += 1
    # distribute maximum losses evenly among highest
    # contributing subperils and set other losses to 0
    loss_max_normed = loss_max / max_loss_count
    for j in range(losses.shape[1]):
        if losses[sidx, j] == loss_max:
            losses[sidx, j] = loss_max_normed
        else:
            losses[sidx, j] = 0.


@njit(cache=True, fastmath=True)
def setmaxloss(losses):
    """Set maximum losses.
    For each sample idx, find the maximum loss across all items and set to zero
    all the losses smaller than the maximum loss. If the maximum loss occurs in `N` items,
    then set the loss in all these items as the maximum loss divided by `N`.

    Args:
        losses (numpy.array[oasis_float]): losses for all item_ids and sample idx.

    Returns:
        numpy.array[oasis_float]: losses for all item_ids and sample idx.
    """
    # losses array layout is [NA, normal sidx (1 to n), special sidx (NUM_IDX)]
    setmaxloss_i(losses, TIV_IDX)
    setmaxloss_i(losses, MAX_LOSS_IDX)
    setmaxloss_i(losses, MEAN_IDX)
    for sidx in range(1, losses.shape[0] - NUM_IDX):
        setmaxloss_i(losses, sidx)

    return losses


@njit(cache=True, fastmath=True)
def split_tiv_classic(gulitems, tiv):
    """Split the total insured value (TIV). If the total loss of all the items
    in `gulitems` exceeds the total insured value, re-scale the losses in the
    same proportion to the losses.

    Args:
        gulitems (numpy.array[oasis_float]): array containing losses of all items.
        tiv (oasis_float): total insured value.
    """
    total_loss = np.sum(gulitems)

    if total_loss > tiv:
        f = tiv / total_loss

        for j in range(gulitems.shape[0]):
            # editing in-place the np array
            gulitems[j] *= f


@njit(cache=True, fastmath=True)
def split_tiv_multiplicative(gulitems, tiv):
    """Split the total insured value (TIV) using a multiplicative formula for the
    total loss as tiv * (1 - (1-A)*(1-B)*(1-C)...), where A, B, C are damage ratios
    computed as the ratio between a sub-peril loss and the tiv. Sub-peril losses
    in gulitems are always back-allocated proportionally to the losses.

    Args:
        gulitems (numpy.array[oasis_float]): array containing losses of all items.
        tiv (oasis_float): total insured value.
    """
    Ngulitems = gulitems.shape[0]
    undamaged_value = 1.
    sum_loss = 0.
    for i in range(Ngulitems):
        undamaged_value *= 1. - gulitems[i] / tiv
        sum_loss += gulitems[i]

    multiplicative_loss = tiv * (1. - undamaged_value)

    if sum_loss > 0.:
        # back-allocate proportionally in any case (i.e., not only if total_loss > tiv)
        f = multiplicative_loss / sum_loss

        for j in range(Ngulitems):
            # editing in-place the np array
            gulitems[j] *= f


@njit(cache=True, fastmath=True)
def compute_mean_loss(bin_scaling, prob_to, bin_mean, bin_count, max_damage_bin_to):
    """Compute the mean ground-up loss and some properties.

    Args:
        bin_scaling (oasis_float): scaling on damage bin values.
        prob_to (numpy.array[oasis_float]): bin maximum probability
        bin_mean (numpy.array[oasis_float]): bin mean damage (`interpolation` column in damagebins file).
        bin_count (int): number of bins.
        max_damage_bin_to (oasis_float): maximum damage value (i.e., `bin_to` of the last damage bin).

    Returns:
        float64, float64, float64, float64: mean ground-up loss, standard deviation of the ground-up loss,
          chance of loss, maximum loss
    """
    # chance_of_loss = 1. - prob_to[0] if bin_mean[0] == 0. else 1.
    chance_of_loss = 1 - prob_to[0] * (1 - (bin_mean[0] > 0))

    gul_mean = 0.
    ctr_var = 0.
    last_prob_to = 0.
    for i in range(bin_count):
        prob_from = last_prob_to
        new_gul = (prob_to[i] - prob_from) * bin_mean[i]
        gul_mean += new_gul
        ctr_var += new_gul * bin_mean[i]
        last_prob_to = prob_to[i]

    gul_mean *= bin_scaling
    ctr_var *= bin_scaling**2.
    # Var(aX) = a**2 E(X^2) - E(aX)**2
    std_dev = sqrt(max(ctr_var - gul_mean**2., 0.))
    max_loss = max_damage_bin_to * bin_scaling

    return gul_mean, std_dev, chance_of_loss, max_loss
