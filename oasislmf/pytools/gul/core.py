"""
This file contains the core mathematical functions used in gulpy.

"""
import numpy as np
from numba import njit
from math import sqrt  # faster than numpy.sqrt

from oasislmf.pytools.gul.common import STD_DEV_IDX, NUM_IDX


@njit(cache=True, fastmath=False, error_model="numpy")
def get_gul(bin_from, bin_to, bin_mean, prob_from, prob_to, rval, tiv):
    """Compute the ground-up loss using linear or quadratic interpolaiton if necessary.

    Args:
        bin_from (oasis_float): bin minimum damage.
        bin_to (oasis_float): bin maximum damage.
        bin_mean (oasis_float): bin mean damage (`interpolation` column in damagebins file).
        prob_from (oasis_float): bin minimum probability
        prob_to (oasis_float): bin maximum probability
        rval (float64): the random cdf value.
        tiv (oasis_float): total insured value.

    Returns:
        float64: the computed ground-up loss
    """
    bin_width = bin_to - bin_from

    # point-like bin
    if bin_width == 0.:
        gul = tiv * bin_to

        return gul

    bin_height = prob_to - prob_from
    rval_bin_offset = rval - prob_from

    # linear interpolation
    x = np.float64((bin_mean - bin_from) / bin_width)
    # if np.int(np.round(x * 100000)) == 50000:
    if np.abs(x - 0.5) <= 5e-6:
        # this condition requires 1 less operation
        gul = tiv * (bin_from + rval_bin_offset * bin_width / bin_height)

        return gul

    # quadratic interpolation
    # MT: I haven't re-derived the algorithm for this case; not sure where the parabola vertex is set
    aa = 3. * bin_height / bin_width**2 * (2. * x - 1.)
    bb = 2. * bin_height / bin_width * (2. - 3. * x)
    cc = - rval_bin_offset

    gul = tiv * (bin_from + (sqrt(bb**2. - 4. * aa * cc) - bb) / (2. * aa))

    return gul


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
    Nsamples, Nitems = losses.shape

    # the main loop starts from STD_DEV
    for i in range(NUM_IDX + STD_DEV_IDX, Nsamples, 1):
        loss_max = 0.
        max_loss_count = 0

        # find maximum losses and count occurrences
        for j in range(Nitems):
            if losses[i, j] > loss_max:
                loss_max = losses[i, j]
                max_loss_count = 1
            elif losses[i, j] == loss_max:
                max_loss_count += 1

        # distribute maximum losses evenly among highest
        # contributing subperils and set other losses to 0
        loss_max_normed = loss_max / max_loss_count
        for j in range(Nitems):
            if losses[i, j] == loss_max:
                losses[i, j] = loss_max_normed
            else:
                losses[i, j] = 0.

    return losses


@njit(cache=True, fastmath=True)
def split_tiv(gulitems, tiv):
    """Split the total insured value (TIV). If the total loss of all the items
    in `gulitems` exceeds the total insured value, re-scale the losses in the
    same proportion to the losses.

    Args:
        gulitems (numpy.array[oasis_float]): array containing losses of all items.
        tiv (oasis_float): total insured value,
    """
    total_loss = np.sum(gulitems)

    if total_loss > tiv:
        f = tiv / total_loss

        for j in range(gulitems.shape[0]):
            # editing in-place the np array
            gulitems[j] *= f


@njit(cache=True, fastmath=True)
def compute_mean_loss(tiv, prob_to, bin_mean, bin_count, max_damage_bin_to):
    """Compute the mean ground-up loss and some properties.

    Args:
        tiv (oasis_float): total insured value.
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

    gul_mean *= tiv
    ctr_var *= tiv**2.
    std_dev = sqrt(max(ctr_var - gul_mean**2., 0.))
    max_loss = max_damage_bin_to * tiv

    return gul_mean, std_dev, chance_of_loss, max_loss
