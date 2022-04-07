"""
This file contains the core mathematical functions used in gulpy.

"""
import numpy as np
from numba import njit
from math import sqrt  # faster than numpy.sqrt

from oasislmf.pytools.gul.common import STD_DEV_IDX, NUM_IDX


@njit(cache=True, fastmath=False)
def get_gul(bin_from, bin_to, bin_mean, prob_from, prob_to, rval, tiv):
    """Compute the ground-up loss.  using linear interpolation or quadratic interpolaiton.

    Args:
        bin_from (_type_): _description_
        bin_to (_type_): _description_
        bin_mean (_type_): _description_
        prob_from (_type_): _description_
        prob_to (_type_): _description_
        rval (_type_): _description_
        tiv (_type_): _description_

    Returns:
        float64 : the computed ground-up loss
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
def setmaxloss(loss):
    """Set maximum loss.
    For each sample, find the maximum loss across all items.

    """
    Nsamples, Nitems = loss.shape

    # the main loop starts from STD_DEV
    for i in range(NUM_IDX + STD_DEV_IDX, Nsamples, 1):
        loss_max = 0.
        max_loss_count = 0

        # find maximum loss and count occurrences
        for j in range(Nitems):
            if loss[i, j] > loss_max:
                loss_max = loss[i, j]
                max_loss_count = 1
            elif loss[i, j] == loss_max:
                max_loss_count += 1

        # distribute maximum losses evenly among highest
        # contributing subperils and set other losses to 0
        loss_max_normed = loss_max / max_loss_count
        for j in range(Nitems):
            if loss[i, j] == loss_max:
                loss[i, j] = loss_max_normed
            else:
                loss[i, j] = 0.

    return loss


@njit(cache=True, fastmath=True)
def split_tiv(gulitems, tiv):
    # if the total loss exceeds the tiv
    # then split tiv in the same proportions to the losses
    if tiv > 0:
        total_loss = np.sum(gulitems)

        nitems = gulitems.shape[0]
        if total_loss > tiv:
            for j in range(nitems):
                # editing in-place the np array
                gulitems[j] *= tiv / total_loss


@njit(cache=True, fastmath=True)
def compute_mean_loss(tiv, prob_to, bin_mean, bin_count, max_damage_bin_to):
    """Compute the mean ground-up loss and some properties.

    Args:
        tiv (_type_): _description_
        prob_to (_type_): _description_
        bin_mean (_type_): _description_
        bin_count (_type_): _description_
        max_damage_bin_to (_type_): _description_

    Returns:
        _type_: _description_
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
