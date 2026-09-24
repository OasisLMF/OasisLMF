"""This file contains the core mathematical functions used in gulpy."""
from math import factorial, sqrt  # sqrt is faster than numpy.sqrt

import numpy as np
from numba import njit

from oasislmf.pytools.gul.random import _interpolate_lookup


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

    disc = max(bb**2. + 4. * aa * rval_bin_offset, 0.)
    sqrt_disc = sqrt(disc)

    if bb > 0.:
        t = 2. * rval_bin_offset / (bb + sqrt_disc)
    else:
        t = (sqrt_disc - bb) / (2. * aa)

    gul = bin_scaling * (bin_from + t)
    return gul


@njit(cache=True, fastmath=True, inline='always')
def apply_alloc_rule(item_losses, alloc_rule, tiv):
    """Apply the back-allocation cap to one cross-item vector, in place.

    ``item_losses`` is the losses of every item on a coverage at ONE (sample, building) -- the
    axis the rules reduce over. The fused path passes a length-1 slice, the whole-coverage path
    passes the full vector; that difference is the only thing separating them, so both call this
    rather than each spelling the rules out.

    Order matters and is the one write_losses established: setmaxloss first, then the tiv split.

    Args:
        item_losses (numpy.array[oasis_float]): one loss per item, edited in place.
        alloc_rule (int): 0 none, 1 classic split, 2 setmaxloss then classic, 3 multiplicative.
        tiv (float): the coverage tiv the split caps against; no split when it is 0.
    """
    if alloc_rule == 2:
        setmaxloss_items(item_losses)
    if tiv > 0:
        if alloc_rule == 1 or alloc_rule == 2:
            split_tiv_classic(item_losses, tiv)
        elif alloc_rule == 3:
            split_tiv_multiplicative(item_losses, tiv)


@njit(cache=True, fastmath=True)
def setmaxloss_items(item_losses):
    """Keep only the largest loss across items, shared evenly where it ties.

    Args:
        item_losses (numpy.array[oasis_float]): one loss per item, edited in place.
    """
    loss_max = 0.
    max_loss_count = 0

    # find maximum losses and count occurrences
    for j in range(item_losses.shape[0]):
        if item_losses[j] > loss_max:
            loss_max = item_losses[j]
            max_loss_count = 1
        elif item_losses[j] == loss_max:
            max_loss_count += 1
    # distribute maximum losses evenly among highest
    # contributing subperils and set other losses to 0
    loss_max_normed = loss_max / max_loss_count
    for j in range(item_losses.shape[0]):
        if item_losses[j] == loss_max:
            item_losses[j] = loss_max_normed
        else:
            item_losses[j] = 0.


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


# Hermite terms kept when converting a copula correlation into a loss correlation. Cost is linear
# in this and the series converges geometrically in rho, so it only matters for a sharply
# non-linear curve at a high correlation. Measured against simulation: a smooth curve is exact at
# 5 terms for any rho; the worst case tried (a two-point curve at rho 0.9) is 6.5% low at 5 terms,
# 1.6% low at 10 and exact at 20. 10 keeps the residual under ~1% of the reported std_dev, and it
# errs LOW -- the truncated terms are all positive.
HERMITE_TERMS = 10
# sqrt(k!) for k = 1..HERMITE_TERMS, the normalisation of the probabilists' Hermite basis
SQRT_FACTORIAL = np.array([sqrt(float(factorial(k))) for k in range(1, HERMITE_TERMS + 1)])


@njit(cache=True, fastmath=True)
def accumulate_hermite_coeffs(bin_scaling, prob_to, bin_mean, bin_count, weight,
                              x_min, inv_factor, norm_inv_cdf, coeffs):
    """Add ``weight`` times one damage CDF's Hermite coefficients into ``coeffs``.

    ``coeffs[k - 1]`` accumulates ``a_k = E[L He_k(X)] / sqrt(k!)`` for the loss ``L = g(Phi(X))``
    that this CDF defines, with ``g`` piecewise constant at ``bin_mean * bin_scaling`` -- the same
    approximation :func:`compute_mean_loss` makes, so the two agree. Using
    ``integral He_k phi = -He_{k-1} phi``, a bin contributes
    ``loss * [He_{k-1}(x_lo) phi(x_lo) - He_{k-1}(x_hi) phi(x_hi)] / sqrt(k!)``.

    Callers add one CDF per hazard bin weighted by that bin's probability, which gives the
    coefficients of ``E[L | damage uniform]`` -- the only part of the loss the damage copula can
    correlate. A caller with a single effective CDF passes weight 1.

    Args:
        bin_scaling (float): scaling on damage bin values, as passed to compute_mean_loss.
        prob_to (numpy.array[oasis_float]): bin maximum probability.
        bin_mean (numpy.array[oasis_float]): bin mean damage.
        bin_count (int): number of bins.
        weight (float): probability of the hazard bin this CDF belongs to; 1 for a single CDF.
        x_min (float): lower probability bound of the inverse-normal lookup table.
        inv_factor (float): scaling that indexes that table.
        norm_inv_cdf (numpy.array[float64]): the inverse-normal table. Deliberately the same one
            the copula itself inverts through, so the correction cannot disagree with the draws.
        coeffs (numpy.array[float64]): length HERMITE_TERMS, added to in place.
    """
    # He_{k-1} at the lower edge of the current bin, k = 1..HERMITE_TERMS, starting at -infinity
    # where the density is zero, so only the upper edge of the first bin contributes.
    he_lo = np.zeros(HERMITE_TERMS)
    he_hi = np.zeros(HERMITE_TERMS)
    phi_lo = 0.
    table_len = len(norm_inv_cdf)
    for i in range(bin_count):
        x_hi = _interpolate_lookup(prob_to[i], x_min, inv_factor, norm_inv_cdf, table_len)
        phi_hi = 0.3989422804014327 * np.exp(-0.5 * x_hi * x_hi)
        he_hi[0] = 1.
        for k in range(1, HERMITE_TERMS):
            he_hi[k] = x_hi * he_hi[k - 1] - (k - 1) * (he_hi[k - 2] if k >= 2 else 0.)
        loss = bin_mean[i] * bin_scaling * weight
        for k in range(HERMITE_TERMS):
            coeffs[k] += loss * (he_lo[k] * phi_lo - he_hi[k] * phi_hi) / SQRT_FACTORIAL[k]
        for k in range(HERMITE_TERMS):
            he_lo[k] = he_hi[k]
        phi_lo = phi_hi


@njit(cache=True, fastmath=True, inline='always')
def loss_correlation(coeffs, variance, copula_rho):
    """Turn a copula correlation into the correlation it induces between two buildings' LOSSES.

    Both buildings' uniforms come from standard normals with correlation ``copula_rho``. For
    ``L = g(Phi(X))`` expanded in Hermite polynomials, Mehler's formula gives exactly
    ``Cov(L_i, L_j) = sum_k a_k^2 rho^k`` while ``Var(L)`` is the marginal variance, so the loss
    correlation is that sum over the variance. Two consequences: it is never above ``copula_rho``,
    which is why using the copula value directly overstates the variance of a sum by up to ~39%;
    and where the loss also depends on something drawn independently per building (the hazard
    intensity in a full Monte Carlo run) that noise raises ``variance`` without contributing to
    the sum, which is exactly how it should dilute the correlation.

    Args:
        coeffs (numpy.array[float64]): ``a_k`` for k = 1..HERMITE_TERMS.
        variance (float): the marginal variance of the loss, i.e. compute_mean_loss's std_dev
            squared. Carries the independent per-building noise as well.
        copula_rho (float): the correlation applied to the damage draws.

    Returns:
        float: the loss correlation, in [0, 1].
    """
    if variance <= 0. or copula_rho <= 0.:
        return 0.
    total = 0.
    rho_k = 1.
    for k in range(HERMITE_TERMS):
        rho_k *= copula_rho
        total += coeffs[k] * coeffs[k] * rho_k
    r = total / variance
    # the truncated series cannot exceed the marginal, but round-off can
    return min(max(r, 0.), 1.)
