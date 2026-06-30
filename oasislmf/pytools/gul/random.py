"""
This file contains the utilities for generating random numbers in gulpy.

"""

import logging
from math import sqrt

import numpy as np
from numba import njit
from scipy.stats import norm

logger = logging.getLogger(__name__)


GROUP_ID_HASH_CODE = np.int64(1543270363)
EVENT_ID_HASH_CODE = np.int64(1943272559)
HASH_MOD_CODE = np.int64(2147483648)

HAZARD_GROUP_ID_HASH_CODE = np.int64(1143271949)
HAZARD_EVENT_ID_HASH_CODE = np.int64(1243274353)
HAZARD_HASH_MOD_CODE = np.int64(1957483729)

# --- Philox4x32 counter-based RNG (random_generator 2) ----------------------
# Constants from the Random123 library (philox4x32). A counter-based generator is a
# keyed pseudo-random function value_j = F(key, j): keying it with a group's per-event
# seed yields a reproducible, order-independent stream WITHOUT the per-row
# np.random.seed() reseed that dominates the MersenneTwister/Latin Hypercube cost.
PHILOX_M0 = np.uint64(0xD2511F53)
PHILOX_M1 = np.uint64(0xCD9E8D57)
PHILOX_W0 = np.uint32(0x9E3779B9)
PHILOX_W1 = np.uint32(0xBB67AE85)
PHILOX_U32_MASK = np.uint64(0xFFFFFFFF)
PHILOX_SHIFT32 = np.uint64(32)
PHILOX_INV32 = np.float64(1.0) / np.float64(2.0 ** 32)
PHILOX_STREAM_JITTER = np.uint32(0)
PHILOX_STREAM_SHUFFLE = np.uint32(1)

# parameters for get_corr_rval in a normal cdf
x_min = 1e-16
x_max = 1 - 1e-16
norm_inv_N = 1000000
cdf_min = -20
cdf_max = 20.
inv_factor = (norm_inv_N - 1) / (x_max - x_min)
norm_factor = (norm_inv_N - 1) / (cdf_max - cdf_min)


@njit(cache=True, fastmath=True)
def generate_hash(group_id, event_id, base_seed=0):
    """Generate hash for a given `group_id`, `event_id` pair for the vulnerability pdf.

    Args:
        group_id (int): group id.
        event_id (int]): event id.
        base_seed (int, optional): base random seed. Defaults to 0.

    Returns:
        int64: hash
    """
    hash = (base_seed + (group_id * GROUP_ID_HASH_CODE) % HASH_MOD_CODE +
            (event_id * EVENT_ID_HASH_CODE) % HASH_MOD_CODE) % HASH_MOD_CODE

    return hash


@njit(cache=True, fastmath=True)
def generate_hash_hazard(hazard_group_id, event_id, base_seed=0):
    """Generate hash for a given `hazard_group_id`, `event_id` pair for the hazard pdf.

    Args:
        hazard_group_id (int): group id.
        event_id (int]): event id.
        base_seed (int, optional): base random seed. Defaults to 0.

    Returns:
        int64: hash
    """
    hash = (base_seed + (hazard_group_id * HAZARD_GROUP_ID_HASH_CODE) % HAZARD_HASH_MOD_CODE +
            (event_id * HAZARD_EVENT_ID_HASH_CODE) % HAZARD_HASH_MOD_CODE) % HAZARD_HASH_MOD_CODE

    return hash


def get_random_generator(random_generator):
    """Get the random generator function.

    Args:
        random_generator (int): random generator function id.

    Returns:
        The random generator function.

    """
    # define random generator function
    if random_generator == 0:
        logger.info("Random generator: MersenneTwister")
        return random_MersenneTwister

    elif random_generator == 1:
        logger.info("Random generator: Latin Hypercube")
        return random_LatinHypercube

    elif random_generator == 2:
        logger.info("Random generator: Latin Hypercube on Philox4x32-7 (counter-based)")
        return random_LatinHypercube_Philox7

    else:
        raise ValueError(f"No random generator exists for random_generator={random_generator}.")


EVENT_ID_HASH_CODE = np.int64(1943_272_559)
PERIL_CORRELATION_GROUP_HASH = np.int64(1836311903)
HASH_MOD_CODE = np.int64(2147483648)


@njit(cache=True, fastmath=True)
def generate_correlated_hash_vector(unique_peril_correlation_groups, event_id, correlated_hashes, base_seed=0):
    """Generate hashes for all peril correlation groups for a given `event_id`.

    Args:
        unique_peril_correlation_groups (List[int]): list of the unique peril correlation groups.
        event_id (int): event id.
        base_seed (int, optional): base random seed. Defaults to 0.
        correlated_hashes: empty buffer for the output (size of max group id not the number of group id
    """
    unique_peril_index = 0
    unique_peril_len = unique_peril_correlation_groups.shape[0]
    for i in range(1, correlated_hashes.shape[0]):
        if unique_peril_correlation_groups[unique_peril_index] == i:
            correlated_hashes[i] = (
                base_seed +
                (unique_peril_correlation_groups[unique_peril_index] * PERIL_CORRELATION_GROUP_HASH) % HASH_MOD_CODE +
                (event_id * EVENT_ID_HASH_CODE) % HASH_MOD_CODE
            ) % HASH_MOD_CODE
            unique_peril_index += 1
            if unique_peril_index == unique_peril_len:
                break


def compute_norm_inv_cdf_lookup(cdf_min, cdf_max, N):
    return norm.ppf(np.linspace(cdf_min, cdf_max, N))


def compute_norm_cdf_lookup(x_min, x_max, N):
    return norm.cdf(np.linspace(x_min, x_max, N))


@njit(cache=True, fastmath=True)
def get_norm_cdf_cell_nb(x, x_min, x_max, N):
    return int((x - x_min) * (N - 1) // (x_max - x_min))


@njit(cache=True, fastmath=True)
def _interpolate_lookup(value, range_start, factor, table, N):
    """Linear interpolation lookup into a precomputed table.

    Instead of removing the fractional position to pick the lower table entry,
    this blends the two surrounding entries using the fractional distance,
    reducing lookup error in the distribution tails
    """
    pos = (value - range_start) * factor
    index_lower = int(pos)
    if index_lower < 0:
        index_lower = 0
    elif index_lower > N - 2:
        index_lower = N - 2
    frac = pos - index_lower
    return table[index_lower] + frac * (table[index_lower + 1] - table[index_lower])


@njit(cache=True, fastmath=True)
def get_corr_rval(x_unif, y_unif, rho, x_min, x_max, N, norm_inv_cdf, cdf_min,
                  cdf_max, norm_cdf, Nsamples, z_unif):
    sqrt_rho = sqrt(rho)
    sqrt_1_minus_rho = sqrt(1. - rho)
    inv_factor = (N - 1) / (x_max - x_min)
    norm_factor = (N - 1) / (cdf_max - cdf_min)

    for i in range(Nsamples):
        x_norm = _interpolate_lookup(x_unif[i], x_min, inv_factor, norm_inv_cdf, N)
        y_norm = _interpolate_lookup(y_unif[i], x_min, inv_factor, norm_inv_cdf, N)
        z_norm = sqrt_rho * x_norm + sqrt_1_minus_rho * y_norm

        z_unif[i] = _interpolate_lookup(z_norm, cdf_min, norm_factor, norm_cdf, N)


@njit(cache=True, fastmath=True)
def get_corr_rval_float(x_unif, y_unif, rho, x_min, norm_inv_cdf, inv_factor, cdf_min,
                        norm_cdf, norm_factor, Nsamples, z_unif):
    """
    this calculate the new correlated values like in get_corr_rval but with precomputed inv_factor and norm_factor
    inv_factor = (N - 1) / (x_max - x_min)
    norm_factor = (N - 1) / (cdf_max - cdf_min)
    """
    sqrt_rho = sqrt(rho)
    sqrt_1_minus_rho = sqrt(1. - rho)
    N = len(norm_inv_cdf)

    for i in range(Nsamples):
        x_norm = _interpolate_lookup(x_unif[i], x_min, inv_factor, norm_inv_cdf, N)
        y_norm = _interpolate_lookup(y_unif[i], x_min, inv_factor, norm_inv_cdf, N)
        z_norm = sqrt_rho * x_norm + sqrt_1_minus_rho * y_norm

        z_unif[i] = _interpolate_lookup(z_norm, cdf_min, norm_factor, norm_cdf, N)


@njit(cache=True, fastmath=True)
def random_MersenneTwister(seeds, n, skip_seeds=0):
    """Generate random numbers using the default Mersenne Twister algorithm.

    Args:
        seeds (List[int64]): List of seeds.
        n (int): number of random samples to generate for each seed.
        skip_seeds (int): number of seeds to skip starting from the beginning
          of the `seeds` array. For skipped seeds no random numbers are generated
          and the output rndms will contain zeros at their corresponding row.
          Default is 0, i.e. no seeds are skipped.

    Returns:
        rndms (array[float]): 2-d array of shape (number of seeds, n) 
          containing the random values generated for each seed.
        rndms_idx (Dict[int64, int]): mapping between `seed` and the 
          row in rndms that stores the corresponding random values.
    """
    Nseeds = len(seeds)
    rndms = np.zeros((Nseeds, n), dtype='float64')

    for seed_i in range(skip_seeds, Nseeds, 1):
        # set the seed
        np.random.seed(seeds[seed_i])

        # draw all random numbers at once (vectorized)
        rndms[seed_i, :] = np.random.random(n)

    return rndms


@njit(cache=True, fastmath=True)
def random_LatinHypercube(seeds, n, skip_seeds=0):
    """Generate random numbers using the Latin Hypercube algorithm.

    Args:
        seeds (List[int64]): List of seeds.
        n (int): number of random samples to generate for each seed.

    Returns:
        rndms (array[float]): 2-d array of shape (number of seeds, n) 
          containing the random values generated for each seed.
        rndms_idx (Dict[int64, int]): mapping between `seed` and the 
          row in rndms that stores the corresponding random values.
        skip_seeds (int): number of seeds to skip starting from the beginning
          of the `seeds` array. For skipped seeds no random numbers are generated
          and the output rndms will contain zeros at their corresponding row.
          Default is 0, i.e. no seeds are skipped.

    Notes:
        Implementation follows scipy.stats.qmc.LatinHypercube v1.8.0.
        Following scipy notation, here we assume `centered=False` all the times:
        instead of taking `samples=0.5*np.ones(n)`, here we always
        draw uniform random samples in order to initialise `samples`.
    """
    Nseeds = len(seeds)
    rndms = np.zeros((Nseeds, n), dtype='float64')
    # define arrays here and re-use them later
    samples = np.zeros(n, dtype='float64')
    perms = np.zeros(n, dtype='float64')

    for seed_i in range(skip_seeds, Nseeds, 1):
        # set the seed
        np.random.seed(seeds[seed_i])

        # draw all random numbers at once (vectorized)
        samples[:] = np.random.random(n)

        # re-generate permutations array
        perms[:] = np.arange(1., np.float64(n + 1))

        # in-place shuffle permutations
        np.random.shuffle(perms)

        # vectorized Latin Hypercube transformation
        rndms[seed_i, :] = (perms - samples) / float(n)

    return rndms


@njit(cache=True, fastmath=True)
def _philox4x32_7(c0, c1, c2, c3, k0, k1):
    """Compute one Philox4x32-7 block (4 uint32 outputs) for a counter and key.

    Implements the Random123 philox4x32 round function with 7 rounds — the documented
    Monte-Carlo-safe minimum (passes TestU01 BigCrush). The round function is validated
    against the official Random123 known-answer test vectors in the unit tests.

    Args:
        c0, c1, c2, c3 (uint32): the 128-bit counter words.
        k0, k1 (uint32): the 64-bit key words.

    Returns:
        tuple(uint32, uint32, uint32, uint32): the four output words.
    """
    for _r in range(7):
        p0 = PHILOX_M0 * np.uint64(c0)
        hi0 = np.uint32(p0 >> PHILOX_SHIFT32)
        lo0 = np.uint32(p0 & PHILOX_U32_MASK)
        p1 = PHILOX_M1 * np.uint64(c2)
        hi1 = np.uint32(p1 >> PHILOX_SHIFT32)
        lo1 = np.uint32(p1 & PHILOX_U32_MASK)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = np.uint32(k0 + PHILOX_W0)
        k1 = np.uint32(k1 + PHILOX_W1)
    return c0, c1, c2, c3


# Latin Hypercube on Philox4x32 (random_generator 2).
#
# Reproduces the Latin Hypercube math of `random_LatinHypercube`
# (rndms = (perms - jitter) / n) but draws randomness from Philox instead of a
# per-row-reseeded Mersenne Twister. For each seed two independent Philox streams are
# used, keyed by the seed and separated by a counter "stream tag" word: stream 0 for
# the within-stratum jitter and stream 1 for the Fisher-Yates permutation. The result
# is a valid Latin Hypercube sample (exactly one point per stratum), deterministic and
# order-independent per seed (= per group_id/event_id), and NOT bit-identical to the
# Mersenne-Twister-based generators.


@njit(cache=True, fastmath=True)
def random_LatinHypercube_Philox7(seeds, n, skip_seeds=0):
    """Latin Hypercube on Philox4x32-7 (random_generator=2).

    See the module comment above `random_LatinHypercube_Philox7` for the algorithm.

    Args:
        seeds (array[int]): per-row seeds (a hash of group_id/event_id).
        n (int): number of samples to generate for each seed.
        skip_seeds (int): number of leading rows to skip (left as zeros); correlation
          arrays pass 1.

    Returns:
        rndms (array[float64]): 2-d array of shape (len(seeds), n) of LH samples in (0, 1].
    """
    Nseeds = len(seeds)
    rndms = np.zeros((Nseeds, n), dtype=np.float64)
    perms = np.empty(n, dtype=np.float64)
    inv_n = np.float64(1.0) / np.float64(n)
    nfull = n - (n & 3)
    zero = np.uint32(0)
    for i in range(skip_seeds, Nseeds):
        s = np.uint64(seeds[i])
        k0 = np.uint32(s & PHILOX_U32_MASK)
        k1 = np.uint32(s >> PHILOX_SHIFT32)

        for k in range(n):
            perms[k] = np.float64(k + 1)

        # Fisher-Yates permutation of perms, driven by the shuffle stream (4 swaps/block).
        # Head/tail split (mirrors the jitter loop below): the nfull_shuf bulk swaps run
        # guard-free in groups of 4; only the final partial block needs the idx>=1 guards.
        # The (Philox word -> idx) pairing is identical to a flat per-swap loop, so the
        # permutation (and therefore the output) is unchanged.
        nshuf = n - 1
        nfull_shuf = nshuf - (nshuf & 3)
        ctr = np.uint32(0)
        idx = n - 1
        c = 0
        while c < nfull_shuf:
            w0, w1, w2, w3 = _philox4x32_7(ctr, PHILOX_STREAM_SHUFFLE, zero, zero, k0, k1)
            ctr = np.uint32(ctr + 1)
            jj = int(np.float64(w0) * PHILOX_INV32 * np.float64(idx + 1))
            t = perms[idx]
            perms[idx] = perms[jj]
            perms[jj] = t
            jj = int(np.float64(w1) * PHILOX_INV32 * np.float64(idx))
            t = perms[idx - 1]
            perms[idx - 1] = perms[jj]
            perms[jj] = t
            jj = int(np.float64(w2) * PHILOX_INV32 * np.float64(idx - 1))
            t = perms[idx - 2]
            perms[idx - 2] = perms[jj]
            perms[jj] = t
            jj = int(np.float64(w3) * PHILOX_INV32 * np.float64(idx - 2))
            t = perms[idx - 3]
            perms[idx - 3] = perms[jj]
            perms[jj] = t
            idx -= 4
            c += 4
        if idx >= 1:
            w0, w1, w2, w3 = _philox4x32_7(ctr, PHILOX_STREAM_SHUFFLE, zero, zero, k0, k1)
            jj = int(np.float64(w0) * PHILOX_INV32 * np.float64(idx + 1))
            t = perms[idx]
            perms[idx] = perms[jj]
            perms[jj] = t
            idx -= 1
            if idx >= 1:
                jj = int(np.float64(w1) * PHILOX_INV32 * np.float64(idx + 1))
                t = perms[idx]
                perms[idx] = perms[jj]
                perms[jj] = t
                idx -= 1
            if idx >= 1:
                jj = int(np.float64(w2) * PHILOX_INV32 * np.float64(idx + 1))
                t = perms[idx]
                perms[idx] = perms[jj]
                perms[jj] = t
                idx -= 1

        # combine perms with the jitter stream (4 outputs/block)
        ctr = np.uint32(0)
        k = 0
        while k < nfull:
            w0, w1, w2, w3 = _philox4x32_7(ctr, PHILOX_STREAM_JITTER, zero, zero, k0, k1)
            ctr = np.uint32(ctr + 1)
            rndms[i, k] = (perms[k] - np.float64(w0) * PHILOX_INV32) * inv_n
            rndms[i, k + 1] = (perms[k + 1] - np.float64(w1) * PHILOX_INV32) * inv_n
            rndms[i, k + 2] = (perms[k + 2] - np.float64(w2) * PHILOX_INV32) * inv_n
            rndms[i, k + 3] = (perms[k + 3] - np.float64(w3) * PHILOX_INV32) * inv_n
            k += 4
        if k < n:
            w0, w1, w2, w3 = _philox4x32_7(ctr, PHILOX_STREAM_JITTER, zero, zero, k0, k1)
            rndms[i, k] = (perms[k] - np.float64(w0) * PHILOX_INV32) * inv_n
            k += 1
            if k < n:
                rndms[i, k] = (perms[k] - np.float64(w1) * PHILOX_INV32) * inv_n
                k += 1
            if k < n:
                rndms[i, k] = (perms[k] - np.float64(w2) * PHILOX_INV32) * inv_n
                k += 1
    return rndms
