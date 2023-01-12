"""
This file contains the utilities for generating random numbers in gulpy.

"""

from math import sqrt
import logging
import numpy as np
from scipy.stats import norm
from numba import njit

logger = logging.getLogger(__name__)


GROUP_ID_HASH_CODE = np.int64(1543270363)
EVENT_ID_HASH_CODE = np.int64(1943272559)
HASH_MOD_CODE = np.int64(2147483648)

HAZ_GROUP_ID_HASH_CODE = np.int64(1343271947)
HAZ_EVENT_ID_HASH_CODE = np.int64(1743274343)
HAZ_HASH_MOD_CODE = np.int64(2157483709)


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
def generate_hash_haz(group_id, event_id, base_seed=0):
    """Generate hash for a given `group_id`, `event_id` pair for the hazard pdf.

    Args:
        group_id (int): group id.
        event_id (int]): event id.
        base_seed (int, optional): base random seed. Defaults to 0.

    Returns:
        int64: hash
    """
    hash = (base_seed + (group_id * HAZ_GROUP_ID_HASH_CODE) % HAZ_HASH_MOD_CODE +
            (event_id * HAZ_EVENT_ID_HASH_CODE) % HAZ_HASH_MOD_CODE) % HAZ_HASH_MOD_CODE

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

    else:
        raise ValueError(f"No random generator exists for random_generator={random_generator}.")


EVENT_ID_HASH_CODE = np.int64(1943_272_559)
PERIL_CORRELATION_GROUP_HASH = np.int64(1836311903)
HASH_MOD_CODE = np.int64(2147483648)


@njit(cache=True, fastmath=True)
def generate_correlated_hash_vector(unique_peril_correlation_groups, event_id, base_seed=0):
    """Generate hashes for all peril correlation groups for a given `event_id`.

    Args:
        unique_peril_correlation_groups (List[int]): list of the unique peril correlation groups.
        event_id (int): event id.
        base_seed (int, optional): base random seed. Defaults to 0.

    Returns:
        List[int64]: hashes
    """
    Nperil_correlation_groups = unique_peril_correlation_groups.shape[0]
    correlated_hashes = np.empty(Nperil_correlation_groups + 1, dtype='int64')
    correlated_hashes[0] = 0

    for i in range(1, Nperil_correlation_groups + 1):
        correlated_hashes[i] = (
            base_seed +
            (unique_peril_correlation_groups[i - 1] * PERIL_CORRELATION_GROUP_HASH) % HASH_MOD_CODE +
            (event_id * EVENT_ID_HASH_CODE) % HASH_MOD_CODE
        ) % HASH_MOD_CODE

    return correlated_hashes


def compute_norm_inv_cdf_lookup(arr_min, arr_max, arr_N):
    return norm.ppf(np.linspace(arr_min, arr_max, arr_N))


def compute_norm_cdf_lookup(arr_min, arr_max, arr_N):
    return norm.cdf(np.linspace(arr_min, arr_max, arr_N))


@njit(cache=True, fastmath=True)
def get_norm_cdf_cell_nb(x, arr_min, arr_max, arr_N):
    return int((x - arr_min) * (arr_N - 1) // (arr_max - arr_min))


@njit(cache=True, fastmath=True)
def get_corr_rval(x_unif, y_unif, rho, arr_min, arr_max, arr_N, norm_inv_cdf, arr_min_cdf,
                  arr_max_cdf, arr_N_cdf, norm_cdf, Nsamples, z_unif):

    sqrt_rho = sqrt(rho)
    sqrt_1_minus_rho = sqrt(1. - rho)

    for i in range(Nsamples):
        x_norm = norm_inv_cdf[get_norm_cdf_cell_nb(x_unif[i], arr_min, arr_max, arr_N)]
        y_norm = norm_inv_cdf[get_norm_cdf_cell_nb(y_unif[i], arr_min, arr_max, arr_N)]
        z_norm = sqrt_rho * x_norm + sqrt_1_minus_rho * y_norm

        z_unif[i] = norm_cdf[get_norm_cdf_cell_nb(z_norm, arr_min_cdf, arr_max_cdf, arr_N_cdf)]


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

        # draw the random numbers
        for j in range(n):
            # by default in numba this should be Mersenne-Twister
            rndms[seed_i, j] = np.random.uniform(0., 1.)

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

        # draw the random numbers and re-generate permutations array
        for i in range(n):
            samples[i] = np.random.uniform(0., 1.)
            perms[i] = i + 1

        # in-place shuffle permutations
        np.random.shuffle(perms)

        for j in range(n):
            rndms[seed_i, j] = (perms[j] - samples[j]) / float(n)

    return rndms
