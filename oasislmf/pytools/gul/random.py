"""
This file contains the utilities for generating random numbers in gulpy.

"""

import logging
import numpy as np
from numba import njit

logger = logging.getLogger(__name__)


GROUP_ID_HASH_CODE = np.int64(1543270363)
EVENT_ID_HASH_CODE = np.int64(1943272559)
HASH_MOD_CODE = np.int64(2147483648)


@njit(cache=True, fastmath=True)
def generate_hash(group_id, event_id, base_seed=0):
    """Generate hash for a given `group_id`, `event_id` pair.

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
def generate_correlated_hash(event_id, base_seed=0):
    """Generate hash for an `event_id`.

    Args:
        event_id (int): event id.
        base_seed (int, optional): base random seed. Defaults to 0.

    Returns:
        int64: hash
    """
    hash = (base_seed + (event_id * EVENT_ID_HASH_CODE) % HASH_MOD_CODE) % HASH_MOD_CODE

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


@njit(cache=True, fastmath=True)
def random_MersenneTwister(seeds, n):
    """Generate random numbers using the default Mersenne Twister algorithm.

    Args:
        seeds (List[int64]): List of seeds.
        n (int): number of random samples to generate for each seed.

    Returns:
        rndms (array[float]): 2-d array of shape (number of seeds, n) 
          containing the random values generated for each seed.
        rndms_idx (Dict[int64, int]): mapping between `seed` and the 
          row in rndms that stores the corresponding random values.
    """
    Nseeds = len(seeds)
    rndms = np.zeros((Nseeds, n), dtype='float64')

    for seed_i, seed in enumerate(seeds):
        # set the seed
        np.random.seed(seed)

        # draw the random numbers
        for j in range(n):
            # by default in numba this should be Mersenne-Twister
            rndms[seed_i, j] = np.random.uniform(0., 1.)

    return rndms


@njit(cache=True, fastmath=True)
def random_LatinHypercube(seeds, n):
    """Generate random numbers using the Latin Hypercube algorithm.

    Args:
        seeds (List[int64]): List of seeds.
        n (int): number of random samples to generate for each seed.

    Returns:
        rndms (array[float]): 2-d array of shape (number of seeds, n) 
          containing the random values generated for each seed.
        rndms_idx (Dict[int64, int]): mapping between `seed` and the 
          row in rndms that stores the corresponding random values.

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

    for seed_i, seed in enumerate(seeds):
        # set the seed
        np.random.seed(seed)

        # draw the random numbers and re-generate permutations array
        for i in range(n):
            samples[i] = np.random.uniform(0., 1.)
            perms[i] = i + 1

        # in-place shuffle permutations
        np.random.shuffle(perms)

        for j in range(n):
            rndms[seed_i, j] = (perms[j] - samples[j]) / float(n)

    return rndms
