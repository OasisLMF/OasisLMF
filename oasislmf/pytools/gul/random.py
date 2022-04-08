"""
This file contains the utilities for generating random numbers in gulpy.

"""

import logging
import numpy as np
from numba import njit

logger = logging.getLogger(__name__)


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
    rndms_idx = {}

    for seed_i, seed in enumerate(seeds):
        # set the seed
        np.random.seed(seed)

        # draw the random numbers
        for j in range(n):
            # by default in numba this should be Mersenne-Twister
            rndms[seed_i, j] = np.random.uniform(0., 1.)

        rndms_idx[seed] = seed_i

    return rndms, rndms_idx


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
    rndms_idx = {}

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

        rndms_idx[seed] = seed_i

    return rndms, rndms_idx
