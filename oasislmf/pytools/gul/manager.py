"""
This file is the entry point for the gul command for the package.

"""

from oasislmf.pytools.getmodel.common import oasis_float
from oasislmf.pytools.getmodel.manager import get_mean_damage_bins, get_items

import os
import numpy as np
from scipy.stats import qmc

import logging
logger = logging.getLogger(__name__)


def get_coverages(input_path, ignore_file_type=set()):
    """
    Loads the coverages from the coverages file.

    Args:
        input_path: (str) the path containing the coverage file
        ignore_file_type: set(str) file extension to ignore when loading

    Returns: np.array[oasis_float]
        coverages array
    """
    input_files = set(os.listdir(input_path))

    # TODO: store default filenames (e.g., coverages.bin) in a parameters file

    if "coverages.bin" in input_files and "bin" not in ignore_file_type:
        logger.debug(f"loading {os.path.join(input_path, 'coverages.csv')}")
        coverages = np.fromfile(os.path.join(input_path, "coverages.bin"), dtype=oasis_float)

    elif "coverages.csv" in input_files and "csv" not in ignore_file_type:
        logger.debug(f"loading {os.path.join(input_path, 'coverages.csv')}")
        coverages = np.genfromtxt(os.path.join(input_path, "coverages.csv"), dtype=oasis_float, delimiter=",")

    else:
        raise FileNotFoundError(f'coverages file not found at {input_path}')

    return coverages


def generate_rands(N=100, method='uniform', d=1, rng=None, seed=None):
    """
    Generate random numbers.
    
    Args:
        N: int, optional
            Number of random numbers to generate.
        method: str, optional
            Random generation method to use for drawing samples (see Notes).
        d: int, optional
            Dimensions, which some methods have (e.g., Latin Hypercube and Sobol).
        rng: numpy.random.Generator, optional
            Random number generator. If provided, it uses that generator to draw 
            random samples between 0. and 1. If None (default), the function creates
            a random number generator internally using `seed`.
        seed: int or list of int, optional
            Random seed to initialise the random number generator if the user provides
            no random number generator in `rng`. Default: 123456.
            If a list of integers is provided, the function creates as many random 
            number generators as the length of the list and draws `N` from each
            random number generator. 
            
    Returns: np.array[oasis_float]
        The random numbers.
        
    Notes:
        Currently, the implemented methods are:
            'uniform': uniform distribution between 0 and 1 (uses scipy).
            'LHS'    : Latin Hypercube Sampling (uses the smt package).
            
        Docs of the smt package for LHS are at:
        https://smt.readthedocs.io/en/latest/_src_docs/sampling_methods/lhs.html
    
    """
    if not rng:
        if isinstance(seed, list):
            rng = [np.random.default_rng(seed_i) for seed_i in seed]
        else:
            rng = np.random.default_rng(seed)

    if isinstance(rng, list):
        if method == 'uniform':
            # uniform sampling
            rndm = np.array([rng_.uniform(0, 1, size=N) for rng_ in rng])  # .reshape((d, N))

        elif method == 'LHS':
            # latin hypercube sampling
            samplers = [qmc.LatinHypercube(d=1, seed=rng_) for rng_ in rng]
            rndm = np.array([sampler.random(N) for sampler in samplers])

        elif method == 'Sobol':
            # Sobol sequences
            samplers = [qmc.Sobol(d=1, scramble=False, seed=rng_) for rng_ in rng]

            # check if N is power of 2
            N_base2 = np.log2(N)

            if N_base2.is_integer():
                # N is a power of 2
                rndm = np.array([sampler.random_base2(m=int(N_base2)) for sampler in samplers])
            else:
                # print("WARNING: Sobol sequences can be affected in their balance for sample numbers that are not powers of 2")
                # print("         See notes at https://scipy.github.io/devdocs/reference/generated/scipy.stats.qmc.Sobol.html")
                # some techniques may be required to generate large number of sobol sequences
                # sobol also has a maximum number of dimensions d=21201.
                rndm = np.array([sampler.random(N) for sampler in samplers])

    else:
        if method == 'uniform':
            # uniform sampling
            rndm = rng.uniform(0, 1, size=d * N)  # .reshape((d, N))

        elif method == 'LHS':
            # latin hypercube sampling
            sampler = qmc.LatinHypercube(d=d, seed=rng)
            rndm = sampler.random(N)

        elif method == 'Sobol':
            # Sobol sequences
            sampler = qmc.Sobol(d=d, scramble=False, seed=rng)

            # check if N is power of 2
            N_base2 = np.log2(N)

            if N_base2.is_integer():
                # N is a power of 2
                rndm = sampler.random_base2(m=int(N_base2))
            else:
                # print("WARNING: Sobol sequences can be affected in their balance for sample numbers that are not powers of 2")
                # print("         See notes at https://scipy.github.io/devdocs/reference/generated/scipy.stats.qmc.Sobol.html")
                # some techniques may be required to generate large number of sobol sequences
                # sobol also has a maximum number of dimensions d=21201.

                rndm = sampler.random(N)

    # ...add further methods here...

    return rndm.reshape(-1)


GROUP_ID_HASH_CODE = np.int64(1543270363)
EVENT_ID_HASH_CODE = np.int64(1943272559)

HASH_MOD_CODE = np.int64(2147483648)


def generate_hash(group_id, event_id, rand_seed=0, correlated=False):
    """
    Generate hash for group_id, event_id

    Args:
        group_id ([type]): [description]
        event_id ([type]): [description]
        rand_seed (int, optional): [description]. Defaults to 0.
        correlated (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    hashed = rand_seed
    hashed += np.mod(group_id * GROUP_ID_HASH_CODE, HASH_MOD_CODE)

    if correlated:
        return hashed

    hashed += np.mod(event_id, * EVENT_ID_HASH_CODE, HASH_MOD_CODE)

    return hashed


def run(run_dir, ignore_file_type, sample_size,):
    """
    Runs the main process of the gul calculation.

    Args:
        run_dir: (str) the directory of where the process is running
        ignore_file_type: set(str) file extension to ignore when loading

    """
    logger.info("starting gulpy")

    static_path = os.path.join(run_dir, 'static')
    input_path = os.path.join(run_dir, 'input')
    ignore_file_type = set(ignore_file_type)

    static_path = 'static/'
    # TODO: store static_path in a paraparameters file
    damage_bins = get_mean_damage_bins(static_path)

    input_path = 'input/'
    # TODO: store input_path in a paraparameters file
    items = get_items(input_path)
    coverages = get_coverages(input_path)
    Ncoverages = coverages

    # get random numbers
    # getRands rnd(opt.rndopt, opt.rand_vector_size, opt.rand_seed);
    # getRands rnd0(opt.rndopt, opt.rand_vector_size, opt.rand_seed);
    Nrands = sample_size * Ncoverages

    seed = 123456  # substitute with desired value
    rands = generate_rands(N=100, method='uniform', d=1, rng=None, seed=seed)
    # run gulcalc

    logger.info("gulpy is finished")

    return 0
