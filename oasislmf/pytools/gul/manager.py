"""
This file is the entry point for the gul command for the package.

"""

from oasislmf.pytools.getmodel.common import oasis_float
from oasislmf.pytools.getmodel.manager import get_mean_damage_bins, get_items

import os
import numpy as np
import logging
logger = logging.getLogger(__name__)


def get_coverages(input_path, ignore_file_type=set()):
    """
    Loads the coverages from the coverages file.

    Args:
        input_path: (str) the path containing the coverage file
        ignore_file_type: set(str) file extension to ignore when loading

    Returns: (np.array[oasis_float])
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


def run(run_dir, ignore_file_type, **kwargs):
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

    # get random numbers
    # getRands rnd(opt.rndopt, opt.rand_vector_size, opt.rand_seed);
    # getRands rnd0(opt.rndopt, opt.rand_vector_size, opt.rand_seed);

    # run gulcalc

    logger.info("gulpy is finished")

    return 0
