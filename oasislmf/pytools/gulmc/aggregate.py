"""
This file contains specific functionality needed for aggregate vulnerabilities.

"""
import logging

import numba as nb
import numpy as np
import pandas as pd
from numba import njit
from numba.typed import Dict, List
from numba.types import int32 as nb_int32

from oasis_data_manager.filestore.backends.base import BaseStorage
from oasislmf.pytools.common.data import areaperil_int, nb_areaperil_int, nb_oasis_float, aggregatevulnerability_dtype, vulnerability_weight_dtype

logger = logging.getLogger(__name__)

AGG_VULN_WEIGHTS_KEY_TYPE = nb.types.Tuple((nb.from_dtype(areaperil_int), nb.types.int32))
AGG_VULN_WEIGHTS_VAL_TYPE = nb.types.float32


@njit(cache=True)
def gen_empty_agg_vuln_to_vuln_ids():
    """Generate empty map to store the definitions of aggregate vulnerability functions.

    Returns:
        dict[int, list[int]]: map of aggregate vulnerability id to list of vulnerability ids.
    """
    return Dict.empty(nb_int32, List.empty_list(nb_int32))


@njit(cache=True)
def gen_empty_areaperil_vuln_idx_to_weights():
    """Generate empty map to store the weights of individual vulnerability functions in each aggregate vulnerability.

    Returns:
        dict[AGG_VULN_WEIGHTS_KEY_TYPE, AGG_VULN_WEIGHTS_VAL_TYPE]: map of areaperil_id, vulnerability id to weight.
    """
    return Dict.empty(AGG_VULN_WEIGHTS_KEY_TYPE, AGG_VULN_WEIGHTS_VAL_TYPE)


def read_aggregate_vulnerability(storage: BaseStorage, ignore_file_type=set()):
    """Load the aggregate vulnerability definitions from file.

    Args:
        storage: (BaseStorage) the storage manager for fetching model data
        ignore_file_type (Set[str]): file extension to ignore when loading.

    Returns:
        np.array[AggregateVulnerability]: aggregate vulnerability table.
    """
    input_files = set(storage.listdir())

    if "aggregate_vulnerability.bin" in input_files and "bin" not in ignore_file_type:
        logger.debug(f"loading {storage.get_storage_url('aggregate_vulnerability.bin', encode_params=False)}")
        with storage.open('aggregate_vulnerability.bin') as f:
            aggregate_vulnerability = np.memmap(f, dtype=aggregatevulnerability_dtype, mode='r')

    elif "aggregate_vulnerability.csv" in input_files and "csv" not in ignore_file_type:
        logger.debug(f"loading {storage.get_storage_url('aggregate_vulnerability.csv', encode_params=False)}")
        with storage.open('aggregate_vulnerability.csv') as f:
            aggregate_vulnerability = np.loadtxt(f, dtype=aggregatevulnerability_dtype, delimiter=",", skiprows=1, ndmin=1)

    else:
        aggregate_vulnerability = None
        logging.warning(
            f"Aggregate vulnerability table not found at {storage.get_storage_url('', encode_params=False)[0]}. Continuing without aggregate vulnerability definitions.")

    return aggregate_vulnerability


def read_vulnerability_weights(storage: BaseStorage, ignore_file_type=set()):
    """Load the vulnerability weights definitions from file.

    Args:
        storage: (BaseStorage) the storage manager for fetching model data
        ignore_file_type (Set[str]): file extension to ignore when loading.

    Returns:
        np.array[VulnerabilityWeight]: vulnerability weights table.
    """
    input_files = set(storage.listdir())

    if "weights.bin" in input_files and "bin" not in ignore_file_type:
        logger.debug(f"loading {storage.get_storage_url('weights.bin', encode_params=False)}")
        with storage.open("weights.bin") as f:
            aggregate_weights = np.memmap(f, dtype=vulnerability_weight_dtype, mode='r')

    elif "weights.csv" in input_files and "csv" not in ignore_file_type:
        logger.debug(f"loading {storage.get_storage_url('weights.csv', encode_params=False)}")
        with storage.open("weights.csv") as f:
            aggregate_weights = np.loadtxt(f, dtype=vulnerability_weight_dtype, delimiter=",", skiprows=1, ndmin=1)

    else:
        aggregate_weights = None
        logging.warning(
            f"Vulnerability weights not found at {storage.get_storage_url('', encode_params=False)[0]}. Continuing without vulnerability weights definitions.")

    return aggregate_weights


def process_aggregate_vulnerability(aggregate_vulnerability):
    """Rearrange aggregate vulnerability definitions from tabular format to a map between aggregate
    vulnerability id and the list of vulnerability ids that it is made of.

    Args:
        aggregate_vulnerability (np.array[AggregateVulnerability]): aggregate vulnerability table.

    Returns:
        dict[int, list[int]]: map of aggregate vulnerability id to list of vulnerability ids.
    """
    agg_vuln_to_vuln_ids = gen_empty_agg_vuln_to_vuln_ids()

    if aggregate_vulnerability is not None:

        agg_vuln_df = pd.DataFrame(aggregate_vulnerability)
        # init agg_vuln_to_vuln_ids to allow numba to compile later functions
        # vulnerability_id and aggregate_vulnerability_id are remapped to the internal ids
        # using the vulnd_dict map that contains only the vulnerability_id used in this portfolio.

        # here we read all aggregate vulnerability_id, then, after processing the items file,
        # we will filter out the aggregate vulnerability that are not used in this portfolio.
        for agg, grp in agg_vuln_df.groupby('aggregate_vulnerability_id'):
            agg_vuln_id = nb_int32(agg)

            if agg_vuln_id not in agg_vuln_to_vuln_ids:
                agg_vuln_to_vuln_ids[agg_vuln_id] = List.empty_list(nb_int32)

            for entry in grp['vulnerability_id'].to_list():
                agg_vuln_to_vuln_ids[agg_vuln_id].append(nb_int32(entry))

    return agg_vuln_to_vuln_ids


@nb.njit(cache=True)
def process_vulnerability_weights(areaperil_vuln_i_to_weight, vuln_dict, aggregate_weights):
    """
    Polpulate the useful (areaperil_id, vulnerability_i) in areaperil_vuln_i_to_weight with the weight from aggregate_weights

    Args:
        areaperil_vuln_i_to_weight: dict of useful (areaperil_id, vulnerability_i) to 0. (weight placeholder to be updated)
        vuln_dict: vuln_dict (Tuple[Dict[int, int]): vulnerability dictionary, vuln_id => vuln_i.
        aggregate_weights (np.array[VulnerabilityWeight]): vulnerability weights table.
    """
    for i in range(len(aggregate_weights)):
        rec = aggregate_weights[i]
        if rec['vulnerability_id'] in vuln_dict:
            key = (nb_areaperil_int(rec['areaperil_id']), vuln_dict[rec['vulnerability_id']])
            if key in areaperil_vuln_i_to_weight:
                areaperil_vuln_i_to_weight[key] = nb_oasis_float(rec['weight'])
