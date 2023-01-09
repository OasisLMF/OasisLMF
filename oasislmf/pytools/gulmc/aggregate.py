"""
This file contains specific functionality needed for aggregate vulnerabilities.

"""
import logging
import os

import numba as nb
import numpy as np
import pandas as pd
from numba import njit
from numba.typed import Dict, List
from numba.types import int32 as nb_int32
from numba.types import uint32 as nb_uint32

from oasislmf.pytools.getmodel.common import areaperil_int
from oasislmf.pytools.gulmc.common import (AggregateVulnerability,
                                           VulnerabilityWeight)

AGG_VULN_WEIGHTS_KEY_TYPE = nb.types.Tuple((nb.from_dtype(areaperil_int), nb.types.int32))
AGG_VULN_WEIGHTS_VAL_TYPE = nb.types.int32

logger = logging.getLogger(__name__)


@njit(cache=True)
def gen_empty_agg_vuln_to_vuln_ids():
    """TODO

    Returns:
        _type_: _description_
    """
    return Dict.empty(nb_int32, List.empty_list(nb_int32))


@njit(cache=True)
def gen_empty_areaperil_vuln_ids_to_weights():
    """TODO

    Returns:
        _type_: _description_
    """
    return Dict.empty(AGG_VULN_WEIGHTS_KEY_TYPE, AGG_VULN_WEIGHTS_VAL_TYPE)


def read_aggregate_vulnerability(path, ignore_file_type=set()):
    """Load the aggregate vulnerability definitions from file.
    TODO: check docs

    Args:
        path (str): the path pointing to the file
        ignore_file_type (Set[str]): file extension to ignore when loading.

    Returns:
        Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]]
        vulnerability dictionary, vulnerability IDs, areaperil to vulnerability index dictionary,
        areaperil ID to vulnerability index array, areaperil ID to vulnerability array
    """
    input_files = set(os.listdir(path))

    if "aggregate_vulnerability.bin" in input_files and "bin" not in ignore_file_type:
        fname = os.path.join(path, 'aggregate_vulnerability.bin')
        logger.debug(f"loading {fname}")
        aggregate_vulnerability = np.memmap(fname, dtype=AggregateVulnerability, mode='r')

    elif "aggregate_vulnerability.csv" in input_files and "csv" not in ignore_file_type:
        fname = os.path.join(path, 'aggregate_vulnerability.csv')
        logger.debug(f"loading {fname}")
        aggregate_vulnerability = np.loadtxt(fname, dtype=AggregateVulnerability, delimiter=",", skiprows=1, ndmin=1)

    else:
        aggregate_vulnerability = None
        logging.warning('Aggregate vulnerability table not found at {path}. Continuing without aggregate vulnerability definitions.')

    return aggregate_vulnerability


def read_vulnerability_weights(path, ignore_file_type=set()):
    """Load the aggregate vulnerability weights definitions from file.
    TODO: check docs

    Args:
        path (str): the path pointing to the file
        ignore_file_type (Set[str]): file extension to ignore when loading.

    Returns:
        Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]]
        vulnerability dictionary, vulnerability IDs, areaperil to vulnerability index dictionary,
        areaperil ID to vulnerability index array, areaperil ID to vulnerability array
    """
    input_files = set(os.listdir(path))

    if "weights.bin" in input_files and "bin" not in ignore_file_type:
        fname = os.path.join(path, 'weights.bin')
        logger.debug(f"loading {fname}")
        aggregate_weights = np.memmap(fname, dtype=VulnerabilityWeight, mode='r')

    elif "weights.csv" in input_files and "csv" not in ignore_file_type:
        fname = os.path.join(path, 'weights.csv')
        logger.debug(f"loading {fname}")
        aggregate_weights = np.loadtxt(fname, dtype=VulnerabilityWeight, delimiter=",", skiprows=1, ndmin=1)

    else:
        aggregate_weights = None
        logging.warning('Vulnerability weights not found at {path}. Continuing without vulnerability weights definitions.')

    return aggregate_weights


def process_aggregate_vulnerability(aggregate_vulnerability):
    """
    TODO: add docs

    Args:
        aggregate_vulnerability (_type_): _description_

    Returns:
        _type_: _description_
    """
    agg_vuln_to_vuln_id = gen_empty_agg_vuln_to_vuln_ids()

    if aggregate_vulnerability is not None:

        agg_vuln_df = pd.DataFrame(aggregate_vulnerability)
        # init agg_vuln_to_vuln_ids to allow numba to compile later functions
        # vulnerability_id and aggregate_vulnerability_id are remapped to the internal ids
        # using the vulnd_dict map that contains only the vulnerability_id used in this portfolio.

        # here we read all aggregate vulnerability_id, then, after processing the items file,
        # we will filter out the aggregate vulnerability that are not used in this portfolio.
        for agg, grp in agg_vuln_df.groupby('aggregate_vulnerability_id'):
            agg_idx = nb_int32(agg)

            if agg_idx not in agg_vuln_to_vuln_id:
                agg_vuln_to_vuln_id[agg_idx] = List.empty_list(nb_int32)

            for entry in grp['vulnerability_id'].to_list():
                agg_vuln_to_vuln_id[agg_idx].append(nb_int32(entry))

    return agg_vuln_to_vuln_id


def process_vulnerability_weights(aggregate_weights, agg_vuln_to_vuln_id):
    """
    TODO: add docs

    Args:
        aggregate_weights (_type_): _description_
        agg_vuln_to_vuln_id (_type_): _description_

    Returns:
        _type_: _description_
    """
    areaperil_vuln_id_to_weight = gen_empty_areaperil_vuln_ids_to_weights()

    if aggregate_weights is not None:
        agg_weight_df = pd.DataFrame(aggregate_weights)

        if len(agg_vuln_to_vuln_id) > 0:
            # at least one aggregate vulnerability is defined
            for agg, grp in agg_weight_df.groupby(['areaperil_id', 'vulnerability_id']):
                areaperil_vuln_id_to_weight[(nb_uint32(agg[0]), nb_int32(agg[1]))] = nb_int32(grp['weight'].to_list()[0])

    return areaperil_vuln_id_to_weight


@njit(cache=True)
def map_agg_vuln_ids_to_agg_vuln_idxs(agg_vulns, agg_vuln_to_vuln_ids, vuln_dict):
    """For each aggregate vulnerability listed in `agg_vulns`, map the individual vulnerability_ids that compose it
    to the indices where they are stored in `vuln_array`.

    TODO: update docs

    Args:
        agg_vulns (List[int32])
    """
    agg_vuln_to_vuln_idxs = Dict.empty(nb_int32, List.empty_list(nb_int32))

    for agg in agg_vulns:
        agg_vuln_to_vuln_idxs[agg] = List([vuln_dict[vuln] for vuln in agg_vuln_to_vuln_ids[agg]])

    return agg_vuln_to_vuln_idxs


@njit(cache=True)
def map_areaperil_vuln_id_to_weight_to_areaperil_vuln_idx_to_weight(areaperil_to_vulns, areaperil_vuln_id_to_weight, vuln_dict):
    """Make map from aggregate vulnerability id to the list of sub-vulnerability ids of which they are composed, where the
    value of the sub-vulnerability id is the internal pointer to the dense array where they are stored.
    TODO: update docs
    """
    ap_vuln_idx_weights = Dict.empty(AGG_VULN_WEIGHTS_KEY_TYPE, AGG_VULN_WEIGHTS_VAL_TYPE)

    if len(areaperil_vuln_id_to_weight) > 0:
        for ap in areaperil_to_vulns:
            for vuln in areaperil_to_vulns[ap]:
                if tuple((ap, vuln)) in areaperil_vuln_id_to_weight:
                    # if the weights file contains this definition, add this entry to the map.
                    # otherwise, the weight of this (areaperil_id, vulnerability_id) is assumed to be zero during loss calculation.
                    ap_vuln_idx_weights[tuple((ap, vuln_dict[vuln]))] = areaperil_vuln_id_to_weight[tuple((ap, vuln))]

    return ap_vuln_idx_weights
