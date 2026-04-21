"""
This file contains specific functionality needed for aggregate vulnerabilities.

"""
import logging

import numba as nb
import numpy as np
import pandas as pd
from oasis_data_manager.filestore.backends.base import BaseStorage
from oasislmf.pytools.common.data import nb_areaperil_int, nb_oasis_float, oasis_int, aggregatevulnerability_dtype, vulnerability_weight_dtype
from oasislmf.pytools.common.id_index import build as id_index_build
from oasislmf.pytools.common.hashmap import unpack as hm_unpack, _find_key as hm_find_key, NOT_FOUND as HM_NOT_FOUND

logger = logging.getLogger(__name__)


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
        logger.warning(
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
        logger.warning(
            f"Vulnerability weights not found at {storage.get_storage_url('', encode_params=False)[0]}. Continuing without vulnerability weights definitions.")

    return aggregate_weights


def process_aggregate_vulnerability(aggregate_vulnerability):
    """Rearrange aggregate vulnerability definitions from tabular format into CRS arrays
    mapping aggregate vulnerability id to the list of vulnerability ids that compose it.

    Args:
        aggregate_vulnerability (np.array[AggregateVulnerability]): aggregate vulnerability table.

    Returns:
        agg_vuln_ids (np.array[oasis_int]): sorted array of aggregate vulnerability ids.
        agg_vuln_id_ja_offsets (np.array[oasis_int]): jagged array offsets. Row i spans
            agg_vuln_id_ja_vuln_ids[agg_vuln_id_ja_offsets[i]:agg_vuln_id_ja_offsets[i+1]].
        agg_vuln_id_ja_vuln_ids (np.array[oasis_int]): flat jagged array of constituent vulnerability ids.
    """
    if aggregate_vulnerability is not None and len(aggregate_vulnerability) > 0:
        agg_vuln_df = pd.DataFrame(aggregate_vulnerability)
        # Group by aggregate_vulnerability_id, preserving insertion order of sub-vulns.
        # Sort groups so agg_vuln_ids is sorted for binary search in generate_item_map.
        grouped = agg_vuln_df.groupby('aggregate_vulnerability_id', sort=True)
        n_groups = len(grouped)
        n_entries = len(agg_vuln_df)

        agg_vuln_ids = np.empty(n_groups, dtype=oasis_int)
        agg_vuln_id_ja_vuln_ids = np.empty(n_entries, dtype=oasis_int)
        agg_vuln_id_ja_offsets = np.empty(n_groups + 1, dtype=oasis_int)
        agg_vuln_id_ja_offsets[0] = 0
        ptr = 0

        for i, (agg_id, grp) in enumerate(grouped):
            agg_vuln_ids[i] = agg_id
            sub_vulns = grp['vulnerability_id'].values
            n_sub = len(sub_vulns)
            agg_vuln_id_ja_vuln_ids[ptr:ptr + n_sub] = sub_vulns
            ptr += n_sub
            agg_vuln_id_ja_offsets[i + 1] = ptr

    else:
        agg_vuln_ids = np.empty(0, dtype=oasis_int)
        agg_vuln_id_ja_vuln_ids = np.empty(0, dtype=oasis_int)
        agg_vuln_id_ja_offsets = np.zeros(1, dtype=oasis_int)

    agg_vuln_id_ja_id_ind = id_index_build(agg_vuln_ids)
    return agg_vuln_ids, agg_vuln_id_ja_id_ind, agg_vuln_id_ja_offsets, agg_vuln_id_ja_vuln_ids


@nb.njit(cache=True)
def process_vulnerability_weights(areaperil_agg_vuln_idx_ja_areaperil_ids, areaperil_agg_vuln_idx_ja_data,
                                  vuln_map, vuln_map_keys, aggregate_weights):
    """
    Populate the weight field in the merged data array by matching aggregate_weights records.

    Args:
        areaperil_agg_vuln_idx_ja_areaperil_ids (np.array[areaperil_int]): areaperil_id for each entry.
        areaperil_agg_vuln_idx_ja_data (np.array[agg_vuln_idx_weight_dtype]): merged (vuln_idx, weight) per entry.
        vuln_map (np.ndarray[uint8]): packed hashmap table mapping vuln_id to dense index.
        vuln_map_keys (np.ndarray[int32]): array of unique vulnerability ids (hashmap keys).
        aggregate_weights (np.array[VulnerabilityWeight]): vulnerability weights table.
    """
    hm_info, hm_lookup, hm_index = hm_unpack(vuln_map)
    n_entries = len(areaperil_agg_vuln_idx_ja_data)
    for i in range(len(aggregate_weights)):
        rec = aggregate_weights[i]
        slot = hm_find_key(hm_info, hm_lookup, hm_index, vuln_map_keys, rec['vulnerability_id'])
        if slot != HM_NOT_FOUND:
            dense_idx = hm_index[slot]
            ap_id = nb_areaperil_int(rec['areaperil_id'])
            for j in range(n_entries):
                if areaperil_agg_vuln_idx_ja_areaperil_ids[j] == ap_id and areaperil_agg_vuln_idx_ja_data[j]['vuln_idx'] == dense_idx:
                    areaperil_agg_vuln_idx_ja_data[j]['weight'] = nb_oasis_float(rec['weight'])
