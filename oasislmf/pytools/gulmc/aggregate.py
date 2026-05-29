"""
This file contains specific functionality needed for aggregate vulnerabilities.

"""
import logging
import os

import numba as nb
import numpy as np
import pandas as pd
from oasis_data_manager.filestore.backends.base import BaseStorage
from oasislmf.utils.data import analysis_settings_loader
from oasislmf.pytools.common.data import (
    areaperil_int, nb_areaperil_int, nb_oasis_int, nb_oasis_float, oasis_float, oasis_int,
    aggregatevulnerability_dtype, vulnerability_weight_dtype,
)
from oasislmf.pytools.common.id_index import build as id_index_build
from oasislmf.pytools.common.hashmap import (
    init_dict as hm_init_dict, unpack as hm_unpack, rehash as hm_rehash,
    _try_add_key as hm_try_add_key, _find_key as hm_find_key,
    i_add_key_fail as hm_i_add_key_fail,
    new_slot_bit as hm_new_slot_bit, slot_mask as hm_slot_mask,
    NOT_FOUND as HM_NOT_FOUND,
)

logger = logging.getLogger(__name__)

# Key for the (areaperil_id, vuln_idx) -> weight hashmap built in
# process_vulnerability_weights. Defined at module level so the structured-record
# fnv1a/key_eq overloads compile once.
agg_weight_key_dtype = np.dtype([('areaperil_id', areaperil_int), ('vuln_idx', oasis_int)])


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

    Builds a (areaperil_id, vuln_idx) -> weight hashmap from aggregate_weights once, then
    iterates entries with one O(1) lookup each. Total cost: O(W + E).

    Iterating from the weight side handles the case where the same (ap, vuln_idx) pair
    appears in multiple entries (two distinct aggregate definitions sharing a sub-vuln on
    the same areaperil): every such entry gets the same weight, since each is looked up
    independently.

    Args:
        areaperil_agg_vuln_idx_ja_areaperil_ids (np.array[areaperil_int]): areaperil_id for each entry.
        areaperil_agg_vuln_idx_ja_data (np.array[agg_vuln_idx_weight_dtype]): merged (vuln_idx, weight) per entry.
        vuln_map (np.ndarray[uint8]): packed hashmap table mapping vuln_id to dense index.
        vuln_map_keys (np.ndarray[int32]): array of unique vulnerability ids (hashmap keys).
        aggregate_weights (np.array[VulnerabilityWeight]): vulnerability weights table.
    """
    n_weights = len(aggregate_weights)
    n_entries = len(areaperil_agg_vuln_idx_ja_data)
    if n_weights == 0 or n_entries == 0:
        return

    # Build (areaperil_id, vuln_idx) -> weight hashmap from aggregate_weights.
    # Pre-sized to fit all weights so the load-factor guard never fires; only the
    # rehash path on Robin-Hood collision is kept.
    weight_key_table = np.empty(n_weights, dtype=agg_weight_key_dtype)
    weight_values = np.empty(n_weights, dtype=oasis_float)
    weight_table = hm_init_dict(n_weights)
    w_info, w_lookup, w_index = hm_unpack(weight_table)

    vm_info, vm_lookup, vm_index = hm_unpack(vuln_map)
    n_valid = 0
    for i in range(n_weights):
        rec = aggregate_weights[i]
        vslot = hm_find_key(vm_info, vm_lookup, vm_index, vuln_map_keys, rec['vulnerability_id'])
        if vslot == HM_NOT_FOUND:
            continue
        weight_key_table[n_valid]['areaperil_id'] = nb_areaperil_int(rec['areaperil_id'])
        weight_key_table[n_valid]['vuln_idx'] = nb_oasis_int(vm_index[vslot])
        weight_values[n_valid] = nb_oasis_float(rec['weight'])
        result = hm_try_add_key(w_info, w_lookup, w_index, weight_key_table,
                                weight_key_table[n_valid], n_valid)
        while result == hm_i_add_key_fail:
            weight_table = hm_rehash(weight_table, weight_key_table)
            w_info, w_lookup, w_index = hm_unpack(weight_table)
            result = hm_try_add_key(w_info, w_lookup, w_index, weight_key_table,
                                    weight_key_table[n_valid], n_valid)
        if result & hm_new_slot_bit:
            n_valid += 1
        else:
            # Duplicate (ap, vuln_idx) in aggregate_weights: preserve the old "last wins"
            # behavior by overwriting the previously-stored weight at its original position.
            weight_values[w_index[result & hm_slot_mask]] = nb_oasis_float(rec['weight'])

    # Apply weights to entries with O(1) lookup each.
    lookup_key = np.empty(1, dtype=agg_weight_key_dtype)[0]
    for j in range(n_entries):
        lookup_key['areaperil_id'] = areaperil_agg_vuln_idx_ja_areaperil_ids[j]
        lookup_key['vuln_idx'] = areaperil_agg_vuln_idx_ja_data[j]['vuln_idx']
        eslot = hm_find_key(w_info, w_lookup, w_index, weight_key_table, lookup_key)
        if eslot != HM_NOT_FOUND:
            areaperil_agg_vuln_idx_ja_data[j]['weight'] = weight_values[w_index[eslot]]


def get_vuln_rngadj(run_dir, vuln_map, vuln_map_keys):
    """
    Loads vulnerability adjustments from the analysis settings file.

    Args:
        run_dir (str): path to the run directory (used to load the analysis settings)
        vuln_map (np.ndarray[uint8]): packed hashmap table mapping vuln_id to dense index.
        vuln_map_keys (np.ndarray[int32]): array of unique vulnerability ids (hashmap keys).

    Returns: (np.ndarray[oasis_float]) vulnerability adjustments array, indexed by dense vuln index.
    """
    settings_path = os.path.join(run_dir, "analysis_settings.json")
    vuln_adj = np.ones(len(vuln_map_keys), dtype=oasis_float)
    if not os.path.exists(settings_path):
        logger.debug(f"analysis_settings.json not found in {run_dir}.")
        return vuln_adj
    vulnerability_adjustments_field = analysis_settings_loader(settings_path).get('vulnerability_adjustments', None)
    if vulnerability_adjustments_field is not None:
        adjustments = vulnerability_adjustments_field.get('adjustments', None)
    else:
        adjustments = None
    if adjustments is None:
        logger.debug(f"vulnerability_adjustments not found in {settings_path}.")
        return vuln_adj
    hm_info, hm_lookup, hm_index = hm_unpack(vuln_map)
    for key, value in adjustments.items():
        slot = hm_find_key(hm_info, hm_lookup, hm_index, vuln_map_keys, np.int32(int(key)))
        if slot != HM_NOT_FOUND:
            idx = hm_index[slot]
            vuln_adj[idx] = nb_oasis_float(value)
    return vuln_adj
