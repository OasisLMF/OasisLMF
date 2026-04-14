"""
This file contains specific functionality to read and process items files.
"""
import logging
import os

import numpy as np
import numba as nb
from numba.typed import Dict, List
from numba.types import int32 as nb_int32
from numba.types import int8 as nb_int8

from oasislmf.pytools.common.data import areaperil_int, nb_areaperil_int, nb_oasis_int, oasis_float, oasis_int, items_dtype
from oasislmf.pytools.common.id_index import get_idx, NOT_FOUND
from oasislmf.pytools.common.hashmap import (
    init_dict, unpack, rehash, _try_add_key,
    index_dtype, i_add_key_fail, new_slot_bit,
    NOT_FOUND as HM_NOT_FOUND,
)
from oasislmf.pytools.gul.utils import append_to_dict_value
from oasislmf.pytools.gulmc.common import ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE


logger = logging.getLogger(__name__)


def read_items(input_path, ignore_file_type=set(), dynamic_footprint=False, legacy=False):
    """Load the items from the items file.

    Args:
        input_path (str): the path pointing to the file
        ignore_file_type (Set[str]): file extension to ignore when loading.

    Returns:
        Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]]
          vulnerability dictionary, vulnerability IDs, areaperil to vulnerability index dictionary,
          areaperil ID to vulnerability index array, areaperil ID to vulnerability array
    """
    input_files = set(os.listdir(input_path))

    if "items.bin" in input_files and "bin" not in ignore_file_type:
        items_fname = os.path.join(input_path, 'items.bin')
        logger.debug(f"loading {items_fname}")
        items = np.memmap(items_fname, dtype=items_dtype, mode='r')

    elif "items.csv" in input_files and "csv" not in ignore_file_type:
        items_fname = os.path.join(input_path, 'items.csv')
        logger.debug(f"loading {items_fname}")
        items = np.loadtxt(items_fname, dtype=items_dtype, delimiter=",", skiprows=1, ndmin=1)

    else:
        raise FileNotFoundError(f'items file not found at {input_path}')

    return items


@nb.njit(cache=True, fastmath=True)
def generate_item_map(items, coverages,
                      agg_vuln_id_ja_id_ind, agg_vuln_id_ja_offsets, agg_vuln_id_ja_vuln_ids):
    """Generate item_map; requires items to be sorted.

    Args:
        items (numpy.ndarray): 1-d structured array storing item data.
            items need to be sorted by increasing areaperil_id, vulnerability_id
            in order to output the items in correct order. Must have an 'areaperil_agg_vuln_idx' field.
        coverages (numpy.ndarray): coverage id to information on items
        agg_vuln_id_ja_id_ind (np.array[oasis_int]): id_index structure built from sorted aggregate vulnerability ids.
        agg_vuln_id_ja_offsets (np.array[oasis_int]): jagged array offsets for agg_vuln_id_ja_vuln_ids.
        agg_vuln_id_ja_vuln_ids (np.array[oasis_int]): flat jagged array of constituent vulnerability ids.

    Returns:
        item_map (Dict[ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE]): dict storing
            the mapping between areaperil_id, vulnerability_id to item.
        areaperil_ids_map (Dict[int, Dict[int, int]]) dict storing the mapping between each
            areaperil_id and all the vulnerability ids associated with it.
        vuln_map (np.ndarray[uint8]): packed hashmap table mapping vuln_id to dense index.
        vuln_map_keys (np.ndarray[int32]): array of unique vulnerability ids (hashmap keys, trimmed to n_unique).
        areaperil_agg_vuln_idx_ja_offsets (np.array[oasis_int]): jagged array offsets. Block i spans
            areaperil_agg_vuln_idx_ja_vuln_idxs[offsets[i]:offsets[i+1]].
        areaperil_agg_vuln_idx_ja_vuln_idxs (np.array[oasis_int]): flat jagged array of dense vulnerability
            indices for aggregate vulnerabilities.
        areaperil_agg_vuln_idx_ja_weights (np.array[oasis_float]): flat jagged array of vulnerability weights,
            aligned with vuln_idxs. Initialized to 0, populated later by process_vulnerability_weights.
        areaperil_agg_vuln_idx_ja_areaperil_ids (np.array[areaperil_int]): areaperil_id for each entry,
            parallel to vuln_idxs. Used by process_vulnerability_weights to locate weight slots.

    """
    # --- Hashmap for iterative vuln_id -> dense_index construction ---
    # Upper bound on unique vuln_ids: all non-aggregate + all aggregate sub-vulns
    max_unique_vulns = len(items) + len(agg_vuln_id_ja_vuln_ids)
    vuln_key_table = np.empty(max(max_unique_vulns, 1), dtype=np.int32)
    vuln_table = init_dict(max_unique_vulns)
    hm_info, hm_lookup, hm_index = unpack(vuln_table)
    n_unique = index_dtype(0)

    # --- Main pass: build all structures, assign dense indices on the fly ---
    item_map = Dict.empty(ITEM_MAP_KEY_TYPE, List.empty_list(ITEM_MAP_VALUE_TYPE))
    areaperil_ids_map = Dict.empty(nb_areaperil_int, Dict.empty(nb_int32, nb_int8))

    # Compute upper bound on jagged array size for aggregate vulnerabilities.
    n_agg_vulns = len(agg_vuln_id_ja_offsets) - 1
    max_sub_vulns = nb_int32(0)
    for i in range(n_agg_vulns):
        n = agg_vuln_id_ja_offsets[i + 1] - agg_vuln_id_ja_offsets[i]
        if n > max_sub_vulns:
            max_sub_vulns = n
    max_agg_vuln_size = max(len(items) * max_sub_vulns, 1)

    # Jagged arrays for aggregate vulnerability dense indices and weights
    areaperil_agg_vuln_idx_ja_vuln_idxs = np.empty(max_agg_vuln_size, dtype=oasis_int)
    areaperil_agg_vuln_idx_ja_weights = np.zeros(max_agg_vuln_size, dtype=oasis_float)
    areaperil_agg_vuln_idx_ja_areaperil_ids = np.empty(max_agg_vuln_size, dtype=areaperil_int)
    areaperil_agg_vuln_idx_ja_offsets = np.empty(len(items) + 1, dtype=oasis_int)
    ja_ptr = nb_int32(0)
    n_agg_vuln_groups = nb_int32(0)
    areaperil_agg_vuln_idx_ja_offsets[0] = 0

    # Track last (areaperil_id, vulnerability_id) for deduplication.
    # Items are sorted by (areaperil_id, vulnerability_id), so duplicates are contiguous.
    last_areaperil_id = nb_areaperil_int(0)
    last_vulnerability_id = nb_int32(-1)
    last_block_idx = nb_int32(-1)

    for j, item in enumerate(items):
        areaperil_id = item['areaperil_id']
        vulnerability_id = item['vulnerability_id']

        append_to_dict_value(item_map, tuple((areaperil_id, vulnerability_id)), j, ITEM_MAP_VALUE_TYPE)
        coverages[item['coverage_id']]['max_items'] += 1

        # Check if this vulnerability_id is an aggregate (id_index lookup)
        agg_idx = get_idx(agg_vuln_id_ja_id_ind, vulnerability_id)
        is_aggregate = agg_idx != NOT_FOUND

        if is_aggregate:
            is_new_pair = (areaperil_id != last_areaperil_id or vulnerability_id != last_vulnerability_id)
            if is_new_pair:
                sub_start = agg_vuln_id_ja_offsets[agg_idx]
                sub_end = agg_vuln_id_ja_offsets[agg_idx + 1]
                n_sub = sub_end - sub_start
                last_block_idx = n_agg_vuln_groups
                last_areaperil_id = areaperil_id
                last_vulnerability_id = vulnerability_id
                for si in range(sub_start, sub_end):
                    k = si - sub_start
                    sub_vuln_id = np.int32(agg_vuln_id_ja_vuln_ids[si])
                    # Insert sub-vuln_id into hashmap, get dense index directly
                    vuln_key_table[n_unique] = sub_vuln_id
                    if hm_info[1] >= hm_info[2]:
                        vuln_table = rehash(vuln_table, vuln_key_table)
                        hm_info, hm_lookup, hm_index = unpack(vuln_table)
                    result = _try_add_key(hm_info, hm_lookup, hm_index, vuln_key_table, n_unique)
                    while result == i_add_key_fail:
                        vuln_table = rehash(vuln_table, vuln_key_table)
                        hm_info, hm_lookup, hm_index = unpack(vuln_table)
                        result = _try_add_key(hm_info, hm_lookup, hm_index, vuln_key_table, n_unique)
                    if result & new_slot_bit:
                        dense_idx = nb_oasis_int(n_unique)
                        n_unique += index_dtype(1)
                    else:
                        dense_idx = nb_oasis_int(hm_index[result])
                    areaperil_agg_vuln_idx_ja_vuln_idxs[ja_ptr + k] = dense_idx
                    areaperil_agg_vuln_idx_ja_areaperil_ids[ja_ptr + k] = areaperil_id
                ja_ptr += n_sub
                n_agg_vuln_groups += 1
                areaperil_agg_vuln_idx_ja_offsets[n_agg_vuln_groups] = ja_ptr

            item['areaperil_agg_vuln_idx'] = last_block_idx
        else:
            # Insert vuln_id into hashmap, get dense index directly
            vuln_key_table[n_unique] = np.int32(vulnerability_id)
            if hm_info[1] >= hm_info[2]:
                vuln_table = rehash(vuln_table, vuln_key_table)
                hm_info, hm_lookup, hm_index = unpack(vuln_table)
            result = _try_add_key(hm_info, hm_lookup, hm_index, vuln_key_table, n_unique)
            while result == i_add_key_fail:
                vuln_table = rehash(vuln_table, vuln_key_table)
                hm_info, hm_lookup, hm_index = unpack(vuln_table)
                result = _try_add_key(hm_info, hm_lookup, hm_index, vuln_key_table, n_unique)
            if result & new_slot_bit:
                item['vulnerability_idx'] = nb_oasis_int(n_unique)
                n_unique += index_dtype(1)
            else:
                item['vulnerability_idx'] = nb_oasis_int(hm_index[result])

        if areaperil_id not in areaperil_ids_map:
            areaperil_ids_map[areaperil_id] = Dict.empty(nb_int32, nb_int8)
        areaperil_ids_map[areaperil_id][vulnerability_id] = 0

    return (item_map, areaperil_ids_map, vuln_table, vuln_key_table[:n_unique],
            areaperil_agg_vuln_idx_ja_offsets[:n_agg_vuln_groups + 1],
            areaperil_agg_vuln_idx_ja_vuln_idxs[:ja_ptr], areaperil_agg_vuln_idx_ja_weights[:ja_ptr],
            areaperil_agg_vuln_idx_ja_areaperil_ids[:ja_ptr])
