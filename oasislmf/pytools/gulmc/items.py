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
def generate_item_map(items, coverages, valid_areaperil_id,
                      agg_vuln_id_ja_id_ind, agg_vuln_id_ja_offsets, agg_vuln_id_ja_vuln_ids):
    """Generate item_map; requires items to be sorted.

    Args:
        items (numpy.ndarray): 1-d structured array storing item data.
            items need to be sorted by increasing areaperil_id, vulnerability_id
            in order to output the items in correct order. Must have an 'areaperil_agg_vuln_idx' field.
        coverages (numpy.ndarray): coverage id to information on items
        valid_areaperil_id (numpy.ndarray[int32]): list of non-filtered area_peril_id (None is no filter)
        agg_vuln_id_ja_id_ind (np.array[oasis_int]): id_index structure built from sorted aggregate vulnerability ids.
        agg_vuln_id_ja_offsets (np.array[oasis_int]): jagged array offsets for agg_vuln_id_ja_vuln_ids.
        agg_vuln_id_ja_vuln_ids (np.array[oasis_int]): flat jagged array of constituent vulnerability ids.

    Returns:
        item_map (Dict[ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE]): dict storing
            the mapping between areaperil_id, vulnerability_id to item.
        areaperil_ids_map (Dict[int, Dict[int, int]]) dict storing the mapping between each
            areaperil_id and all the vulnerability ids associated with it.
        vuln_dict (Dict[int, int]): vuln id to vuln idx for each vulnerability.
        areaperil_agg_vuln_idx_ja_offsets (np.array[oasis_int]): jagged array offsets. Block i spans
            areaperil_agg_vuln_idx_ja_vuln_idxs[offsets[i]:offsets[i+1]].
        areaperil_agg_vuln_idx_ja_vuln_idxs (np.array[oasis_int]): flat jagged array of dense vulnerability
            indices for aggregate vulnerabilities.
        areaperil_agg_vuln_idx_ja_weights (np.array[oasis_float]): flat jagged array of vulnerability weights,
            aligned with vuln_idxs. Initialized to 0, populated later by process_vulnerability_weights.
        areaperil_agg_vuln_idx_ja_areaperil_ids (np.array[areaperil_int]): areaperil_id for each entry,
            parallel to vuln_idxs. Used by process_vulnerability_weights to locate weight slots.

    """
    item_map = Dict.empty(ITEM_MAP_KEY_TYPE, List.empty_list(ITEM_MAP_VALUE_TYPE))
    areaperil_ids_map = Dict.empty(nb_areaperil_int, Dict.empty(nb_int32, nb_int8))
    vuln_dict = Dict()
    vuln_idx = 0

    # Compute upper bound on jagged array size for aggregate vulnerabilities.
    # Each unique (areaperil_id, agg_vuln_id) pair contributes len(sub_vulns) entries.
    # Upper bound: number of items * max sub-vulns per aggregate.
    n_agg_vulns = len(agg_vuln_id_ja_offsets) - 1
    max_sub_vulns = nb_int32(0)
    for i in range(n_agg_vulns):
        n = agg_vuln_id_ja_offsets[i + 1] - agg_vuln_id_ja_offsets[i]
        if n > max_sub_vulns:
            max_sub_vulns = n
    max_agg_vuln_size = max(len(items) * max_sub_vulns, 1)

    # Jagged arrays for aggregate vulnerability indices and weights
    areaperil_agg_vuln_idx_ja_vuln_idxs = np.empty(max_agg_vuln_size, dtype=oasis_int)
    areaperil_agg_vuln_idx_ja_weights = np.zeros(max_agg_vuln_size, dtype=oasis_float)
    areaperil_agg_vuln_idx_ja_areaperil_ids = np.empty(max_agg_vuln_size, dtype=areaperil_int)
    # Jagged array offsets: one entry per block + final sentinel. Max blocks = len(items).
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

        if valid_areaperil_id is not None and item['areaperil_id'] not in valid_areaperil_id:
            continue

        append_to_dict_value(item_map, tuple((areaperil_id, vulnerability_id)), j, ITEM_MAP_VALUE_TYPE)
        coverages[item['coverage_id']]['max_items'] += 1

        # Check if this vulnerability_id is an aggregate (id_index lookup)
        agg_idx = get_idx(agg_vuln_id_ja_id_ind, vulnerability_id)
        is_aggregate = agg_idx != NOT_FOUND

        # Populate vuln_dict, to map all used vuln_id to vuln_i
        if item['vulnerability_id'] not in vuln_dict:
            if is_aggregate:
                sub_start = agg_vuln_id_ja_offsets[agg_idx]
                sub_end = agg_vuln_id_ja_offsets[agg_idx + 1]
                for si in range(sub_start, sub_end):
                    sub_vuln_id = agg_vuln_id_ja_vuln_ids[si]
                    if sub_vuln_id not in vuln_dict:
                        vuln_dict[sub_vuln_id] = nb_oasis_int(vuln_idx)
                        vuln_idx += 1
            else:
                vuln_dict[item['vulnerability_id']] = nb_oasis_int(vuln_idx)
                vuln_idx += 1

        if is_aggregate:
            is_new_pair = (areaperil_id != last_areaperil_id or vulnerability_id != last_vulnerability_id)
            if is_new_pair:
                # First time seeing this (areaperil, agg_vuln) — allocate a jagged array block
                sub_start = agg_vuln_id_ja_offsets[agg_idx]
                sub_end = agg_vuln_id_ja_offsets[agg_idx + 1]
                n_sub = sub_end - sub_start
                last_block_idx = n_agg_vuln_groups
                last_areaperil_id = areaperil_id
                last_vulnerability_id = vulnerability_id
                for si in range(sub_start, sub_end):
                    k = si - sub_start
                    areaperil_agg_vuln_idx_ja_vuln_idxs[ja_ptr + k] = vuln_dict[agg_vuln_id_ja_vuln_ids[si]]
                    areaperil_agg_vuln_idx_ja_areaperil_ids[ja_ptr + k] = areaperil_id
                ja_ptr += n_sub
                n_agg_vuln_groups += 1
                areaperil_agg_vuln_idx_ja_offsets[n_agg_vuln_groups] = ja_ptr

            item['areaperil_agg_vuln_idx'] = last_block_idx
        else:
            item['vulnerability_idx'] = vuln_dict[item['vulnerability_id']]

        if areaperil_id not in areaperil_ids_map:
            areaperil_ids_map[areaperil_id] = Dict.empty(nb_int32, nb_int8)
        areaperil_ids_map[areaperil_id][vulnerability_id] = 0

    return (item_map, areaperil_ids_map, vuln_dict,
            areaperil_agg_vuln_idx_ja_offsets[:n_agg_vuln_groups + 1],
            areaperil_agg_vuln_idx_ja_vuln_idxs[:ja_ptr], areaperil_agg_vuln_idx_ja_weights[:ja_ptr],
            areaperil_agg_vuln_idx_ja_areaperil_ids[:ja_ptr])
