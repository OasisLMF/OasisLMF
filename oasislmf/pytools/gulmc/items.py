"""
This file contains specific functionality to read and process items files.
"""
import logging
import os

import numpy as np
import numba as nb
from numba.types import int32 as nb_int32

from oasislmf.pytools.common.data import areaperil_int, nb_areaperil_int, nb_oasis_int, oasis_int, items_dtype
from oasislmf.pytools.common.id_index import build as id_index_build, get_idx, NOT_FOUND
from oasislmf.pytools.common.hashmap import (
    init_dict, unpack, rehash, _try_add_key,
    i_add_key_fail, new_slot_bit, slot_mask,
    NOT_FOUND as HM_NOT_FOUND,
    HM_INFO_N_VALID,
)
from oasislmf.pytools.getmodel.manager import encode_peril_id
from oasislmf.pytools.gulmc.common import ItemAdjustment, agg_vuln_idx_weight_dtype


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
    """Generate item_map as a two-level jagged array; requires items to be sorted.

    Items must be sorted by (areaperil_id, vulnerability_id). The function builds a
    two-level CSR structure that replaces the former Numba Dict item_map and areaperil_ids_map:

        Level 0: areaperil_id → areaperil_ind  (via id_index)
        Level 1: areaperil_ind → vuln_ids      (areaperil_to_vuln_ja_offsets / areaperil_to_vuln_ja_vuln_ids)
        Level 2: pair position → item indices   (areaperil_to_vuln_ja_vuln_ja_offsets / areaperil_to_vuln_ja_vuln_ja_item_idxs)

    Args:
        items (numpy.ndarray): 1-d structured array storing item data, sorted by
            (areaperil_id, vulnerability_id). Must have 'areaperil_agg_vuln_idx' field.
        coverages (numpy.ndarray): coverage id to information on items.
        agg_vuln_id_ja_id_ind (np.array): id_index for aggregate vulnerability ids.
        agg_vuln_id_ja_offsets (np.array[oasis_int]): jagged array offsets for agg_vuln_id_ja_vuln_ids.
        agg_vuln_id_ja_vuln_ids (np.array[oasis_int]): flat jagged array of constituent vulnerability ids.

    Returns:
        areaperil_to_vuln_ja_areaperil_ids (np.array[areaperil_int]): sorted unique areaperil_ids.
        areaperil_to_vuln_ja_offsets (np.array[oasis_int]): L1 CSR offsets (N_areaperil + 1).
        areaperil_to_vuln_ja_vuln_ids (np.array[int32]): vuln_id at each pair position.
        areaperil_to_vuln_ja_vuln_ja_offsets (np.array[oasis_int]): L2 CSR offsets (N_pairs + 1).
        areaperil_to_vuln_ja_vuln_ja_item_idxs (np.array[oasis_int]): flat item indices into items array.
        vuln_map (np.ndarray[uint8]): packed hashmap table mapping vuln_id to dense index.
        vuln_map_keys (np.ndarray[int32]): array of unique vulnerability ids.
        areaperil_agg_vuln_idx_ja_offsets (np.array[oasis_int]): CSR offsets for aggregate vulns.
        areaperil_agg_vuln_idx_ja_data (np.array[agg_vuln_idx_weight_dtype]): merged (vuln_idx, weight) per entry.
        areaperil_agg_vuln_idx_ja_areaperil_ids (np.array[areaperil_int]): areaperil_id per aggregate entry.
    """
    N = len(items)

    # --- Pass 1: count unique areaperil_ids and unique (areaperil_id, vuln_id) pairs ---
    n_unique_areaperils = nb_int32(0)
    n_unique_pairs = nb_int32(0)
    prev_ap = areaperil_int.type(0)
    prev_vuln = nb_int32(-1)
    for j in range(N):
        ap = items[j]['areaperil_id']
        vuln = items[j]['vulnerability_id']
        if ap != prev_ap or vuln != prev_vuln:
            n_unique_pairs += 1
            if ap != prev_ap:
                n_unique_areaperils += 1
            prev_ap = ap
            prev_vuln = vuln

    # --- Allocate jagged array structures ---
    # L0: unique areaperil_ids (for id_index build, done after this function)
    unique_areaperil_ids = np.empty(n_unique_areaperils, dtype=areaperil_int)
    # L1: areaperil → vuln pairs
    areaperil_to_vuln_ja_offsets = np.empty(n_unique_areaperils + 1, dtype=oasis_int)
    areaperil_to_vuln_ja_vuln_ids = np.empty(n_unique_pairs, dtype=np.int32)
    # L2: pair → item indices
    areaperil_to_vuln_ja_vuln_ja_offsets = np.empty(n_unique_pairs + 1, dtype=oasis_int)
    areaperil_to_vuln_ja_vuln_ja_item_idxs = np.empty(N, dtype=oasis_int)

    # --- Hashmap for iterative vuln_id -> dense_index construction ---
    max_unique_vulns = N + len(agg_vuln_id_ja_vuln_ids)
    vuln_key_table = np.empty(max(max_unique_vulns, 1), dtype=np.int32)
    vuln_table = init_dict(max_unique_vulns)
    hm_info, hm_lookup, hm_index = unpack(vuln_table)

    # --- Aggregate vulnerability jagged arrays ---
    n_agg_vulns = len(agg_vuln_id_ja_offsets) - 1
    max_sub_vulns = nb_int32(0)
    for i in range(n_agg_vulns):
        n = agg_vuln_id_ja_offsets[i + 1] - agg_vuln_id_ja_offsets[i]
        if n > max_sub_vulns:
            max_sub_vulns = n
    max_agg_vuln_size = max(N * max_sub_vulns, 1)

    areaperil_agg_vuln_idx_ja_data = np.zeros(max_agg_vuln_size, dtype=agg_vuln_idx_weight_dtype)
    areaperil_agg_vuln_idx_ja_areaperil_ids = np.empty(max_agg_vuln_size, dtype=areaperil_int)
    areaperil_agg_vuln_idx_ja_offsets = np.empty(N + 1, dtype=oasis_int)
    ja_ptr = nb_int32(0)
    n_agg_vuln_groups = nb_int32(0)
    areaperil_agg_vuln_idx_ja_offsets[0] = 0

    # --- Pass 2: build all structures ---
    ap_idx = nb_int32(-1)        # current areaperil index
    pair_idx = nb_int32(-1)      # current pair index
    item_ptr = nb_int32(0)       # pointer into item_idxs
    prev_ap = areaperil_int.type(0)
    prev_vuln = nb_int32(-1)
    last_block_idx = nb_int32(-1)

    areaperil_to_vuln_ja_vuln_ja_offsets[0] = 0

    for j in range(N):
        ap = items[j]['areaperil_id']
        vuln = items[j]['vulnerability_id']

        is_new_pair = (ap != prev_ap or vuln != prev_vuln)
        if is_new_pair:
            # Close previous pair's item range
            if pair_idx >= 0:
                areaperil_to_vuln_ja_vuln_ja_offsets[pair_idx + 1] = item_ptr

            pair_idx += 1
            areaperil_to_vuln_ja_vuln_ids[pair_idx] = vuln

            if ap != prev_ap:
                # Close previous areaperil's vuln range
                if ap_idx >= 0:
                    areaperil_to_vuln_ja_offsets[ap_idx + 1] = pair_idx
                ap_idx += 1
                unique_areaperil_ids[ap_idx] = ap
                areaperil_to_vuln_ja_offsets[ap_idx] = pair_idx

            prev_ap = ap
            prev_vuln = vuln

        # Store item index
        areaperil_to_vuln_ja_vuln_ja_item_idxs[item_ptr] = oasis_int.type(j)
        item_ptr += 1
        coverages[items[j]['coverage_id']]['max_items'] += 1

        # --- Vulnerability hashmap + aggregate processing (unchanged logic) ---
        agg_idx = get_idx(agg_vuln_id_ja_id_ind, vuln)
        is_aggregate = agg_idx != NOT_FOUND

        if is_aggregate:
            if is_new_pair:
                sub_start = agg_vuln_id_ja_offsets[agg_idx]
                sub_end = agg_vuln_id_ja_offsets[agg_idx + 1]
                n_sub = sub_end - sub_start
                last_block_idx = n_agg_vuln_groups
                for si in range(sub_start, sub_end):
                    k = si - sub_start
                    sub_vuln_id = np.int32(agg_vuln_id_ja_vuln_ids[si])
                    result = _try_add_key(hm_info, hm_lookup, hm_index, vuln_key_table, sub_vuln_id)
                    while result == i_add_key_fail:
                        vuln_table = rehash(vuln_table, vuln_key_table)
                        hm_info, hm_lookup, hm_index = unpack(vuln_table)
                        result = _try_add_key(hm_info, hm_lookup, hm_index, vuln_key_table, sub_vuln_id)
                    dense_idx = nb_oasis_int(hm_index[result & slot_mask])
                    areaperil_agg_vuln_idx_ja_data[ja_ptr + k]['vuln_idx'] = dense_idx
                    areaperil_agg_vuln_idx_ja_areaperil_ids[ja_ptr + k] = ap
                ja_ptr += n_sub
                n_agg_vuln_groups += 1
                areaperil_agg_vuln_idx_ja_offsets[n_agg_vuln_groups] = ja_ptr

            items[j]['areaperil_agg_vuln_idx'] = last_block_idx
        else:
            vuln_id_int32 = np.int32(vuln)
            result = _try_add_key(hm_info, hm_lookup, hm_index, vuln_key_table, vuln_id_int32)
            while result == i_add_key_fail:
                vuln_table = rehash(vuln_table, vuln_key_table)
                hm_info, hm_lookup, hm_index = unpack(vuln_table)
                result = _try_add_key(hm_info, hm_lookup, hm_index, vuln_key_table, vuln_id_int32)
            items[j]['vulnerability_idx'] = nb_oasis_int(hm_index[result & slot_mask])

    # Close final pair and areaperil
    if pair_idx >= 0:
        areaperil_to_vuln_ja_vuln_ja_offsets[pair_idx + 1] = item_ptr
    if ap_idx >= 0:
        areaperil_to_vuln_ja_offsets[ap_idx + 1] = pair_idx + 1

    return (unique_areaperil_ids, areaperil_to_vuln_ja_offsets,
            areaperil_to_vuln_ja_vuln_ids, areaperil_to_vuln_ja_vuln_ja_offsets,
            areaperil_to_vuln_ja_vuln_ja_item_idxs,
            vuln_table, vuln_key_table[:hm_info[HM_INFO_N_VALID]],
            areaperil_agg_vuln_idx_ja_offsets[:n_agg_vuln_groups + 1],
            areaperil_agg_vuln_idx_ja_data[:ja_ptr],
            areaperil_agg_vuln_idx_ja_areaperil_ids[:ja_ptr])


@nb.njit(cache=True)
def build_cdf_group_indices(vuln_ja_offsets, vuln_ja_item_idxs, items, dynamic_footprint):
    """Assign a sequential index to each unique CDF-producing group.

    A CDF group is a set of items that share identical vulnerability CDFs. For non-dynamic
    models, each (areaperil, vuln_id) pair — which corresponds to one position in the
    item_map jagged array — gets a single index. For dynamic models, items within the same
    pair may have different intensity_adjustment values that produce different CDFs, so each
    unique adjustment gets its own sub-index.

    Numba compiles two specializations based on whether dynamic_footprint is None or not.

    Args:
        vuln_ja_offsets (np.array[oasis_int]): L2 CSR offsets (N_pairs + 1).
        vuln_ja_item_idxs (np.array[oasis_int]): flat item indices.
        items (np.ndarray): items table (must have 'intensity_adjustment' for dynamic).
        dynamic_footprint: None for static footprints, truthy for dynamic.

    Returns:
        item_cdf_group_idx (np.array[int64]): maps item_idx → CDF group index.
        n_cdf_groups (int): total number of unique CDF groups.
    """
    item_cdf_group_idx = np.empty(len(items), dtype=np.int64)
    n_pairs = len(vuln_ja_offsets) - 1
    cdf_group_cache_id = 0

    if dynamic_footprint is None:
        for k in range(n_pairs):
            start = vuln_ja_offsets[k]
            end = vuln_ja_offsets[k + 1]
            for pos in range(start, end):
                item_cdf_group_idx[vuln_ja_item_idxs[pos]] = cdf_group_cache_id
            cdf_group_cache_id += 1
    else:
        adj_key_storage = np.empty(max(len(items), 1), dtype=oasis_int)
        adj_cache_ids = np.empty(max(len(items), 1), dtype=np.int64)

        for k in range(n_pairs):
            start = vuln_ja_offsets[k]
            end = vuln_ja_offsets[k + 1]
            n_pair_items = end - start
            # fresh hashmap per pair (maps intensity_adjustment → dense index)
            adj_table = init_dict(max(n_pair_items, 1))
            adj_info, adj_lookup, adj_index = unpack(adj_table)

            for pos in range(start, end):
                item_idx = vuln_ja_item_idxs[pos]
                adj = items[item_idx]['intensity_adjustment']

                result = _try_add_key(adj_info, adj_lookup, adj_index, adj_key_storage, adj)
                while result == i_add_key_fail:
                    adj_table = rehash(adj_table, adj_key_storage)
                    adj_info, adj_lookup, adj_index = unpack(adj_table)
                    result = _try_add_key(adj_info, adj_lookup, adj_index, adj_key_storage, adj)

                dense_idx = adj_index[result & slot_mask]
                if result & new_slot_bit:  # new unique adjustment
                    adj_cache_ids[dense_idx] = cdf_group_cache_id
                    cdf_group_cache_id += 1
                item_cdf_group_idx[item_idx] = adj_cache_ids[dense_idx]

    return item_cdf_group_idx, cdf_group_cache_id


def get_dynamic_footprint_adjustments(input_path):
    """Generate intensity adjustment array for dynamic footprint models.

    Args:
        input_path (str): location of the generated adjustments file.

    Returns:
        numpy array with itemid and adjustment factors
    """
    adjustments_fn = os.path.join(input_path, 'item_adjustments.csv')
    if os.path.isfile(adjustments_fn):
        adjustments_tb = np.loadtxt(adjustments_fn, dtype=ItemAdjustment, delimiter=",", skiprows=1, ndmin=1)
    else:
        items_fp = os.path.join(input_path, 'items.csv')
        items_tb = np.loadtxt(items_fp, dtype=items_dtype, delimiter=",", skiprows=1, ndmin=1)
        adjustments_tb = np.array([(i[0], 0, 0) for i in items_tb], dtype=ItemAdjustment)

    return adjustments_tb


def get_peril_id(input_path):
    """
    Get peril_id associated with item_id

    Args:
        input_path (str): The directory path where the 'gul_summary_map.csv' file is located.

    Returns:
        np.ndarray: A structured NumPy array with the following fields:
            - 'item_id' (oasis_int): The item ID as an integer.
            - 'peril_id' (oasis_int): The encoded peril ID as an integer.
    """
    from oasislmf.pytools.common.data import load_as_ndarray

    read_dtype = np.dtype([('item_id', oasis_int), ('peril_id', 'U3')])
    raw = load_as_ndarray(input_path, 'gul_summary_map', read_dtype,
                          col_map={'item_id': 'item_id', 'peril_id': 'peril_id'})

    result = np.empty(len(raw), dtype=np.dtype([('item_id', oasis_int), ('peril_id', oasis_int)]))
    result['item_id'] = raw['item_id']
    result['peril_id'] = np.vectorize(encode_peril_id)(raw['peril_id'])

    return result
