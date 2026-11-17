"""This file contains specific functionality to read and process items files."""
import logging
import os

import numpy as np
import numba as nb
from numba.types import int32 as nb_int32

from oasislmf.pytools.common.data import areaperil_int, nb_oasis_int, oasis_int, items_dtype
from oasislmf.pytools.common.blended_areaperil import select_placement
from oasislmf.pytools.gul.random import PHILOX_SHIFT32, PHILOX_U32_MASK, _lh_philox_block
from oasislmf.pytools.common.id_index import get_idx, NOT_FOUND
from oasislmf.pytools.common.hashmap import (
    init_dict, unpack, rehash, _try_add_key,
    i_add_key_fail, new_slot_bit, slot_mask,
    HM_INFO_N_VALID,
)
from oasislmf.pytools.getmodel.manager import encode_peril_id
from oasislmf.pytools.gulmc.common import ItemAdjustment, agg_vuln_idx_weight_dtype


logger = logging.getLogger(__name__)


def read_items(input_path, ignore_file_type=set()):
    """Load the items from the items file.

    Args:
        input_path (str): the path pointing to the file
        ignore_file_type (Set[str]): file extension to ignore when loading.

    Returns:
        np.ndarray[items_dtype]: items table loaded from items.bin (memmap'd) or items.csv.
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


def build_item_placements(items, item_location_keys, item_building_ids,
                          blended_areaperil_placement_ja_offsets, blended_areaperil_placement_ja_cum_weights,
                          blended_areaperil_placement_ja_areaperil_j, sample_size):
    """Draw each building's placement for every sample, and record which areaperil each item reads.

    Placement is where a building stands, which does not change from one event to the next (D8),
    so it is drawn once per run rather than per event: one Latin Hypercube block of ``sample_size``
    uniforms per (location, building), from a key that has no event in it. Every item of the
    building maps the same uniforms through its own blended areaperil, whose cumulative weights the
    load validation guarantees are identical across the location's perils -- so a building's wind
    and surge items always agree on the placement, and only the areaperil each reads differs.

    Only items whose blended areaperil has more than one placement get a run; every other item has
    an empty one, which reads as areaperil_j 0 for every sample.

    Args:
        items (np.ndarray): items table with 'blended_areaperil_idx' and 'packed_buildings'.
        item_location_keys (np.ndarray[uint64]): per item, its location's placement key.
        item_building_ids (np.ndarray[int]): per item, its ``building_id`` (1 unless the location
            was disaggregated into one item per building).
        blended_areaperil_placement_ja_offsets (np.ndarray[oasis_int]): placements per blended areaperil.
        blended_areaperil_placement_ja_cum_weights (np.ndarray[float64]): running placement weight.
        blended_areaperil_placement_ja_areaperil_j (np.ndarray[int32]): placement -> j, -1 uncovered.
        sample_size (int): number of samples S.

    Returns:
        tuple: ``item_placement_ja_offsets`` (N_items + 1, int64) and
        ``item_placement_ja_areaperil_j`` (flat, int8 or int16): per item, per building b and sample
        s at ``offsets[item_idx] + (b - 1) * S + (s - 1)``, the j of the item areaperil it reads, or
        -1 when no areaperil covers the placement (or float weights leave it unplaced).
    """
    n_items = items.shape[0]
    blended_idx = items['blended_areaperil_idx']
    n_placements = (blended_areaperil_placement_ja_offsets[blended_idx + 1]
                    - blended_areaperil_placement_ja_offsets[blended_idx])
    n_buildings = np.abs(items['packed_buildings']).astype(np.int64)
    run_length = np.where(n_placements > 1, n_buildings * sample_size, 0)
    item_placement_ja_offsets = np.zeros(n_items + 1, dtype=np.int64)
    np.cumsum(run_length, out=item_placement_ja_offsets[1:])

    max_j = int(blended_areaperil_placement_ja_areaperil_j.max(initial=0))
    j_dtype = np.int8 if max_j < np.iinfo(np.int8).max else np.int16
    item_placement_ja_areaperil_j = np.empty(item_placement_ja_offsets[-1], dtype=j_dtype)
    if item_placement_ja_offsets[-1] > 0:
        _fill_item_placements(blended_idx, n_buildings, item_location_keys,
                              item_building_ids.astype(np.int64), item_placement_ja_offsets,
                              blended_areaperil_placement_ja_offsets, blended_areaperil_placement_ja_cum_weights,
                              blended_areaperil_placement_ja_areaperil_j, sample_size, item_placement_ja_areaperil_j)
    return item_placement_ja_offsets, item_placement_ja_areaperil_j


@nb.njit(cache=True)
def _fill_item_placements(blended_idx, n_buildings, item_location_keys, item_building_ids,
                          item_placement_ja_offsets,
                          placement_ja_offsets, placement_ja_cum_weights, placement_ja_areaperil_j,
                          sample_size, item_placement_ja_areaperil_j):
    """Kernel of ``build_item_placements``: one Philox block per (item, building), mapped to j."""
    uniforms = np.empty(sample_size, dtype=np.float64)
    perm_scratch = np.empty(sample_size, dtype=np.float64)
    for item_idx in range(blended_idx.shape[0]):
        start = item_placement_ja_offsets[item_idx]
        if item_placement_ja_offsets[item_idx + 1] == start:
            continue
        b_idx = blended_idx[item_idx]
        p_start = placement_ja_offsets[b_idx]
        p_end = placement_ja_offsets[b_idx + 1]
        cum_weights = placement_ja_cum_weights[p_start:p_end]
        key = item_location_keys[item_idx]
        k0 = np.uint32(key & PHILOX_U32_MASK)
        k1 = np.uint32(key >> PHILOX_SHIFT32)
        for b in range(n_buildings[item_idx]):
            # the building's number within its location: building_id for one item per building,
            # the packed building index for one item carrying them all
            _lh_philox_block(k0, k1, item_building_ids[item_idx] - 1 + b, sample_size, perm_scratch, uniforms)
            out = start + b * sample_size
            for s in range(sample_size):
                p = select_placement(cum_weights, uniforms[s])
                item_placement_ja_areaperil_j[out + s] = placement_ja_areaperil_j[p_start + p] if p < p_end - p_start else -1


def build_item_areaperils(items, blended_areaperil_ja_offsets, blended_areaperil_ja_areaperil_ids):
    """Expand each item into its item areaperils and sort them into item map order.

    An item reads one item areaperil per distinct areaperil of its blended areaperil; item
    areaperil ``item_areaperil_ja_offsets[item_idx] + j`` is the item's j-th. The item map visits
    them grouped by areaperil, then by vulnerability, then by item: with K = 1 everywhere that is
    exactly the (areaperil_id, vulnerability_id) order the items are sorted in, so traversal order,
    rng assignment and CDF group numbering are unchanged.

    Args:
        items (np.ndarray): items table, sorted by (areaperil_id, vulnerability_id), with
            'blended_areaperil_idx'.
        blended_areaperil_ja_offsets (np.ndarray[oasis_int]): CSR offsets over a blended
            areaperil's areaperils.
        blended_areaperil_ja_areaperil_ids (np.ndarray[areaperil_int]): footprint areaperil ids.

    Returns:
        tuple: ``item_areaperil_ja_offsets`` (N_items + 1), and the item areaperils in item map
        order as four parallel arrays: areaperil id, vulnerability id, item index and item
        areaperil index.
    """
    n_items = items.shape[0]
    blended_idx = items['blended_areaperil_idx']
    item_k = (blended_areaperil_ja_offsets[blended_idx + 1] - blended_areaperil_ja_offsets[blended_idx]).astype(np.int64)
    item_areaperil_ja_offsets = np.zeros(n_items + 1, dtype=oasis_int)
    np.cumsum(item_k, out=item_areaperil_ja_offsets[1:])
    n_item_areaperils = int(item_areaperil_ja_offsets[-1])

    entry_item_idx = np.repeat(np.arange(n_items, dtype=oasis_int), item_k)
    # j of each item areaperil: its position within its item's run
    entry_j = np.arange(n_item_areaperils, dtype=np.int64) - item_areaperil_ja_offsets[entry_item_idx]
    entry_areaperil_id = blended_areaperil_ja_areaperil_ids[blended_areaperil_ja_offsets[blended_idx[entry_item_idx]] + entry_j]
    entry_vuln_id = items['vulnerability_id'][entry_item_idx]

    order = np.lexsort((entry_item_idx, entry_vuln_id, entry_areaperil_id))
    return (item_areaperil_ja_offsets,
            np.ascontiguousarray(entry_areaperil_id[order]),
            np.ascontiguousarray(entry_vuln_id[order]),
            np.ascontiguousarray(entry_item_idx[order]),
            order.astype(oasis_int))


@nb.njit(cache=True, fastmath=True)
def generate_item_map(entry_areaperil_ids, entry_vuln_ids, entry_item_idxs, entry_item_areaperil_idxs,
                      items, n_item_areaperils,
                      agg_vuln_id_ja_id_ind, agg_vuln_id_ja_offsets, agg_vuln_id_ja_vuln_ids):
    """Build the item map from the item areaperils, and assign the dense vulnerability indices.

    The item map is a jagged array from each footprint areaperil an item can read to the item
    areaperils that read it::

        item_map_ja_id_ind:      areaperil_id -> areaperil index        (built by the caller)
        item_map_ja_offsets:     areaperil index -> range of entries
        item_map_ja_item_idxs / item_map_ja_item_areaperil_idxs: one entry per item areaperil

    The entries must already be in item map order (``build_item_areaperils``). Dense vulnerability
    indices are assigned in that order, so with K = 1 they come out as before.

    Args:
        entry_areaperil_ids (np.array[areaperil_int]): areaperil id per entry, in item map order.
        entry_vuln_ids (np.array[int32]): vulnerability id per entry.
        entry_item_idxs (np.array[oasis_int]): item index per entry.
        entry_item_areaperil_idxs (np.array[oasis_int]): item areaperil index per entry.
        items (np.ndarray): items table; 'vulnerability_idx' is written for non-aggregate items.
        n_item_areaperils (int): number of item areaperils.
        agg_vuln_id_ja_id_ind (np.array): id_index for aggregate vulnerability ids.
        agg_vuln_id_ja_offsets (np.array[oasis_int]): jagged array offsets for agg_vuln_id_ja_vuln_ids.
        agg_vuln_id_ja_vuln_ids (np.array[oasis_int]): flat jagged array of constituent vulnerability ids.

    Returns:
        item_map_ja_areaperil_ids (np.array[areaperil_int]): sorted unique footprint areaperil ids.
        item_map_ja_offsets (np.array[oasis_int]): CSR offsets (N_areaperil + 1).
        item_map_ja_item_idxs (np.array[oasis_int]): item index per entry.
        item_map_ja_item_areaperil_idxs (np.array[oasis_int]): item areaperil index per entry.
        item_areaperil_ja_areaperil_agg_vuln_idxs (np.array[oasis_int]): per item areaperil, its
            block in ``areaperil_agg_vuln_idx_ja_*``, or -1 when its vulnerability is not aggregate.
        vuln_map (np.ndarray[uint8]): packed hashmap table mapping vuln_id to dense index.
        vuln_map_keys (np.ndarray[int32]): array of unique vulnerability ids.
        areaperil_agg_vuln_idx_ja_offsets (np.array[oasis_int]): CSR offsets for aggregate vulns.
        areaperil_agg_vuln_idx_ja_data (np.array[agg_vuln_idx_weight_dtype]): merged (vuln_idx, weight) per entry.
        areaperil_agg_vuln_idx_ja_areaperil_ids (np.array[areaperil_int]): areaperil_id per aggregate entry.
    """
    n_entries = entry_areaperil_ids.shape[0]

    # --- Pass 1: count areaperils, and size the aggregate-vuln output exactly -----------------
    n_unique_areaperils = nb_int32(0)
    n_agg_pair_groups = nb_int32(0)    # number of unique (areaperil, agg_vuln_id) blocks
    n_agg_entries_total = nb_int32(0)  # total sub-vuln entries written across all blocks
    for e in range(n_entries):
        new_areaperil = e == 0 or entry_areaperil_ids[e] != entry_areaperil_ids[e - 1]
        if new_areaperil:
            n_unique_areaperils += 1
        if new_areaperil or entry_vuln_ids[e] != entry_vuln_ids[e - 1]:
            agg_idx = get_idx(agg_vuln_id_ja_id_ind, entry_vuln_ids[e])
            if agg_idx != NOT_FOUND:
                n_agg_pair_groups += 1
                n_agg_entries_total += nb_int32(agg_vuln_id_ja_offsets[agg_idx + 1] - agg_vuln_id_ja_offsets[agg_idx])

    item_map_ja_areaperil_ids = np.empty(n_unique_areaperils, dtype=areaperil_int)
    item_map_ja_offsets = np.empty(n_unique_areaperils + 1, dtype=oasis_int)
    item_map_ja_item_idxs = np.empty(n_entries, dtype=oasis_int)
    item_map_ja_item_areaperil_idxs = np.empty(n_entries, dtype=oasis_int)
    item_areaperil_ja_areaperil_agg_vuln_idxs = np.full(n_item_areaperils, -1, dtype=oasis_int)

    # --- Hashmap for iterative vuln_id -> dense_index construction ---------------------------
    max_unique_vulns = n_entries + len(agg_vuln_id_ja_vuln_ids)
    vuln_key_table = np.empty(max(max_unique_vulns, 1), dtype=np.int32)
    vuln_table = init_dict(max_unique_vulns)
    hm_info, hm_lookup, hm_index = unpack(vuln_table)

    # --- Aggregate vulnerability jagged arrays (sized exactly from pass 1) -------------------
    agg_alloc = max(n_agg_entries_total, 1)  # 0 → 1 to keep array shapes valid
    areaperil_agg_vuln_idx_ja_data = np.zeros(agg_alloc, dtype=agg_vuln_idx_weight_dtype)
    areaperil_agg_vuln_idx_ja_areaperil_ids = np.empty(agg_alloc, dtype=areaperil_int)
    areaperil_agg_vuln_idx_ja_offsets = np.empty(n_agg_pair_groups + 1, dtype=oasis_int)
    ja_ptr = nb_int32(0)
    n_agg_vuln_groups = nb_int32(0)
    areaperil_agg_vuln_idx_ja_offsets[0] = 0

    # --- Pass 2: build all structures ---------------------------------------------------------
    ap_idx = nb_int32(-1)
    last_block_idx = nb_int32(-1)
    for e in range(n_entries):
        ap = entry_areaperil_ids[e]
        vuln = entry_vuln_ids[e]
        new_areaperil = e == 0 or ap != entry_areaperil_ids[e - 1]
        is_new_pair = new_areaperil or vuln != entry_vuln_ids[e - 1]
        if new_areaperil:
            ap_idx += 1
            item_map_ja_areaperil_ids[ap_idx] = ap
            item_map_ja_offsets[ap_idx] = e

        item_idx = entry_item_idxs[e]
        item_map_ja_item_idxs[e] = item_idx
        item_map_ja_item_areaperil_idxs[e] = entry_item_areaperil_idxs[e]

        agg_idx = get_idx(agg_vuln_id_ja_id_ind, vuln)
        if agg_idx != NOT_FOUND:
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
            item_areaperil_ja_areaperil_agg_vuln_idxs[entry_item_areaperil_idxs[e]] = last_block_idx
        else:
            vuln_id_int32 = np.int32(vuln)
            result = _try_add_key(hm_info, hm_lookup, hm_index, vuln_key_table, vuln_id_int32)
            while result == i_add_key_fail:
                vuln_table = rehash(vuln_table, vuln_key_table)
                hm_info, hm_lookup, hm_index = unpack(vuln_table)
                result = _try_add_key(hm_info, hm_lookup, hm_index, vuln_key_table, vuln_id_int32)
            items[item_idx]['vulnerability_idx'] = nb_oasis_int(hm_index[result & slot_mask])

    item_map_ja_offsets[n_unique_areaperils] = n_entries

    return (item_map_ja_areaperil_ids, item_map_ja_offsets,
            item_map_ja_item_idxs, item_map_ja_item_areaperil_idxs,
            item_areaperil_ja_areaperil_agg_vuln_idxs,
            vuln_table, vuln_key_table[:hm_info[HM_INFO_N_VALID]],
            areaperil_agg_vuln_idx_ja_offsets[:n_agg_vuln_groups + 1],
            areaperil_agg_vuln_idx_ja_data[:ja_ptr],
            areaperil_agg_vuln_idx_ja_areaperil_ids[:ja_ptr])


@nb.njit(cache=True)
def build_cdf_group_indices(item_map_ja_offsets, item_map_ja_item_idxs, item_map_ja_item_areaperil_idxs,
                            items, n_item_areaperils, dynamic_footprint):
    """Assign a sequential CDF group index to each item areaperil.

    A CDF group is a set of item areaperils that share identical vulnerability CDFs: the same
    areaperil and vulnerability, and under a dynamic footprint the same intensity adjustment.
    Groups are numbered in item map order, the order the cache is filled in.

    Numba compiles two specializations based on whether dynamic_footprint is None or not.

    Args:
        item_map_ja_offsets (np.array[oasis_int]): item map CSR offsets.
        item_map_ja_item_idxs (np.array[oasis_int]): item index per entry.
        item_map_ja_item_areaperil_idxs (np.array[oasis_int]): item areaperil index per entry.
        items (np.ndarray): items table ('vulnerability_id', and 'intensity_adjustment' when dynamic).
        n_item_areaperils (int): number of item areaperils.
        dynamic_footprint: None for static footprints, truthy for dynamic.

    Returns:
        item_areaperil_ja_cdf_group_idxs (np.array[int64]): CDF group index per item areaperil.
        n_cdf_groups (int): total number of CDF groups.
    """
    item_areaperil_ja_cdf_group_idxs = np.empty(n_item_areaperils, dtype=np.int64)
    n_areaperils = item_map_ja_offsets.shape[0] - 1
    n_cdf_groups = 0

    if dynamic_footprint is None:
        for ap_idx in range(n_areaperils):
            for e in range(item_map_ja_offsets[ap_idx], item_map_ja_offsets[ap_idx + 1]):
                # entries of an areaperil are sorted by vulnerability, so a pair starts where it changes
                same_pair = e > item_map_ja_offsets[ap_idx] and (
                    items[item_map_ja_item_idxs[e]]['vulnerability_id'] == items[item_map_ja_item_idxs[e - 1]]['vulnerability_id'])
                if not same_pair:
                    n_cdf_groups += 1
                item_areaperil_ja_cdf_group_idxs[item_map_ja_item_areaperil_idxs[e]] = n_cdf_groups - 1
    else:
        n_entries = item_map_ja_item_idxs.shape[0]
        adj_key_storage = np.empty(max(n_entries, 1), dtype=oasis_int)
        adj_cache_ids = np.empty(max(n_entries, 1), dtype=np.int64)
        for ap_idx in range(n_areaperils):
            pair_start = item_map_ja_offsets[ap_idx]
            ap_end = item_map_ja_offsets[ap_idx + 1]
            while pair_start < ap_end:
                vuln = items[item_map_ja_item_idxs[pair_start]]['vulnerability_id']
                pair_end = pair_start + 1
                while pair_end < ap_end and items[item_map_ja_item_idxs[pair_end]]['vulnerability_id'] == vuln:
                    pair_end += 1
                # fresh hashmap per (areaperil, vulnerability) pair: intensity_adjustment -> dense index
                adj_table = init_dict(max(pair_end - pair_start, 1))
                adj_info, adj_lookup, adj_index = unpack(adj_table)
                for e in range(pair_start, pair_end):
                    adj = items[item_map_ja_item_idxs[e]]['intensity_adjustment']
                    result = _try_add_key(adj_info, adj_lookup, adj_index, adj_key_storage, adj)
                    while result == i_add_key_fail:
                        adj_table = rehash(adj_table, adj_key_storage)
                        adj_info, adj_lookup, adj_index = unpack(adj_table)
                        result = _try_add_key(adj_info, adj_lookup, adj_index, adj_key_storage, adj)
                    dense_idx = adj_index[result & slot_mask]
                    if result & new_slot_bit:  # new unique adjustment
                        adj_cache_ids[dense_idx] = n_cdf_groups
                        n_cdf_groups += 1
                    item_areaperil_ja_cdf_group_idxs[item_map_ja_item_areaperil_idxs[e]] = adj_cache_ids[dense_idx]
                pair_start = pair_end

    return item_areaperil_ja_cdf_group_idxs, n_cdf_groups


def item_is_aggregate_vuln(item_areaperil_ja_offsets, item_areaperil_ja_areaperil_agg_vuln_idxs):
    """Per item, whether its vulnerability is aggregate.

    Being aggregate is a property of the vulnerability, so all of an item's item areaperils agree
    and the first one is read. The block index itself is per item areaperil, because the aggregate
    weights are keyed by areaperil. An item with no item areaperil (K = 0: every placement of its
    blended areaperil uncovered) is never computed, and reads as not aggregate.
    """
    is_aggregate = np.zeros(item_areaperil_ja_offsets.shape[0] - 1, dtype=bool)
    has_areaperil = item_areaperil_ja_offsets[1:] > item_areaperil_ja_offsets[:-1]
    is_aggregate[has_areaperil] = item_areaperil_ja_areaperil_agg_vuln_idxs[item_areaperil_ja_offsets[:-1][has_areaperil]] >= 0
    return is_aggregate


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
    else:  # Fall back to the items file (items.bin or items.csv) with zero adjustments.
        items_tb = read_items(input_path)
        adjustments_tb = np.zeros(len(items_tb), dtype=ItemAdjustment)
        adjustments_tb['item_id'] = items_tb['item_id']

    return adjustments_tb


def get_peril_id(input_path):
    """Get peril_id associated with item_id

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
