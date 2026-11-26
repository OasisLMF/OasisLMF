"""This file contains specific functionality to read and process items files."""
import logging
import os

import numpy as np
import numba as nb
from numba.types import int32 as nb_int32

from oasislmf.pytools.common.data import areaperil_int, nb_oasis_int, oasis_int, items_dtype, unique_sorted
from oasislmf.pytools.common.blended_areaperil import select_placement
from oasislmf.pytools.gul.random import PHILOX_SHIFT32, PHILOX_U32_MASK, _lh_philox_block
from oasislmf.pytools.common.id_index import get_idx, NOT_FOUND
from oasislmf.pytools.common.hashmap import (
    init_dict, unpack, rehash, _try_add_key,
    i_add_key_fail, slot_mask,
    HM_INFO_N_VALID,
)
from oasislmf.pytools.getmodel.manager import encode_peril_id
from oasislmf.utils.exceptions import OasisException
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
                          blended_areaperil_placement_schemes, sample_size):
    """Draw every building's placement for every sample, once per run, shared by its location's items.

    Placement is where a building stands, which does not change from one event to the next,
    so it is drawn once rather than per event: one Latin Hypercube block of ``sample_size``
    uniforms per (location, building), from a key that has no event in it, mapped to a placement
    index through the location's cumulative weights. A location's items -- its perils and coverage
    types -- all read that one placement, each through its own blended areaperil's map to the
    areaperil it covers (``blended_areaperil_placement_ja_areaperil_j``). So they always agree on
    where the building stands, and the placement is stored once per location rather than per item.

    That needs one set of positions per location: every blended item of a location must name the
    same placement scheme, which is checked here.

    Only items whose blended areaperil has more than one placement are listed; every other item
    reads areaperil_j 0 throughout and costs nothing here.

    Args:
        items (np.ndarray): items table with 'blended_areaperil_idx' and 'packed_buildings'.
        item_location_keys (np.ndarray[uint64]): per item, its location's placement key.
        item_building_ids (np.ndarray[int]): per item, its ``building_id`` (1 unless the location
            was disaggregated into one item per building).
        blended_areaperil_placement_ja_offsets (np.ndarray[oasis_int]): placements per blended areaperil.
        blended_areaperil_placement_ja_cum_weights (np.ndarray[float64]): running placement weight.
        blended_areaperil_placement_schemes (np.ndarray[int64] | None): placement scheme of each
            blended areaperil. None skips the one-scheme-per-location check, for a structure that
            passed it when it was built.
        sample_size (int): number of samples S.

    Returns:
        dict: the arrays below, by name. A location's code is ``(b - 1) * S + (s - 1)`` for its
        building b and sample s (both 1-based).

        - ``placement_item_idxs`` (int32, ascending): the items that read a placement.
        - ``placement_item_starts`` (int64): for each, where its first building's samples start in
          ``location_placements``, building b and sample s being at ``start + (b - 1) * S + (s - 1)``.
        - ``location_placements``: per (location, building, sample) the placement index, equal to
          the number of placements when float weights leave the building unplaced. Its dtype is
          the narrowest unsigned type that holds that index.
        - ``placement_item_locations`` (int32) and ``placement_item_code_starts`` (int64): for each
          item that reads a placement, its location and the code of its first building's first
          sample within that location.
        - The inverse of ``location_placements``, which lists the samples that land on a given
          placement: ``placement_location_ja_offsets`` (int64, per location plus one) delimits
          each location's run of ``placement_location_ja_placements``, the placements its
          samples land on (ascending, of ``location_placements``'s dtype; unplaced samples are
          left out); ``placement_location_ja_code_offsets`` (int64, one past the last) delimits
          each listed placement's run of ``placement_location_ja_codes``, the location codes on
          it, ascending, in the narrowest unsigned type that holds them.
    """
    blended_idx = items['blended_areaperil_idx']
    n_placements = (blended_areaperil_placement_ja_offsets[blended_idx + 1]
                    - blended_areaperil_placement_ja_offsets[blended_idx]).astype(np.int64)
    placement_item_idxs = np.flatnonzero(n_placements > 1).astype(np.int32)
    if placement_item_idxs.shape[0] == 0:
        return empty_placements()

    item_blended_idx = blended_idx[placement_item_idxs]
    first_building = item_building_ids[placement_item_idxs].astype(np.int64)
    n_buildings = np.abs(items['packed_buildings'][placement_item_idxs]).astype(np.int64)
    location_keys, item_location = np.unique(item_location_keys[placement_item_idxs], return_inverse=True)
    n_locations = location_keys.shape[0]

    # one scheme per location: its blended areaperils (one per peril or coverage type) must list the
    # same positions, or its items could not share one placement
    if blended_areaperil_placement_schemes is not None:
        schemes = blended_areaperil_placement_schemes[item_blended_idx]
        location_scheme = np.empty(n_locations, dtype=schemes.dtype)
        location_scheme[item_location] = schemes
        mismatch = location_scheme[item_location] != schemes
        if mismatch.any():
            bad = np.unique(item_location[mismatch])
            raise OasisException(
                f"blended areaperils: {len(bad)} location(s) read blended areaperils of different placement schemes "
                f"(placement keys {location_keys[bad][:5].tolist()}); every blended areaperil of a location must "
                f"name the same placement scheme")

    # a location's run covers its building numbers 1..n, the most any of its items reaches
    location_n_buildings = np.zeros(n_locations, dtype=np.int64)
    np.maximum.at(location_n_buildings, item_location, first_building - 1 + n_buildings)
    location_starts = np.zeros(n_locations + 1, dtype=np.int64)
    np.cumsum(location_n_buildings * sample_size, out=location_starts[1:])
    placement_item_starts = location_starts[item_location] + (first_building - 1) * sample_size

    location_blended_idx = np.empty(n_locations, dtype=item_blended_idx.dtype)
    location_blended_idx[item_location] = item_blended_idx
    max_index = int(n_placements.max())  # the unplaced index equals the placement count
    placement_dtype = np.uint8 if max_index <= np.iinfo(np.uint8).max else (
        np.uint16 if max_index <= np.iinfo(np.uint16).max else np.int32)
    location_placements = np.empty(location_starts[-1], dtype=placement_dtype)
    _fill_location_placements(location_keys, location_n_buildings, location_starts, location_blended_idx,
                              blended_areaperil_placement_ja_offsets, blended_areaperil_placement_ja_cum_weights,
                              sample_size, location_placements)

    # the inverse: per location, which codes land on each placement
    location_n_placements = (blended_areaperil_placement_ja_offsets[location_blended_idx + 1]
                             - blended_areaperil_placement_ja_offsets[location_blended_idx]).astype(np.int64)
    counts = np.empty(max_index + 1, dtype=np.int64)
    location_n_distinct = np.empty(n_locations, dtype=np.int64)
    n_placed = _count_location_placements(location_starts, location_n_placements, location_placements, counts,
                                          location_n_distinct)
    ja_offsets = np.zeros(n_locations + 1, dtype=np.int64)
    np.cumsum(location_n_distinct, out=ja_offsets[1:])
    max_code = int(location_n_buildings.max()) * sample_size - 1
    code_dtype = np.uint16 if max_code <= np.iinfo(np.uint16).max else (
        np.uint32 if max_code <= np.iinfo(np.uint32).max else np.int64)
    ja_placements = np.empty(ja_offsets[-1], dtype=placement_dtype)
    ja_code_offsets = np.empty(ja_offsets[-1] + 1, dtype=np.int64)
    ja_codes = np.empty(n_placed, dtype=code_dtype)
    _fill_location_placement_codes(location_starts, location_n_placements, location_placements, counts,
                                   ja_placements, ja_code_offsets, ja_codes)
    logger.info(f"placements: {n_locations} locations, {location_placements.shape[0]} samples on "
                f"{ja_placements.shape[0]} (location, placement) pairs")

    return {
        'placement_item_idxs': placement_item_idxs,
        'placement_item_starts': placement_item_starts,
        'location_placements': location_placements,
        'placement_item_locations': item_location.astype(np.int32),
        'placement_item_code_starts': (first_building - 1) * sample_size,
        'placement_location_ja_offsets': ja_offsets,
        'placement_location_ja_placements': ja_placements,
        'placement_location_ja_code_offsets': ja_code_offsets,
        'placement_location_ja_codes': ja_codes,
    }


def build_areaperil_placements(blended_areaperil_ja_offsets, blended_areaperil_placement_ja_offsets,
                               blended_areaperil_placement_ja_areaperil_j):
    """The inverse of ``blended_areaperil_placement_ja_areaperil_j``: per blended areaperil and
    areaperil_j, the placements that read it. Several placements can read one areaperil.

    Returns:
        tuple: ``blended_areaperil_ja_placement_offsets`` (oasis_int), indexed like
        ``blended_areaperil_ja`` (position ``blended_areaperil_ja_offsets[b] + areaperil_j``) plus
        one, delimiting ``blended_areaperil_ja_placements`` (int32), each one's placements as
        indexes within its blended areaperil, ascending. Uncovered placements are not listed.
    """
    n_positions = int(blended_areaperil_ja_offsets[-1])
    counts = np.zeros(n_positions + 1, dtype=np.int64)
    _count_areaperil_placements(blended_areaperil_ja_offsets, blended_areaperil_placement_ja_offsets,
                                blended_areaperil_placement_ja_areaperil_j, counts)
    offsets = np.zeros(n_positions + 1, dtype=oasis_int)
    np.cumsum(counts[:n_positions], out=offsets[1:])
    placements = np.empty(int(offsets[-1]), dtype=np.int32)
    _fill_areaperil_placements(blended_areaperil_ja_offsets, blended_areaperil_placement_ja_offsets,
                               blended_areaperil_placement_ja_areaperil_j, offsets, counts, placements)
    return offsets, placements


@nb.njit(cache=True)
def _count_areaperil_placements(ja_offsets, placement_ja_offsets, placement_ja_areaperil_j, counts):
    for blend in range(placement_ja_offsets.shape[0] - 1):
        for pos in range(placement_ja_offsets[blend], placement_ja_offsets[blend + 1]):
            areaperil_j = placement_ja_areaperil_j[pos]
            if areaperil_j >= 0:
                counts[ja_offsets[blend] + areaperil_j] += 1


@nb.njit(cache=True)
def _fill_areaperil_placements(ja_offsets, placement_ja_offsets, placement_ja_areaperil_j, offsets, cursor, placements):
    """``cursor`` is overwritten, as each position's write cursor."""
    for a in range(offsets.shape[0] - 1):
        cursor[a] = offsets[a]
    for blend in range(placement_ja_offsets.shape[0] - 1):
        start = placement_ja_offsets[blend]
        for pos in range(start, placement_ja_offsets[blend + 1]):
            areaperil_j = placement_ja_areaperil_j[pos]
            if areaperil_j >= 0:
                a = ja_offsets[blend] + areaperil_j
                placements[cursor[a]] = pos - start
                cursor[a] += 1


def empty_placements():
    """``build_item_placements``'s arrays when no item reads a placement."""
    return {
        'placement_item_idxs': np.zeros(0, dtype=np.int32),
        'placement_item_starts': np.zeros(0, dtype=np.int64),
        'location_placements': np.zeros(0, dtype=np.uint8),
        'placement_item_locations': np.zeros(0, dtype=np.int32),
        'placement_item_code_starts': np.zeros(0, dtype=np.int64),
        'placement_location_ja_offsets': np.zeros(1, dtype=np.int64),
        'placement_location_ja_placements': np.zeros(0, dtype=np.uint8),
        'placement_location_ja_code_offsets': np.zeros(1, dtype=np.int64),
        'placement_location_ja_codes': np.zeros(0, dtype=np.uint16),
    }


@nb.njit(cache=True)
def _count_location_placements(location_starts, location_n_placements, location_placements, counts, location_n_distinct):
    """First pass of the inversion: per location, how many distinct placements its samples land on.

    ``counts`` is scratch of at least the largest placement count plus one. Returns the number of
    placed samples, the unplaced ones (index equal to the placement count) being left out.
    """
    n_placed = 0
    for loc in range(location_n_placements.shape[0]):
        n_placements = location_n_placements[loc]
        for p in range(n_placements + 1):
            counts[p] = 0
        for i in range(location_starts[loc], location_starts[loc + 1]):
            counts[location_placements[i]] += 1
        n_distinct = 0
        for p in range(n_placements):
            if counts[p] > 0:
                n_distinct += 1
                n_placed += counts[p]
        location_n_distinct[loc] = n_distinct
    return n_placed


@nb.njit(cache=True)
def _fill_location_placement_codes(location_starts, location_n_placements, location_placements, counts,
                                   ja_placements, ja_code_offsets, ja_codes):
    """Second pass of the inversion: a counting sort of each location's samples by placement.

    Codes are written in ascending order within a placement, as the samples are visited in code
    order.
    """
    d = 0
    pos = 0
    for loc in range(location_n_placements.shape[0]):
        n_placements = location_n_placements[loc]
        for p in range(n_placements + 1):
            counts[p] = 0
        start = location_starts[loc]
        for i in range(start, location_starts[loc + 1]):
            counts[location_placements[i]] += 1
        # counts become each placement's write cursor
        for p in range(n_placements):
            if counts[p] > 0:
                ja_placements[d] = p
                ja_code_offsets[d] = pos
                n = counts[p]
                counts[p] = pos
                pos += n
                d += 1
        for i in range(start, location_starts[loc + 1]):
            p = location_placements[i]
            if p < n_placements:
                ja_codes[counts[p]] = i - start
                counts[p] += 1
    ja_code_offsets[d] = pos


@nb.njit(cache=True)
def _fill_location_placements(location_keys, location_n_buildings, location_starts, location_blended_idx,
                              placement_ja_offsets, placement_ja_cum_weights, sample_size, location_placements):
    """Kernel of ``build_item_placements``: one Philox block per (location, building), mapped to a placement."""
    uniforms = np.empty(sample_size, dtype=np.float64)
    perm_scratch = np.empty(sample_size, dtype=np.float64)
    for loc in range(location_keys.shape[0]):
        b_idx = location_blended_idx[loc]
        cum_weights = placement_ja_cum_weights[placement_ja_offsets[b_idx]:placement_ja_offsets[b_idx + 1]]
        key = location_keys[loc]
        k0 = np.uint32(key & PHILOX_U32_MASK)
        k1 = np.uint32(key >> PHILOX_SHIFT32)
        for building in range(location_n_buildings[loc]):
            # the building's number within its location is the block coordinate: building_id for one
            # item per building, the packed building index for one item carrying them all
            _lh_philox_block(k0, k1, building, sample_size, perm_scratch, uniforms)
            out = location_starts[loc] + building * sample_size
            for s in range(sample_size):
                location_placements[out + s] = select_placement(cum_weights, uniforms[s])


mixed_cdf_group_dtype = np.dtype([('blended_areaperil_idx', oasis_int),
                                  ('vulnerability_id', np.int32),
                                  ('intensity_adjustment', oasis_int),
                                  ('return_period', np.int64),
                                  ('vulnerability_idx', oasis_int)])


def _group_keys(blended_areaperil_idx, vulnerability_id, intensity_adjustment, return_period, dynamic_footprint):
    columns = [blended_areaperil_idx, vulnerability_id]
    if dynamic_footprint:
        columns += [intensity_adjustment, return_period]
    return np.stack([np.asarray(column, dtype=np.int64) for column in columns], axis=1)


def _unique_rows(keys):
    """``np.unique(keys, axis=0, return_inverse=True)`` for a 2-d int64 array, much faster.

    Each row becomes one int64 code, its columns' ranks in mixed radix, which orders the codes as
    the rows sort lexicographically, so the result is the same. Row-wise unique sorts the rows as
    structured records and is an order of magnitude slower. Falls back to it should the code
    overflow.
    """
    if keys.shape[0] == 0:
        return keys.copy(), np.zeros(0, dtype=np.int64)
    codes = np.zeros(keys.shape[0], dtype=np.int64)
    n_codes = 1
    for column in range(keys.shape[1]):
        values, ranks = np.unique(keys[:, column], return_inverse=True)
        n_codes *= values.shape[0]
        if n_codes >= 1 << 62:
            unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
            return unique_keys, inverse.reshape(-1)
        codes = codes * values.shape[0] + ranks.reshape(-1)
    _, first, inverse = np.unique(codes, return_index=True, return_inverse=True)
    return keys[first], inverse.reshape(-1)


def _cell_blends(blended_areaperil_ja_offsets, blended_areaperil_ja_areaperil_ids, blended_areaperil_placement_schemes):
    """The size-1 blended areaperils (no placement scheme), as their areaperil ids sorted and their indices in that order."""
    cell_blend_idxs = np.flatnonzero(blended_areaperil_placement_schemes == -1)
    cell_ids = blended_areaperil_ja_areaperil_ids[blended_areaperil_ja_offsets[cell_blend_idxs]]
    order = np.argsort(cell_ids, kind='stable')
    return cell_ids[order], cell_blend_idxs[order]


@nb.njit(cache=True)
def _fill_cell_codes(key_blend_idxs, key_tail_ranks, blended_areaperil_ja_offsets, ja_blend_ranks, n_tails, codes):
    """Write the code of every blended key's size-1 groups, K per key (see ``build_item_mixed_cdf_groups``)."""
    pos = 0
    for key in range(key_blend_idxs.shape[0]):
        b = key_blend_idxs[key]
        tail_rank = key_tail_ranks[key]
        for a in range(blended_areaperil_ja_offsets[b], blended_areaperil_ja_offsets[b + 1]):
            codes[pos] = ja_blend_ranks[a] * n_tails + tail_rank
            pos += 1


def build_item_mixed_cdf_groups(items, dynamic_footprint, blended_areaperil_ja_offsets, blended_areaperil_ja_areaperil_ids,
                                blended_areaperil_placement_schemes):
    """Group the items that report the same effective damage cdf into mixed cdf groups.

    An item's effective damage cdf is its areaperils' cdfs mixed by the probability its building
    stands on each. It depends only on its blended areaperil, its vulnerability and, under a dynamic
    footprint, its intensity adjustment, so the items agreeing on those share it and every
    per-areaperil structure. An areaperil read directly (its own size-1 blended areaperil) is the
    case of one areaperil of probability 1.

    The cdf of a size-1 group, an areaperil's own, does not depend on protection: protection only
    decides whether an item of it is live. A blended group's mixture does, since it decides which
    of its areaperils count, so under a dynamic footprint its key also holds the protection return
    period.

    A blended group's mixture is made of its areaperils' own cdfs, so every areaperil it reads has
    its size-1 group, with no item when no item reads that areaperil directly (``build_blended_areaperils``
    with ``cell_blends`` gives each such areaperil its size-1 blended areaperil).

    Groups are numbered in key order, and ``items`` is sorted by (areaperil_id, vulnerability_id),
    so with one group per (areaperil, vulnerability) a group's items, taken in item order, come in
    the order the items themselves are sorted in.

    Returns:
        tuple: ``item_mixed_cdf_groups`` (int32, the group of each item), ``mixed_cdf_groups``
        (mixed_cdf_group_dtype, one row per group; ``vulnerability_idx`` is filled in later by
        ``generate_item_map``), and the items of each group as ``mixed_cdf_group_items_ja_offsets`` /
        ``mixed_cdf_group_items_ja_item_idxs`` (in item order).
    """
    n_items = items.shape[0]
    blended_idx = np.asarray(items['blended_areaperil_idx'], dtype=np.int64)
    is_cell_blend = blended_areaperil_placement_schemes == -1
    if dynamic_footprint:
        intensity_adjustment = items['intensity_adjustment']
        return_period = np.where(is_cell_blend[blended_idx], 0, items['return_period'])
    else:
        intensity_adjustment = return_period = None
    item_keys = _group_keys(blended_idx, items['vulnerability_id'], intensity_adjustment, return_period, dynamic_footprint)

    # The groups are the items' keys plus, for every blended key, the size-1 group of each of its
    # areaperils: (that areaperil's size-1 blended areaperil, the key's vulnerability and intensity
    # adjustment, no protection). That is K rows per blended key, millions on a state-wide blend,
    # so no row is built: a key is one int64 code, its blended areaperil's rank times the number
    # of tails plus its tail's rank, the tail being the key after its blended areaperil. Codes
    # sort as the rows do, so the groups keep key order. The tails are few, every one an item's,
    # and the code cannot overflow, as both ranks are below 2**31.
    n_key_columns = 3 if dynamic_footprint else 2
    blended_keys, _ = _unique_rows(np.ascontiguousarray(item_keys[~is_cell_blend[blended_idx], :n_key_columns]))
    cell_tails = blended_keys[:, 1:]
    if dynamic_footprint:
        cell_tails = np.concatenate([cell_tails, np.zeros((cell_tails.shape[0], 1), dtype=np.int64)], axis=1)
    tails, tail_ranks = _unique_rows(np.ascontiguousarray(np.concatenate([item_keys[:, 1:], cell_tails])))
    n_tails = np.int64(max(tails.shape[0], 1))

    # the size-1 blended areaperil of each areaperil the blended keys read, ranked, per blend table position
    key_blend_idxs = unique_sorted(blended_keys[:, 0])
    starts = blended_areaperil_ja_offsets[key_blend_idxs].astype(np.int64)
    k = blended_areaperil_ja_offsets[key_blend_idxs + 1] - starts
    ja_positions = np.repeat(starts - (np.cumsum(k) - k), k) + np.arange(k.sum(), dtype=np.int64)
    sorted_cell_ids, cell_blend_idxs = _cell_blends(blended_areaperil_ja_offsets, blended_areaperil_ja_areaperil_ids,
                                                    blended_areaperil_placement_schemes)
    ja_cell_blend_idxs = cell_blend_idxs[np.searchsorted(sorted_cell_ids, blended_areaperil_ja_areaperil_ids[ja_positions])]
    blend_values = unique_sorted(np.concatenate([item_keys[:, 0], ja_cell_blend_idxs]))
    ja_blend_ranks = np.zeros(blended_areaperil_ja_areaperil_ids.shape[0], dtype=np.int64)
    ja_blend_ranks[ja_positions] = np.searchsorted(blend_values, ja_cell_blend_idxs)
    del ja_positions, ja_cell_blend_idxs

    key_k = (blended_areaperil_ja_offsets[blended_keys[:, 0] + 1] - blended_areaperil_ja_offsets[blended_keys[:, 0]]).astype(np.int64)
    codes = np.empty(n_items + int(key_k.sum()), dtype=np.int64)
    codes[:n_items] = np.searchsorted(blend_values, item_keys[:, 0]) * n_tails + tail_ranks[:n_items]
    _fill_cell_codes(blended_keys[:, 0], tail_ranks[n_items:], blended_areaperil_ja_offsets, ja_blend_ranks, n_tails,
                     codes[n_items:])
    del ja_blend_ranks
    item_codes = codes[:n_items].copy()
    codes.sort()
    is_first = np.ones(codes.shape[0], dtype=np.bool_)
    np.not_equal(codes[1:], codes[:-1], out=is_first[1:])
    group_codes = codes[is_first]
    del codes, is_first

    item_mixed_cdf_groups = np.searchsorted(group_codes, item_codes).astype(np.int32)
    group_blend_ranks, group_tail_ranks = np.divmod(group_codes, n_tails)
    del group_codes
    mixed_cdf_groups = np.zeros(group_blend_ranks.shape[0], dtype=mixed_cdf_group_dtype)
    mixed_cdf_groups['blended_areaperil_idx'] = blend_values[group_blend_ranks]
    mixed_cdf_groups['vulnerability_id'] = tails[:, 0][group_tail_ranks]
    if dynamic_footprint:
        mixed_cdf_groups['intensity_adjustment'] = tails[:, 1][group_tail_ranks]
        mixed_cdf_groups['return_period'] = tails[:, 2][group_tail_ranks]
    mixed_cdf_group_items_ja_offsets = np.zeros(mixed_cdf_groups.shape[0] + 1, dtype=oasis_int)
    np.cumsum(np.bincount(item_mixed_cdf_groups, minlength=mixed_cdf_groups.shape[0]), out=mixed_cdf_group_items_ja_offsets[1:])
    mixed_cdf_group_items_ja_item_idxs = np.argsort(item_mixed_cdf_groups, kind='stable').astype(oasis_int)
    return item_mixed_cdf_groups, mixed_cdf_groups, mixed_cdf_group_items_ja_offsets, mixed_cdf_group_items_ja_item_idxs


@nb.njit(cache=True)
def _group_key_tail_less(group, vulnerability_id, intensity_adjustment, return_period):
    """Whether a group's key, after its blended areaperil, sorts before (vulnerability, adjustment, protection)."""
    if group['vulnerability_id'] != vulnerability_id:
        return group['vulnerability_id'] < vulnerability_id
    if group['intensity_adjustment'] != intensity_adjustment:
        return group['intensity_adjustment'] < intensity_adjustment
    return group['return_period'] < return_period


@nb.njit(cache=True)
def _find_cell_group(mixed_cdf_groups, blend_group_starts, cell_blend_idx, vulnerability_id, intensity_adjustment):
    """The size-1 group of (size-1 blended areaperil, vulnerability, adjustment) with no protection, -1 if there is none."""
    if cell_blend_idx < 0:
        return -1
    lo = blend_group_starts[cell_blend_idx]
    hi = blend_group_starts[cell_blend_idx + 1]
    while lo < hi:
        mid = (lo + hi) // 2
        if _group_key_tail_less(mixed_cdf_groups[mid], vulnerability_id, intensity_adjustment, 0):
            lo = mid + 1
        else:
            hi = mid
    if lo == blend_group_starts[cell_blend_idx + 1]:
        return -1
    group = mixed_cdf_groups[lo]
    if (group['vulnerability_id'] != vulnerability_id or group['intensity_adjustment'] != intensity_adjustment
            or group['return_period'] != 0):
        return -1
    return lo


@nb.njit(cache=True)
def _fill_group_areaperils(mixed_cdf_groups, blended_areaperil_ja_offsets, blended_areaperil_ja_areaperil_ids,
                           ja_cell_blend_idxs, blend_group_starts,
                           entry_group, entry_areaperil_id, entry_vuln_id, cell_groups):
    """Each group's areaperils in group order, with their size-1 group (see ``build_mixed_cdf_group_areaperils``)."""
    e = 0
    for g in range(mixed_cdf_groups.shape[0]):
        group = mixed_cdf_groups[g]
        b = group['blended_areaperil_idx']
        vulnerability_id = group['vulnerability_id']
        intensity_adjustment = group['intensity_adjustment']
        for a in range(blended_areaperil_ja_offsets[b], blended_areaperil_ja_offsets[b + 1]):
            areaperil_id = blended_areaperil_ja_areaperil_ids[a]
            entry_group[e] = g
            entry_areaperil_id[e] = areaperil_id
            entry_vuln_id[e] = vulnerability_id
            cell_groups[e] = _find_cell_group(mixed_cdf_groups, blend_group_starts, ja_cell_blend_idxs[a],
                                              vulnerability_id, intensity_adjustment)
            e += 1


def build_mixed_cdf_group_areaperils(mixed_cdf_groups, blended_areaperil_ja_offsets, blended_areaperil_ja_areaperil_ids,
                                     blended_areaperil_placement_schemes):
    """Expand each mixed cdf group into its group areaperils and sort them into item map order.

    A group reads one group areaperil per distinct areaperil of its blended areaperil; group
    areaperil ``mixed_cdf_group_areaperil_ja_offsets[mixed_cdf_group_idx] + j`` is the group's j-th.
    The item map visits them grouped by areaperil, then by vulnerability, then by group.

    The groups' intensity adjustment and protection are 0 without a dynamic footprint, so the
    size-1 groups are found the same way with and without one.

    Args:
        mixed_cdf_groups (np.ndarray[mixed_cdf_group_dtype]): mixed cdf groups, in key order.
        blended_areaperil_ja_offsets (np.ndarray[oasis_int]): CSR offsets over a blended
            areaperil's areaperils.
        blended_areaperil_ja_areaperil_ids (np.ndarray[areaperil_int]): footprint areaperil ids.
        blended_areaperil_placement_schemes (np.ndarray[int64]): per blended areaperil, its
            placement scheme, -1 for an areaperil's own size-1 one.

    Returns:
        tuple: ``mixed_cdf_group_areaperil_ja_offsets`` (N_mixed_cdf_groups + 1),
        ``mixed_cdf_group_areaperil_ja_cell_groups`` (per group areaperil, the size-1 group of its
        areaperil under the group's vulnerability, which is the group itself for a size-1 group),
        and the group areaperils in item map order as four parallel arrays: areaperil id,
        vulnerability id, group index and group areaperil index.
    """
    n_mixed_cdf_groups = mixed_cdf_groups.shape[0]
    blended_idx = mixed_cdf_groups['blended_areaperil_idx'].astype(np.int64)
    mixed_cdf_group_k = (blended_areaperil_ja_offsets[blended_idx + 1] - blended_areaperil_ja_offsets[blended_idx]).astype(np.int64)
    mixed_cdf_group_areaperil_ja_offsets = np.zeros(n_mixed_cdf_groups + 1, dtype=oasis_int)
    np.cumsum(mixed_cdf_group_k, out=mixed_cdf_group_areaperil_ja_offsets[1:])
    n_mixed_cdf_group_areaperils = int(mixed_cdf_group_areaperil_ja_offsets[-1])

    # the size-1 group of each group areaperil's areaperil: same vulnerability and intensity
    # adjustment, no protection. build_item_mixed_cdf_groups made one for every areaperil read.
    # Groups are in key order, so a blended areaperil's groups are one run, sorted by the rest of the key.
    # an areaperil's size-1 blended areaperil, per blend table position (-1 for none): looked up
    # once per position rather than once per group reading it, i.e. once per vulnerability
    sorted_cell_ids, cell_blend_idxs = _cell_blends(blended_areaperil_ja_offsets, blended_areaperil_ja_areaperil_ids,
                                                    blended_areaperil_placement_schemes)
    if sorted_cell_ids.shape[0]:
        pos = np.minimum(np.searchsorted(sorted_cell_ids, blended_areaperil_ja_areaperil_ids), sorted_cell_ids.shape[0] - 1)
        ja_cell_blend_idxs = np.where(sorted_cell_ids[pos] == blended_areaperil_ja_areaperil_ids, cell_blend_idxs[pos], -1)
        del pos
    else:
        ja_cell_blend_idxs = np.full(blended_areaperil_ja_areaperil_ids.shape[0], -1, dtype=np.int64)
    blend_group_starts = np.searchsorted(mixed_cdf_groups['blended_areaperil_idx'],
                                         np.arange(blended_areaperil_ja_offsets.shape[0], dtype=np.int64))
    entry_group = np.empty(n_mixed_cdf_group_areaperils, dtype=oasis_int)
    entry_areaperil_id = np.empty(n_mixed_cdf_group_areaperils, dtype=areaperil_int)
    entry_vuln_id = np.empty(n_mixed_cdf_group_areaperils, dtype=mixed_cdf_groups.dtype['vulnerability_id'])
    mixed_cdf_group_areaperil_ja_cell_groups = np.empty(n_mixed_cdf_group_areaperils, dtype=np.int32)
    _fill_group_areaperils(mixed_cdf_groups, blended_areaperil_ja_offsets, blended_areaperil_ja_areaperil_ids,
                           ja_cell_blend_idxs, blend_group_starts,
                           entry_group, entry_areaperil_id, entry_vuln_id, mixed_cdf_group_areaperil_ja_cell_groups)
    if (mixed_cdf_group_areaperil_ja_cell_groups < 0).any():
        raise OasisException("gulmc structure: a blended areaperil reads an areaperil with no mixed cdf group of its own")

    order = np.lexsort((entry_group, entry_vuln_id, entry_areaperil_id))
    return (mixed_cdf_group_areaperil_ja_offsets,
            mixed_cdf_group_areaperil_ja_cell_groups,
            np.ascontiguousarray(entry_areaperil_id[order]),
            np.ascontiguousarray(entry_vuln_id[order]),
            np.ascontiguousarray(entry_group[order]),
            order.astype(oasis_int))


@nb.njit(cache=True, fastmath=True)
def generate_item_map(entry_areaperil_ids, entry_vuln_ids, entry_mixed_cdf_group_idxs, entry_mixed_cdf_group_areaperil_idxs,
                      mixed_cdf_groups, n_mixed_cdf_group_areaperils,
                      agg_vuln_id_ja_id_ind, agg_vuln_id_ja_offsets, agg_vuln_id_ja_vuln_ids):
    """Build the item map from the group areaperils, and assign the dense vulnerability indices.

    The item map is a jagged array from each footprint areaperil a mixed cdf group can read to the
    group areaperils that read it::

        item_map_ja_id_ind:      areaperil_id -> areaperil index        (built by the caller)
        item_map_ja_offsets:     areaperil index -> range of entries
        item_map_ja_mixed_cdf_group_idxs / item_map_ja_mixed_cdf_group_areaperil_idxs: one entry per group areaperil

    The entries must already be in item map order (``build_mixed_cdf_group_areaperils``). Dense vulnerability
    indices are assigned in that order, so with K = 1 they come out as before.

    Args:
        entry_areaperil_ids (np.array[areaperil_int]): areaperil id per entry, in item map order.
        entry_vuln_ids (np.array[int32]): vulnerability id per entry.
        entry_mixed_cdf_group_idxs (np.array[oasis_int]): mixed cdf group index per entry.
        entry_mixed_cdf_group_areaperil_idxs (np.array[oasis_int]): group areaperil index per entry.
        mixed_cdf_groups (np.ndarray[mixed_cdf_group_dtype]): mixed cdf groups; 'vulnerability_idx' is written for
            non-aggregate ones.
        n_mixed_cdf_group_areaperils (int): number of group areaperils.
        agg_vuln_id_ja_id_ind (np.array): id_index for aggregate vulnerability ids.
        agg_vuln_id_ja_offsets (np.array[oasis_int]): jagged array offsets for agg_vuln_id_ja_vuln_ids.
        agg_vuln_id_ja_vuln_ids (np.array[oasis_int]): flat jagged array of constituent vulnerability ids.

    Returns:
        item_map_ja_areaperil_ids (np.array[areaperil_int]): sorted unique footprint areaperil ids.
        item_map_ja_offsets (np.array[oasis_int]): CSR offsets (N_areaperil + 1).
        item_map_ja_mixed_cdf_group_idxs (np.array[oasis_int]): mixed cdf group index per entry.
        item_map_ja_mixed_cdf_group_areaperil_idxs (np.array[oasis_int]): group areaperil index per entry.
        mixed_cdf_group_areaperil_ja_areaperil_agg_vuln_idxs (np.array[oasis_int]): aggregate block per group areaperil, -1 if not aggregate.
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
    item_map_ja_mixed_cdf_group_idxs = np.empty(n_entries, dtype=oasis_int)
    item_map_ja_mixed_cdf_group_areaperil_idxs = np.empty(n_entries, dtype=oasis_int)
    mixed_cdf_group_areaperil_ja_areaperil_agg_vuln_idxs = np.full(n_mixed_cdf_group_areaperils, -1, dtype=oasis_int)

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

        mixed_cdf_group_idx = entry_mixed_cdf_group_idxs[e]
        item_map_ja_mixed_cdf_group_idxs[e] = mixed_cdf_group_idx
        item_map_ja_mixed_cdf_group_areaperil_idxs[e] = entry_mixed_cdf_group_areaperil_idxs[e]

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
            mixed_cdf_group_areaperil_ja_areaperil_agg_vuln_idxs[entry_mixed_cdf_group_areaperil_idxs[e]] = last_block_idx
        else:
            vuln_id_int32 = np.int32(vuln)
            result = _try_add_key(hm_info, hm_lookup, hm_index, vuln_key_table, vuln_id_int32)
            while result == i_add_key_fail:
                vuln_table = rehash(vuln_table, vuln_key_table)
                hm_info, hm_lookup, hm_index = unpack(vuln_table)
                result = _try_add_key(hm_info, hm_lookup, hm_index, vuln_key_table, vuln_id_int32)
            mixed_cdf_groups[mixed_cdf_group_idx]['vulnerability_idx'] = nb_oasis_int(hm_index[result & slot_mask])

    item_map_ja_offsets[n_unique_areaperils] = n_entries

    return (item_map_ja_areaperil_ids, item_map_ja_offsets,
            item_map_ja_mixed_cdf_group_idxs, item_map_ja_mixed_cdf_group_areaperil_idxs,
            mixed_cdf_group_areaperil_ja_areaperil_agg_vuln_idxs,
            vuln_table, vuln_key_table[:hm_info[HM_INFO_N_VALID]],
            areaperil_agg_vuln_idx_ja_offsets[:n_agg_vuln_groups + 1],
            areaperil_agg_vuln_idx_ja_data[:ja_ptr],
            areaperil_agg_vuln_idx_ja_areaperil_ids[:ja_ptr])


@nb.njit(cache=True)
def _normalised_member_weights(agg_offsets, agg_weights, out):
    """Each aggregate block's member weights over their positive total, or equal when none is positive."""
    for block in range(agg_offsets.shape[0] - 1):
        total = 0.
        for m in range(agg_offsets[block], agg_offsets[block + 1]):
            if agg_weights[m] > 0:
                total += agg_weights[m]
        n_members = agg_offsets[block + 1] - agg_offsets[block]
        for m in range(agg_offsets[block], agg_offsets[block + 1]):
            if total > 0:
                out[m] = agg_weights[m] / total if agg_weights[m] > 0 else 0.
            else:
                out[m] = 1. / n_members


def build_vuln_member_blocks(n_vulns, vuln_idx_to_cond_idx, areaperil_agg_vuln_idx_ja_offsets, areaperil_agg_vuln_idx_ja_data,
                             mixed_cdf_group_areaperil_ja_agg_blocks, mixed_cdf_groups, mixed_cdf_group_areaperil_ja_offsets):
    """Give every group areaperil the vulnerabilities its damage cdf is a weighted sum of.

    An aggregate vulnerability is a weighted sum of member vulnerabilities, with weights that
    depend on the areaperil; any other vulnerability is the one-member case of weight 1. Block
    ``v`` for ``v < n_vulns`` is that single member, so an ordinary group areaperil's block is its
    vulnerability index, and the aggregate blocks follow. Weights are normalised here: the members'
    positive weights over their total, or equal weights when none is positive.

    A member's row is its row in the cdf table it is read from: the vulnerability index in
    ``vuln_cdf``, or, for a conditional vulnerability (only ever a dependent's), its row in
    ``cond_cdf``.

    Args:
        n_vulns (int): number of dense vulnerability indices.
        vuln_idx_to_cond_idx (np.array[int64]): dense vulnerability index -> conditional row, or -1.
        areaperil_agg_vuln_idx_ja_offsets (np.array[oasis_int]): aggregate blocks' member offsets.
        areaperil_agg_vuln_idx_ja_data (np.array[agg_vuln_idx_weight_dtype]): (vuln_idx, weight) per member.
        mixed_cdf_group_areaperil_ja_agg_blocks (np.array[oasis_int]): aggregate block per group
            areaperil, -1 when its vulnerability is not aggregate.
        mixed_cdf_groups (np.ndarray[mixed_cdf_group_dtype]): with 'vulnerability_idx' filled in.
        mixed_cdf_group_areaperil_ja_offsets (np.array[oasis_int]): per group, its group areaperils.

    Returns:
        tuple: ``vuln_member_ja_offsets``, ``vuln_member_ja_rows`` (int32), ``vuln_member_ja_weights``
        (float64), and ``mixed_cdf_group_areaperil_ja_member_blocks`` (int32, per group areaperil).
    """
    n_agg_blocks = areaperil_agg_vuln_idx_ja_offsets.shape[0] - 1
    single_rows = np.arange(n_vulns, dtype=np.int64)
    single_rows = np.where(vuln_idx_to_cond_idx >= 0, vuln_idx_to_cond_idx, single_rows)
    agg_weights = np.empty(areaperil_agg_vuln_idx_ja_data.shape[0], dtype=np.float64)
    _normalised_member_weights(areaperil_agg_vuln_idx_ja_offsets, areaperil_agg_vuln_idx_ja_data['weight'].astype(np.float64),
                               agg_weights)
    vuln_member_ja_offsets = np.concatenate([np.arange(n_vulns, dtype=oasis_int),
                                             (n_vulns + areaperil_agg_vuln_idx_ja_offsets[:n_agg_blocks + 1]).astype(oasis_int)])
    vuln_member_ja_rows = np.concatenate([single_rows, areaperil_agg_vuln_idx_ja_data['vuln_idx']]).astype(np.int32)
    vuln_member_ja_weights = np.concatenate([np.ones(n_vulns, dtype=np.float64), agg_weights])
    entry_group = np.repeat(np.arange(mixed_cdf_groups.shape[0]), np.diff(mixed_cdf_group_areaperil_ja_offsets))
    mixed_cdf_group_areaperil_ja_member_blocks = np.where(
        mixed_cdf_group_areaperil_ja_agg_blocks >= 0, n_vulns + mixed_cdf_group_areaperil_ja_agg_blocks,
        mixed_cdf_groups['vulnerability_idx'][entry_group]).astype(np.int32)
    return vuln_member_ja_offsets, vuln_member_ja_rows, vuln_member_ja_weights, mixed_cdf_group_areaperil_ja_member_blocks


def item_is_aggregate_vuln(item_mixed_cdf_groups, mixed_cdf_group_areaperil_ja_offsets, group_areaperil_is_aggregate):
    """Per item, whether its vulnerability is aggregate.

    Being aggregate is a property of the vulnerability, so all of a group's group areaperils agree
    and the first one is read. An item with no group areaperil (K = 0: every placement of its
    blended areaperil uncovered) is never computed, and reads as not aggregate.
    """
    is_aggregate = np.zeros(mixed_cdf_group_areaperil_ja_offsets.shape[0] - 1, dtype=bool)
    has_areaperil = mixed_cdf_group_areaperil_ja_offsets[1:] > mixed_cdf_group_areaperil_ja_offsets[:-1]
    is_aggregate[has_areaperil] = group_areaperil_is_aggregate[mixed_cdf_group_areaperil_ja_offsets[:-1][has_areaperil]]
    return is_aggregate[item_mixed_cdf_groups]


def item_is_computed(item_mixed_cdf_groups, mixed_cdf_group_areaperil_ja_offsets):
    """Per item, whether it reads any areaperil; one that reads none (every placement uncovered) is never computed."""
    return (np.diff(mixed_cdf_group_areaperil_ja_offsets) > 0)[item_mixed_cdf_groups]


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
