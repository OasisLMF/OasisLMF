__all__ = [
    'extract_financial_structure',
    'load_financial_structure',
    'create_financial_structure',
    'load_static',
]

import logging
import os

import numpy as np
from numba import from_dtype, njit


from oasislmf.pytools.common.data import (load_as_ndarray, load_as_array, almost_equal,
                                          fm_policytc_dtype,
                                          fm_profile_dtype, fm_profile_step_dtype,
                                          fm_programme_dtype,
                                          fm_xref_dtype,
                                          items_dtype,
                                          oasis_int, nb_oasis_int, oasis_float, null_index)
from .common import (allowed_allocation_rule, need_extras, need_tiv_policy)

logger = logging.getLogger(__name__)

# financial structure processed array
nodes_array_dtype = from_dtype(np.dtype([('node_id', np.uint64),
                                         ('level_id', oasis_int),
                                         ('agg_id', oasis_int),
                                         ('layer_len', oasis_int),
                                         ('cross_layer_profile', oasis_int),
                                         ('profile_len', oasis_int),
                                         ('profiles', oasis_int),
                                         ('loss', oasis_int),
                                         ('net_loss', oasis_int),
                                         ('extra', oasis_int),
                                         ('is_reallocating', np.uint8),
                                         ('parent_len', oasis_int),
                                         ('parent', oasis_int),
                                         ('children', oasis_int),
                                         ('output_ids', oasis_int),
                                         ]))

compute_info_dtype = from_dtype(np.dtype([('allocation_rule', oasis_int),
                                          ('max_level', oasis_int),
                                          ('max_layer', oasis_int),
                                          ('node_len', oasis_int),
                                          ('children_len', oasis_int),
                                          ('parents_len', oasis_int),
                                          ('profile_len', oasis_int),
                                          ('loss_len', oasis_int),
                                          ('extra_len', oasis_int),
                                          ('compute_len', oasis_int),
                                          ('start_level', oasis_int),
                                          ('items_len', oasis_int),
                                          ('output_len', oasis_int),
                                          ('stepped', np.bool_),
                                          ]))
profile_index_dtype = from_dtype(np.dtype([('i_start', oasis_int),
                                           ('i_end', oasis_int),
                                           ]))

# Profile entry dtype for CSR storage of programme_node_to_profiles
profile_entry_dtype = np.dtype([('layer_id', oasis_int),
                                ('i_start', oasis_int),
                                ('i_end', oasis_int)])


def load_static(static_path):
    """
    Load the raw financial data from static_path as numpy ndarray
    first check if .bin file is present then try .cvs
    try loading profile_step before falling back to normal profile,

    :param static_path: str
            static_path
    :return:
        programme : link between nodes
        policytc : info on layer
        profile : policy profile can be profile_step or profile
        xref : node to output_id
        items : items (item_id and coverage_id mapping)
        coverages : Tiv value for each coverage id
    :raise:
        FileNotFoundError if one of the static is missing
    """
    programme = load_as_ndarray(static_path, 'fm_programme', fm_programme_dtype)
    policytc = load_as_ndarray(static_path, 'fm_policytc', fm_policytc_dtype)
    profile = load_as_ndarray(static_path, 'fm_profile_step', fm_profile_step_dtype, False)
    if len(profile) == 0:
        profile = load_as_ndarray(static_path, 'fm_profile', fm_profile_dtype)
        stepped = None
    else:
        stepped = True
    xref = load_as_ndarray(static_path, 'fm_xref', fm_xref_dtype)

    items = load_as_ndarray(static_path, 'items', items_dtype, must_exist=False)[['item_id', 'coverage_id']]
    coverages = load_as_array(static_path, 'coverages', oasis_float, must_exist=False)
    if np.unique(items['coverage_id']).shape[0] != coverages.shape[0]:
        # one of the file is missing we default to empty array
        items = np.empty(0, dtype=items_dtype)
        coverages = np.empty(0, dtype=oasis_float)

    return programme, policytc, profile, stepped, xref, items, coverages


@njit(cache=True)
def does_nothing(profile):
    """
    evaluate if the profile is just doing nothing to the loss.
    this allows to save some memory and compulation time and memory during the calculation
    :param profile: np.array of fm_profile_dtype or fm_profile_step_dtype
            profile
    :return:
        boolean : True is profile is actually doing nothing
    """
    return ((profile['calcrule_id'] == 100) or
            (profile['calcrule_id'] == 12 and almost_equal(profile['deductible1'], 0)) or
            (profile['calcrule_id'] == 15 and almost_equal(profile['limit1'], 1)) or
            (profile['calcrule_id'] == 16 and almost_equal(profile['deductible1'], 0)) or
            (profile['calcrule_id'] == 34 and almost_equal(profile['deductible1'], 0)
                and almost_equal(profile['attachment1'], 0)
                and almost_equal(profile['share1'], 1))
            )


@njit(cache=True)
def idx_to_node(node_idx, node_level_start, start_level, max_level):
    """Convert a flat node index back to (level, agg_id) tuple."""
    for level in range(start_level, max_level + 1):
        level_start = node_level_start[level]
        level_end = node_level_start[level + 1]
        if level_start < node_idx <= level_end:
            return (nb_oasis_int(level), nb_oasis_int(node_idx - level_start))
    # Fallback (shouldn't happen)
    return (nb_oasis_int(0), nb_oasis_int(0))


@njit(cache=True)
def get_all_children_csr(node_idx, children_indptr, children_data, items_only, max_nodes):
    """CSR version of get_all_children using NumPy arrays.

    Args:
        node_idx: Starting node index
        children_indptr, children_data: CSR arrays for parent->children relationship
        items_only: If True, only return leaf nodes (items)
        max_nodes: Maximum possible nodes (for pre-allocation)

    Returns:
        Tuple of (result_array, result_len) - node indices of children
    """
    result = np.empty(max_nodes, dtype=oasis_int)
    stack = np.empty(max_nodes, dtype=oasis_int)
    result_len = 0
    stack_len = 1
    stack[0] = node_idx

    while stack_len > 0:
        stack_len -= 1
        current = stack[stack_len]
        start = children_indptr[current]
        end = children_indptr[current + 1]

        if start < end:  # has children
            if not items_only:
                result[result_len] = current
                result_len += 1
            for i in range(start, end):
                stack[stack_len] = children_data[i]
                stack_len += 1
        else:  # leaf node
            result[result_len] = current
            result_len += 1

    return result, result_len


@njit(cache=True)
def get_all_parent_csr(start_nodes, start_len, parents_indptr, parents_data, target_level, node_level_start, max_nodes):
    """CSR version of get_all_parent using NumPy arrays.

    Args:
        start_nodes: Array of starting node indices
        start_len: Number of valid entries in start_nodes
        parents_indptr, parents_data: CSR arrays for child->parents relationship
        target_level: Stop at nodes at this level
        node_level_start: Array to convert index to level
        max_nodes: Maximum possible nodes (for pre-allocation)

    Returns:
        Tuple of (result_array, result_len) - unique node indices at target_level
    """
    result = np.empty(max_nodes, dtype=oasis_int)
    stack = np.empty(max_nodes, dtype=oasis_int)
    visited = np.zeros(max_nodes, dtype=np.uint8)  # Track unique results
    result_len = 0
    stack_len = 0

    # Initialize stack with start nodes
    for i in range(start_len):
        stack[stack_len] = start_nodes[i]
        stack_len += 1

    while stack_len > 0:
        stack_len -= 1
        current = stack[stack_len]
        start = parents_indptr[current]
        end = parents_indptr[current + 1]

        if start < end:  # has parents
            for i in range(start, end):
                parent_idx = parents_data[i]
                stack[stack_len] = parent_idx
                stack_len += 1
            # Check if current is at target level
            for level in range(node_level_start.shape[0] - 1):
                if node_level_start[level] < current <= node_level_start[level + 1]:
                    if level == target_level and visited[current] == 0:
                        visited[current] = 1
                        result[result_len] = current
                        result_len += 1
                    break
        else:
            # No parents - check if at or below target level
            for level in range(node_level_start.shape[0] - 1):
                if node_level_start[level] < current <= node_level_start[level + 1]:
                    if level <= target_level and visited[current] == 0:
                        visited[current] = 1
                        result[result_len] = current
                        result_len += 1
                    break

    return result, result_len


@njit(cache=True)
def is_multi_peril(fm_programme):
    for i in range(fm_programme.shape[0]):
        if fm_programme[i]['level_id'] == 1 and fm_programme[i]['from_agg_id'] != fm_programme[i]['to_agg_id']:
            return True
    else:
        return False


@njit(cache=True)
def get_tiv_csr(children_indices, children_len, items, coverages, node_level_start, start_level):
    """CSR-compatible version of get_tiv using node indices.

    Args:
        children_indices: Array of child node indices (item level nodes)
        children_len: Number of valid entries in children_indices
        items: Items array mapping item_id to coverage_id
        coverages: Coverage values
        node_level_start: Array for converting index to level/agg_id
        start_level: The start level (item level)

    Returns:
        Total insured value for the children
    """
    used_cov = np.zeros_like(coverages, dtype=np.uint8)
    tiv = 0
    item_level_start = node_level_start[start_level]

    for i in range(children_len):
        node_idx = children_indices[i]
        # Convert node index to agg_id (item_id for item level)
        agg_id = node_idx - item_level_start
        coverage_i = items[agg_id - 1]['coverage_id'] - 1
        if not used_cov[coverage_i]:
            used_cov[coverage_i] = 1
            tiv += coverages[coverage_i]
    return tiv


@njit(cache=True)
def prepare_profile_simple(profile, tiv):
    # if use TIV convert calcrule to fix deductible
    if profile['calcrule_id'] == 4:
        profile['calcrule_id'] = 1
        profile['deductible1'] *= tiv

    elif profile['calcrule_id'] == 6:
        profile['calcrule_id'] = 12
        profile['deductible1'] *= tiv

    elif profile['calcrule_id'] == 18:
        profile['calcrule_id'] = 2
        profile['deductible1'] *= tiv

    elif profile['calcrule_id'] == 21:
        profile['calcrule_id'] = 13
        profile['deductible1'] *= tiv

    elif profile['calcrule_id'] == 9:
        profile['calcrule_id'] = 1
        profile['deductible1'] *= profile['limit1']
    elif profile['calcrule_id'] == 15:
        if profile['limit1'] >= 1:
            profile['calcrule_id'] = 12


@njit(cache=True)
def prepare_profile_stepped(profile, tiv):
    # if use TIV convert calcrule to fix deductible
    if profile['calcrule_id'] == 27:
        profile['trigger_start'] *= tiv
        if profile['trigger_end'] == 1:
            profile['trigger_end'] = np.inf
        else:
            profile['trigger_end'] *= tiv
        loss = min(max(profile['payout_start'] * tiv - profile['deductible1'], 0), profile['limit1'])
        cond_loss = min(loss * profile['scale2'], profile['limit2'])
        profile['payout_start'] = (loss + cond_loss) * (1 + profile['scale1'])

    elif profile['calcrule_id'] == 28:
        profile['trigger_start'] *= tiv
        if profile['trigger_end'] == 1:
            profile['trigger_end'] = np.inf
        else:
            profile['trigger_end'] *= tiv
        profile['scale1'] += 1

        if profile['payout_start'] == 0:  # backward compatibility v1.22.x
            profile['calcrule_id'] = 281

    elif profile['calcrule_id'] == 29:
        profile['calcrule_id'] = 27
        profile['trigger_start'] *= tiv
        if profile['trigger_end'] == 1:
            profile['trigger_end'] = np.inf
        else:
            profile['trigger_end'] *= tiv
        loss = max(profile['payout_start'] * tiv - profile['deductible1'], 0)
        cond_loss = min(loss * profile['scale2'], profile['limit2'])
        profile['payout_start'] = (loss + cond_loss) * (1 + profile['scale1'])

    elif profile['calcrule_id'] == 30:
        profile['calcrule_id'] = 27
        profile['trigger_start'] *= tiv
        if profile['trigger_end'] == 1:
            profile['trigger_end'] = np.inf
        else:
            profile['trigger_end'] *= tiv
        loss = max(profile['payout_start'] * profile['limit1'] - profile['deductible1'], 0)
        cond_loss = min(loss * profile['scale2'], profile['limit2'])
        profile['payout_start'] = (loss + cond_loss) * (1 + profile['scale1'])

    elif profile['calcrule_id'] == 31:
        profile['calcrule_id'] = 27
        profile['trigger_start'] *= tiv
        if profile['trigger_end'] == 1:
            profile['trigger_end'] = np.inf
        else:
            profile['trigger_end'] *= tiv
        loss = max(profile['payout_start'] - profile['deductible1'], 0)
        cond_loss = min(loss * profile['scale2'], profile['limit2'])
        profile['payout_start'] = (loss + cond_loss) * (1 + profile['scale1'])

    elif profile['calcrule_id'] == 32:
        profile['scale1'] += 1

    elif profile['calcrule_id'] == 37:
        profile['trigger_start'] *= tiv
        if profile['trigger_end'] == 1:
            profile['trigger_end'] = np.inf
        else:
            profile['trigger_end'] *= tiv
        profile['scale1'] += 1

    elif profile['calcrule_id'] == 38:
        profile['trigger_start'] *= tiv
        if profile['trigger_end'] == 1:
            profile['trigger_end'] = np.inf
        else:
            profile['trigger_end'] *= tiv
        profile['scale1'] += 1

    else:
        prepare_profile_simple(profile, tiv)


@njit(cache=True)
def extract_financial_structure(allocation_rule, fm_programme, fm_policytc, fm_profile, stepped, fm_xref, items, coverages):
    """
    :param allocation_rule:
        option to indicate out the loss are allocated to the output
    :param fm_programme:
        structure of the levels
    :param fm_policytc:
        structure of the layers and policy_id to apply
    :param fm_profile:
        definition of the policy_id
    :param fm_xref:
        mapping between the output of the allocation and output item_id
    :return:
        compute_infos:
        nodes_array:
        node_parents_array:
        node_profiles_array:
        output_array:
    """
    ##### profile_id_to_profile_index ####
    # policies may have multiple step, create a mapping between profile_id and the start and end index in fm_profile file
    max_profile_id = np.max(fm_profile['profile_id'])
    profile_id_to_profile_index = np.empty(max_profile_id + 1, dtype=profile_index_dtype)
    # is_tiv_profile[profile_id] = 1 if profile requires TIV calculation
    is_tiv_profile = np.zeros(max_profile_id + 1, dtype=np.uint8)
    last_profile_id = 0  # real profile_id start at 1
    for i in range(fm_profile.shape[0]):
        if fm_profile[i]['calcrule_id'] in need_tiv_policy:
            is_tiv_profile[fm_profile[i]['profile_id']] = 1
        profile_id_to_profile_index[fm_profile[i]['profile_id']]['i_end'] = i + 1
        if last_profile_id != fm_profile[i]['profile_id']:
            profile_id_to_profile_index[fm_profile[i]['profile_id']]['i_start'] = i
            last_profile_id = fm_profile[i]['profile_id']

    # in fm_programme check if multi-peril and get size of each levels
    max_level = np.max(fm_programme['level_id'])
    level_node_len = np.zeros(max_level + 1, dtype=oasis_int)
    multi_peril = False
    for i in range(fm_programme.shape[0]):
        programme = fm_programme[i]
        if programme['level_id'] == 1 and programme['from_agg_id'] != programme['to_agg_id']:
            multi_peril = True
        if level_node_len[programme['level_id'] - 1] < programme['from_agg_id']:
            level_node_len[programme['level_id'] - 1] = programme['from_agg_id']

        if level_node_len[programme['level_id']] < programme['to_agg_id']:
            level_node_len[programme['level_id']] = programme['to_agg_id']

    ##### fm_policytc (level_id agg_id layer_id => profile_id) #####
    # Pre-pass: Count TIV profile duplicates needed (first occurrence uses original, subsequent need copies)
    # tiv_first_seen[profile_id] = 1 after first occurrence
    tiv_first_seen = np.zeros(max_profile_id + 1, dtype=np.uint8)
    num_tiv_duplicates = 0
    for i in range(fm_policytc.shape[0]):
        policytc = fm_policytc[i]
        profile_id = nb_oasis_int(policytc['profile_id'])
        if is_tiv_profile[profile_id]:
            if tiv_first_seen[profile_id]:
                # Subsequent occurrence - needs duplicate profile entries
                i_start = profile_id_to_profile_index[profile_id]['i_start']
                i_end = profile_id_to_profile_index[profile_id]['i_end']
                num_tiv_duplicates += i_end - i_start
            else:
                tiv_first_seen[profile_id] = 1

    # Create expanded fm_profile with TIV duplicates pre-allocated
    original_fm_profile_len = fm_profile.shape[0]
    if num_tiv_duplicates > 0:
        new_fm_profile = np.empty(original_fm_profile_len + num_tiv_duplicates, dtype=fm_profile.dtype)
        new_fm_profile[:original_fm_profile_len] = fm_profile[:]
        fm_profile = new_fm_profile

    # fm_xref
    if multi_peril:  # if single peril we can skip item level computation (level 0)
        start_level = nb_oasis_int(0)
    else:
        start_level = nb_oasis_int(1)

    if start_level == max_level:  # there is only one level we can switch the computation as if it were a0
        allocation_rule = 0

    if allocation_rule == 0:
        out_level = nb_oasis_int(max_level)
    else:
        out_level = start_level

    # Compute node_level_start for array-based indexing
    # node_level_start[level] gives the starting index for nodes at that level
    node_level_start = np.zeros(level_node_len.shape[0] + 1, oasis_int)
    for i in range(start_level, level_node_len.shape[0]):
        node_level_start[i + 1] = node_level_start[i] + level_node_len[i]
    total_nodes = node_level_start[-1] + 1

    # Build profiles CSR directly from fm_policytc (no intermediate dict)
    # Pass 1: Count profiles per node
    profiles_count = np.zeros(total_nodes, dtype=oasis_int)
    for i in range(fm_policytc.shape[0]):
        policytc = fm_policytc[i]
        node_idx = node_level_start[policytc['level_id']] + policytc['agg_id']
        profiles_count[node_idx] += 1

    # Build profiles_indptr
    profiles_indptr = np.zeros(total_nodes + 1, dtype=oasis_int)
    for i in range(total_nodes):
        profiles_indptr[i + 1] = profiles_indptr[i] + profiles_count[i]

    # Allocate profiles_data with structured dtype (layer_id, i_start, i_end)
    profiles_data = np.empty(profiles_indptr[-1], dtype=profile_entry_dtype)

    # Pass 2: Fill profiles_data directly (handling TIV duplicates)
    profiles_cursor = np.zeros(total_nodes, dtype=oasis_int)
    tiv_first_seen[:] = 0  # Reset for second pass
    i_new_fm_profile = original_fm_profile_len

    for i in range(fm_policytc.shape[0]):
        policytc = fm_policytc[i]
        profile_id = nb_oasis_int(policytc['profile_id'])
        node_idx = node_level_start[policytc['level_id']] + policytc['agg_id']

        i_start = profile_id_to_profile_index[profile_id]['i_start']
        i_end = profile_id_to_profile_index[profile_id]['i_end']

        # Handle TIV profile duplication
        if is_tiv_profile[profile_id]:
            if tiv_first_seen[profile_id]:
                # Subsequent occurrence - create duplicate in expanded fm_profile
                old_i_start, old_i_end = i_start, i_end
                i_start = i_new_fm_profile
                for j in range(old_i_start, old_i_end):
                    fm_profile[i_new_fm_profile] = fm_profile[j]
                    i_new_fm_profile += 1
                i_end = i_new_fm_profile
            else:
                tiv_first_seen[profile_id] = 1

        # Fill CSR data
        pos = profiles_indptr[node_idx] + profiles_cursor[node_idx]
        profiles_data[pos]['layer_id'] = policytc['layer_id']
        profiles_data[pos]['i_start'] = i_start
        profiles_data[pos]['i_end'] = i_end
        profiles_cursor[node_idx] += 1

    # Sort profiles within each node by layer_id (in-place bubble sort)
    for node_idx in range(total_nodes):
        start = profiles_indptr[node_idx]
        end = profiles_indptr[node_idx + 1]
        n = end - start
        if n > 1:
            # Bubble sort - simple and works well for small n
            for j in range(n):
                for k in range(start, end - 1 - j):
                    if profiles_data[k]['layer_id'] > profiles_data[k + 1]['layer_id']:
                        # Swap entries
                        temp_layer = profiles_data[k]['layer_id']
                        temp_i_start = profiles_data[k]['i_start']
                        temp_i_end = profiles_data[k]['i_end']
                        profiles_data[k]['layer_id'] = profiles_data[k + 1]['layer_id']
                        profiles_data[k]['i_start'] = profiles_data[k + 1]['i_start']
                        profiles_data[k]['i_end'] = profiles_data[k + 1]['i_end']
                        profiles_data[k + 1]['layer_id'] = temp_layer
                        profiles_data[k + 1]['i_start'] = temp_i_start
                        profiles_data[k + 1]['i_end'] = temp_i_end

    ##### xref #####
    # Create 2D array mapping (agg_id, layer_id) => output_id for out_level nodes
    # output_id_arr[agg_id, layer_id] = output_id (0 = no output)
    max_agg_id_out = level_node_len[out_level]
    max_layer_id = np.max(fm_xref['layer_id']) if fm_xref.shape[0] > 0 else 1
    output_id_arr = np.zeros((max_agg_id_out + 1, max_layer_id + 1), dtype=oasis_int)

    output_len = 0
    for i in range(fm_xref.shape[0]):
        xref = fm_xref[i]
        if output_len < xref['output']:
            output_len = nb_oasis_int(xref['output'])

        output_id_arr[xref['agg_id'], xref['layer_id']] = xref['output']

    ##### programme ####
    # node_layers will contain the number of layers for each node
    # Using array indexed by node_level_start[level] + agg_id (0 = not set)
    node_layers_arr = np.zeros(total_nodes, dtype=oasis_int)
    # node_cross_layers tracks cross-layer nodes (0 = false, 1 = true)
    node_cross_layers_arr = np.zeros(total_nodes, dtype=np.uint8)
    # layer_source tracks where each node gets its layer list from (index to source node)
    # If layer_source[idx] == idx, node uses its own profiles from programme_node_to_profiles
    # Otherwise, it inherits from layer_source[idx]
    layer_source = np.arange(total_nodes, dtype=oasis_int)  # default: each node is its own source

    # fill up node_layers with the number of policies for each node
    for programme in fm_programme:
        parent_level = nb_oasis_int(programme['level_id'])
        parent_agg = nb_oasis_int(programme['to_agg_id'])
        parent_idx = node_level_start[parent_level] + parent_agg
        if node_layers_arr[parent_idx] == 0:
            # Use CSR format: profiles_indptr[idx+1] - profiles_indptr[idx] = count
            node_layers_arr[parent_idx] = profiles_indptr[parent_idx + 1] - profiles_indptr[parent_idx]
            # layer_source[parent_idx] = parent_idx already (uses own profiles)

    # Build parent/child CSR arrays directly from fm_programme (no intermediate dicts)
    # This uses a two-pass approach: count first, then fill

    # Pass 1: Count children per parent and parents per child
    children_count = np.zeros(total_nodes, dtype=oasis_int)
    parents_count = np.zeros(total_nodes, dtype=oasis_int)
    children_len = 1
    parents_len = 0

    for i in range(fm_programme.shape[0]):
        programme = fm_programme[i]
        if programme['level_id'] > start_level:
            parent_idx = node_level_start[programme['level_id']] + programme['to_agg_id']

            if programme['from_agg_id'] > 0:
                child_idx = node_level_start[programme['level_id'] - 1] + programme['from_agg_id']
            else:
                child_idx = node_level_start[start_level] + (-programme['from_agg_id'])

            children_count[parent_idx] += 1
            parents_count[child_idx] += 1
            parents_len += 1

    # Build CSR indptr arrays
    children_indptr = np.zeros(total_nodes + 1, dtype=oasis_int)
    for i in range(total_nodes):
        children_indptr[i + 1] = children_indptr[i] + children_count[i]
        if children_count[i] > 0:
            children_len += 1 + children_count[i]

    parents_indptr = np.zeros(total_nodes + 1, dtype=oasis_int)
    for i in range(total_nodes):
        parents_indptr[i + 1] = parents_indptr[i] + parents_count[i]

    # Allocate CSR data arrays
    children_data = np.empty(children_indptr[-1], dtype=oasis_int)
    parents_data = np.empty(parents_indptr[-1], dtype=oasis_int)

    # Pass 2: Fill CSR data arrays (iterate level by level like original to preserve order)
    children_cursor = np.zeros(total_nodes, dtype=oasis_int)
    parents_cursor = np.zeros(total_nodes, dtype=oasis_int)

    # Iterate from max_level down to start_level to match original parent ordering
    for level in range(max_level, start_level, -1):
        for i in range(fm_programme.shape[0]):
            programme = fm_programme[i]
            if programme['level_id'] == level:
                parent_idx = node_level_start[programme['level_id']] + programme['to_agg_id']

                if programme['from_agg_id'] > 0:
                    child_idx = node_level_start[programme['level_id'] - 1] + programme['from_agg_id']
                else:
                    child_idx = node_level_start[start_level] + (-programme['from_agg_id'])

                # Add child to parent's children list
                c_pos = children_indptr[parent_idx] + children_cursor[parent_idx]
                children_data[c_pos] = child_idx
                children_cursor[parent_idx] += 1

                # Add parent to child's parents list (insert at front for correct order)
                # We fill from the end to maintain insertion order (like insert(0, parent))
                p_start = parents_indptr[child_idx]
                p_end = parents_indptr[child_idx + 1]
                p_pos = p_end - 1 - parents_cursor[child_idx]
                parents_data[p_pos] = parent_idx
                parents_cursor[child_idx] += 1

    # Now process layer propagation and cross-layer detection using CSR
    # Go through each level from top to bottom
    for level in range(max_level, start_level, -1):
        for i in range(fm_programme.shape[0]):
            programme = fm_programme[i]
            if programme['level_id'] == level:
                parent_idx = node_level_start[programme['level_id']] + programme['to_agg_id']

                if programme['from_agg_id'] > 0:
                    child_idx = node_level_start[programme['level_id'] - 1] + programme['from_agg_id']
                else:
                    child_idx = node_level_start[start_level] + (-programme['from_agg_id'])

                if node_layers_arr[child_idx] == 0 or node_layers_arr[child_idx] <= node_layers_arr[parent_idx]:
                    node_layers_arr[child_idx] = node_layers_arr[parent_idx]
                    # Child inherits layer source from parent (follow chain if parent also inherits)
                    layer_source[child_idx] = layer_source[parent_idx]
                elif node_layers_arr[child_idx] > node_layers_arr[parent_idx]:  # cross layer node
                    # Use CSR-based get_all_parent_csr instead of dict-based get_all_parent
                    start_nodes = np.array([parent_idx], dtype=oasis_int)
                    grand_parents_arr, grand_parents_len = get_all_parent_csr(
                        start_nodes, 1, parents_indptr, parents_data,
                        max_level, node_level_start, total_nodes)
                    for gp_i in range(grand_parents_len):
                        gp_idx = grand_parents_arr[gp_i]
                        if node_layers_arr[gp_idx] < node_layers_arr[child_idx]:
                            node_cross_layers_arr[gp_idx] = nb_oasis_int(1)
                            node_layers_arr[parent_idx] = node_layers_arr[child_idx]

    # compute number of steps (steps), max size of node to compute (compute_len)
    # Note: node_level_start was computed earlier for array-based indexing
    steps = max_level + (1 - start_level)
    compute_len = node_level_start[-1] + steps + level_node_len[-1] + 1

    # Compute output_array_size using array-based iteration
    output_array_size = 0
    for agg_id in range(1, level_node_len[out_level] + 1):
        node_idx = node_level_start[out_level] + agg_id
        output_array_size += node_layers_arr[node_idx]

    nodes_array = np.empty(node_level_start[-1] + 1, dtype=nodes_array_dtype)
    node_parents_array = np.empty(parents_len, dtype=oasis_int)
    node_profiles_array = np.zeros(fm_policytc.shape[0] + 1, dtype=profile_index_dtype)
    output_array = np.zeros(output_array_size, dtype=oasis_int)

    node_i = 1
    children_i = 1
    parents_i = 0
    profile_i = 1
    loss_i = 0
    extra_i = 0
    output_i = 0

    for level in range(start_level, max_level + 1):
        for agg_id in range(1, level_node_len[level] + 1):
            node_programme = (nb_oasis_int(level), nb_oasis_int(agg_id))
            node = nodes_array[node_i]
            node['node_id'] = node_i
            node_i += 1
            node['level_id'] = level
            node['agg_id'] = agg_id

            # layers
            node_idx = node_level_start[level] + agg_id
            node['layer_len'] = node_layers_arr[node_idx]
            node['cross_layer_profile'] = 0  # set default to 0 change if it is a cross_layer_profile after
            node['loss'], loss_i = loss_i, loss_i + node['layer_len']
            if level == start_level:
                node['net_loss'], loss_i = loss_i, loss_i + 1

            node['extra'] = null_index
            node['is_reallocating'] = 0

            # children - use CSR format
            num_children = children_indptr[node_idx + 1] - children_indptr[node_idx]
            if num_children > 0:
                node['children'], children_i = children_i, children_i + 1 + num_children
            else:
                node['children'] = 0

            # parent - use CSR format
            p_start = parents_indptr[node_idx]
            p_end = parents_indptr[node_idx + 1]
            num_parents = p_end - p_start
            if num_parents > 0:
                node['parent_len'] = num_parents
                node['parent'] = parents_i
                for pi in range(p_start, p_end):
                    node_parents_array[parents_i], parents_i = parents_data[pi], nb_oasis_int(parents_i + 1)
            else:
                node['parent_len'] = 0

            # profiles - use CSR format
            prof_start = profiles_indptr[node_idx]
            prof_end = profiles_indptr[node_idx + 1]
            num_profiles = prof_end - prof_start
            if num_profiles > 0:
                if node_cross_layers_arr[node_idx]:
                    node['profile_len'] = node['layer_len']
                    node['cross_layer_profile'] = 1
                else:
                    node['profile_len'] = num_profiles
                node['profiles'] = profile_i

                # Iterate through profiles from CSR (already sorted by layer_id)
                for prof_idx in range(prof_start, prof_end):
                    layer_id = profiles_data[prof_idx]['layer_id']
                    i_start = profiles_data[prof_idx]['i_start']
                    i_end = profiles_data[prof_idx]['i_end']

                    node_profile, profile_i = node_profiles_array[profile_i], profile_i + 1
                    node_profile['i_start'] = i_start
                    node_profile['i_end'] = i_end

                    # if use TIV we compute it and precompute % TIV values
                    for profile_index in range(i_start, i_end):
                        if fm_profile[profile_index]['calcrule_id'] in need_tiv_policy:
                            all_children_arr, all_children_len = get_all_children_csr(
                                node_idx, children_indptr, children_data, True, total_nodes)
                            tiv = get_tiv_csr(all_children_arr, all_children_len, items, coverages,
                                              node_level_start, start_level)
                            break
                    else:
                        tiv = 0

                    for profile_index in range(i_start, i_end):
                        profile = fm_profile[profile_index]
                        if stepped is None:
                            prepare_profile_simple(profile, tiv)
                        else:
                            prepare_profile_stepped(profile, tiv)
                        if does_nothing(profile):
                            # only non step policy can "does_nothing" so this is safe
                            node_profile['i_end'] = node_profile['i_start']

                    # check if we need to compute extras (min and max ded policies)
                    for profile_index in range(i_start, i_end):
                        if fm_profile[profile_index]['calcrule_id'] in need_extras:
                            node['is_reallocating'] = allocation_rule == 2 and fm_profile[profile_index]['calcrule_id'] != 27

                            items_child_arr, items_child_len = get_all_children_csr(
                                node_idx, children_indptr, children_data, True, total_nodes)
                            all_parent_arr, all_parent_len = get_all_parent_csr(
                                items_child_arr, items_child_len, parents_indptr, parents_data,
                                level, node_level_start, total_nodes)

                            for pi in range(all_parent_len):
                                parent_node_idx = all_parent_arr[pi]
                                all_children_arr, all_children_len = get_all_children_csr(
                                    parent_node_idx, children_indptr, children_data, False, total_nodes)
                                for ci in range(all_children_len):
                                    child_node_idx = all_children_arr[ci]
                                    child = nodes_array[child_node_idx]
                                    if child['extra'] == null_index:
                                        child['extra'], extra_i = extra_i, extra_i + node['layer_len']

                            break

            else:  # item level has no profile
                node['profile_len'] = 1
                node['profiles'] = 0

            if level == out_level:
                # Check if any layer for this agg_id has an output_id
                has_output = False
                for layer_idx in range(output_id_arr.shape[1]):
                    if output_id_arr[agg_id, layer_idx] != 0:
                        has_output = True
                        break
                if has_output:
                    node['output_ids'], output_i = output_i, output_i + node['layer_len']
                    # Get the source node for layers (could be this node or an ancestor)
                    source_idx = layer_source[node_idx]
                    # Use CSR format directly with source_idx
                    src_prof_start = profiles_indptr[source_idx]
                    src_prof_end = profiles_indptr[source_idx + 1]
                    for i in range(src_prof_end - src_prof_start):
                        layer_id = profiles_data[src_prof_start + i]['layer_id']
                        if output_id_arr[agg_id, layer_id] != 0:
                            output_array[node['output_ids'] + i] = output_id_arr[agg_id, layer_id]
                else:
                    raise KeyError("Some output nodes are missing output_ids")

    compute_infos = np.empty(1, dtype=compute_info_dtype)
    compute_info = compute_infos[0]
    compute_info['allocation_rule'] = allocation_rule
    compute_info['max_level'] = max_level
    compute_info['node_len'] = node_i
    compute_info['children_len'] = children_i
    compute_info['parents_len'] = parents_i
    compute_info['profile_len'] = profile_i
    compute_info['loss_len'] = loss_i
    compute_info['extra_len'] = extra_i
    compute_info['compute_len'] = compute_len
    compute_info['start_level'] = start_level
    compute_info['items_len'] = level_node_len[0]
    compute_info['output_len'] = output_len
    compute_info['stepped'] = stepped is not None
    compute_info['max_layer'] = max(nodes_array['layer_len'][1:])

    return compute_infos, nodes_array, node_parents_array, node_profiles_array, output_array, fm_profile


def create_financial_structure(allocation_rule, static_path):
    """
    :param allocation_rule: int
            back-allocation rule
    :param static_path: string
            path to the static files
    :return:
        compute_queue : the step of the computation to perform on each event
        node_indexes : map node to index of item in result array
        index_dependencies : map node to its dependent indexes
        node_profile : map node to profile
        output_item_index : list of item_id, index to put in the output
    """

    if allocation_rule not in allowed_allocation_rule:
        raise ValueError(f"allocation_rule must be in {allowed_allocation_rule}, found {allocation_rule}")
    if allocation_rule == 3:
        allocation_rule = 2

    fm_programme, fm_policytc, fm_profile, stepped, fm_xref, items, coverages = load_static(static_path)
    financial_structure = extract_financial_structure(allocation_rule, fm_programme, fm_policytc, fm_profile,
                                                      stepped, fm_xref, items, coverages)
    compute_info, nodes_array, node_parents_array, node_profiles_array, output_array, fm_profile = financial_structure
    logger.info(f'nodes_array has {len(nodes_array)} elements')
    logger.info(f'compute_info : {dict(zip(compute_info.dtype.names, compute_info[0]))}')

    np.save(os.path.join(static_path, f'compute_info_{allocation_rule}'), compute_info)
    np.save(os.path.join(static_path, f'nodes_array_{allocation_rule}'), nodes_array)
    np.save(os.path.join(static_path, f'node_parents_array_{allocation_rule}'), node_parents_array)
    np.save(os.path.join(static_path, f'node_profiles_array_{allocation_rule}'), node_profiles_array)
    np.save(os.path.join(static_path, f'output_array_{allocation_rule}'), output_array)
    np.save(os.path.join(static_path, 'fm_profile'), fm_profile)


def load_financial_structure(allocation_rule, static_path):
    compute_info = np.load(os.path.join(static_path, f'compute_info_{allocation_rule}.npy'), mmap_mode='r')
    nodes_array = np.load(os.path.join(static_path, f'nodes_array_{allocation_rule}.npy'), mmap_mode='r')
    node_parents_array = np.load(os.path.join(static_path, f'node_parents_array_{allocation_rule}.npy'), mmap_mode='r')
    node_profiles_array = np.load(os.path.join(static_path, f'node_profiles_array_{allocation_rule}.npy'), mmap_mode='r')
    output_array = np.load(os.path.join(static_path, f'output_array_{allocation_rule}.npy'), mmap_mode='r')
    fm_profile = np.load(os.path.join(static_path, 'fm_profile.npy'), mmap_mode='r')

    return compute_info, nodes_array, node_parents_array, node_profiles_array, output_array, fm_profile
