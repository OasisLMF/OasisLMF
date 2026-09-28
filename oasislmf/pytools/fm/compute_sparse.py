"""Financial Module (FM) Sparse Computation Engine
================================================

This module implements the core loss computation for the Oasis Financial Module using
sparse array storage. It processes insurance/reinsurance losses through a hierarchical
node structure, applying financial terms (deductibles, limits, etc.) at each level.

Architecture Overview
---------------------
The FM computation uses a bottom-up traversal of a tree structure where:
- Leaf nodes (items) contain ground-up losses from the GUL stream
- Internal nodes aggregate losses from children and apply financial profiles
- The root represents the final insured/reinsured loss output

Storage Model
-------------
Data is stored using a CSR (Compressed Sparse Row) inspired format:
- sidx_indptr[i:i+1] gives the range of sample indices for node i
- sidx_val contains the actual sample index values (sidx)
- loss_indptr[i:i+1] gives the range of loss values for loss pointer i
- loss_val contains the actual loss values aligned with sidx_val
- extras_indptr/extras_val similarly store deductible/overlimit/underlimit values

Computation Flow
----------------
For each event:

1. Read losses from input stream into sparse arrays
2. For each level (bottom to top):

   a. Aggregate children losses into parent nodes
   b. Apply financial profiles (calc rules) to compute insured loss
   c. Back-allocate results to base children (for allocation rules 1 & 2)
   d. Queue parent nodes for the next level

3. Output final losses to stream

Key Concepts
------------
- profile_len: Number of profiles for a node (may differ from layer_len for step policies)
- layer_len: Number of layers in output
- cross_layer_profile: When True, one profile applies to merged loss across all layers
- base_children: The leaf-level descendants of a node (used for back allocation)
- allocation_rule: 0=no allocation, 1=proportional to input, 2=proportional to output
"""

from oasislmf.pytools.common.data import oasis_float, oasis_int, null_index
from oasislmf.pytools.common.event_stream import (MAX_LOSS_IDX, MEAN_IDX, NUM_SPECIAL_SIDX, TIV_IDX,
                                                  decode_local_sidx)
from .policy import calc
from .policy_extras import calc as calc_extra
from .common import EXTRA_SIDX_COUNT, compute_idx_dtype, DEDUCTIBLE, UNDERLIMIT, OVERLIMIT
from .back_allocation import back_alloc_a2, back_alloc_extra_a2, back_alloc_layer, back_alloc_layer_extra

from numba import njit, objmode
import numpy as np
import os
import time
import logging
logger = logging.getLogger(__name__)

# Verify the dense per-node temporaries are clean when a node starts. They are cleared over
# the exact sidx a node touched rather than wholesale, which is only correct if the clearing
# set covers every writer -- and the writers span this module and back_allocation. numba folds
# this constant at compile time, so the scan costs nothing while it is False. Turn it on and
# run tests/fm/test_fmpy.py + tests/pytools/fm to prove the set rather than argue it.
DEBUG_TEMPS = False

# Per-level profiling of compute_event. numba cannot call time.perf_counter in nopython, and
# objmode is far too slow per node -- but at the level boundary it runs 2x per level per event,
# which is free. Counters are plain integer adds. numba folds the constant, so this costs
# nothing while it is False. Enable it and set FM_PROFILE_OUT to a path to get a TSV.
DEBUG_PROFILE = False
PROFILE_LEVELS = 24
# 0 seconds  1 nodes  2 output values  3 aggregate-path  4 adopt-path  5 leaf-path
PROFILE_METRICS = 6


@njit(cache=True)
def assert_temps_clean(temp_node_loss, temp_node_extras, node_id):
    """Abort if a dense temporary still holds a value written for an earlier node."""
    for profile_i in range(temp_node_loss.shape[0]):
        for i in range(temp_node_loss.shape[1]):
            if temp_node_loss[profile_i, i] != 0:
                print("DIRTY temp_node_loss: node", node_id, "profile", profile_i,
                      "idx", i, "value", temp_node_loss[profile_i, i])
                raise ValueError("temp_node_loss dirty at node entry")
    for profile_i in range(temp_node_extras.shape[0]):
        for i in range(temp_node_extras.shape[1]):
            for j in range(temp_node_extras.shape[2]):
                if temp_node_extras[profile_i, i, j] != 0:
                    print("DIRTY temp_node_extras: node", node_id, "profile", profile_i,
                          "idx", i, "col", j, "value", temp_node_extras[profile_i, i, j])
                    raise ValueError("temp_node_extras dirty at node entry")


@njit(cache=True, inline='always')
def collapses_buildings(node, child, site_collapse_level, building_packing):
    """Whether aggregating ``child`` into ``node`` crosses the building-packing collapse point.

    ``site_collapse_level`` is the last level whose terms apply per building, and 0 is a
    legitimate value, not a sentinel: an input set with no location terms writes no risk-keyed
    level, so the buildings merge as soon as the items are aggregated. That is why
    ``building_packing`` gates this rather than a truthiness test on the level.
    """
    return (building_packing
            and child['level_id'] <= site_collapse_level < node['level_id'])


@njit(cache=True)
def get_base_children(node, children, nodes_array, temp_children_queue):
    """Find all leaf-level (base) descendants of a node using breadth-first traversal.

    Base children are the nodes at the lowest level that have no children themselves.
    These are needed for back allocation - when we apply financial terms at a higher
    level, we need to distribute the resulting losses back to the original items.

    Algorithm:
    1. Start with the direct children of the node
    2. For each child, check if it has its own children
    3. If yes, add those grandchildren to the queue and continue
    4. If no, this is a base child - store it at the front of the array
    5. Continue until all descendants are processed

    The result is temp_children_queue[0:base_child_i] containing all base children.

    Args:
        node: The parent node whose base children we want to find
        children: Array tracking children for each node (children[node['children']] = count,
                  followed by child node IDs)
        nodes_array: Array of all node information
        temp_children_queue: Working array to store the traversal queue and final results

    Returns:
        int: Number of base children found
    """
    children_count = children[node['children']]
    if children_count:
        temp_children_queue[:children_count] = children[node['children'] + 1: node['children'] + children_count + 1]
        temp_children_queue[children_count] = null_index
        queue_end = children_count
        queue_i = 0
        base_child_i = 0
        while temp_children_queue[queue_i] != null_index:
            parent = nodes_array[temp_children_queue[queue_i]]
            children_count = children[parent['children']]
            if children_count:
                temp_children_queue[queue_end: queue_end + children_count] = children[parent['children'] + 1: parent['children'] + children_count + 1]
                queue_end += children_count
                temp_children_queue[queue_end] = null_index
            else:
                temp_children_queue[base_child_i] = temp_children_queue[queue_i]
                base_child_i += 1
            queue_i += 1
    else:
        temp_children_queue[0] = node['node_id']
        base_child_i = 1
    return base_child_i


@njit(cache=True, fastmath=True)
def collapse_packed_storage(node, compute_idx, max_sidx_val, has_net_loss,
                            sidx_indexes, sidx_indptr, sidx_val,
                            loss_indptr, loss_val, extras_indptr, extras_val,
                            collapse_loss, collapse_extras, collapse_net):
    """Sum one node's building blocks onto the local sidx they decode to, in fresh storage.

    Storage cannot be collapsed where it lies: the arrays are one arena and a node's range ends
    where the next one starts. So append at the bump pointer and repoint the node, exactly as an
    aggregation does for a parent; the old slice becomes dead space the capacity bound allows for.

    Returns:
        bool: True if the node carried packed indices and has been collapsed, False if it was
        already ordinary and nothing was written.
    """
    node_sidx_i = sidx_indexes[node['node_id']]
    start = sidx_indptr[node_sidx_i]
    val_count = sidx_indptr[node_sidx_i + 1] - start
    if val_count == 0:
        return False

    packed = False
    for val_i in range(val_count):
        sidx = sidx_val[start + val_i]
        if sidx > max_sidx_val or sidx < -NUM_SPECIAL_SIDX:
            packed = True
            break
    if not packed:
        return False

    has_extras = node['extra'] != null_index
    layer_count = node['layer_len']
    collapse_loss[:layer_count].fill(0)
    if has_extras:
        collapse_extras[:layer_count].fill(0)

    for layer_i in range(layer_count):
        leaf_loss_start = loss_indptr[node['loss'] + layer_i]
        for val_i in range(val_count):
            local_sidx = decode_local_sidx(sidx_val[start + val_i], max_sidx_val)
            collapse_loss[layer_i, local_sidx] += loss_val[leaf_loss_start + val_i]
        if has_extras:
            leaf_extra_start = extras_indptr[node['extra'] + layer_i]
            for val_i in range(val_count):
                local_sidx = decode_local_sidx(sidx_val[start + val_i], max_sidx_val)
                for extra_i in range(3):
                    collapse_extras[layer_i, local_sidx, extra_i] += extras_val[leaf_extra_start + val_i, extra_i]

    if has_net_loss:
        collapse_net.fill(0)
        leaf_net_start = loss_indptr[node['net_loss']]
        for val_i in range(val_count):
            local_sidx = decode_local_sidx(sidx_val[start + val_i], max_sidx_val)
            collapse_net[local_sidx] += loss_val[leaf_net_start + val_i]

    _emit_collapsed(node, compute_idx, max_sidx_val, has_net_loss, has_extras, layer_count,
                    sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val,
                    extras_indptr, extras_val, collapse_loss, collapse_extras, collapse_net)
    return True


@njit(cache=True, fastmath=True)
def _emit_collapsed(node, compute_idx, max_sidx_val, has_net_loss, has_extras, layer_count,
                    sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val,
                    extras_indptr, extras_val, collapse_loss, collapse_extras, collapse_net):
    """Append the canonical collapsed slice for ``node`` and repoint it there.

    The index array is written in stream order -- the specials then 1..S -- rather than sorted,
    so the writer's positional read of -5/-3/-1 holds without anything having to sort.
    """
    new_start = compute_idx['sidx_ptr_i']
    sidx_indexes[node['node_id']] = compute_idx['sidx_i']
    compute_idx['sidx_i'] += 1
    for special_idx in (MAX_LOSS_IDX, TIV_IDX, MEAN_IDX):
        sidx_val[compute_idx['sidx_ptr_i']] = special_idx
        compute_idx['sidx_ptr_i'] += 1
    for sample_idx in range(1, max_sidx_val + 1):
        sidx_val[compute_idx['sidx_ptr_i']] = sample_idx
        compute_idx['sidx_ptr_i'] += 1
    sidx_indptr[compute_idx['sidx_i']] = compute_idx['sidx_ptr_i']
    new_val_count = compute_idx['sidx_ptr_i'] - new_start

    if has_net_loss:
        loss_indptr[node['net_loss']] = compute_idx['loss_ptr_i']
        for val_i in range(new_val_count):
            loss_val[compute_idx['loss_ptr_i']] = collapse_net[sidx_val[new_start + val_i]]
            compute_idx['loss_ptr_i'] += 1

    for layer_i in range(layer_count):
        loss_indptr[node['loss'] + layer_i] = compute_idx['loss_ptr_i']
        for val_i in range(new_val_count):
            loss_val[compute_idx['loss_ptr_i']] = collapse_loss[layer_i, sidx_val[new_start + val_i]]
            compute_idx['loss_ptr_i'] += 1
        if has_extras:
            extras_indptr[node['extra'] + layer_i] = compute_idx['extras_ptr_i']
            for val_i in range(new_val_count):
                for extra_i in range(3):
                    extras_val[compute_idx['extras_ptr_i'], extra_i] = collapse_extras[
                        layer_i, sidx_val[new_start + val_i], extra_i]
                compute_idx['extras_ptr_i'] += 1


@njit(cache=True, fastmath=True)
def collapse_site_node(compute_node, storage_node, children, nodes_array, temp_children_queue,
                       compute_idx, max_sidx_val, keep_input_loss, start_level, collapse_leaves,
                       sidx_indexes, sidx_indptr, sidx_val,
                       loss_indptr, loss_val, extras_indptr, extras_val,
                       collapse_loss, collapse_extras, collapse_net):
    """Merge the buildings at the last level whose terms apply per building.

    Runs once this node has applied its terms and back-allocated, so every per-building term is
    in and the allocation that had to land per building already has. Everything above is keyed on
    acc_id or CondTag and cannot tell buildings apart, so from here the stream is ordinary.

    Doing it here rather than from the first level above keeps it local: a site node collapses its
    own items, where a Cond/Pol node above would walk its whole subtree to find them and every
    further level would walk it again to find nothing left to do. The site level is always present
    when it is needed -- buildings only stay apart because some site level applies a term per
    building, and without one the reader sums them away instead.

    Order matters between the two halves. The leaves go first and the node last: back-allocation
    from above looks its factors up by sample index, so the node and the leaves under it have to
    cross together, and a node collapsed ahead of its leaves would hand back-allocation packed
    indices and read building 1's for every building.

    ``collapse_leaves`` is false only for allocation rule 0, where nothing back-allocates and no
    leaf is ever read again, so the node's own storage is all that has to come out collapsed.
    """
    # net_loss is allocated only for start_level nodes, and nodes_array is np.empty, so the field
    # is garbage on anything above it -- the node collapsed here often is.
    if collapse_leaves:
        base_children_count = get_base_children(storage_node, children, nodes_array, temp_children_queue)
        for base_child_i in range(base_children_count):
            leaf = nodes_array[temp_children_queue[base_child_i]]
            collapse_packed_storage(leaf, compute_idx, max_sidx_val,
                                    keep_input_loss and leaf['level_id'] == start_level,
                                    sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val,
                                    extras_indptr, extras_val, collapse_loss, collapse_extras, collapse_net)

    collapse_packed_storage(storage_node, compute_idx, max_sidx_val,
                            keep_input_loss and storage_node['level_id'] == start_level,
                            sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val,
                            extras_indptr, extras_val, collapse_loss, collapse_extras, collapse_net)
    if compute_node['node_id'] != storage_node['node_id']:
        sidx_indexes[compute_node['node_id']] = sidx_indexes[storage_node['node_id']]
        for layer_i in range(compute_node['layer_len']):
            loss_indptr[compute_node['loss'] + layer_i] = loss_indptr[storage_node['loss'] + layer_i]


@njit(cache=True)
def first_time_layer(profile_count, base_children_count, temp_children_queue, compute_idx, nodes_array,
                     sidx_indptr, sidx_indexes,
                     loss_indptr, loss_val
                     ):
    """Initialize multi-layer loss storage for base children when first encountering layered computation.

    When a node has multiple layers but its children were computed with only one layer,
    we need to create separate loss arrays for each layer. This copies the layer 0 loss
    to layers 1..N-1 so that back allocation can work independently on each layer.

    This is a lazy initialization - we only create the extra layer storage when we
    actually need it, which saves memory for nodes that never reach multi-layer parents.

    Args:
        profile_count: Number of profiles/layers to create
        base_children_count: Number of base children to process
        temp_children_queue: Array containing base children node IDs
        compute_idx: Computation state tracking pointers
        nodes_array: Array of node information
        sidx_indptr: Sample index pointers
        sidx_indexes: Node to sample index mapping
        loss_indptr: Loss value pointers
        loss_val: Loss values array
    """
    for base_child_i in range(base_children_count):
        child = nodes_array[temp_children_queue[base_child_i]]
        child_val_count = sidx_indptr[sidx_indexes[child['node_id']] + 1] - sidx_indptr[sidx_indexes[child['node_id']]]
        child_loss_val_layer_0 = loss_val[loss_indptr[child['loss']]:
                                          loss_indptr[child['loss']] + child_val_count]
        for profile_i in range(1, profile_count):
            loss_indptr[child['loss'] + profile_i] = compute_idx['loss_ptr_i']
            loss_val[compute_idx['loss_ptr_i']: compute_idx['loss_ptr_i'] + child_val_count] = child_loss_val_layer_0
            compute_idx['loss_ptr_i'] += child_val_count


@njit(cache=True)
def first_time_layer_extra(profile_count, base_children_count, temp_children_queue, compute_idx, nodes_array,
                           sidx_indptr, sidx_indexes,
                           loss_indptr, loss_val,
                           extras_indptr, extras_val,
                           ):
    """Initialize multi-layer loss AND extras storage for base children.

    Same as first_time_layer but also handles the extras array (deductible, overlimit, underlimit).
    For aggregation cases (single base child), extras are copied from layer 0.
    For back allocation cases (multiple base children), extras are zeroed for new layers
    since each layer will compute its own extras through back allocation.

    Args:
        profile_count: Number of profiles/layers to create
        base_children_count: Number of base children to process
        temp_children_queue: Array containing base children node IDs
        compute_idx: Computation state tracking pointers
        nodes_array: Array of node information
        sidx_indptr: Sample index pointers
        sidx_indexes: Node to sample index mapping
        loss_indptr: Loss value pointers
        loss_val: Loss values array
        extras_indptr: Extras value pointers
        extras_val: Extras values array (deductible, overlimit, underlimit)
    """
    for base_child_i in range(base_children_count):
        child = nodes_array[temp_children_queue[base_child_i]]
        child_val_count = sidx_indptr[sidx_indexes[child['node_id']] + 1] - sidx_indptr[sidx_indexes[child['node_id']]]
        child_loss_val_layer_0 = loss_val[loss_indptr[child['loss']]:
                                          loss_indptr[child['loss']] + child_val_count]
        if base_children_count == 1:  # aggregation case
            child_extra_val_layer_0 = extras_val[extras_indptr[child['extra']]:
                                                 extras_indptr[child['extra']] + child_val_count]
        else:  # back allocation case
            child_extra_val_layer_0 = np.zeros_like(extras_val[extras_indptr[child['extra']]:
                                                               extras_indptr[child['extra']] + child_val_count])

        for profile_i in range(1, profile_count):
            loss_indptr[child['loss'] + profile_i] = compute_idx['loss_ptr_i']
            loss_val[compute_idx['loss_ptr_i']: compute_idx['loss_ptr_i'] + child_val_count] = child_loss_val_layer_0
            compute_idx['loss_ptr_i'] += child_val_count

            extras_indptr[child['extra'] + profile_i] = compute_idx['extras_ptr_i']
            extras_val[compute_idx['extras_ptr_i']: compute_idx['extras_ptr_i'] + child_val_count] = child_extra_val_layer_0
            compute_idx['extras_ptr_i'] += child_val_count


@njit(cache=True, fastmath=True, inline='always')
def mark_node_sidx(key, temp_node_sidx, temp_node_keys, key_count):
    """Record that this node holds ``key``, the first time it is seen.

    Collecting a node's sidx by scanning the dense flag array costs the whole sidx range --
    ``max_buildings * (S + 6)`` under packing -- however few the node holds. That is 21e12
    iterations on a 2.1M-node structure with one 630,510-building location. Keeping the keys as
    they arrive makes collection cost the node's OWN size.

    Returns:
        int: the new key count.
    """
    if not temp_node_sidx[key]:
        temp_node_sidx[key] = True
        temp_node_keys[key_count] = key
        key_count += 1
    return key_count


@njit(cache=True, fastmath=True, inline='always')
def sorted_node_sidx(temp_node_keys, key_count):
    """This node's sidx in ascending order.

    The same order the dense scan produced, since that ran over an ascending range: specials
    most-negative per building first, then samples.
    """
    keys = temp_node_keys[:key_count]
    keys.sort()
    return keys


@njit(cache=True, fastmath=True)
def aggregate_children_extras(node, children_count, nodes_array, children, temp_children_queue, compute_idx,
                              site_collapse_level, building_packing, max_sidx_val,
                              temp_node_sidx, temp_node_keys, sidx_indexes, sidx_indptr, sidx_val,
                              temp_node_loss, loss_indptr, loss_val,
                              temp_node_extras, extras_indptr, extras_val):
    """Aggregate losses AND extras from multiple children into a parent node.

    Similar to aggregate_children but also tracks the "extras" - deductible amount,
    overlimit, and underlimit values that are needed for back allocation when
    financial terms modify losses.

    The extras represent:
    - DEDUCTIBLE: Amount deducted from loss (policy deductible applied)
    - OVERLIMIT: Amount exceeding the policy limit
    - UNDERLIMIT: Remaining capacity under the limit (limit - loss)

    These extras must be aggregated alongside losses so that when back allocation
    occurs, we know how to distribute the deductible/limit effects back to
    individual items proportionally.

    Args:
        node: Parent node being computed
        children_count: Number of direct children
        nodes_array: Array of all node information
        children: Children tracking array
        temp_children_queue: Working array for base children lookup
        compute_idx: Computation state pointers
        site_collapse_level: last level whose terms apply per building; children at or below it
            have their building blocks merged as they are aggregated into a node above it
        building_packing: whether this input set has packed items at all
        max_sidx_val: the stream's sample size, used to decode a packed sidx to its local one
        temp_node_sidx: Dense boolean array marking active sidx values
        temp_node_keys: scratch holding the sidx this node has marked, in arrival order
        sidx_indexes: Maps node_id to sidx array position
        sidx_indptr: Pointers into sidx_val
        sidx_val: Sample index values
        all_sidx: All possible sidx values
        temp_node_loss: Dense array for accumulating losses
        loss_indptr: Pointers into loss_val
        loss_val: Loss values
        temp_node_extras: Dense array [profile, sidx, 3] for extras accumulation
        extras_indptr: Pointers into extras_val
        extras_val: Extras values [deductible, overlimit, underlimit]

    Returns:
        int: Number of sidx values for this node
    """
    sidx_created = False
    node_sidx_start = compute_idx['sidx_ptr_i']
    node_sidx_end = 0
    sidx_indexes[node['node_id']] = compute_idx['sidx_i']
    compute_idx['sidx_i'] += 1

    for profile_i in range(node['profile_len']):
        profile_temp_node_loss = temp_node_loss[profile_i]
        profile_temp_node_extras = temp_node_extras[profile_i]
        key_count = 0

        for children_i in range(node['children'] + 1, node['children'] + children_count + 1):
            child = nodes_array[children[children_i]]
            child_sidx_val = sidx_val[sidx_indptr[sidx_indexes[child['node_id']]]:
                                      sidx_indptr[sidx_indexes[child['node_id']] + 1]]

            if profile_i == 1 and loss_indptr[child['loss'] + profile_i] == loss_indptr[child['loss']]:
                # this is the first time child branch has multiple layer we create views for root children
                base_children_count = get_base_children(child, children, nodes_array, temp_children_queue)
                # print('new layers', child['level_id'], child['agg_id'], node['profile_len'])
                first_time_layer_extra(
                    node['profile_len'], base_children_count, temp_children_queue, compute_idx, nodes_array,
                    sidx_indptr, sidx_indexes,
                    loss_indptr, loss_val,
                    extras_indptr, extras_val,
                )

            child_loss = loss_val[loss_indptr[child['loss'] + profile_i]:
                                  loss_indptr[child['loss'] + profile_i] + child_sidx_val.shape[0]]
            child_extra = extras_val[extras_indptr[child['extra'] + profile_i]:
                                     extras_indptr[child['extra'] + profile_i] + child_sidx_val.shape[0]]
            # print('child', child['level_id'], child['agg_id'], profile_i, loss_indptr[child['loss'] + profile_i], child_loss[0], child['extra'], extras_indptr[child['extra'] + profile_i])
            collapse = collapses_buildings(node, child, site_collapse_level, building_packing)
            for val_i in range(child_sidx_val.shape[0]):
                if collapse:
                    key = decode_local_sidx(child_sidx_val[val_i], max_sidx_val)
                else:
                    key = child_sidx_val[val_i]
                key_count = mark_node_sidx(key, temp_node_sidx, temp_node_keys, key_count)
                profile_temp_node_loss[key] += child_loss[val_i]
                profile_temp_node_extras[key] += child_extra[val_i]
        # print('res', profile_i, profile_temp_node_loss[-3], profile_temp_node_extras[-3])

        loss_indptr[node['loss'] + profile_i] = compute_idx['loss_ptr_i']
        extras_indptr[node['extra'] + profile_i] = compute_idx['extras_ptr_i']
        if sidx_created:
            for node_sidx_cur in range(node_sidx_start, node_sidx_end):
                loss_val[compute_idx['loss_ptr_i']] = profile_temp_node_loss[sidx_val[node_sidx_cur]]
                compute_idx['loss_ptr_i'] += 1
                extras_val[compute_idx['extras_ptr_i']] = profile_temp_node_extras[sidx_val[node_sidx_cur]]
                compute_idx['extras_ptr_i'] += 1
            for key_i in range(key_count):
                temp_node_sidx[temp_node_keys[key_i]] = False

        else:
            node_keys = sorted_node_sidx(temp_node_keys, key_count)
            for key_i in range(key_count):
                sidx = node_keys[key_i]
                sidx_val[compute_idx['sidx_ptr_i']] = sidx
                compute_idx['sidx_ptr_i'] += 1
                # temp_node_sidx is reused by every node of the event, so a node must leave it
                # as it found it. Without this a node BELOW site_collapse_level leaves its
                # PACKED indices set, and the collapsed node above collects them too.
                temp_node_sidx[sidx] = False

                loss_val[compute_idx['loss_ptr_i']] = profile_temp_node_loss[sidx]
                compute_idx['loss_ptr_i'] += 1

                extras_val[compute_idx['extras_ptr_i']] = profile_temp_node_extras[sidx]
                compute_idx['extras_ptr_i'] += 1

            node_sidx_end = compute_idx['sidx_ptr_i']
            node_val_count = node_sidx_end - node_sidx_start
            sidx_indptr[compute_idx['sidx_i']] = compute_idx['sidx_ptr_i']
            sidx_created = True
    # print('node', node['node_id'], node['agg_id'], loss_indptr[node['loss']], temp_node_loss[:node['profile_len'], -3], temp_node_extras[:node['profile_len'], -3])
    # fill up all layer if necessary
    for layer_i in range(node['profile_len'], node['layer_len']):
        loss_indptr[node['loss'] + layer_i] = loss_indptr[node['loss']]
        extras_indptr[node['extra'] + layer_i] = extras_indptr[node['extra']]

    return node_val_count


@njit(cache=True, fastmath=True)
def aggregate_children(node, children_count, nodes_array, children, temp_children_queue, compute_idx,
                       site_collapse_level, building_packing, max_sidx_val,
                       temp_node_sidx, temp_node_keys, sidx_indexes, sidx_indptr, sidx_val,
                       temp_node_loss, loss_indptr, loss_val):
    """Aggregate losses from multiple children into a parent node (without extras tracking).

    This function sums the losses from all children for each sample index (sidx).
    It handles the sparse-to-dense-to-sparse conversion needed for aggregation:

    1. For each child, read its sparse loss values
    2. Accumulate into a dense temporary array (temp_node_loss) indexed by sidx
    3. Convert back to sparse storage for the parent node

    Multi-layer handling:

    - If the parent has multiple profiles but children only have one layer,
      triggers first_time_layer to create layer storage for children
    - Each profile/layer is processed separately

    The function also creates the sidx array for the parent node (union of all
    children's sidx values) on the first profile iteration.

    Args:
        node: Parent node being computed
        children_count: Number of direct children
        nodes_array: Array of all node information
        children: Children tracking array (count + child IDs per node)
        temp_children_queue: Working array for base children lookup
        compute_idx: Computation state pointers (sidx_i, sidx_ptr_i, loss_ptr_i, etc.)
        site_collapse_level: last level whose terms apply per building; children at or below it
            have their building blocks merged as they are aggregated into a node above it
        building_packing: whether this input set has packed items at all
        max_sidx_val: the stream's sample size, used to decode a packed sidx to its local one
        temp_node_sidx: Dense boolean array marking which sidx values have data
        temp_node_keys: scratch holding the sidx this node has marked, in arrival order
        sidx_indexes: Maps node_id to its sidx array position
        sidx_indptr: Pointers into sidx_val for each node
        sidx_val: Sample index values
        all_sidx: Ordered array of all possible sidx values for iteration
        temp_node_loss: Dense array [profile, sidx] for accumulating child losses
        loss_indptr: Pointers into loss_val
        loss_val: Loss values aligned with sidx_val

    Returns:
        int: Number of sidx values (node_val_count) for this node
    """
    sidx_created = False
    node_sidx_start = compute_idx['sidx_ptr_i']
    node_sidx_end = 0
    sidx_indexes[node['node_id']] = compute_idx['sidx_i']
    compute_idx['sidx_i'] += 1
    for profile_i in range(node['profile_len']):
        profile_temp_node_loss = temp_node_loss[profile_i]
        key_count = 0
        for children_i in range(node['children'] + 1, node['children'] + children_count + 1):
            child = nodes_array[children[children_i]]
            child_sidx_val = sidx_val[sidx_indptr[sidx_indexes[child['node_id']]]:
                                      sidx_indptr[sidx_indexes[child['node_id']] + 1]]
            if profile_i == 1 and loss_indptr[child['loss'] + profile_i] == loss_indptr[child['loss']]:
                # this is the first time child branch has multiple layer we create views for root children
                base_children_count = get_base_children(child, children, nodes_array, temp_children_queue)
                first_time_layer(
                    node['profile_len'], base_children_count, temp_children_queue, compute_idx, nodes_array,
                    sidx_indptr, sidx_indexes,
                    loss_indptr, loss_val
                )
            child_loss = loss_val[loss_indptr[child['loss'] + profile_i]:
                                  loss_indptr[child['loss'] + profile_i] + child_sidx_val.shape[0]]

            collapse = collapses_buildings(node, child, site_collapse_level, building_packing)
            for val_i in range(child_sidx_val.shape[0]):
                if collapse:
                    key = decode_local_sidx(child_sidx_val[val_i], max_sidx_val)
                else:
                    key = child_sidx_val[val_i]
                key_count = mark_node_sidx(key, temp_node_sidx, temp_node_keys, key_count)
                profile_temp_node_loss[key] += child_loss[val_i]

        loss_indptr[node['loss'] + profile_i] = compute_idx['loss_ptr_i']
        if sidx_created:
            for node_sidx_cur in range(node_sidx_start, node_sidx_end):
                loss_val[compute_idx['loss_ptr_i']] = profile_temp_node_loss[sidx_val[node_sidx_cur]]
                compute_idx['loss_ptr_i'] += 1
            for key_i in range(key_count):
                temp_node_sidx[temp_node_keys[key_i]] = False

        else:
            node_keys = sorted_node_sidx(temp_node_keys, key_count)
            for key_i in range(key_count):
                sidx = node_keys[key_i]
                sidx_val[compute_idx['sidx_ptr_i']] = sidx
                compute_idx['sidx_ptr_i'] += 1
                temp_node_sidx[sidx] = False

                loss_val[compute_idx['loss_ptr_i']] = profile_temp_node_loss[sidx]
                compute_idx['loss_ptr_i'] += 1

            node_sidx_end = compute_idx['sidx_ptr_i']
            node_val_count = node_sidx_end - node_sidx_start
            sidx_indptr[compute_idx['sidx_i']] = compute_idx['sidx_ptr_i']
            sidx_created = True

    # fill up all layer if necessary
    for layer_i in range(node['profile_len'], node['layer_len']):
        loss_indptr[node['loss'] + layer_i] = loss_indptr[node['loss']]

    return node_val_count


@njit(cache=True)
def set_parent_next_compute(parent_id, child_id, nodes_array, children, computes, compute_idx):
    """Register a parent node for computation at the next level.

    As we process nodes at the current level, we track which parent nodes will
    need to be computed next. This function:

    1. Adds the child to the parent's children list
    2. If this is the first child seen for this parent, adds the parent to the
       compute queue for the next level

    The children array uses a count-then-values format:
    - children[parent['children']] = count of children
    - children[parent['children'] + 1..count] = child node IDs

    Args:
        parent_id: Node ID of the parent to queue
        child_id: Node ID of the child being linked
        nodes_array: Array of all node information
        children: Children tracking array
        computes: Queue of nodes to compute (current level followed by next level)
        compute_idx: Contains next_compute_i pointing to end of queue
    """
    parent = nodes_array[parent_id]
    parent_children_count = children[parent['children']] + 1
    children[parent['children']] = parent_children_count
    children[parent['children'] + parent_children_count] = child_id
    if parent_children_count == 1:  # first time parent is seen
        computes[compute_idx['next_compute_i']] = parent_id
        compute_idx['next_compute_i'] += 1


@njit(cache=True, fastmath=True)
def load_net_value(computes, compute_idx, nodes_array,
                   sidx_indptr, sidx_indexes,
                   loss_indptr, loss_val):
    """Convert gross losses to net losses for output streaming.

    Net loss = input loss - insured loss (what remains after insurance pays)

    For multi-layer policies, net loss at layer i = net loss at layer i-1 minus
    the insured loss paid by layer i. This creates a waterfall where each layer
    pays from what remains after previous layers.

    This function iterates through the output nodes and replaces the gross loss
    values with net loss values in-place, then resets the compute index so the
    stream writer outputs from the beginning.

    Called when net_loss output is requested instead of or in addition to gross loss.
    """
    net_compute_i = 0
    while computes[net_compute_i]:
        node_i, net_compute_i = computes[net_compute_i], net_compute_i + 1
        node = nodes_array[node_i]
        # net loss layer i is initial loss - sum of all layer up to i
        node_val_count = sidx_indptr[sidx_indexes[node['node_id']] + 1] - sidx_indptr[sidx_indexes[node['node_id']]]
        node_ba_val_prev = loss_val[loss_indptr[node['net_loss']]: loss_indptr[node['net_loss']] + node_val_count]
        for layer_i in range(node['layer_len']):
            node_ba_val_cur = loss_val[loss_indptr[node['loss'] + layer_i]: loss_indptr[node['loss'] + layer_i] + node_val_count]
            # print(node['agg_id'], layer_i, loss_indptr[node['loss'] + layer_i], node_ba_val_prev, node_ba_val_cur)
            node_ba_val_cur[:] = np.maximum(node_ba_val_prev - node_ba_val_cur, 0)
            node_ba_val_prev = node_ba_val_cur
    compute_idx['level_start_compute_i'] = 0


@njit(cache=True, fastmath=True, inline='always')
def effective_max_buildings(compute_info):
    """How many building blocks the sidx-indexed arrays actually have to span.

    ``max_buildings`` says what the STREAM can carry; this says what reaches fm storage. When no
    level applies terms per building there is nothing to collapse later, so FMReader sums the
    buildings away as it reads (see ``collapse_on_read`` in manager.run_synchronous_sparse, which
    must stay the same predicate) and every stored sidx is local.

    It has to be ONE function: ``all_sidx`` is the iteration domain and ``temp_node_sidx`` is
    indexed by its values, so sizing them from two separate reads of ``max_buildings`` lets them
    disagree -- and the scan then runs off the end of the dense temporaries, unchecked.
    """
    max_buildings = max(1, compute_info['max_buildings'])
    if max_buildings > 1 and compute_info['site_collapse_level'] < max(1, compute_info['start_level']):
        return 1
    return max_buildings


@njit(cache=True, fastmath=True, error_model="numpy")
def compute_event(compute_info,
                  keep_input_loss,
                  nodes_array,
                  node_parents_array,
                  node_profiles_array,
                  len_array, max_sidx_val, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, extras_indptr, extras_val,
                  children,
                  computes,
                  compute_idx,
                  item_parent_i,
                  fm_profile,
                  stepped,
                  profile):
    """Compute insured losses for a single event through the entire financial structure.

    This is the main computation function that processes one event's losses through
    all levels of the insurance/reinsurance hierarchy. Results are stored in-place
    in loss_val.

    Algorithm Overview
    ------------------
    The computation proceeds bottom-up through the financial structure levels::

        For each level (starting from items, going up to final output):
            For each node to compute at this level:
                1. AGGREGATE: Sum losses from children nodes
                   - Multiple children: aggregate into temp arrays, create parent sidx
                   - Single child: reuse child's storage (optimization)
                   - No children (item level): use input losses directly

                2. APPLY PROFILE: Apply financial terms to the aggregated loss
                   - For each profile (may be 1 per layer or 1 cross-layer):
                     - Run calc/calc_extra with the profile's calc rules
                     - Handles deductibles, limits, shares, etc.

                3. BACK ALLOCATE: Distribute results back to base children
                   - Rule 0: No allocation (output at aggregate level)
                   - Rule 1: Proportional to original input loss
                   - Rule 2: Proportional to computed loss (pro-rata)

                4. QUEUE PARENTS: Register parent nodes for next level computation

    Cross-Layer Profiles
    --------------------
    When cross_layer_profile=True, losses from all layers are first merged,
    the profile is applied to the total, then results are back-allocated
    proportionally to each layer.

    Args:
        compute_info: Computation metadata (levels, allocation rule, etc.)
        keep_input_loss: If True, preserve input loss for net loss calculation
        nodes_array: Static node information (agg_id, level_id, pointers)
        node_parents_array: Parent node IDs for each node
        node_profiles_array: Profile metadata (i_start, i_end into fm_profile)
        len_array: Size for dense temporary arrays
        max_sidx_val: Maximum sample index value
        sidx_indexes: Node to sidx array position mapping
        sidx_indptr: CSR-style pointers into sidx_val
        sidx_val: Sample index values
        loss_indptr: CSR-style pointers into loss_val
        loss_val: Loss values (modified in place)
        extras_indptr: CSR-style pointers into extras_val
        extras_val: Extras [deductible, overlimit, underlimit]
        children: Dynamic children tracking per node
        computes: Queue of nodes to process
        compute_idx: Computation state (current position, pointers)
        item_parent_i: Tracks which parent index for multi-parent items
        fm_profile: Array of financial profile terms
        stepped: True/None flag for stepped policies (None for JIT compatibility)
        profile: (PROFILE_LEVELS, PROFILE_METRICS) scratch accumulated when DEBUG_PROFILE is
            set, and untouched otherwise -- numba folds the constant away.
    """
    # =========================================================================
    # INITIALIZATION: Set up computation state and temporary arrays
    # =========================================================================
    compute_idx['sidx_i'] = compute_idx['next_compute_i']
    compute_idx['sidx_ptr_i'] = compute_idx['loss_ptr_i'] = sidx_indptr[compute_idx['next_compute_i']]
    compute_idx['extras_ptr_i'] = 0
    compute_idx['compute_i'] = 0

    # Dense boolean array: temp_node_sidx[sidx] = True if this sidx has a value
    # Used during aggregation to track which samples have data
    temp_node_sidx = np.zeros(len_array, dtype=oasis_int)
    # the sidx a node actually holds, in arrival order
    temp_node_keys = np.zeros(len_array, dtype=oasis_int)

    # Temporary storage for profile calculation output (loss after applying terms)
    temp_node_loss_sparse = np.zeros(len_array, dtype=oasis_float)

    # For cross-layer profiles: merged loss across all layers before applying terms
    temp_node_loss_layer_merge = np.zeros(len_array, dtype=oasis_float)

    # After cross-layer back allocation: loss for each layer [layer, sidx]
    temp_node_loss_layer_ba = np.zeros((compute_info['max_layer'], len_array), dtype=oasis_float)

    # Dense accumulator for children aggregation, then reused for back alloc factors
    # Shape: [layer/profile, sidx] - uses float64 for precision during summation
    temp_node_loss = np.zeros((compute_info['max_layer'], len_array), dtype=np.float64)

    # Dense accumulator for extras during aggregation [layer, sidx, extra_type]
    # extra_type: 0=DEDUCTIBLE, 1=OVERLIMIT, 2=UNDERLIMIT
    temp_node_extras = np.zeros((compute_info['max_layer'], len_array, 3), dtype=oasis_float)

    # For cross-layer: merged extras before and after profile application
    temp_node_extras_layer_merge = np.zeros((len_array, 3), dtype=oasis_float)
    temp_node_extras_layer_merge_save = np.zeros((len_array, 3), dtype=oasis_float)

    # Working queue for BFS traversal to find base children
    temp_children_queue = np.empty(nodes_array.shape[0], dtype=oasis_int)

    # Scratch for collapsing a packed leaf, indexed by the COLLAPSED sample index, so it spans
    # max_sidx_val + 6 rather than the packed range.
    collapse_len = max_sidx_val + 6
    collapse_loss = np.zeros((compute_info['max_layer'], collapse_len), dtype=np.float64)
    collapse_extras = np.zeros((compute_info['max_layer'], collapse_len, 3), dtype=oasis_float)
    collapse_net = np.zeros(collapse_len, dtype=np.float64)

    # Every sidx a node can carry, ascending -- iterating it is what orders a parent's sidx
    # array, so it must cover every value that can arrive. Under packing that includes each
    # building's block: specials NUM_SPECIAL_SIDX lower per building, samples at (b-1)*S+1..b*S.
    # max_buildings is 1 for an ordinary run, reducing this to (-5, -3, -1, 1..S).
    n_buildings = effective_max_buildings(compute_info)
    all_sidx = np.empty(n_buildings * (max_sidx_val + EXTRA_SIDX_COUNT), dtype=oasis_int)
    i = 0
    for b in range(n_buildings, 0, -1):
        shift = (b - 1) * NUM_SPECIAL_SIDX
        all_sidx[i] = MAX_LOSS_IDX - shift     # -5: maximum loss
        all_sidx[i + 1] = TIV_IDX - shift      # -3: total insured value
        all_sidx[i + 2] = MEAN_IDX - shift     # -1: mean/expected loss
        i += EXTRA_SIDX_COUNT
    all_sidx[i:] = np.arange(1, n_buildings * max_sidx_val + 1)  # sample indices, building-major

    # Last level whose terms apply per building; children at or below it merge their blocks when
    # aggregated into a node above it. 0 for an ordinary run, making the checks below no-ops.
    site_collapse_level = compute_info['site_collapse_level']
    building_packing = compute_info['max_buildings'] > 1

    # Pre-compute allocation rule flags for efficiency
    is_allocation_rule_a0 = compute_info['allocation_rule'] == 0
    is_allocation_rule_a1 = compute_info['allocation_rule'] == 1
    is_allocation_rule_a2 = compute_info['allocation_rule'] == 2

    # =========================================================================
    # MAIN LOOP: Process each level bottom-up
    # =========================================================================
    _t0 = 0.0
    for level in range(compute_info['start_level'], compute_info['max_level'] + 1):
        if DEBUG_PROFILE:
            with objmode(_t0='f8'):
                _t0 = time.perf_counter()
        # Level boundary: next_compute_i points past current level nodes
        # Setting to index+1 creates a "null terminator" (computes[i]=0) that stops the while loop
        compute_idx['next_compute_i'] += 1
        compute_idx['level_start_compute_i'] = compute_idx['compute_i']

        # ---------------------------------------------------------------------
        # Process all nodes queued for this level
        # ---------------------------------------------------------------------
        while computes[compute_idx['compute_i']]:
            compute_node = nodes_array[computes[compute_idx['compute_i']]]
            compute_idx['compute_i'] += 1
            children_count = children[compute_node['children']]
            if DEBUG_TEMPS:
                assert_temps_clean(temp_node_loss, temp_node_extras, compute_node['node_id'])

            # =================================================================
            # STEP 1: AGGREGATE - Gather losses from children into this node
            # =================================================================
            # Three cases:
            # - children_count > 1: Sum all children's losses (true aggregation)
            # - children_count == 1: Single child, can reuse its storage
            # - children_count == 0: Item level, losses already loaded from stream
            if children_count:
                # A single child is normally adopted wholesale, which would carry its building blocks
                # through and skip the collapse. Common shape: a site node over one coverage type.
                if children_count == 1 and building_packing:
                    only_child = nodes_array[children[compute_node['children'] + 1]]
                    must_collapse = collapses_buildings(compute_node, only_child, site_collapse_level,
                                                        building_packing)
                else:
                    must_collapse = False

                if children_count > 1 or must_collapse:
                    storage_node = compute_node
                    if storage_node['extra'] == null_index:
                        node_val_count = aggregate_children(
                            storage_node, children_count, nodes_array, children, temp_children_queue, compute_idx,
                            site_collapse_level, building_packing, max_sidx_val,
                            temp_node_sidx, temp_node_keys, sidx_indexes, sidx_indptr, sidx_val,
                            temp_node_loss, loss_indptr, loss_val
                        )
                    else:
                        node_val_count = aggregate_children_extras(
                            storage_node, children_count, nodes_array, children, temp_children_queue, compute_idx,
                            site_collapse_level, building_packing, max_sidx_val,
                            temp_node_sidx, temp_node_keys, sidx_indexes, sidx_indptr, sidx_val,
                            temp_node_loss, loss_indptr, loss_val,
                            temp_node_extras, extras_indptr, extras_val
                        )
                    node_sidx = sidx_val[compute_idx['sidx_ptr_i'] - node_val_count: compute_idx['sidx_ptr_i']]
                    if DEBUG_PROFILE and level < PROFILE_LEVELS:
                        profile[level, 3] += 1

                else:  # only 1 child
                    storage_node = nodes_array[children[compute_node['children'] + 1]]
                    # positive sidx are the same as child
                    node_sidx = sidx_val[sidx_indptr[sidx_indexes[storage_node['node_id']]]:sidx_indptr[sidx_indexes[storage_node['node_id']] + 1]]
                    if DEBUG_PROFILE and level < PROFILE_LEVELS:
                        profile[level, 4] += 1
                    node_val_count = node_sidx.shape[0]

                    if compute_node['profile_len'] > 1 and loss_indptr[storage_node['loss'] + 1] == loss_indptr[storage_node['loss']]:
                        # first time layer, we need to create view for storage_node and copy loss and extra
                        node_loss = loss_val[loss_indptr[storage_node['loss']]:loss_indptr[storage_node['loss']] + node_val_count]

                        for profile_i in range(1, compute_node['profile_len']):
                            loss_indptr[storage_node['loss'] + profile_i] = compute_idx['loss_ptr_i']
                            loss_val[compute_idx['loss_ptr_i']: compute_idx['loss_ptr_i'] + node_val_count] = node_loss
                            compute_idx['loss_ptr_i'] += node_val_count

                        if compute_node['extra'] != null_index:
                            node_extras = extras_val[extras_indptr[storage_node['extra']]:extras_indptr[storage_node['extra']] + node_val_count]
                            for profile_i in range(1, compute_node['profile_len']):
                                extras_indptr[storage_node['extra'] + profile_i] = compute_idx['extras_ptr_i']
                                extras_val[compute_idx['extras_ptr_i']: compute_idx['extras_ptr_i'] + node_val_count] = node_extras
                                compute_idx['extras_ptr_i'] += node_val_count

                        base_children_count = get_base_children(storage_node, children, nodes_array, temp_children_queue)
                        if base_children_count > 1:
                            if compute_node['extra'] != null_index:
                                first_time_layer_extra(
                                    compute_node['profile_len'], base_children_count, temp_children_queue, compute_idx,
                                    nodes_array,
                                    sidx_indptr, sidx_indexes,
                                    loss_indptr, loss_val,
                                    extras_indptr, extras_val,
                                )
                            else:
                                first_time_layer(
                                    compute_node['profile_len'], base_children_count, temp_children_queue, compute_idx,
                                    nodes_array,
                                    sidx_indptr, sidx_indexes,
                                    loss_indptr, loss_val,
                                )

                    if children[storage_node['children']] and compute_node['extra'] != null_index:
                        # child is not base child so back allocation
                        # we need to keep track of extra before profile
                        for profile_i in range(compute_node['profile_len']):
                            if compute_node['cross_layer_profile']:
                                copy_extra = True
                            else:
                                node_profile = node_profiles_array[compute_node['profiles'] + profile_i]
                                copy_extra = node_profile['i_start'] < node_profile['i_end']
                            if copy_extra:
                                node_extras = extras_val[extras_indptr[storage_node['extra'] + profile_i]:
                                                         extras_indptr[storage_node['extra'] + profile_i] + node_val_count]

                                for val_i in range(node_val_count):
                                    temp_node_extras[profile_i, node_sidx[val_i]] = node_extras[val_i]

            else:  # if no children and in compute then layer 1 loss is already set
                # we create space for extras and copy loss from layer 1 to other layer if they exist
                storage_node = compute_node
                node_sidx = sidx_val[sidx_indptr[sidx_indexes[storage_node['node_id']]]:
                                     sidx_indptr[sidx_indexes[storage_node['node_id']] + 1]]
                node_val_count = node_sidx.shape[0]
                if DEBUG_PROFILE and level < PROFILE_LEVELS:
                    profile[level, 5] += 1
                node_loss = loss_val[loss_indptr[storage_node['loss']]:
                                     loss_indptr[storage_node['loss']] + node_val_count]

                if compute_node['extra'] != null_index:  # for layer 1 (profile_i=0)
                    extras_indptr[storage_node['extra']] = compute_idx['extras_ptr_i']
                    node_extras = extras_val[compute_idx['extras_ptr_i']: compute_idx['extras_ptr_i'] + node_val_count]
                    node_extras.fill(0)
                    compute_idx['extras_ptr_i'] += node_val_count

                for profile_i in range(1, compute_node['profile_len']):  # if base level already has layers
                    loss_indptr[storage_node['loss'] + profile_i] = compute_idx['loss_ptr_i']
                    loss_val[compute_idx['loss_ptr_i']: compute_idx['loss_ptr_i'] + node_val_count] = node_loss
                    compute_idx['loss_ptr_i'] += node_val_count

                    if compute_node['extra'] != null_index:
                        extras_indptr[storage_node['extra'] + profile_i] = compute_idx['extras_ptr_i']
                        extras_val[compute_idx['extras_ptr_i']: compute_idx['extras_ptr_i'] + node_val_count].fill(0)
                        compute_idx['extras_ptr_i'] += node_val_count

                for layer_i in range(compute_node['profile_len'], compute_node['layer_len']):
                    # fill up all layer if necessary
                    loss_indptr[storage_node['loss'] + layer_i] = loss_indptr[storage_node['loss']]
                    if compute_node['extra'] != null_index:
                        extras_indptr[storage_node['extra'] + layer_i] = extras_indptr[storage_node['extra']]

                if keep_input_loss:
                    loss_indptr[storage_node['net_loss']] = compute_idx['loss_ptr_i']
                    loss_val[compute_idx['loss_ptr_i']: compute_idx['loss_ptr_i'] + node_val_count] = node_loss
                    compute_idx['loss_ptr_i'] += node_val_count

            base_children_count = 0  # Lazy-initialized when needed for back allocation

            # =================================================================
            # STEP 2: APPLY CROSS-LAYER PROFILE (if applicable)
            # =================================================================
            # Cross-layer profiles apply financial terms to the sum of all layers,
            # then distribute the result back proportionally to each layer.
            # This is used for aggregate limits/deductibles that span layers.
            if compute_node['cross_layer_profile']:
                node_profile = node_profiles_array[compute_node['profiles']]
                if node_profile['i_start'] < node_profile['i_end']:
                    if compute_node['extra'] != null_index:
                        temp_node_loss_layer_merge[:node_val_count].fill(0)
                        temp_node_extras_layer_merge[:node_val_count].fill(0)

                        for layer_i in range(compute_node['layer_len']):
                            temp_node_loss_layer_merge[:node_val_count] += loss_val[
                                loss_indptr[storage_node['loss'] + layer_i]:
                                loss_indptr[storage_node['loss'] + layer_i] + node_val_count
                            ]
                            temp_node_extras_layer_merge[:node_val_count] += extras_val[
                                extras_indptr[storage_node['extra'] + layer_i]:
                                extras_indptr[storage_node['extra'] + layer_i] + node_val_count
                            ]
                        loss_in = temp_node_loss_layer_merge[:node_val_count]
                        loss_out = temp_node_loss_sparse[:node_val_count]
                        temp_node_extras_layer_merge_save[:node_val_count] = temp_node_extras_layer_merge[
                            :node_val_count]  # save values as they are overwriten

                        for profile_step_i in range(node_profile['i_start'], node_profile['i_end']):
                            calc_extra(fm_profile[profile_step_i],
                                       loss_out,
                                       loss_in,
                                       temp_node_extras_layer_merge[:, DEDUCTIBLE],
                                       temp_node_extras_layer_merge[:, OVERLIMIT],
                                       temp_node_extras_layer_merge[:, UNDERLIMIT],
                                       stepped)
                        # print(level, compute_node['agg_id'], base_children_count, fm_profile[profile_step_i]['calcrule_id'],
                        #       loss_indptr[storage_node['loss']], loss_in, '=>', loss_out)
                        # print(temp_node_extras_layer_merge_save[node_sidx[0], DEDUCTIBLE], '=>', temp_node_extras_layer_merge[0, DEDUCTIBLE], extras_indptr[storage_node['extra']])
                        # print(temp_node_extras_layer_merge_save[node_sidx[0], OVERLIMIT], '=>', temp_node_extras_layer_merge[0, OVERLIMIT])
                        # print(temp_node_extras_layer_merge_save[node_sidx[0], UNDERLIMIT], '=>', temp_node_extras_layer_merge[0, UNDERLIMIT])
                        back_alloc_layer_extra(compute_node['layer_len'], node_val_count, storage_node['loss'], storage_node['extra'],
                                               loss_in, loss_out, loss_indptr, loss_val,
                                               temp_node_loss_layer_ba,
                                               extras_indptr, extras_val,
                                               temp_node_extras_layer_merge, temp_node_extras_layer_merge_save
                                               )

                    else:
                        temp_node_loss_layer_merge[:node_val_count].fill(0)
                        for layer_i in range(compute_node['layer_len']):
                            temp_node_loss_layer_merge[:node_val_count] += loss_val[
                                loss_indptr[storage_node['loss'] + layer_i]:
                                loss_indptr[storage_node['loss'] + layer_i] + node_val_count
                            ]
                        loss_in = temp_node_loss_layer_merge[:node_val_count]
                        loss_out = temp_node_loss_sparse[:node_val_count]
                        for profile_step_i in range(node_profile['i_start'], node_profile['i_end']):
                            calc(fm_profile[profile_step_i],
                                 loss_out,
                                 loss_in,
                                 stepped)
                        # print(level, compute_node['agg_id'], base_children_count, fm_profile[profile_step_i]['calcrule_id'],
                        #       loss_indptr[storage_node['loss']], loss_in, '=>', loss_out)
                        back_alloc_layer(compute_node['layer_len'], node_val_count, storage_node['loss'],
                                         loss_in, loss_out, loss_indptr, loss_val, temp_node_loss_layer_ba)

            # =================================================================
            # STEP 3: APPLY PER-LAYER PROFILES AND BACK ALLOCATE
            # =================================================================
            # For each profile/layer:
            # 1. Get the appropriate profile (same for all if cross_layer, else per-layer)
            # 2. Apply calc rules (deductible, limit, share, etc.) if profile has steps
            # 3. Back allocate the computed loss to base children
            for profile_i in range(compute_node['profile_len']):
                if compute_node['cross_layer_profile']:
                    node_profile = node_profiles_array[compute_node['profiles']]
                else:
                    node_profile = node_profiles_array[compute_node['profiles'] + profile_i]

                if node_profile['i_start'] < node_profile['i_end']:
                    loss_in = loss_val[loss_indptr[storage_node['loss'] + profile_i]:
                                       loss_indptr[storage_node['loss'] + profile_i] + node_val_count]
                    loss_out = temp_node_loss_sparse[:node_val_count]

                    if compute_node['extra'] != null_index:
                        extra = extras_val[extras_indptr[storage_node['extra'] + profile_i]:
                                           extras_indptr[storage_node['extra'] + profile_i] + node_val_count]

                        if compute_node['cross_layer_profile']:
                            loss_out = temp_node_loss_layer_ba[profile_i][:node_val_count]
                        else:
                            for profile_step_i in range(node_profile['i_start'], node_profile['i_end']):
                                calc_extra(fm_profile[profile_step_i],
                                           loss_out,
                                           loss_in,
                                           extra[:, DEDUCTIBLE],
                                           extra[:, OVERLIMIT],
                                           extra[:, UNDERLIMIT],
                                           stepped)
                                # print(compute_node['level_id'], 'fm_profile', fm_profile[profile_step_i])
                                # print(level, compute_node['agg_id'], base_children_count, profile_i, fm_profile[profile_step_i]['calcrule_id'],
                                #       loss_indptr[storage_node['loss'] + profile_i], loss_in, '=>', loss_out)
                                # print(temp_node_extras[profile_i, node_sidx[0], DEDUCTIBLE], '=>', extra[0, DEDUCTIBLE], extras_indptr[storage_node['extra'] + profile_i])
                                # print(temp_node_extras[profile_i, node_sidx[0], OVERLIMIT], '=>', extra[0, OVERLIMIT])
                                # print(temp_node_extras[profile_i, node_sidx[0], UNDERLIMIT], '=>', extra[0, UNDERLIMIT])
                        if not base_children_count:
                            base_children_count = get_base_children(storage_node, children, nodes_array,
                                                                    temp_children_queue)
                            # back_alloc's one-base-child shortcut writes the post-profile loss straight to loss_in,
                            # valid only when that child IS the storage node. The forced aggregation above breaks
                            # that, so tell it which case this is.
                            storage_is_base_child = (
                                base_children_count == 1
                                and nodes_array[temp_children_queue[0]]['node_id'] == storage_node['node_id'])
                            if is_allocation_rule_a2:
                                ba_children_count = base_children_count
                            else:
                                # Rules below a2 hand back_alloc a count of 1 whatever the node
                                # really has, which means "take the one-base-child shortcut". The
                                # flag has to agree: pairing a forced 1 with a flag derived from
                                # the real count asks for a state that means nothing, and skips a
                                # shortcut the rule asked for.
                                ba_children_count = 1
                                storage_is_base_child = True

                        back_alloc_extra_a2(ba_children_count, storage_is_base_child, temp_children_queue, nodes_array, profile_i,
                                            node_val_count, node_sidx, sidx_indptr, sidx_indexes, sidx_val,
                                            loss_in, loss_out, temp_node_loss, loss_indptr, loss_val,
                                            extra, temp_node_extras, extras_indptr, extras_val)
                    else:
                        if compute_node['cross_layer_profile']:
                            loss_out = temp_node_loss_layer_ba[profile_i][:node_val_count]
                        else:
                            for profile_step_i in range(node_profile['i_start'], node_profile['i_end']):
                                calc(fm_profile[profile_step_i],
                                     loss_out,
                                     loss_in,
                                     stepped)
                                # print(compute_node['level_id'], 'fm_profile', fm_profile[profile_step_i])
                                # print(level, compute_node['agg_id'], base_children_count, profile_i, fm_profile[profile_step_i]['calcrule_id'],
                                #       loss_indptr[storage_node['loss'] + profile_i], loss_in, '=>', loss_out)
                        if not base_children_count:
                            base_children_count = get_base_children(storage_node, children, nodes_array,
                                                                    temp_children_queue)
                            # back_alloc's one-base-child shortcut writes the post-profile loss straight to loss_in,
                            # valid only when that child IS the storage node. The forced aggregation above breaks
                            # that, so tell it which case this is.
                            storage_is_base_child = (
                                base_children_count == 1
                                and nodes_array[temp_children_queue[0]]['node_id'] == storage_node['node_id'])
                            if is_allocation_rule_a2:
                                ba_children_count = base_children_count
                            else:
                                # Rules below a2 hand back_alloc a count of 1 whatever the node
                                # really has, which means "take the one-base-child shortcut". The
                                # flag has to agree: pairing a forced 1 with a flag derived from
                                # the real count asks for a state that means nothing, and skips a
                                # shortcut the rule asked for.
                                ba_children_count = 1
                                storage_is_base_child = True

                        back_alloc_a2(ba_children_count, storage_is_base_child, temp_children_queue, nodes_array, profile_i,
                                      node_val_count, node_sidx, sidx_indptr, sidx_indexes, sidx_val,
                                      loss_in, loss_out, temp_node_loss, loss_indptr, loss_val)

            # =================================================================
            # STEP 4: QUEUE PARENTS FOR NEXT LEVEL
            # =================================================================
            if level != compute_info['max_level']:
                # Two cases for finding parents:
                # 1. Node has direct parent(s) in node_parents_array
                # 2. Node is an aggregation - find parents via base children
                if compute_node['parent_len']:
                    # Direct parent: use storage_node (may differ from compute_node for single-child)
                    parent_id = node_parents_array[compute_node['parent']]
                    set_parent_next_compute(
                        parent_id, storage_node['node_id'],
                        nodes_array, children, computes, compute_idx)
                else:
                    # No direct parent: this node aggregates multiple items that may have
                    # different parents. Find each base child's next parent.
                    if not base_children_count:
                        base_children_count = get_base_children(storage_node, children, nodes_array, temp_children_queue)

                    for base_child_i in range(base_children_count):
                        child = nodes_array[temp_children_queue[base_child_i]]
                        parent_id = node_parents_array[child['parent'] + item_parent_i[child['node_id']]]
                        item_parent_i[child['node_id']] += 1
                        set_parent_next_compute(
                            parent_id, child['node_id'],
                            nodes_array, children, computes, compute_idx)
            elif is_allocation_rule_a1:
                # Allocation Rule 1: Final output proportional to INPUT (ground-up) loss
                # At the top level, redistribute the total insured loss to each item
                # based on its original contribution before any financial terms
                if not base_children_count:
                    base_children_count = get_base_children(storage_node, children, nodes_array, temp_children_queue)
                if base_children_count > 1:
                    # a1 reuses temp_node_loss as an accumulator, but the aggregate's sums for
                    # this node are still sitting in it at node_sidx, and the += below would
                    # build on top of them. a2 assigns rather than accumulates, so only a1
                    # needs this. Entries outside node_sidx are already zero -- the previous
                    # node cleared its own, which DEBUG_TEMPS verifies.
                    for val_i in range(node_val_count):
                        temp_node_loss[:, node_sidx[val_i]] = 0
                    for base_child_i in range(base_children_count):
                        child = nodes_array[temp_children_queue[base_child_i]]

                        child_sidx_start = sidx_indptr[sidx_indexes[child['node_id']]]
                        child_sidx_end = sidx_indptr[sidx_indexes[child['node_id']] + 1]
                        child_val_count = child_sidx_end - child_sidx_start

                        child_sidx = sidx_val[child_sidx_start: child_sidx_end]
                        child_loss = loss_val[loss_indptr[child['net_loss']]: loss_indptr[child['net_loss']] + child_val_count]

                        for val_i in range(child_val_count):
                            temp_node_loss[:, child_sidx[val_i]] += child_loss[val_i]

                    for profile_i in range(compute_node['profile_len']):
                        node_loss = loss_val[loss_indptr[storage_node['loss'] + profile_i]:
                                             loss_indptr[storage_node['loss'] + profile_i] + node_val_count]
                        for val_i in range(node_val_count):
                            if node_loss[val_i]:
                                temp_node_loss[profile_i, node_sidx[val_i]] = node_loss[val_i] / temp_node_loss[profile_i, node_sidx[val_i]]
                            else:
                                temp_node_loss[profile_i, node_sidx[val_i]] = 0

                        for base_child_i in range(base_children_count):
                            child = nodes_array[temp_children_queue[base_child_i]]

                            child_sidx_start = sidx_indptr[sidx_indexes[child['node_id']]]
                            child_sidx_end = sidx_indptr[sidx_indexes[child['node_id']] + 1]
                            child_val_count = child_sidx_end - child_sidx_start

                            child_sidx = sidx_val[child_sidx_start: child_sidx_end]
                            child_loss = loss_val[loss_indptr[child['loss'] + profile_i]: loss_indptr[child['loss'] + profile_i] + child_val_count]
                            child_net = loss_val[loss_indptr[child['net_loss']]: loss_indptr[child['net_loss']] + child_val_count]

                            for val_i in range(child_val_count):
                                child_loss[val_i] = child_net[val_i] * temp_node_loss[profile_i, child_sidx[val_i]]

                    # The only write this node makes outside node_sidx: the base children are
                    # leaves, so below the collapse level their sidx are still packed while the
                    # node's own are collapsed. Clear across every layer, matching the `[:, ...]`
                    # accumulation above.
                    for base_child_i in range(base_children_count):
                        child = nodes_array[temp_children_queue[base_child_i]]
                        child_sidx = sidx_val[sidx_indptr[sidx_indexes[child['node_id']]]:
                                              sidx_indptr[sidx_indexes[child['node_id']] + 1]]
                        for val_i in range(child_sidx.shape[0]):
                            temp_node_loss[:, child_sidx[val_i]] = 0
            elif is_allocation_rule_a0:
                # Allocation Rule 0: No back allocation - output at aggregate level only
                # Just ensure compute_node points to the correct storage location
                if compute_node['node_id'] != storage_node['node_id']:
                    sidx_indexes[compute_node['node_id']] = sidx_indexes[storage_node['node_id']]
                    for profile_i in range(compute_node['profile_len']):
                        loss_indptr[compute_node['loss'] + profile_i] = loss_indptr[storage_node['loss'] + profile_i]

            if DEBUG_PROFILE and level < PROFILE_LEVELS:
                profile[level, 1] += 1
                profile[level, 2] += node_val_count

            if building_packing and compute_node['level_id'] == site_collapse_level:
                collapse_site_node(
                    compute_node, storage_node, children, nodes_array, temp_children_queue,
                    compute_idx, max_sidx_val, keep_input_loss, compute_info['start_level'],
                    not is_allocation_rule_a0,
                    sidx_indexes, sidx_indptr, sidx_val,
                    loss_indptr, loss_val, extras_indptr, extras_val,
                    collapse_loss, collapse_extras, collapse_net
                )

            # The dense temporaries are scratch for exactly one node: the aggregate's sums, the
            # pre-terms extras back_alloc reads, and the factors it writes all live and die here.
            # Clearing the node's own sidx costs its size; the fill it replaces cost the whole
            # packed range, which is sized for the portfolio's largest location, not this node.
            for val_i in range(node_val_count):
                temp_node_loss[:, node_sidx[val_i]] = 0
                temp_node_extras[:, node_sidx[val_i]] = 0

        compute_idx['compute_i'] += 1
        if DEBUG_PROFILE and level < PROFILE_LEVELS:
            with objmode(_t1='f8'):
                _t1 = time.perf_counter()
            profile[level, 0] += _t1 - _t0

    item_parent_i.fill(1)
    # print(compute_info['max_level'], next_compute_i, compute_i, next_compute_i-compute_i, computes[compute_i:compute_i + 2], computes[next_compute_i - 1: next_compute_i + 1])
    if compute_info['allocation_rule'] != 0:
        compute_idx['level_start_compute_i'] = 0


def init_variable(compute_info, max_sidx_val, temp_dir, low_memory, keep_input_loss):
    """Initialize all arrays needed for FM computation.

    Creates the sparse storage arrays for sample indices, losses, and extras.
    These use a CSR-like format where:

    - ``*_indptr`` arrays point to the start of each node's data
    - ``*_val`` arrays contain the actual values

    The loss and extras arrays share the same indexing as sidx - each node's
    loss[i] corresponds to sidx[i]. This allows using sidx_indexes to track
    the length of values for any node.

    Args:
        compute_info: Metadata with array size requirements
        max_sidx_val: Maximum sample index (determines array sizing)
        temp_dir: Directory for memory-mapped files (low_memory mode)
        low_memory: If True, use memory-mapped files instead of RAM
        keep_input_loss (bool): whether net_loss storage is in use -- allocation rule 1, or any
            net-loss output mode at any allocation rule. It costs one further packed slice per
            packable node, so reserving it unconditionally charges every gross run for storage it
            never writes.

    Returns:
        Tuple of all initialized arrays needed by compute_event
    """
    # Nodes up to the collapse level hold one block per building, so they need max_buildings
    # times the room. The arrays are one arena filled by a bump allocator, so this is a
    # capacity bound, not a per-node stride. It has to be right: numba does not bounds-check,
    # so an arena too small corrupts the heap instead of raising.
    # Only what reaches storage, not what the stream can carry: a run with nothing to collapse
    # is read collapsed, and then the factor buys nothing while the dense temporaries are scanned
    # per node per event.
    max_buildings = int(effective_max_buildings(compute_info))
    collapsed_on_read = max_buildings != max(1, int(compute_info['max_buildings']))
    packable_nodes = int(compute_info['packable_node_len'])

    # int(): max_sidx_val arrives from the stream header as an int32, and NEP 50 keeps a
    # numpy int32 times a Python int in int32. Every arena size is derived from this, and
    # they run to 3.3e9 slots at S=100 on a 630k-building book -- the product wraps negative
    # and np.zeros rejects it. The arena is indexed with int64 throughout, so only this
    # arithmetic needs widening.
    max_sidx_count = int(max_sidx_val) + EXTRA_SIDX_COUNT
    # dense temporaries are indexed by sidx *value*, and a packed item's sidx runs up to
    # max_buildings * max_sidx_val with its specials wrapping onto the tail, so they span the
    # whole packed range
    len_array = max_buildings * (max_sidx_val + 6)

    # One packed slice per packable node, budgeted as the SUM of their building counts rather
    # than their count times the largest location in the portfolio -- see packable_building_slots.
    # It is the whole count and not (count - 1) because under an allocation rule above 0
    # collapse_site_node appends a collapsed copy rather than shrinking the original in place,
    # which the arena cannot do; that copy is what the base allowance covers.
    #
    # Forced to 0 alongside max_buildings when the stream is collapsed on read, where no packed
    # sidx can reach the arena at all.
    extra_slots = 0 if collapsed_on_read else int(compute_info['packable_building_slots']) * max_sidx_count
    # The sidx arena owes one packed slice per node; loss and extras owe one per layer, which
    # packable_layer_slots sums per level rather than charging every slice the deepest layering in
    # the portfolio. net_loss adds one further slice per node -- not per layer -- and only when it
    # is in use, which is allocation rule 1 or any net-loss output mode; the caller resolves that
    # and passes it in, so an ordinary gross run no longer reserves a net_loss copy it never
    # writes.
    extra_layer_slots = 0 if collapsed_on_read else int(compute_info['packable_layer_slots']) * max_sidx_count
    if keep_input_loss:
        extra_layer_slots += extra_slots

    # The arena is indexed with int64 throughout -- every *_indptr is np.int64 and compute_idx's
    # bump pointers are Python int -- but the SIZES are computed from compute_info, whose fields
    # are oasis_int (int32). Under NEP 50 an int32 scalar times a Python int stays int32, so the
    # product wraps before the int64 term is added and np.zeros is handed a negative length. At
    # S=100 on a 630k-building book that is a 3.55e9-slot arena: well inside int64, and nowhere
    # near int32. Take the sizes into Python ints before any arithmetic.
    node_slots = int(compute_info['node_len']) * max_sidx_count + extra_slots
    loss_slots = int(compute_info['loss_len']) * max_sidx_count + extra_layer_slots
    extra_arena_slots = int(compute_info['extra_len']) * max_sidx_count + extra_layer_slots

    if low_memory:
        sidx_val = np.memmap(os.path.join(temp_dir, "sidx_val.bin"), mode='w+',
                             shape=(node_slots,), dtype=oasis_int)
        loss_val = np.memmap(os.path.join(temp_dir, "loss_val.bin"), mode='w+',
                             shape=(loss_slots,), dtype=oasis_float)
        extras_val = np.memmap(os.path.join(temp_dir, "extras_val.bin"), mode='w+',
                               shape=(extra_arena_slots, 3), dtype=oasis_float)
    else:
        sidx_val = np.zeros(node_slots, dtype=oasis_int)
        loss_val = np.zeros(loss_slots, dtype=oasis_float)
        extras_val = np.zeros((extra_arena_slots, 3), dtype=oasis_float)

    # One entry per allocation, not per node: collapse_site_node appends a collapsed slice for
    # each packed leaf rather than shrinking it in place, and each of those takes a further entry.
    sidx_indptr = np.zeros(compute_info['node_len'] + packable_nodes + 1, dtype=np.int64)
    loss_indptr = np.zeros(compute_info['loss_len'] + 1, dtype=np.int64)
    extras_indptr = np.zeros(compute_info['extra_len'] + 1, dtype=np.int64)

    sidx_indexes = np.empty(compute_info['node_len'], dtype=oasis_int)
    children = np.zeros(compute_info['children_len'], dtype=np.uint32)
    computes = np.zeros(compute_info['compute_len'], dtype=np.uint32)

    pass_through = np.zeros(compute_info['items_len'] + 1, dtype=oasis_float)
    item_parent_i = np.ones(compute_info['items_len'] + 1, dtype=np.uint32)

    compute_idx = np.empty(1, dtype=compute_idx_dtype)[0]
    compute_idx['next_compute_i'] = 0

    return (max_sidx_val, max_sidx_count, len_array, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val,
            pass_through, extras_indptr, extras_val, children, computes, item_parent_i, compute_idx)


@njit(cache=True)
def reset_variable(children, compute_idx, computes):
    """Reset the per event array

    Args:
        children: array of all the children with loss value for each node
        compute_idx: single element named array containing all the pointer needed to tract the computation (compute_idx_dtype)
        computes: array of node to compute
    """
    computes[:compute_idx['next_compute_i']].fill(0)
    children.fill(0)
    compute_idx['next_compute_i'] = 0
