"""
Financial Module (FM) Sparse Computation Engine
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
from oasislmf.pytools.common.event_stream import MAX_LOSS_IDX, MEAN_IDX, TIV_IDX
from .policy import calc
from .policy_extras import calc as calc_extra
from .common import EXTRA_SIDX_COUNT, compute_idx_dtype, DEDUCTIBLE, UNDERLIMIT, OVERLIMIT
from .back_allocation import back_alloc_a2, back_alloc_extra_a2, back_alloc_layer, back_alloc_layer_extra

from numba import njit
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)


@njit(cache=True)
def get_base_children(node, children, nodes_array, temp_children_queue):
    """
    Find all leaf-level (base) descendants of a node using breadth-first traversal.

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


@njit(cache=True)
def first_time_layer(profile_count, base_children_count, temp_children_queue, compute_idx, nodes_array,
                     sidx_indptr, sidx_indexes,
                     loss_indptr, loss_val
                     ):
    """
    Initialize multi-layer loss storage for base children when first encountering layered computation.

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
    """
    Initialize multi-layer loss AND extras storage for base children.

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


@njit(cache=True, fastmath=True)
def aggregate_children_extras(node, children_count, nodes_array, children, temp_children_queue, compute_idx,
                              temp_node_sidx, sidx_indexes, sidx_indptr, sidx_val, all_sidx,
                              temp_node_loss, loss_indptr, loss_val,
                              temp_node_extras, extras_indptr, extras_val):
    """
    Aggregate losses AND extras from multiple children into a parent node.

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
        temp_node_sidx: Dense boolean array marking active sidx values
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
            for val_i in range(child_sidx_val.shape[0]):
                temp_node_sidx[child_sidx_val[val_i]] = True
                profile_temp_node_loss[child_sidx_val[val_i]] += child_loss[val_i]
                profile_temp_node_extras[child_sidx_val[val_i]] += child_extra[val_i]
        # print('res', profile_i, profile_temp_node_loss[-3], profile_temp_node_extras[-3])

        loss_indptr[node['loss'] + profile_i] = compute_idx['loss_ptr_i']
        extras_indptr[node['extra'] + profile_i] = compute_idx['extras_ptr_i']
        if sidx_created:
            for node_sidx_cur in range(node_sidx_start, node_sidx_end):
                loss_val[compute_idx['loss_ptr_i']] = profile_temp_node_loss[sidx_val[node_sidx_cur]]
                compute_idx['loss_ptr_i'] += 1
                extras_val[compute_idx['extras_ptr_i']] = profile_temp_node_extras[sidx_val[node_sidx_cur]]
                compute_idx['extras_ptr_i'] += 1

        else:
            for sidx in all_sidx:
                if temp_node_sidx[sidx]:
                    sidx_val[compute_idx['sidx_ptr_i']] = sidx
                    compute_idx['sidx_ptr_i'] += 1

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
                       temp_node_sidx, sidx_indexes, sidx_indptr, sidx_val, all_sidx,
                       temp_node_loss, loss_indptr, loss_val):
    """
    Aggregate losses from multiple children into a parent node (without extras tracking).

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
        temp_node_sidx: Dense boolean array marking which sidx values have data
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

            for val_i in range(child_sidx_val.shape[0]):
                temp_node_sidx[child_sidx_val[val_i]] = True
                profile_temp_node_loss[child_sidx_val[val_i]] += child_loss[val_i]

        loss_indptr[node['loss'] + profile_i] = compute_idx['loss_ptr_i']
        if sidx_created:
            for node_sidx_cur in range(node_sidx_start, node_sidx_end):
                loss_val[compute_idx['loss_ptr_i']] = profile_temp_node_loss[sidx_val[node_sidx_cur]]
                compute_idx['loss_ptr_i'] += 1

        else:
            for sidx in all_sidx:
                if temp_node_sidx[sidx]:
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
    """
    Register a parent node for computation at the next level.

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
    """
    Convert gross losses to net losses for output streaming.

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
                  stepped):
    """
    Compute insured losses for a single event through the entire financial structure.

    This is the main computation function that processes one event's losses through
    all levels of the insurance/reinsurance hierarchy. Results are stored in-place
    in loss_val.

    Algorithm Overview
    ------------------
    The computation proceeds bottom-up through the financial structure levels:

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

    # Ordered list of all sidx values: special indices first (-5, -3, -1), then 1..max
    # This ordering ensures consistent iteration during sparse-to-dense conversion
    all_sidx = np.empty(max_sidx_val + EXTRA_SIDX_COUNT, dtype=oasis_int)
    all_sidx[0] = MAX_LOSS_IDX   # -5: maximum loss
    all_sidx[1] = TIV_IDX        # -3: total insured value
    all_sidx[2] = MEAN_IDX       # -1: mean/expected loss
    all_sidx[3:] = np.arange(1, max_sidx_val + 1)  # sample indices 1..N

    # Pre-compute allocation rule flags for efficiency
    is_allocation_rule_a0 = compute_info['allocation_rule'] == 0
    is_allocation_rule_a1 = compute_info['allocation_rule'] == 1
    is_allocation_rule_a2 = compute_info['allocation_rule'] == 2

    # =========================================================================
    # MAIN LOOP: Process each level bottom-up
    # =========================================================================
    for level in range(compute_info['start_level'], compute_info['max_level'] + 1):
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

            # =================================================================
            # STEP 1: AGGREGATE - Gather losses from children into this node
            # =================================================================
            # Three cases:
            # - children_count > 1: Sum all children's losses (true aggregation)
            # - children_count == 1: Single child, can reuse its storage
            # - children_count == 0: Item level, losses already loaded from stream
            if children_count:
                if children_count > 1:
                    storage_node = compute_node
                    temp_node_loss.fill(0)
                    if storage_node['extra'] == null_index:
                        node_val_count = aggregate_children(
                            storage_node, children_count, nodes_array, children, temp_children_queue, compute_idx,
                            temp_node_sidx, sidx_indexes, sidx_indptr, sidx_val, all_sidx,
                            temp_node_loss, loss_indptr, loss_val
                        )
                    else:
                        temp_node_extras.fill(0)
                        node_val_count = aggregate_children_extras(
                            storage_node, children_count, nodes_array, children, temp_children_queue, compute_idx,
                            temp_node_sidx, sidx_indexes, sidx_indptr, sidx_val, all_sidx,
                            temp_node_loss, loss_indptr, loss_val,
                            temp_node_extras, extras_indptr, extras_val
                        )
                    node_sidx = sidx_val[compute_idx['sidx_ptr_i'] - node_val_count: compute_idx['sidx_ptr_i']]

                else:  # only 1 child
                    storage_node = nodes_array[children[compute_node['children'] + 1]]
                    # positive sidx are the same as child
                    node_sidx = sidx_val[sidx_indptr[sidx_indexes[storage_node['node_id']]]:sidx_indptr[sidx_indexes[storage_node['node_id']] + 1]]
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
                node_loss = loss_val[loss_indptr[storage_node['loss']]:
                                     loss_indptr[storage_node['loss']] + node_val_count]

                if compute_node['extra'] != null_index:  # for layer 1 (profile_i=0)
                    extras_indptr[storage_node['extra']] = compute_idx['extras_ptr_i']
                    node_extras = extras_val[compute_idx['extras_ptr_i']: compute_idx['extras_ptr_i'] + node_val_count]
                    node_extras.fill(0)
                    temp_node_extras[0].fill(0)
                    compute_idx['extras_ptr_i'] += node_val_count

                for profile_i in range(1, compute_node['profile_len']):  # if base level already has layers
                    loss_indptr[storage_node['loss'] + profile_i] = compute_idx['loss_ptr_i']
                    loss_val[compute_idx['loss_ptr_i']: compute_idx['loss_ptr_i'] + node_val_count] = node_loss
                    compute_idx['loss_ptr_i'] += node_val_count

                    if compute_node['extra'] != null_index:
                        extras_indptr[storage_node['extra'] + profile_i] = compute_idx['extras_ptr_i']
                        extras_val[compute_idx['extras_ptr_i']: compute_idx['extras_ptr_i'] + node_val_count].fill(0)
                        compute_idx['extras_ptr_i'] += node_val_count
                        temp_node_extras[profile_i].fill(0)

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
                        temp_node_loss_layer_merge.fill(0)
                        temp_node_extras_layer_merge.fill(0)

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
                        temp_node_loss_layer_merge.fill(0)
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
                            if is_allocation_rule_a2:
                                ba_children_count = base_children_count
                            else:
                                ba_children_count = 1

                        back_alloc_extra_a2(ba_children_count, temp_children_queue, nodes_array, profile_i,
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
                            if is_allocation_rule_a2:
                                ba_children_count = base_children_count
                            else:
                                ba_children_count = 1

                        back_alloc_a2(ba_children_count, temp_children_queue, nodes_array, profile_i,
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
                    temp_node_loss.fill(0)
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
                            child_net = loss_val[loss_indptr[child['net_loss'] + profile_i]: loss_indptr[child['net_loss'] + profile_i] + child_val_count]

                            for val_i in range(child_val_count):
                                child_loss[val_i] = child_net[val_i] * temp_node_loss[profile_i, child_sidx[val_i]]
            elif is_allocation_rule_a0:
                # Allocation Rule 0: No back allocation - output at aggregate level only
                # Just ensure compute_node points to the correct storage location
                if compute_node['node_id'] != storage_node['node_id']:
                    sidx_indexes[compute_node['node_id']] = sidx_indexes[storage_node['node_id']]
                    for profile_i in range(compute_node['profile_len']):
                        loss_indptr[compute_node['loss'] + profile_i] = loss_indptr[storage_node['loss'] + profile_i]

        compute_idx['compute_i'] += 1

    item_parent_i.fill(1)
    # print(compute_info['max_level'], next_compute_i, compute_i, next_compute_i-compute_i, computes[compute_i:compute_i + 2], computes[next_compute_i - 1: next_compute_i + 1])
    if compute_info['allocation_rule'] != 0:
        compute_idx['level_start_compute_i'] = 0


def init_variable(compute_info, max_sidx_val, temp_dir, low_memory):
    """
    Initialize all arrays needed for FM computation.

    Creates the sparse storage arrays for sample indices, losses, and extras.
    These use a CSR-like format where:
    - *_indptr arrays point to the start of each node's data
    - *_val arrays contain the actual values

    The loss and extras arrays share the same indexing as sidx - each node's
    loss[i] corresponds to sidx[i]. This allows using sidx_indexes to track
    the length of values for any node.

    Args:
        compute_info: Metadata with array size requirements
        max_sidx_val: Maximum sample index (determines array sizing)
        temp_dir: Directory for memory-mapped files (low_memory mode)
        low_memory: If True, use memory-mapped files instead of RAM

    Returns:
        Tuple of all initialized arrays needed by compute_event
    """
    max_sidx_count = max_sidx_val + EXTRA_SIDX_COUNT
    len_array = max_sidx_val + 6

    if low_memory:
        sidx_val = np.memmap(os.path.join(temp_dir, "sidx_val.bin"), mode='w+',
                             shape=(compute_info['node_len'] * max_sidx_count), dtype=oasis_int)
        loss_val = np.memmap(os.path.join(temp_dir, "loss_val.bin"), mode='w+',
                             shape=(compute_info['loss_len'] * max_sidx_count), dtype=oasis_float)
        extras_val = np.memmap(os.path.join(temp_dir, "extras_val.bin"), mode='w+',
                               shape=(compute_info['extra_len'] * max_sidx_count, 3), dtype=oasis_float)
    else:
        sidx_val = np.zeros((compute_info['node_len'] * max_sidx_count), dtype=oasis_int)
        loss_val = np.zeros((compute_info['loss_len'] * max_sidx_count), dtype=oasis_float)
        extras_val = np.zeros((compute_info['extra_len'] * max_sidx_count, 3), dtype=oasis_float)

    sidx_indptr = np.zeros(compute_info['node_len'] + 1, dtype=np.int64)
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
    """
    reset the per event array
    Args:
        children: array of all the children with loss value for each node
        compute_idx: single element named array containing all the pointer needed to tract the computation (compute_idx_dtype)
        computes: array of node to compute
    """
    computes[:compute_idx['next_compute_i']].fill(0)
    children.fill(0)
    compute_idx['next_compute_i'] = 0
