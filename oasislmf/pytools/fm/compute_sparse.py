from .policy import calc
from .policy_extras import calc as calc_extra
from .common import np_oasis_float, np_oasis_int, EXTRA_VALUES, null_index, compute_idx_dtype, DEDUCTIBLE, UNDERLIMIT, OVERLIMIT
from .back_allocation import back_alloc_a2, back_alloc_extra_a2, back_alloc_layer, back_alloc_layer_extra

from numba import njit
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)


@njit(cache=True)
def get_base_children(node, children, nodes_array, temp_children_queue):
    """
    fill up temp_children_queue with all the base children of node from index 0 and return the number of base children
    Args:
        node: top node
        children: array of all the children with loss value for each node
        nodes_array: array of information on all nodes
        temp_children_queue: empty array where we write base children.

    Returns:
        number of base children
    """
    len_children = children[node['children']]
    if len_children:
        temp_children_queue[:len_children] = children[node['children'] + 1: node['children'] + len_children + 1]
        temp_children_queue[len_children] = null_index
        queue_end = len_children
        i = 0
        child_i = 0
        while temp_children_queue[i] != null_index:
            parent = nodes_array[temp_children_queue[i]]
            len_children = children[parent['children']]
            if len_children:
                temp_children_queue[queue_end: queue_end + len_children] = children[parent['children'] + 1: parent['children'] + len_children + 1]
                queue_end += len_children
                temp_children_queue[queue_end] = null_index
            else:
                temp_children_queue[child_i] = temp_children_queue[i]
                child_i += 1
            i += 1
    else:
        temp_children_queue[0] = node['node_id']
        child_i = 1
    return child_i


@njit(cache=True)
def first_time_layer(profile_len, base_children_len, temp_children_queue, compute_idx, nodes_array,
                     sidx_indptr, sidx_indexes,
                     loss_indptr, loss_val
                     ):
    """
    first time there is a back allocation with multiple layer, we duplicate loss and extra from layer 1 to the other layers
    """
    for base_child_i in range(base_children_len):
        child = nodes_array[temp_children_queue[base_child_i]]
        child_val_len = sidx_indptr[sidx_indexes[child['node_id']] + 1] - sidx_indptr[sidx_indexes[child['node_id']]]
        child_loss_val_layer_0 = loss_val[loss_indptr[child['loss']]:
                                          loss_indptr[child['loss']] + child_val_len]
        for p in range(1, profile_len):
            loss_indptr[child['loss'] + p] = compute_idx['loss_ptr_i']
            loss_val[compute_idx['loss_ptr_i']: compute_idx['loss_ptr_i'] + child_val_len] = child_loss_val_layer_0
            compute_idx['loss_ptr_i'] += child_val_len


@njit(cache=True)
def first_time_layer_extra(profile_len, base_children_len, temp_children_queue, compute_idx, nodes_array,
                           sidx_indptr, sidx_indexes,
                           loss_indptr, loss_val,
                           extras_indptr, extras_val,
                           ):
    """
    first time there is a back allocation with multiple layer, we duplicate loss and extra from layer 1 to the other layers
    """
    for base_child_i in range(base_children_len):
        child = nodes_array[temp_children_queue[base_child_i]]
        child_val_len = sidx_indptr[sidx_indexes[child['node_id']] + 1] - sidx_indptr[sidx_indexes[child['node_id']]]
        child_loss_val_layer_0 = loss_val[loss_indptr[child['loss']]:
                                          loss_indptr[child['loss']] + child_val_len]
        for p in range(1, profile_len):
            loss_indptr[child['loss'] + p] = compute_idx['loss_ptr_i']
            loss_val[compute_idx['loss_ptr_i']: compute_idx['loss_ptr_i'] + child_val_len] = child_loss_val_layer_0
            compute_idx['loss_ptr_i'] += child_val_len

            extras_indptr[child['extra'] + p] = compute_idx['extras_ptr_i']
            extras_val[compute_idx['extras_ptr_i']: compute_idx['extras_ptr_i'] + child_val_len] = 0
            compute_idx['extras_ptr_i'] += child_val_len


@njit(cache=True, fastmath=True)
def aggregate_children_extras(node, len_children, nodes_array, children, temp_children_queue, compute_idx,
                              temp_node_sidx, sidx_indexes, sidx_indptr, sidx_val, all_sidx,
                              temp_node_loss, loss_indptr, loss_val,
                              temp_node_extras, extras_indptr, extras_val):
    """
    aggregate the loss and extra of the children of the node that is currently computed
    Args:
        node: node that we compute
        len_children: number of children of the node
        nodes_array: array of information on all nodes
        children: array of all the children with loss value for each node
        temp_children_queue: array storing all the base children of the node
        compute_idx: single element named array containing all the pointer needed to tract the computation (compute_idx_dtype)
        temp_node_sidx: dense array to store if sample id has value for this node
        sidx_indexes: index of sidx for nodes
        sidx_indptr: : index to sidx pointer
        sidx_val: sidx values
        all_sidx: list of all sidx in this computation
        temp_node_loss: dense array storing the sum of children loss
        loss_indptr: index to the loss pointer
        loss_val: loss values
        temp_node_extras: dense array storing the sum of children extra
        extras_indptr: index to the extra pointer
        extras_val: extra values
    """
    sidx_created = False
    node_sidx_start = compute_idx['sidx_ptr_i']
    node_sidx_end = 0
    sidx_indexes[node['node_id']] = compute_idx['sidx_i']
    compute_idx['sidx_i'] += 1

    for p in range(node['profile_len']):
        p_temp_node_loss = temp_node_loss[p]
        p_temp_node_extras = temp_node_extras[p]

        for c in range(node['children'] + 1, node['children'] + len_children + 1):
            child = nodes_array[children[c]]
            child_sidx_val = sidx_val[sidx_indptr[sidx_indexes[child['node_id']]]:
                                      sidx_indptr[sidx_indexes[child['node_id']] + 1]]

            if p == 1 and loss_indptr[child['loss'] + p] == loss_indptr[child['loss']]:
                # this is the first time child branch has multiple layer we create views for root children
                base_children_len = get_base_children(child, children, nodes_array, temp_children_queue)
                # print('new layers', child['level_id'], child['agg_id'], node['profile_len'])
                first_time_layer_extra(
                    node['profile_len'], base_children_len, temp_children_queue, compute_idx, nodes_array,
                    sidx_indptr, sidx_indexes,
                    loss_indptr, loss_val,
                    extras_indptr, extras_val,
                )

            child_loss = loss_val[loss_indptr[child['loss'] + p]:
                                  loss_indptr[child['loss'] + p] + child_sidx_val.shape[0]]
            child_extra = extras_val[extras_indptr[child['extra'] + p]:
                                     extras_indptr[child['extra'] + p] + child_sidx_val.shape[0]]
            # print('child', child['level_id'], child['agg_id'], p, loss_indptr[child['loss'] + p], child_loss[0], child['extra'], extras_indptr[child['extra'] + p])
            for indptr in range(child_sidx_val.shape[0]):
                temp_node_sidx[child_sidx_val[indptr]] = True
                p_temp_node_loss[child_sidx_val[indptr]] += child_loss[indptr]
                p_temp_node_extras[child_sidx_val[indptr]] += child_extra[indptr]
        # print('res', p, p_temp_node_loss[-3], p_temp_node_extras[-3])

        loss_indptr[node['loss'] + p] = compute_idx['loss_ptr_i']
        extras_indptr[node['extra'] + p] = compute_idx['extras_ptr_i']
        if sidx_created:
            for node_sidx_cur in range(node_sidx_start, node_sidx_end):
                loss_val[compute_idx['loss_ptr_i']] = p_temp_node_loss[sidx_val[node_sidx_cur]]
                compute_idx['loss_ptr_i'] += 1
                extras_val[compute_idx['extras_ptr_i']] = p_temp_node_extras[sidx_val[node_sidx_cur]]
                compute_idx['extras_ptr_i'] += 1

        else:
            for sidx in all_sidx:
                if temp_node_sidx[sidx]:
                    sidx_val[compute_idx['sidx_ptr_i']] = sidx
                    compute_idx['sidx_ptr_i'] += 1

                    loss_val[compute_idx['loss_ptr_i']] = p_temp_node_loss[sidx]
                    compute_idx['loss_ptr_i'] += 1

                    extras_val[compute_idx['extras_ptr_i']] = p_temp_node_extras[sidx]
                    compute_idx['extras_ptr_i'] += 1

            node_sidx_end = compute_idx['sidx_ptr_i']
            node_val_len = node_sidx_end - node_sidx_start
            sidx_indptr[compute_idx['sidx_i']] = compute_idx['sidx_ptr_i']
            sidx_created = True
    # print('node', node['node_id'], node['agg_id'], loss_indptr[node['loss']], temp_node_loss[:node['profile_len'], -3], temp_node_extras[:node['profile_len'], -3])
    # fill up all layer if necessary
    for layer in range(node['profile_len'], node['layer_len']):
        loss_indptr[node['loss'] + layer] = loss_indptr[node['loss']]
        extras_indptr[node['extra'] + layer] = extras_indptr[node['extra']]

    return node_val_len


@njit(cache=True, fastmath=True)
def aggregate_children(node, len_children, nodes_array, children, temp_children_queue, compute_idx,
                       temp_node_sidx, sidx_indexes, sidx_indptr, sidx_val, all_sidx,
                       temp_node_loss, loss_indptr, loss_val):
    """
    aggregate the loss of the children of the node that is currently computed
    Args:
        node: node that we compute
        len_children: number of children of the node
        nodes_array: array of information on all nodes
        children: array of all the children with loss value for each node
        temp_children_queue: array storing all the base children of the node
        compute_idx: single element named array containing all the pointer needed to tract the computation (compute_idx_dtype)
        temp_node_sidx: dense array to store if sample id has value for this node
        sidx_indexes: index of sidx for nodes
        sidx_indptr: : index to sidx pointer
        sidx_val: sidx values
        all_sidx: list of all sidx in this computation
        temp_node_loss: dense array storing the sum of children loss
        loss_indptr: index to the loss pointer
        loss_val: loss values
    """
    sidx_created = False
    node_sidx_start = compute_idx['sidx_ptr_i']
    node_sidx_end = 0
    sidx_indexes[node['node_id']] = compute_idx['sidx_i']
    compute_idx['sidx_i'] += 1
    for p in range(node['profile_len']):
        p_temp_node_loss = temp_node_loss[p]
        for c in range(node['children'] + 1, node['children'] + len_children + 1):
            child = nodes_array[children[c]]
            child_sidx_val = sidx_val[sidx_indptr[sidx_indexes[child['node_id']]]:
                                      sidx_indptr[sidx_indexes[child['node_id']] + 1]]
            if p == 1 and loss_indptr[child['loss'] + p] == loss_indptr[child['loss']]:
                # this is the first time child branch has multiple layer we create views for root children
                base_children_len = get_base_children(child, children, nodes_array, temp_children_queue)
                first_time_layer(
                    node['profile_len'], base_children_len, temp_children_queue, compute_idx, nodes_array,
                    sidx_indptr, sidx_indexes,
                    loss_indptr, loss_val
                )
            child_loss = loss_val[loss_indptr[child['loss'] + p]:
                                  loss_indptr[child['loss'] + p] + child_sidx_val.shape[0]]

            for indptr in range(child_sidx_val.shape[0]):
                temp_node_sidx[child_sidx_val[indptr]] = True
                p_temp_node_loss[child_sidx_val[indptr]] += child_loss[indptr]

        loss_indptr[node['loss'] + p] = compute_idx['loss_ptr_i']
        if sidx_created:
            for node_sidx_cur in range(node_sidx_start, node_sidx_end):
                loss_val[compute_idx['loss_ptr_i']] = p_temp_node_loss[sidx_val[node_sidx_cur]]
                compute_idx['loss_ptr_i'] += 1

        else:
            for sidx in all_sidx:
                if temp_node_sidx[sidx]:
                    sidx_val[compute_idx['sidx_ptr_i']] = sidx
                    compute_idx['sidx_ptr_i'] += 1
                    temp_node_sidx[sidx] = False

                    loss_val[compute_idx['loss_ptr_i']] = p_temp_node_loss[sidx]
                    compute_idx['loss_ptr_i'] += 1

            node_sidx_end = compute_idx['sidx_ptr_i']
            node_val_len = node_sidx_end - node_sidx_start
            sidx_indptr[compute_idx['sidx_i']] = compute_idx['sidx_ptr_i']
            sidx_created = True

    # fill up all layer if necessary
    for layer in range(node['profile_len'], node['layer_len']):
        loss_indptr[node['loss'] + layer] = loss_indptr[node['loss']]

    return node_val_len


@njit(cache=True)
def set_parent_next_compute(parent_id, child_id, nodes_array, children, computes, compute_idx):
    """
    Set the parent node that needs to be computed at the next level
    Args:
        parent_id: id of the parent
        child_id: id of the child
        nodes_array: array of information on all nodes
        children: array of all the children with loss value for each node
        computes: array of node that need to be computed.
        compute_idx: single element named array containing all the pointer needed to tract the computation (compute_idx_dtype)
    """
    parent = nodes_array[parent_id]
    parent_children_len = children[parent['children']] + 1
    children[parent['children']] = parent_children_len
    children[parent['children'] + parent_children_len] = child_id
    if parent_children_len == 1:  # first time parent is seen
        computes[compute_idx['next_compute_i']] = parent_id
        compute_idx['next_compute_i'] += 1


@njit(cache=True, fastmath=True)
def compute_event(compute_info,
                  net_loss,
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
    compute an entire event, result losses are stored inplace in loss_val
    Args:
        compute_info: general information on the computation (financial_structure.compute_info_dtype)
        net_loss: if true compute the net loss instead of the insured loss
        nodes_array: array of information on all nodes
        node_parents_array: array of node to parent node
        node_profiles_array: array of profile for each node
        len_array: length of array needed to store loss and extra as dence array
        max_sidx_val: maximum sidx value
        sidx_indptr: index to sidx pointer
        sidx_indexes: index of sidx for nodes
        sidx_val: sidx values
        loss_indptr: index of loss for nodes
        loss_val: loss values
        extras_indptr: index of extra for nodes
        extras_val: extra values
        children: array of all the children with loss value for each node
        computes: array of node that need to be computed. it is filled up as the computation carries on
        compute_idx: single element named array containing all the pointer needed to tract the computation (compute_idx_dtype)
        item_parent_i: each node may have multiple parent that are selected one after another. This array keeps track of which parent index to select
        fm_profile: array of all the profiles
        stepped: (True or None) determine if this computation contain stepped profile (using None instead of False to allow jit compilation)

    """
    compute_idx['sidx_i'] = compute_idx['next_compute_i']
    compute_idx['sidx_ptr_i'] = compute_idx['loss_ptr_i'] = sidx_indptr[compute_idx['next_compute_i']]
    compute_idx['extras_ptr_i'] = 0
    compute_idx['compute_i'] = 0

    # dense array to store if sample id has value for this node
    temp_node_sidx = np.zeros(len_array, dtype=np_oasis_int)

    # sparse array to store the loss after profile is applied
    temp_node_loss_sparse = np.zeros(len_array, dtype=np_oasis_float)

    # sparse array to store the merged loss of all layer
    temp_node_loss_layer_merge = np.zeros(len_array, dtype=np_oasis_float)

    # sparse array to loss after layer back alloc
    temp_node_loss_layer_ba = np.zeros((compute_info['max_layer'], len_array), dtype=np_oasis_float)

    # temp_node_loss: dense array storing the sum of children loss, then the loss factor in children back alloc
    temp_node_loss = np.zeros((compute_info['max_layer'], len_array), dtype=np.float64)

    # temp_node_extras: dense array storing the sum of children extra, then the loss factor in children back alloc
    temp_node_extras = np.zeros((compute_info['max_layer'], len_array, 3), dtype=np_oasis_float)

    # temp_node_extras_layer_merge: sparse array to store the merged extra of all layer
    temp_node_extras_layer_merge = np.zeros((len_array, 3), dtype=np_oasis_float)

    # temp_node_extras_layer_merge_save: sparse array to keep the value of extra when profile is apply on merged layer
    temp_node_extras_layer_merge_save = np.zeros((len_array, 3), dtype=np_oasis_float)

    # temp_children_queue: array storing all the base children of the node
    temp_children_queue = np.empty(nodes_array.shape[0], dtype=np_oasis_int)

    # create all sidx array
    all_sidx = np.empty(max_sidx_val + EXTRA_VALUES, dtype=np_oasis_int)
    all_sidx[0] = -5
    all_sidx[1] = -3
    all_sidx[2] = -1
    all_sidx[3:] = np.arange(1, max_sidx_val + 1)

    is_net_loss_or_allocation_rule_a1 = net_loss or (compute_info['allocation_rule'] == 1)
    is_allocation_rule_a0 = compute_info['allocation_rule'] == 0
    is_allocation_rule_a1 = compute_info['allocation_rule'] == 1
    is_allocation_rule_a2 = compute_info['allocation_rule'] == 2

    for level in range(compute_info['start_level'], compute_info['max_level'] + 1):  # perform the bottom up loss computation
        # print(level, next_compute_i, compute_i, next_compute_i-compute_i, computes[compute_i:compute_i+2], computes[next_compute_i-2: next_compute_i+1])

        compute_idx['next_compute_i'] += 1  # next_compute_i will stay at 0 wich will stop the while loop and start the next level
        compute_idx['level_start_compute_i'] = compute_idx['compute_i']

        while computes[compute_idx['compute_i']]:
            compute_node = nodes_array[computes[compute_idx['compute_i']]]
            compute_idx['compute_i'] += 1
            # compute loss sums and extra sum
            len_children = children[compute_node['children']]

            # step 1 depending on the number of children, we aggregate or copy the loss of the previous level
            # we fill up correctly temp_node_loss and temp_node_extras
            if len_children:  # if children we aggregate or copy
                if len_children > 1:
                    storage_node = compute_node
                    temp_node_loss.fill(0)
                    if storage_node['extra'] == null_index:
                        node_val_len = aggregate_children(
                            storage_node, len_children, nodes_array, children, temp_children_queue, compute_idx,
                            temp_node_sidx, sidx_indexes, sidx_indptr, sidx_val, all_sidx,
                            temp_node_loss, loss_indptr, loss_val
                        )
                    else:
                        temp_node_extras.fill(0)
                        node_val_len = aggregate_children_extras(
                            storage_node, len_children, nodes_array, children, temp_children_queue, compute_idx,
                            temp_node_sidx, sidx_indexes, sidx_indptr, sidx_val, all_sidx,
                            temp_node_loss, loss_indptr, loss_val,
                            temp_node_extras, extras_indptr, extras_val
                        )
                    node_sidx = sidx_val[compute_idx['sidx_ptr_i'] - node_val_len: compute_idx['sidx_ptr_i']]

                else:  # only 1 child
                    storage_node = nodes_array[children[compute_node['children'] + 1]]
                    # positive sidx are the same as child
                    node_sidx = sidx_val[sidx_indptr[sidx_indexes[storage_node['node_id']]]:sidx_indptr[sidx_indexes[storage_node['node_id']] + 1]]
                    node_val_len = node_sidx.shape[0]

                    if compute_node['profile_len'] > 1 and loss_indptr[storage_node['loss'] + 1] == loss_indptr[storage_node['loss']]:
                        # first time layer, we need to create view for storage_node and copy loss and extra
                        node_loss = loss_val[loss_indptr[storage_node['loss']]:loss_indptr[storage_node['loss']] + node_val_len]

                        for p in range(1, compute_node['profile_len']):
                            loss_indptr[storage_node['loss'] + p] = compute_idx['loss_ptr_i']
                            loss_val[compute_idx['loss_ptr_i']: compute_idx['loss_ptr_i'] + node_val_len] = node_loss
                            compute_idx['loss_ptr_i'] += node_val_len

                        if compute_node['extra'] != null_index:
                            node_extras = extras_val[extras_indptr[storage_node['extra']]:extras_indptr[storage_node['extra']] + node_val_len]
                            for p in range(1, compute_node['profile_len']):
                                extras_indptr[storage_node['extra'] + p] = compute_idx['extras_ptr_i']
                                extras_val[compute_idx['extras_ptr_i']: compute_idx['extras_ptr_i'] + node_val_len] = node_extras
                                compute_idx['extras_ptr_i'] += node_val_len

                        base_children_len = get_base_children(storage_node, children, nodes_array, temp_children_queue)
                        if base_children_len > 1:
                            if compute_node['extra'] != null_index:
                                first_time_layer_extra(
                                    compute_node['profile_len'], base_children_len, temp_children_queue, compute_idx,
                                    nodes_array,
                                    sidx_indptr, sidx_indexes,
                                    loss_indptr, loss_val,
                                    extras_indptr, extras_val,
                                )
                            else:
                                first_time_layer(
                                    compute_node['profile_len'], base_children_len, temp_children_queue, compute_idx,
                                    nodes_array,
                                    sidx_indptr, sidx_indexes,
                                    loss_indptr, loss_val,
                                )

                    if children[storage_node['children']] and compute_node['extra'] != null_index:
                        # child is not base child so back allocation
                        # we need to keep track of extra before profile
                        for p in range(compute_node['profile_len']):
                            if compute_node['cross_layer_profile']:
                                copy_extra = True
                            else:
                                node_profile = node_profiles_array[compute_node['profiles'] + p]
                                copy_extra = node_profile['i_start'] < node_profile['i_end']
                            if copy_extra:
                                node_extras = extras_val[extras_indptr[storage_node['extra'] + p]:
                                                         extras_indptr[storage_node['extra'] + p] + node_val_len]

                                for i in range(node_val_len):
                                    temp_node_extras[p, node_sidx[i]] = node_extras[i]

            else:  # if no children and in compute then layer 1 loss is already set
                # we create space for extras and copy loss from layer 1 to other layer if they exist
                storage_node = compute_node
                node_sidx = sidx_val[sidx_indptr[sidx_indexes[storage_node['node_id']]]:
                                     sidx_indptr[sidx_indexes[storage_node['node_id']] + 1]]
                node_val_len = node_sidx.shape[0]
                node_loss = loss_val[loss_indptr[storage_node['loss']]:
                                     loss_indptr[storage_node['loss']] + node_val_len]

                if compute_node['extra'] != null_index:  # for layer 1 (p=0)
                    extras_indptr[storage_node['extra']] = compute_idx['extras_ptr_i']
                    node_extras = extras_val[compute_idx['extras_ptr_i']: compute_idx['extras_ptr_i'] + node_val_len]
                    node_extras.fill(0)
                    temp_node_extras[0].fill(0)
                    compute_idx['extras_ptr_i'] += node_val_len

                for p in range(1, compute_node['profile_len']):  # if base level already has layers
                    loss_indptr[storage_node['loss'] + p] = compute_idx['loss_ptr_i']
                    loss_val[compute_idx['loss_ptr_i']: compute_idx['loss_ptr_i'] + node_val_len] = node_loss
                    compute_idx['loss_ptr_i'] += node_val_len

                    if compute_node['extra'] != null_index:
                        extras_indptr[storage_node['extra'] + p] = compute_idx['extras_ptr_i']
                        extras_val[compute_idx['extras_ptr_i']: compute_idx['extras_ptr_i'] + node_val_len].fill(0)
                        compute_idx['extras_ptr_i'] += node_val_len
                        temp_node_extras[p].fill(0)

                for layer in range(compute_node['profile_len'], compute_node['layer_len']):
                    # fill up all layer if necessary
                    loss_indptr[storage_node['loss'] + layer] = loss_indptr[storage_node['loss']]
                    if compute_node['extra'] != null_index:
                        extras_indptr[storage_node['extra'] + layer] = extras_indptr[storage_node['extra']]

                if is_net_loss_or_allocation_rule_a1:
                    loss_indptr[storage_node['net_loss']] = compute_idx['loss_ptr_i']
                    loss_val[compute_idx['loss_ptr_i']: compute_idx['loss_ptr_i'] + node_val_len] = node_loss
                    compute_idx['loss_ptr_i'] += node_val_len

            base_children_len = 0

            # print('level', level, compute_node['agg_id'], loss_indptr[storage_node['loss']], loss_val[loss_indptr[storage_node['loss']]],
            #       node_profiles_array[compute_node['profiles']], compute_node['extra'] != null_index)

            if compute_node['cross_layer_profile']:
                node_profile = node_profiles_array[compute_node['profiles']]
                if compute_node['extra'] != null_index:
                    temp_node_loss_layer_merge.fill(0)
                    temp_node_extras_layer_merge.fill(0)

                    for l in range(compute_node['layer_len']):
                        temp_node_loss_layer_merge[:node_val_len] += loss_val[
                            loss_indptr[storage_node['loss'] + l]:
                            loss_indptr[storage_node['loss'] + l] + node_val_len
                        ]
                        temp_node_extras_layer_merge[:node_val_len] += extras_val[
                            extras_indptr[storage_node['extra'] + l]:
                            extras_indptr[storage_node['extra'] + l] + node_val_len
                        ]
                    loss_in = temp_node_loss_layer_merge[:node_val_len]
                    loss_out = temp_node_loss_sparse[:node_val_len]
                    temp_node_extras_layer_merge_save[:node_val_len] = temp_node_extras_layer_merge[
                        :node_val_len]  # save values as they are overwriten

                    for profile_index in range(node_profile['i_start'], node_profile['i_end']):
                        calc_extra(fm_profile[profile_index],
                                   loss_out,
                                   loss_in,
                                   temp_node_extras_layer_merge[:, DEDUCTIBLE],
                                   temp_node_extras_layer_merge[:, OVERLIMIT],
                                   temp_node_extras_layer_merge[:, UNDERLIMIT],
                                   stepped)
                    # print(level, compute_node['agg_id'], base_children_len, fm_profile[profile_index]['calcrule_id'],
                    #       loss_indptr[storage_node['loss']], loss_in, '=>', loss_out)
                    # print(temp_node_extras_layer_merge_save[node_sidx[0], DEDUCTIBLE], '=>', temp_node_extras_layer_merge[0, DEDUCTIBLE], extras_indptr[storage_node['extra']])
                    # print(temp_node_extras_layer_merge_save[node_sidx[0], OVERLIMIT], '=>', temp_node_extras_layer_merge[0, OVERLIMIT])
                    # print(temp_node_extras_layer_merge_save[node_sidx[0], UNDERLIMIT], '=>', temp_node_extras_layer_merge[0, UNDERLIMIT])
                    back_alloc_layer_extra(compute_node['layer_len'], node_val_len, storage_node['loss'], storage_node['extra'],
                                           loss_in, loss_out, loss_indptr, loss_val,
                                           temp_node_loss_layer_ba,
                                           extras_indptr, extras_val,
                                           temp_node_extras_layer_merge, temp_node_extras_layer_merge_save
                                           )

                else:
                    temp_node_loss_layer_merge.fill(0)
                    for l in range(compute_node['layer_len']):
                        temp_node_loss_layer_merge[:node_val_len] += loss_val[
                            loss_indptr[storage_node['loss'] + l]:
                            loss_indptr[storage_node['loss'] + l] + node_val_len
                        ]
                    loss_in = temp_node_loss_layer_merge[:node_val_len]
                    loss_out = temp_node_loss_sparse[:node_val_len]
                    for profile_index in range(node_profile['i_start'], node_profile['i_end']):
                        calc(fm_profile[profile_index],
                             loss_out,
                             loss_in,
                             stepped)
                    # print(level, compute_node['agg_id'], base_children_len, fm_profile[profile_index]['calcrule_id'],
                    #       loss_indptr[storage_node['loss']], loss_in, '=>', loss_out)
                    back_alloc_layer(compute_node['layer_len'], node_val_len, storage_node['loss'],
                                     loss_in, loss_out, loss_indptr, loss_val, temp_node_loss_layer_ba)

            # we apply the policy term on each layer
            for p in range(compute_node['profile_len']):
                if compute_node['cross_layer_profile']:
                    node_profile = node_profiles_array[compute_node['profiles']]
                else:
                    node_profile = node_profiles_array[compute_node['profiles'] + p]

                if node_profile['i_start'] < node_profile['i_end']:
                    loss_in = loss_val[loss_indptr[storage_node['loss'] + p]:
                                       loss_indptr[storage_node['loss'] + p] + node_val_len]
                    loss_out = temp_node_loss_sparse[:node_val_len]

                    if compute_node['extra'] != null_index:
                        extra = extras_val[extras_indptr[storage_node['extra'] + p]:
                                           extras_indptr[storage_node['extra'] + p] + node_val_len]

                        if compute_node['cross_layer_profile']:
                            loss_out = temp_node_loss_layer_ba[p]
                        else:
                            for profile_index in range(node_profile['i_start'], node_profile['i_end']):
                                calc_extra(fm_profile[profile_index],
                                           loss_out,
                                           loss_in,
                                           extra[:, DEDUCTIBLE],
                                           extra[:, OVERLIMIT],
                                           extra[:, UNDERLIMIT],
                                           stepped)
                                # print(compute_node['level_id'], 'fm_profile', fm_profile[profile_index])
                                # print(level, compute_node['agg_id'], base_children_len, p, fm_profile[profile_index]['calcrule_id'],
                                #       loss_indptr[storage_node['loss'] + p], loss_in, '=>', loss_out)
                                # print(temp_node_extras[p, node_sidx[0], DEDUCTIBLE], '=>', extra[0, DEDUCTIBLE], extras_indptr[storage_node['extra'] + p])
                                # print(temp_node_extras[p, node_sidx[0], OVERLIMIT], '=>', extra[0, OVERLIMIT])
                                # print(temp_node_extras[p, node_sidx[0], UNDERLIMIT], '=>', extra[0, UNDERLIMIT])
                        if not base_children_len:
                            base_children_len = get_base_children(storage_node, children, nodes_array,
                                                                  temp_children_queue)
                            if is_allocation_rule_a2:
                                ba_children_len = base_children_len
                            else:
                                ba_children_len = 1

                        back_alloc_extra_a2(ba_children_len, temp_children_queue, nodes_array, p,
                                            node_val_len, node_sidx, sidx_indptr, sidx_indexes, sidx_val,
                                            loss_in, loss_out, temp_node_loss, loss_indptr, loss_val,
                                            extra, temp_node_extras, extras_indptr, extras_val)
                    else:
                        if compute_node['cross_layer_profile']:
                            loss_out = temp_node_loss_layer_ba[p]
                        else:
                            for profile_index in range(node_profile['i_start'], node_profile['i_end']):
                                calc(fm_profile[profile_index],
                                     loss_out,
                                     loss_in,
                                     stepped)
                                # print(compute_node['level_id'], 'fm_profile', fm_profile[profile_index])
                                # print(level, compute_node['agg_id'], base_children_len, p, fm_profile[profile_index]['calcrule_id'],
                                #       loss_indptr[storage_node['loss'] + p], loss_in, '=>', loss_out)
                        if not base_children_len:
                            base_children_len = get_base_children(storage_node, children, nodes_array,
                                                                  temp_children_queue)
                            if is_allocation_rule_a2:
                                ba_children_len = base_children_len
                            else:
                                ba_children_len = 1

                        back_alloc_a2(ba_children_len, temp_children_queue, nodes_array, p,
                                      node_val_len, node_sidx, sidx_indptr, sidx_indexes, sidx_val,
                                      loss_in, loss_out, temp_node_loss, loss_indptr, loss_val)

            if level != compute_info['max_level']:
                if compute_node['parent_len']:  # node has direct parent we add it to the compute
                    parent_id = node_parents_array[compute_node['parent']]
                    set_parent_next_compute(
                        parent_id, storage_node['node_id'],
                        nodes_array, children, computes, compute_idx)
                else:  # node has no direct parent, we look into base node for the next parent to put in the compute queue
                    if not base_children_len:
                        base_children_len = get_base_children(storage_node, children, nodes_array, temp_children_queue)

                    for base_child_i in range(base_children_len):
                        child = nodes_array[temp_children_queue[base_child_i]]
                        parent_id = node_parents_array[child['parent'] + item_parent_i[child['node_id']]]
                        item_parent_i[child['node_id']] += 1
                        set_parent_next_compute(
                            parent_id, child['node_id'],
                            nodes_array, children, computes, compute_idx)
            elif is_allocation_rule_a1:
                if not base_children_len:
                    base_children_len = get_base_children(storage_node, children, nodes_array, temp_children_queue)
                if base_children_len > 1:
                    temp_node_loss.fill(0)
                    for base_child_i in range(base_children_len):
                        child = nodes_array[temp_children_queue[base_child_i]]

                        child_sidx_start = sidx_indptr[sidx_indexes[child['node_id']]]
                        child_sidx_end = sidx_indptr[sidx_indexes[child['node_id']] + 1]
                        child_val_len = child_sidx_end - child_sidx_start

                        child_sidx = sidx_val[child_sidx_start: child_sidx_end]
                        child_loss = loss_val[loss_indptr[child['net_loss']]: loss_indptr[child['net_loss']] + child_val_len]

                        for i in range(child_val_len):
                            temp_node_loss[:, child_sidx[i]] += child_loss[i]

                    for p in range(compute_node['profile_len']):
                        node_loss = loss_val[loss_indptr[storage_node['loss'] + p]:
                                             loss_indptr[storage_node['loss'] + p] + node_val_len]
                        for i in range(node_val_len):
                            if node_loss[i]:
                                temp_node_loss[p, node_sidx[i]] = node_loss[i] / temp_node_loss[p, node_sidx[i]]
                            else:
                                temp_node_loss[p, node_sidx[i]] = 0

                        for base_child_i in range(base_children_len):
                            child = nodes_array[temp_children_queue[base_child_i]]

                            child_sidx_start = sidx_indptr[sidx_indexes[child['node_id']]]
                            child_sidx_end = sidx_indptr[sidx_indexes[child['node_id']] + 1]
                            child_val_len = child_sidx_end - child_sidx_start

                            child_sidx = sidx_val[child_sidx_start: child_sidx_end]
                            child_loss = loss_val[loss_indptr[child['loss'] + p]: loss_indptr[child['loss'] + p] + child_val_len]
                            child_net = loss_val[loss_indptr[child['net_loss'] + p]: loss_indptr[child['net_loss'] + p] + child_val_len]

                            for i in range(child_val_len):
                                child_loss[i] = child_net[i] * temp_node_loss[p, child_sidx[i]]
            elif is_allocation_rule_a0:
                if compute_node['node_id'] != storage_node['node_id']:
                    sidx_indexes[compute_node['node_id']] = sidx_indexes[storage_node['node_id']]
                    for p in range(compute_node['profile_len']):
                        loss_indptr[compute_node['loss'] + p] = loss_indptr[storage_node['loss'] + p]

        compute_idx['compute_i'] += 1

    if net_loss:
        # we go through the last level node and replace the gross loss by net loss, then reset compute_i to its value
        net_compute_i = 0
        while computes[net_compute_i]:
            node_i, net_compute_i = computes[net_compute_i], net_compute_i + 1
            node = nodes_array[node_i]
            # net loss layer i is initial loss - sum of all layer up to i
            node_val_len = sidx_indptr[sidx_indexes[node['node_id']] + 1] - sidx_indptr[sidx_indexes[node['node_id']]]
            node_ba_val_prev = loss_val[loss_indptr[node['net_loss']]: loss_indptr[node['net_loss']] + node_val_len]
            for layer in range(node['layer_len']):
                node_ba_val_cur = loss_val[loss_indptr[node['loss'] + layer]: loss_indptr[node['loss'] + layer] + node_val_len]
                # print(node['agg_id'], layer, loss_indptr[node['loss'] + layer], node_ba_val_prev, node_ba_val_cur)
                node_ba_val_cur[:] = np.maximum(node_ba_val_prev - node_ba_val_cur, 0)
                node_ba_val_prev = node_ba_val_cur

    item_parent_i.fill(1)
    # print(compute_info['max_level'], next_compute_i, compute_i, next_compute_i-compute_i, computes[compute_i:compute_i + 2], computes[next_compute_i - 1: next_compute_i + 1])
    if compute_info['allocation_rule'] != 0:
        compute_idx['level_start_compute_i'] = 0


def init_variable(compute_info, max_sidx_val, temp_dir, low_memory):
    """
    extras, loss contains the same index as sidx
    therefore we can use only sidx_indexes to tract the length of each node values
    """
    max_sidx_count = max_sidx_val + EXTRA_VALUES
    len_array = max_sidx_val + 6

    if low_memory:
        sidx_val = np.memmap(os.path.join(temp_dir, "sidx_val.bin"), mode='w+',
                             shape=(compute_info['node_len'] * max_sidx_count), dtype=np_oasis_int)
        loss_val = np.memmap(os.path.join(temp_dir, "loss_val.bin"), mode='w+',
                             shape=(compute_info['loss_len'] * max_sidx_count), dtype=np_oasis_float)
        extras_val = np.memmap(os.path.join(temp_dir, "extras_val.bin"), mode='w+',
                               shape=(compute_info['extra_len'] * max_sidx_count, 3), dtype=np_oasis_float)
    else:
        sidx_val = np.zeros((compute_info['node_len'] * max_sidx_count), dtype=np_oasis_int)
        loss_val = np.zeros((compute_info['loss_len'] * max_sidx_count), dtype=np_oasis_float)
        extras_val = np.zeros((compute_info['extra_len'] * max_sidx_count, 3), dtype=np_oasis_float)

    sidx_indptr = np.zeros(compute_info['node_len'] + 1, dtype=np.int64)
    loss_indptr = np.zeros(compute_info['loss_len'] + 1, dtype=np.int64)
    extras_indptr = np.zeros(compute_info['extra_len'] + 1, dtype=np.int64)

    sidx_indexes = np.empty(compute_info['node_len'], dtype=np_oasis_int)
    children = np.zeros(compute_info['children_len'], dtype=np.uint32)
    computes = np.zeros(compute_info['compute_len'], dtype=np.uint32)

    pass_through = np.zeros(compute_info['items_len'] + 1, dtype=np_oasis_float)
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
