from .policy import calc
from .common import np_oasis_float, np_oasis_int, EXTRA_VALUES

from numba import njit
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)


@njit(cache=True)
def merge_2_sorted_array(storage_array_a, ptr_a, end_a, storage_array_b, ptr_b, end_b, storage_array_res, ptr_res):
    while ptr_a < end_a and ptr_b < end_b:
        if storage_array_a[ptr_a] == storage_array_b[ptr_b]:
            storage_array_res[ptr_res] = storage_array_a[ptr_a]
            ptr_a += 1
            ptr_b += 1

        elif storage_array_a[ptr_a] < storage_array_b[ptr_b]:
            storage_array_res[ptr_res] = storage_array_a[ptr_a]
            ptr_a += 1

        else:
            storage_array_res[ptr_res] = storage_array_b[ptr_b]
            ptr_b += 1
        ptr_res += 1

    while ptr_a < end_a:
        storage_array_res[ptr_res] = storage_array_a[ptr_a]
        ptr_a += 1
        ptr_res += 1

    while ptr_b < end_b:
        storage_array_res[ptr_res] = storage_array_b[ptr_b]
        ptr_b += 1
        ptr_res += 1
    return ptr_res


@njit(cache=True)
def single_array_to_sparse(sidx, sidx_ptr_i, sidx_val, temp_node_sidx,
                           loss_ptr_i, loss_val, temp_node_loss,
                           extras_ptr_i, extras_val, temp_node_extras):
    """
    sidx: values to put in the sparse array
    sidx_ptr_i: index of the next sparse sidx value
    sidx_val: sparse array to store the sidx
    temp_node_sidx: dense array of sidx
        """
    sidx_val[sidx_ptr_i] = sidx
    temp_node_sidx[sidx] = False

    loss_val[loss_ptr_i] = temp_node_loss[sidx]
    temp_node_loss[sidx] = 0

    extras_val[extras_ptr_i] = temp_node_extras[sidx]
    temp_node_extras[sidx] = 0
    return sidx_ptr_i + 1, loss_ptr_i + 1, extras_ptr_i + 1,


@njit(cache=True, fastmath=True, error_model="numpy")
def back_allocate(node, children, nodes_array, sidx_indexes, sidx_indptr, sidx_val,
                  loss_indptr, loss_val, loss_ptr_i,
                  computes, next_compute_i, child_loss_pointer, temp_proportion):
    len_children = children[node['children']]
    if len_children > 1:
        node_sidx_start = sidx_indptr[sidx_indexes[node['node_id']]]
        node_sidx_end = sidx_indptr[sidx_indexes[node['node_id']] + 1]

        for layer in range(node['layer_len']):
            node_ba_loss_cur = loss_indptr[node['ba'] + layer]
            node_sum_loss_cur = loss_indptr[node['loss'] + layer]
            # print(node['node_id'], 'node_ba_loss_cur', (node_sidx_end - node_sidx_start), loss_val[node_ba_loss_cur:node_ba_loss_cur + node_sidx_end - node_sidx_start])
            for node_sidx_cur in range(node_sidx_start, node_sidx_end):
                sidx = sidx_val[node_sidx_cur]
                if loss_val[node_ba_loss_cur]:
                    if loss_val[node_sum_loss_cur]:
                        temp_proportion[sidx] = loss_val[node_ba_loss_cur] / loss_val[node_sum_loss_cur]
                    else:
                        sum_loss = 0
                        for c in range(node['children'] + 1, node['children'] + len_children + 1):
                            child = nodes_array[children[c]]
                            child_sidx_i = np.searchsorted(sidx_val[sidx_indptr[sidx_indexes[child['node_id']]]:sidx_indptr[sidx_indexes[child['node_id']] + 1]], sidx)
                            if sidx == sidx_val[sidx_indptr[sidx_indexes[child['node_id']]] + child_sidx_i]:
                                sum_loss += loss_val[loss_indptr[child['loss'] + layer] + child_sidx_i]
                        if sum_loss:
                            temp_proportion[sidx] = - loss_val[node_ba_loss_cur] / sum_loss
                        else:
                            temp_proportion[sidx] = 0
                else:
                    temp_proportion[sidx] = 0
                node_ba_loss_cur += 1
                node_sum_loss_cur += 1

            for c in range(node['children'] + 1, node['children'] + len_children + 1):
                child = nodes_array[children[c]]
                child_sidx_start = sidx_indptr[sidx_indexes[child['node_id']]]
                child_sidx_end = sidx_indptr[sidx_indexes[child['node_id']] + 1]
                child_loss_cur = loss_indptr[child['loss'] + layer]
                child_il_cur = loss_indptr[child[child_loss_pointer] + layer]
                loss_indptr[child['ba'] + layer] = loss_ptr_i
                for child_sidx_cur in range(child_sidx_start, child_sidx_end):
                    proportion = temp_proportion[sidx_val[child_sidx_cur]]
                    if proportion > 0:
                        loss_val[loss_ptr_i] = loss_val[child_il_cur] * proportion
                    elif proportion == 0:
                        loss_val[loss_ptr_i] = 0
                    else:
                        loss_val[loss_ptr_i] = - loss_val[child_loss_cur] * proportion
                    child_loss_cur +=1
                    child_il_cur += 1
                    loss_ptr_i += 1

        for c in range(node['children'] + 1, node['children'] + len_children + 1):
            computes[next_compute_i], next_compute_i = children[c], next_compute_i + 1

    else:
        computes[next_compute_i], next_compute_i = children[node['children'] + 1], next_compute_i + 1
        child = nodes_array[children[node['children'] + 1]]
        loss_indptr[child['ba']: child['ba'] + node['layer_len']] = loss_indptr[node['ba']: node['ba'] + node['layer_len']]

    return loss_ptr_i, next_compute_i


@njit(cache=True, fastmath=True)
def compute_event(compute_info,
                  net_loss,
                  nodes_array,
                  node_parents_array,
                  node_profiles_array,
                  len_array, max_sidx_val, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, extras_indptr, extras_val,
                  children,
                  computes,
                  next_compute_i,
                  fm_profile,
                  stepped):
    # len_array = computation_structure.len_array
    # max_sidx_val = computation_structure.max_sidx_val
    # sidx_indexes = computation_structure.sidx_indexes
    # sidx_indptr = computation_structure.sidx_indptr
    # sidx_val = computation_structure.sidx_val
    # loss_indptr = computation_structure.loss_indptr
    # loss_val = computation_structure.loss_val
    # extras_indptr = computation_structure.extras_indptr
    # extras_val = computation_structure.extras_val
    # at this point we have as many loss as sidx as there is only 1 layer and 1 loss type 'loss'
    sidx_ptr_i = sidx_indptr[next_compute_i]
    loss_ptr_i = sidx_ptr_i
    extras_ptr_i = 0
    sidx_i = next_compute_i
    compute_i = 0

    temp_node_sidx = np.zeros(len_array, dtype=np.uint8)
    temp_node_loss = np.zeros(len_array, dtype=np_oasis_float)
    temp_node_extras = np.zeros((len_array, 3), dtype=np_oasis_float)

    for level in range(compute_info['start_level'], compute_info['max_level'] + 1):#perform the bottom up loss computation
        # print(level, next_compute_i, compute_i, next_compute_i-compute_i, computes[compute_i:compute_i+2], computes[next_compute_i-2: next_compute_i+1])
        next_compute_i += 1 # next_compute_i will stay at 0 wich will stop the while loop and start the next level
        while computes[compute_i]:
            node, compute_i = nodes_array[computes[compute_i]], compute_i + 1\

            #compute loss sums and extra sum
            len_children = children[node['children']]
            if len_children:#if no children and in compute then layer 1 loss is already set
                if len_children > 1:
                    # we need a new sidx array
                    sidx_created = False
                    node_sidx_start = sidx_ptr_i
                    node_sidx_end = 0
                    sidx_indexes[node['node_id']], sidx_i = sidx_i, sidx_i + 1
                    for p in range(node['profile_len']):
                        loss_indptr[node['loss'] + p] = loss_ptr_i
                        extras_indptr[node['extra'] + p] = extras_ptr_i

                        for c in range(node['children'] + 1, node['children'] + len_children + 1):
                            child = nodes_array[children[c]]
                            child_sidx_index = sidx_indexes[child['node_id']]
                            child_sidx_start = sidx_indptr[child_sidx_index]
                            child_sidx_end =  sidx_indptr[child_sidx_index + 1]
                            child_loss_start = loss_indptr[child['il'] + p]
                            child_extra_start = extras_indptr[child['extra'] + p]

                            for indptr in range(child_sidx_end - child_sidx_start):
                                temp_node_sidx[sidx_val[child_sidx_start + indptr]] = True
                                temp_node_loss[sidx_val[child_sidx_start + indptr]] += loss_val[child_loss_start + indptr]
                                temp_node_extras[sidx_val[child_sidx_start + indptr]] += extras_val[child_extra_start + indptr]
                        if sidx_created:
                            for node_sidx_cur in range(node_sidx_start, node_sidx_end):
                                loss_val[loss_ptr_i] = temp_node_loss[sidx_val[node_sidx_cur]]
                                temp_node_loss[sidx_val[node_sidx_cur]] = 0
                                loss_ptr_i += 1
                                extras_val[extras_ptr_i] = temp_node_extras[sidx_val[node_sidx_cur]]
                                temp_node_extras[sidx_val[node_sidx_cur]] = 0
                                extras_ptr_i += 1

                        else:
                            for sidx in [-3, -1]:
                                if temp_node_sidx[sidx]:
                                    sidx_ptr_i, loss_ptr_i, extras_ptr_i = single_array_to_sparse(sidx, sidx_ptr_i, sidx_val, temp_node_sidx,
                                                                                                   loss_ptr_i, loss_val, temp_node_loss,
                                                                                                   extras_ptr_i, extras_val, temp_node_extras)


                            for sidx in range(1, max_sidx_val+1):
                                if temp_node_sidx[sidx]:
                                    # print('res', node['node_id'], sidx, temp_node_loss[sidx], temp_node_extras[sidx])
                                    sidx_ptr_i, loss_ptr_i, extras_ptr_i = single_array_to_sparse(sidx, sidx_ptr_i, sidx_val, temp_node_sidx,
                                                                                                   loss_ptr_i, loss_val, temp_node_loss,
                                                                                                   extras_ptr_i, extras_val, temp_node_extras)
                            node_sidx_end = sidx_ptr_i
                            node_val_len = node_sidx_end - node_sidx_start
                            sidx_indptr[sidx_i] = sidx_ptr_i
                            sidx_created = True

                    #fill up all layer if necessary
                    for layer in range(node['profile_len'], node['layer_len']):
                        loss_indptr[node['loss'] + layer] = loss_indptr[node['loss']]
                        extras_indptr[node['extra'] + layer] = extras_indptr[node['extra']]

                else:
                    child = nodes_array[children[node['children'] + 1]]
                    loss_indptr[node['loss']:node['loss'] + node['layer_len']] = loss_indptr[child['il']:child['il'] + node['layer_len']]
                    sidx_indexes[node['node_id']] = sidx_indexes[child['node_id']]
                    node_sidx_start = sidx_indptr[sidx_indexes[node['node_id']]]
                    node_sidx_end = sidx_indptr[sidx_indexes[node['node_id']] + 1]
                    node_val_len = node_sidx_end - node_sidx_start

                    for p in range(node['profile_len']):
                        extras_indptr[node['extra'] + p] = extras_ptr_i
                        child_indptr = extras_indptr[child['extra'] + p]
                        extras_val[extras_ptr_i: extras_ptr_i + node_val_len] = extras_val[child_indptr: child_indptr + node_val_len]
                        extras_ptr_i += node_val_len

                    #fill up all layer if necessary
                    for layer in range(node['profile_len'], node['layer_len']):
                        extras_indptr[node['extra'] + layer] = extras_indptr[node['extra']]

            else:
                node_sidx_start = sidx_indptr[sidx_indexes[node['node_id']]]
                node_sidx_end = sidx_indptr[sidx_indexes[node['node_id']] + 1]
                node_val_len = node_sidx_end - node_sidx_start

                for p in range(node['profile_len']):
                    extras_indptr[node['extra'] + p] = extras_ptr_i
                    extras_val[extras_ptr_i: extras_ptr_i + node_val_len].fill(0)
                    extras_ptr_i += node_val_len

                #fill up all layer if necessary
                for layer in range(node['profile_len'], node['layer_len']):
                    extras_indptr[node['extra'] + layer] = extras_indptr[node['extra']]

            #compute il
            # print('length', node['node_id'], sidx_indexes[node['node_id']], sidx_i, node_val_len, sidx_indptr[sidx_indexes[node['node_id']]+1], sidx_indptr[sidx_indexes[node['node_id']]])
            for p in range(node['profile_len']):
                node_profile = node_profiles_array[node['profiles']+p]
                if node_profile['i_start'] < node_profile['i_end']:
                    loss_indptr_start =  loss_indptr[node['loss'] + p]
                    extra_indptr_start =  extras_indptr[node['extra'] + p]

                    loss_in = loss_val[loss_indptr_start: loss_indptr_start + node_val_len]
                    extra = extras_val[extra_indptr_start: extra_indptr_start + node_val_len]
                    loss_indptr[node['il'] + p] = loss_ptr_i
                    loss_ptr_i += node_val_len
                    loss_out = loss_val[loss_indptr[node['il'] + p]: loss_ptr_i]

                    for profile_index in range(node_profile['i_start'], node_profile['i_end']):
                        calc(fm_profile[profile_index],
                             loss_out,
                             loss_in,
                             extra[:, 0],
                             extra[:, 1],
                             extra[:, 2],
                             stepped)
                        # print(node['node_id'], loss_indptr[node['loss'] + p], loss_in, '=>', loss_out)
                else:
                    loss_indptr[node['il'] + p] = loss_indptr[node['loss'] + p]

            for layer in range(node['profile_len'], node['layer_len']):
                loss_indptr[node['il'] + layer] = loss_indptr[node['il']]

            if level == compute_info['max_level']:
                loss_indptr[node['ba']:node['ba'] + node['layer_len']] = loss_indptr[node['il']:node['il'] + node['layer_len']]
                if node['parent_len']:# for allocation 1 we set only the parent in computes
                    parent_id = node_parents_array[node['parent']]
                    computes[next_compute_i], next_compute_i = parent_id, next_compute_i + 1
                else:
                    computes[next_compute_i], next_compute_i = computes[compute_i - 1], next_compute_i + 1
            else:
                # set parent children and next computation
                for pa in range(node['parent_len']):
                    parent_id = node_parents_array[node['parent']+pa]
                    parent = nodes_array[parent_id]
                    parent_children_len = children[parent['children']] + 1
                    if parent_children_len == 1 and not pa:#only parent 0 is a real parent, parent 2 is only used on alloc 1
                        computes[next_compute_i], next_compute_i = parent_id, next_compute_i + 1
                    children[parent['children']] = parent_children_len
                    children[parent['children'] + parent_children_len] = computes[compute_i - 1]
        compute_i += 1

    if compute_info['allocation_rule'] == 0:
        pass
    elif compute_info['allocation_rule'] == 1:
        child_pos = np.zeros(compute_info['items_len'] + 1, dtype = np_oasis_int)
        next_compute_i += 1
        while computes[compute_i]:
            node, compute_i = nodes_array[computes[compute_i]], compute_i + 1
            len_children = children[node['children']]
            node_sidx_start = sidx_indptr[sidx_indexes[node['node_id']]]
            node_sidx_end = sidx_indptr[sidx_indexes[node['node_id']] + 1]
            node_val_len = node_sidx_end - node_sidx_start
            if len_children > 1:
                # we are summing the input loss of level 0 or 1 so there is no layer to take into account
                loss_indptr[node['loss']] = loss_ptr_i
                for node_sidx_cur in range(node_sidx_start, node_sidx_end):
                    sidx = sidx_val[node_sidx_cur]
                    loss_val[loss_ptr_i] = 0
                    for c in range(node['children'] + 1, node['children'] + len_children + 1):
                        child = nodes_array[children[c]]
                        if sidx == sidx_val[sidx_indptr[sidx_indexes[child['node_id']]] + child_pos[children[c]]]:
                            loss_val[loss_ptr_i] += loss_val[loss_indptr[child['loss']] + child_pos[children[c]]]
                            child_pos[children[c]] += 1
                    loss_ptr_i += 1

                for layer in range(1, node['layer_len']):
                    loss_indptr[node['loss'] + layer] = loss_indptr[node['loss']]

            loss_ptr_i, next_compute_i = back_allocate(node, children, nodes_array, sidx_indexes, sidx_indptr, sidx_val,
                                                       loss_indptr, loss_val, loss_ptr_i, computes, next_compute_i, 'loss', temp_node_loss)
        compute_i += 1
    else:
        for level in range(compute_info['max_level'] - compute_info['start_level']):# perform back allocation 2
            # print(level, next_compute_i, compute_i, next_compute_i-compute_i, computes[compute_i:compute_i + 2], computes[next_compute_i - 1: next_compute_i + 1])
            next_compute_i += 1
            while computes[compute_i]:
                node, compute_i = nodes_array[computes[compute_i]], compute_i + 1
                loss_ptr_i, next_compute_i = back_allocate(node,
                                                           children,
                                                           nodes_array,
                                                           sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, loss_ptr_i,
                                                           computes,
                                                           next_compute_i,
                                                           'il',
                                                           temp_node_loss)
            compute_i += 1
    if net_loss:
        # we go through the last level node and replace the gross loss by net loss, then reset compute_i to its value
        out_compute_i = compute_i
        while computes[compute_i]:
            node_i, compute_i= computes[compute_i], compute_i + 1
            node = nodes_array[node_i]
            # net loss layer i is initial loss - sum of all layer up to i
            node_val_len = sidx_indptr[sidx_indexes[node['node_id']] + 1] - sidx_indptr[sidx_indexes[node['node_id']]]
            node_ba_val_prev = loss_val[loss_indptr[node['loss']]: loss_indptr[node['loss']] + node_val_len]
            for layer in range(node['layer_len']):
                node_ba_val_cur = loss_val[loss_indptr[node['ba'] + layer]: loss_indptr[node['ba'] + layer] + node_val_len]
                node_ba_val_cur[:] = np.maximum(node_ba_val_prev - node_ba_val_cur, 0)
                node_ba_val_prev = node_ba_val_cur

        compute_i = out_compute_i

    # print(compute_info['max_level'], next_compute_i, compute_i, next_compute_i-compute_i, computes[compute_i:compute_i + 2], computes[next_compute_i - 1: next_compute_i + 1])
    return compute_i


def init_variable(compute_info, max_sidx_val, temp_dir, low_memory):
    """
    extras, loss contains the same index as sidx
    therefore we can use only sidx_indexes to tract the length of each node values
    """
    max_sidx_count = max_sidx_val + EXTRA_VALUES
    len_array = max_sidx_val + 4

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

    return (max_sidx_val, max_sidx_count, len_array, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, extras_indptr, extras_val,
            children, computes)


@njit(cache=True)
def reset_variable(children, compute_i, computes):
    computes[:compute_i].fill(0)
    children.fill(0)
