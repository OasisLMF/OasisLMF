from .policy import calc
from .common import float_equal_precision, np_oasis_float, np_oasis_int

from numba import njit
import numpy as np
import math
import os
import logging
logger = logging.getLogger(__name__)


@njit(cache=True, fastmath=True, error_model="numpy")
def back_allocate(node, children, nodes_array, losses, loss_indexes, loss_i, computes, next_compute_i, child_loss_pointer):
    len_children = children[node['children']]
    if len_children > 1:
        for layer in range(node['layer_len']):
            proportion = np.empty_like(losses[0])
            use_loss = np.zeros_like(losses[0], dtype=np.uint8)
            ba_loss = losses[loss_indexes[node['ba'] + layer]]
            sum_loss = losses[loss_indexes[node['loss'] + layer]]
            for i in range(proportion.shape[0]):
                if ba_loss[i] < float_equal_precision:
                    proportion[i] = 0
                else:
                    # in case of max deductible sum_loss can be 0 while ba_loss > 0
                    # the historical hack is to then use the original loss of the child instead of the il
                    # TODO: this need to be reworked to avoid loss inconsistencies at the limit sum loss = 0
                    #       when max deductible is triggered (loss reallocated proportionally to children's under_limit)
                    if sum_loss[i] < float_equal_precision:
                        use_loss[i] = 1
                        for c in range(node['children'] + 1, node['children'] + len_children + 1):
                            sum_loss[i] += losses[loss_indexes[nodes_array[children[c]]['loss'] + layer]][i]
                        if sum_loss[i] < float_equal_precision:
                            proportion[i] = 0
                            continue
                    proportion[i] = ba_loss[i] / sum_loss[i]

            for c in range(node['children'] + 1, node['children'] + len_children + 1):
                child = nodes_array[children[c]]
                child_ba = losses[loss_i]
                child_loss = losses[loss_indexes[child['loss'] + layer]]
                child_il = losses[loss_indexes[child[child_loss_pointer] + layer]]
                for i in range(proportion.shape[0]):
                    if use_loss[i]:
                        child_ba[i] = proportion[i] * child_loss[i]
                    else:
                        child_ba[i] = proportion[i] * child_il[i]
                loss_indexes[child['ba'] + layer], loss_i = loss_i, loss_i + 1

        for c in range(node['children'] + 1, node['children'] + len_children + 1):
            computes[next_compute_i], next_compute_i = children[c], next_compute_i + 1

    else:
        computes[next_compute_i], next_compute_i = children[node['children'] + 1], next_compute_i + 1
        child = nodes_array[children[node['children'] + 1]]
        loss_indexes[child['ba']: child['ba'] + node['layer_len']] = loss_indexes[node['ba']: node['ba'] + node['layer_len']]

    return loss_i, next_compute_i


@njit(cache=True, fastmath=True)
def compute_event(compute_info,
                  net_loss,
                  nodes_array,
                  node_parents_array,
                  node_profiles_array,
                  losses,
                  loss_indexes,
                  extras,
                  extra_indexes,
                  children,
                  computes,
                  next_compute_i,
                  fm_profile,
                  stepped):

    loss_i = next_compute_i
    extra_i = 0
    compute_i = 0
    for level in range(compute_info['start_level'], compute_info['max_level'] + 1):#perform the bottom up loss computation
        # print(level, next_compute_i, compute_i, next_compute_i-compute_i, computes[compute_i:compute_i+2], computes[next_compute_i-2: next_compute_i+1])
        next_compute_i += 1 # next_compute_i will stay at 0 wich will stop the while loop and start the next level
        while computes[compute_i]:
            node, compute_i = nodes_array[computes[compute_i]], compute_i + 1

            #compute loss sums and extra sum
            len_children = children[node['children']]
            if len_children:#if no children and in compute then layer 1 loss is already set
                if len_children > 1:
                    for p in range(node['profile_len']):
                        node_loss = losses[loss_i]
                        loss_indexes[node['loss'] + p], loss_i = loss_i, loss_i + 1
                        node_extra = extras[extra_i]
                        extra_indexes[node['extra'] + p], extra_i = extra_i, extra_i + 1

                        node_loss.fill(0)
                        node_extra.fill(0)
                        for c in range(node['children'] + 1, node['children'] + len_children + 1):
                            child = nodes_array[children[c]]
                            node_loss += losses[loss_indexes[child['il'] + p]]
                            node_extra += extras[extra_indexes[child['extra'] + p]]
                    #fill up all layer if necessary
                    for layer in range(node['profile_len'], node['layer_len']):
                        loss_indexes[node['loss'] + layer] = loss_indexes[node['loss']]
                        extra_indexes[node['extra'] + layer] = extra_indexes[node['extra']]
                else:
                    child = nodes_array[children[node['children'] + 1]]
                    loss_indexes[node['loss']:node['loss'] + node['layer_len']] = loss_indexes[child['il']:child['il'] + node['layer_len']]

                    for p in range(node['profile_len']):
                        extras[extra_i][:] = extras[extra_indexes[child['extra'] + p]]
                        extra_indexes[node['extra'] + p], extra_i = extra_i, extra_i + 1

                    #fill up all layer if necessary
                    for layer in range(node['profile_len'], node['layer_len']):
                        extra_indexes[node['extra'] + layer] = extra_indexes[node['extra']]

            else:
                for p in range(node['profile_len']):
                    extras[extra_i].fill(0)
                    extra_indexes[node['extra'] + p], extra_i = extra_i, extra_i + 1

                #fill up all layer if necessary
                for layer in range(node['profile_len'], node['layer_len']):
                    extra_indexes[node['extra'] + layer] = extra_indexes[node['extra']]

            #compute il
            for p in range(node['profile_len']):
                node_profile = node_profiles_array[node['profiles']+p]
                if node_profile['i_start'] < node_profile['i_end']:
                    extra = extras[extra_indexes[node['extra'] + p]]
                    loss_in = losses[loss_indexes[node['loss'] + p]]
                    for profile_index in range(node_profile['i_start'], node_profile['i_end']):
                        calc(fm_profile[profile_index],
                             losses[loss_i],
                             loss_in,
                             extra[0],
                             extra[1],
                             extra[2],
                             stepped)
                    loss_indexes[node['il'] + p], loss_i = loss_i, loss_i + 1
                else:
                    loss_indexes[node['il'] + p] = loss_indexes[node['loss'] + p]

            for layer in range(node['profile_len'], node['layer_len']):
                loss_indexes[node['il'] + layer] = loss_indexes[node['il']]

            if level == compute_info['max_level']:
                loss_indexes[node['ba']:node['ba'] + node['layer_len']] = loss_indexes[node['il']:node['il'] + node['layer_len']]
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
        next_compute_i += 1
        while computes[compute_i]:
            node, compute_i = nodes_array[computes[compute_i]], compute_i + 1
            len_children = children[node['children']]
            if len_children > 1:
                # we are summing the input loss of level 0 or 1 so there is no layer to take into account
                node_loss = losses[loss_i]
                loss_indexes[node['loss']], loss_i = loss_i, loss_i + 1

                node_loss.fill(0)
                for c in range(node['children'] + 1, node['children'] + len_children + 1):
                    child = nodes_array[children[c]]
                    node_loss += losses[loss_indexes[child['loss']]]
                for layer in range(1, node['layer_len']):
                    loss_indexes[node['loss'] + layer] = loss_indexes[node['loss']]
            loss_i, next_compute_i = back_allocate(node, children, nodes_array, losses, loss_indexes, loss_i, computes, next_compute_i, 'loss')
        compute_i += 1
    else:
        for level in range(compute_info['max_level'] - compute_info['start_level']):# perform back allocation 2
            # print(level, next_compute_i, compute_i, next_compute_i-compute_i, computes[compute_i:compute_i + 2], computes[next_compute_i - 1: next_compute_i + 1])
            next_compute_i += 1
            while computes[compute_i]:
                node, compute_i = nodes_array[computes[compute_i]], compute_i + 1
                loss_i, next_compute_i = back_allocate(node,
                                                       children,
                                                       nodes_array,
                                                       losses,
                                                       loss_indexes,
                                                       loss_i,
                                                       computes,
                                                       next_compute_i,
                                                       'il')
            compute_i += 1
    if net_loss:
        # we go through the last level node and replace the gross loss by net loss, then reset compute_i to its value
        out_compute_i = compute_i
        while computes[compute_i]:
            node_i, compute_i= computes[compute_i], compute_i + 1
            node = nodes_array[node_i]
            # net loss layer i is initial loss - sum of all layer up to i
            losses[loss_indexes[node['ba']]] = np.maximum(losses[loss_indexes[node['loss']]] - losses[loss_indexes[node['ba']]], 0)
            for layer in range(1, node['layer_len']):
                losses[loss_indexes[node['ba'] + layer]] = np.maximum((losses[loss_indexes[node['ba'] + layer - 1 ]]
                                                                       - losses[loss_indexes[node['ba'] + layer]]),
                                                                      0)

        compute_i = out_compute_i

    # print(compute_info['max_level'], next_compute_i, compute_i, next_compute_i-compute_i, computes[compute_i:compute_i + 2], computes[next_compute_i - 1: next_compute_i + 1])
    return compute_i, loss_i, extra_i


def init_variable(compute_info, len_sample, temp_dir, low_memory):
    if low_memory:
        losses = np.memmap(os.path.join(temp_dir, "temp_loss.bin"), mode='w+',
                           shape=(compute_info['loss_len'], len_sample), dtype=np_oasis_float)
        extras = np.memmap(os.path.join(temp_dir, "extras.bin"), mode='w+',
                           shape=(compute_info['extra_len'], 3, len_sample), dtype=np_oasis_float)
    else:
        losses = np.zeros((compute_info['loss_len'], len_sample), dtype=np_oasis_float)
        extras = np.zeros((compute_info['loss_len'], 3, len_sample), dtype=np_oasis_float)

    loss_indexes = np.zeros(compute_info['loss_len'], dtype=np.uint32)
    extra_indexes = np.zeros(compute_info['extra_len'], dtype=np.uint32)
    children = np.zeros(compute_info['children_len'], dtype=np.uint32)
    computes = np.zeros(compute_info['compute_len'], dtype=np.uint32)

    return losses, loss_indexes, extras, extra_indexes, children, computes


@njit(cache=True)
def reset_variabe(children, compute_i, computes, loss_i, losses):
    computes[:compute_i].fill(0)
    losses[:loss_i].fill(0)
    children.fill(0)


@njit(cache=True)
def reset_variabe_sparse(event_items_count, event_items, in_indptr_array, output_count):
    event_items[:event_items_count].fill(0)
    in_indptr_array[:event_items_count + 1].fill(0)
    output_count.fill(0)


def init_variable_sparse(compute_info, len_array, compute_size, tempdir):

    # structure to compute loss
    loss_indexes = np.zeros(compute_info['loss_len'], dtype=np.uint32)
    extra_indexes = np.zeros(compute_info['extra_len'], dtype=np.uint32)
    children = np.zeros(compute_info['children_len'], dtype=np.uint32)
    computes = np.zeros(compute_info['compute_len'], dtype=np.uint32)
    losses = np.zeros((compute_info['loss_len'], compute_size), dtype=np_oasis_float)
    extras = np.zeros((compute_info['loss_len'], 3, compute_size), dtype=np_oasis_float)

    # structure to store inputs
    event_items = np.zeros(compute_info['items_len'] + 1, dtype=np.uint32)
    in_indptr_array = np.zeros(compute_info['items_len'] + 1, dtype=np.uint32)
    in_sidx_array = np.empty(compute_info['items_len'] * len_array, dtype=np_oasis_int)
    in_loss_array = np.empty(compute_info['items_len'] * len_array, dtype=np_oasis_float)

    # structure to store output
    output_count = np.zeros(compute_info['output_len'] + 1, dtype=np.uint32)
    output_id_array = np.empty(compute_info['output_len'] * len_array, dtype=np_oasis_int)
    out_sidx = np.empty(compute_info['output_len'] * len_array, dtype=np_oasis_int)
    out_loss = np.empty(compute_info['output_len'] * len_array, dtype=np_oasis_float)

    return (loss_indexes, extra_indexes, children, computes, losses, extras,
            event_items, in_indptr_array, in_sidx_array, in_loss_array,
            output_count, output_id_array, out_sidx, out_loss)


@njit(cache=True)
def sparse_to_table(len_array, compute_size,
                    computes, nodes_array, losses, loss_indexes,
                    event_items_count, event_items, indptr_array, sidx_array, loss_array):
    cur_indptr_array = np.empty(event_items_count, dtype=np.int32)
    cur_indptr_array[:] = indptr_array[:event_items_count]

    for compute_cycle_i in range(math.ceil(len_array / compute_size)):
        # in first round we reserve the len_array index for sidx -1, all sidx are then shifted by 1
        max_sidx = (compute_cycle_i + 1) * compute_size - 2
        compute_i = 0

        for event_item_i in range(event_items_count):
            agg_id = event_items[event_item_i]
            if sidx_array[cur_indptr_array[event_item_i]] <= max_sidx: # check if at least 1 sidx is in range
                node = nodes_array[agg_id]
                loss_indexes[node['loss']: node['loss'] + node['layer_len']] = compute_i
                computes[compute_i] = agg_id
                compute_i += 1

                for indptr in range(cur_indptr_array[event_item_i], indptr_array[event_item_i + 1]):
                    if sidx_array[indptr] <= max_sidx:
                        if sidx_array[indptr] == -3:
                            losses[compute_i][0] = loss_array[indptr]
                        elif sidx_array[indptr] == -1:
                            losses[compute_i][1] = loss_array[indptr]
                        else:
                            losses[compute_i][(sidx_array[indptr] + 1) % compute_size] = loss_array[indptr]
                        cur_indptr_array[event_item_i] += 1
                    else:
                        break
        if compute_i:
            yield compute_cycle_i, compute_i, min(compute_size, len_array - compute_cycle_i * compute_size)


@njit(cache=True)
def table_to_sparse(compute_cycle_i, compute_i, cur_output_index,
                    computes, losses, loss_indexes,
                    output_count, output_id_array, out_sidx, out_loss,
                    compute_size, loss_threshold, nodes_array, output_array):
    """
    for the output sparse we use a coo_matrix  base on:
        the three vector output_id_array, out_sidx, out_loss to store the values
        and out_count to store the number of value that will be useful when writing the final stream
    """
    base_sidx = compute_cycle_i * compute_size - 1
    while computes[compute_i]:
        node = nodes_array[computes[compute_i]]
        for layer in range(node['layer_len']):
            output_id = output_array[node['output_ids'] + layer]
            if output_id:  # if output is not in xref output_id is 0
                loss_index = loss_indexes[node['ba'] + layer]
                if compute_cycle_i == 0:# first sample block
                    output_count[output_id] = 2
                    output_id_array[cur_output_index] = output_id
                    out_sidx[cur_output_index] = -3
                    out_loss[cur_output_index] = losses[loss_index, 0]
                    cur_output_index += 1

                    output_id_array[cur_output_index] = output_id
                    out_sidx[cur_output_index] = -1
                    out_loss[cur_output_index] = losses[loss_index, 1]
                    cur_output_index += 1
                    s_start = 2
                else:
                    s_start = 0

                for s in range(s_start, compute_size):
                    if losses[loss_index, s] > loss_threshold:
                        output_count[output_id] += 1
                        output_id_array[cur_output_index] = output_id
                        out_sidx[cur_output_index] = base_sidx + s
                        out_loss[cur_output_index] = losses[loss_index, s]
                        cur_output_index += 1
        compute_i += 1
    return cur_output_index, compute_i


@njit(cache=True)
def compute_sparse(net_loss, compute_size, loss_threshold, stepped,
                   compute_info, nodes_array, node_parents_array, node_profiles_array, output_array, fm_profile,
                   len_array, event_items_count,
                   loss_indexes, extra_indexes, children, computes, losses, extras,
                   event_items, in_indptr_array, in_sidx_array, in_loss_array,
                   output_count, output_id_array, out_sidx, out_loss):
    cur_output_index = 0
    for compute_cycle_i, compute_i, cur_compute_size in sparse_to_table(len_array, compute_size,
                                                      computes, nodes_array, losses, loss_indexes,
                                                      event_items_count, event_items, in_indptr_array, in_sidx_array,
                                                      in_loss_array):
        compute_i, loss_i, extra_i = compute_event(compute_info,
                                                   net_loss,
                                                   nodes_array,
                                                   node_parents_array,
                                                   node_profiles_array,
                                                   losses,
                                                   loss_indexes,
                                                   extras,
                                                   extra_indexes,
                                                   children,
                                                   computes,
                                                   compute_i,
                                                   fm_profile,
                                                   stepped)
        cur_output_index, compute_i = table_to_sparse(compute_cycle_i, compute_i, cur_output_index,
                                           computes, losses, loss_indexes,
                                           output_count, output_id_array, out_sidx, out_loss,
                                           cur_compute_size, loss_threshold, nodes_array, output_array)
        reset_variabe(children, compute_i, computes, loss_i, losses)
    return cur_output_index