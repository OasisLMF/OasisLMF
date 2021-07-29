from .policy import calc
from .common import float_equal_precision, np_oasis_float

from numba import njit
import numpy as np
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
def reset_variable(children, compute_i, computes, loss_i, losses):
    computes[:compute_i].fill(0)
    losses[:loss_i].fill(0)
    children.fill(0)
