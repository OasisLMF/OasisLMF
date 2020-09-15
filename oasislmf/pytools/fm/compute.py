from .financial_structure import TEMP_STORAGE, OUTPUT_STORAGE, TOP_UP, STORE_LOSS_SUM_OPTION,\
    PROFILE, IL_PER_GUL, IL_PER_SUB_IL, PROPORTION, COPY
from .policy import calc
from .common import float_equal_precision, np_oasis_float, null_index
from .queue import QueueTerminated

from numba import njit, boolean
import numpy as np
import logging
logger = logging.getLogger(__name__)


@njit(cache=True, fastmath=True, error_model="numpy")
def compute_event(compute_queue, dependencies, input_loss, input_index, profile,
                  temp_loss, temp_index, losses_sum, deductibles, over_limit, under_limit,
                  output_loss, output_index):
    temp_index.fill(-1)
    output_index.fill(-1)

    losses = [input_loss,
              temp_loss,
              output_loss
              ]
    index = [input_index,
             temp_index,
             output_index
             ]
    index_new = np.zeros(3, dtype=np.uint32)

    for node in compute_queue:
        if node['computation_id'] == PROFILE:
            for dependency in dependencies[node['dependencies_index_start']: node['dependencies_index_end']]:
                if index[dependency['storage']][dependency['index']] != null_index:
                    node_index = index_new[node['storage']]
                    index_new[node['storage']] += 1
                    index[node['storage']][node['index']] = node_index
                    break
            else:
                continue

            loss_sum = losses_sum[node_index]
            loss_sum.fill(0)
            deductibles[node_index].fill(0)
            over_limit[node_index].fill(0)
            under_limit[node_index].fill(0)

            for dependency in dependencies[node['dependencies_index_start']: node['dependencies_index_end']]:
                dependency_index = index[dependency['storage']][dependency['index']]
                if dependency_index != null_index:
                    loss_sum += losses[dependency['storage']][dependency_index]
                    if dependency['storage'] == TEMP_STORAGE:
                        deductibles[node['index']] += deductibles[dependency_index]
                        over_limit[node['index']] += over_limit[dependency_index]
                        under_limit[node['index']] += under_limit[dependency_index]

            calc(profile[node['profile']],
                 losses[node['storage']][node_index],
                 loss_sum,
                 deductibles[node_index],
                 over_limit[node_index],
                 under_limit[node_index])

        elif node['computation_id'] == IL_PER_GUL:
            node_dependencies = dependencies[node['dependencies_index_start']: node['dependencies_index_end']]
            top_node = node_dependencies[0]
            if index[top_node['storage']][top_node['index']] != null_index:
                node_index = index_new[node['storage']]
                index_new[node['storage']] += 1
                index[node['storage']][node['index']] = node_index
                node_loss = losses[node['storage']][node_index]

                top_loss = losses[top_node['storage']][index[top_node['storage']][top_node['index']]]
                for dependency_node in node_dependencies[1:]:
                    dependency_index = index[dependency['storage']][dependency['index']]
                    if dependency_index != null_index:
                        node_loss += losses[dependency_node['storage']][dependency_index]

                for i in range(top_loss.shape[0]):
                    if top_loss[i] < float_equal_precision:
                        node_loss[i] = 0
                    else:
                        node_loss[i] = top_loss[i] / node_loss[i]

        elif node['computation_id'] == IL_PER_SUB_IL:
            ba_node, il_node = dependencies[node['dependencies_index_start']: node['dependencies_index_end']]
            if index[ba_node['storage']][ba_node['index']] != null_index:
                node_index = index_new[node['storage']]
                index_new[node['storage']] += 1
                index[node['storage']][node['index']] = node_index
                node_loss = losses[node['storage']][node_index]

                ba_loss = losses[ba_node['storage']][index[ba_node['storage']][ba_node['index']]]
                if il_node['storage'] == TEMP_STORAGE:
                    il_loss = losses_sum[index[il_node['storage']][il_node['index']]]
                else:
                    il_loss = losses[il_node['storage']][index[il_node['storage']][il_node['index']]]

                for i in range(node_loss.shape[0]):
                    if ba_loss[i] < float_equal_precision:
                        node_loss[i] = 0
                    else:
                        node_loss[i] = ba_loss[i] / il_loss[i]

        elif node['computation_id'] == PROPORTION:
            top_node, il_node = dependencies[node['dependencies_index_start']: node['dependencies_index_end']]
            top_index = index[top_node['storage']][top_node['index']]
            il_index = index[il_node['storage']][il_node['index']]
            if il_index != null_index and top_index != null_index:
                node_index = index_new[node['storage']]
                index_new[node['storage']] += 1
                index[node['storage']][node['index']] = node_index
                losses[node['storage']][node_index] = losses[top_node['storage']][top_index] * losses[il_node['storage']][il_index]

        elif node['computation_id'] == COPY:
            copy_node = dependencies[node['dependencies_index_start']]
            if index[copy_node['storage']][copy_node['index']] != null_index:
                node_index = index_new[node['storage']]
                index_new[node['storage']] += 1
                index[node['storage']][node['index']] = node_index
                losses[node['storage']][node_index] = losses[copy_node['storage']][copy_node['index']]

#@njit(cache=True)
import os
def init_intermediary_variable(storage_to_len, len_sample, options, temp_dir):
    temp_loss = np.memmap(os.path.join(temp_dir, "temp_loss.bin"), mode='w+', shape=(storage_to_len[TEMP_STORAGE], len_sample), dtype=np_oasis_float)
    temp_index = np.empty((storage_to_len[TEMP_STORAGE],), dtype=np.uint32)

    if options[STORE_LOSS_SUM_OPTION]:
        losses_sum = np.memmap(os.path.join(temp_dir, "losses_sum.bin"), mode='w+', shape=(storage_to_len[TOP_UP], len_sample), dtype=np_oasis_float)
    else:
        losses_sum = temp_loss

    deductibles = np.memmap(os.path.join(temp_dir, "deductibles.bin"), mode='w+', shape=(storage_to_len[TOP_UP], len_sample), dtype=np_oasis_float)
    over_limit = np.memmap(os.path.join(temp_dir, "over_limit.bin"), mode='w+', shape=(storage_to_len[TOP_UP], len_sample), dtype=np_oasis_float)
    under_limit = np.memmap(os.path.join(temp_dir, "under_limit.bin"), mode='w+', shape=(storage_to_len[TOP_UP], len_sample), dtype=np_oasis_float)

    return temp_loss, temp_index, losses_sum, deductibles, over_limit, under_limit


#@njit(cache=True)
def init_loss_variable(storage_to_len, storage, len_sample, temp_dir):
    loss = np.memmap(os.path.join(temp_dir, f"loss_{storage}.bin"), mode='w+', shape=(storage_to_len[storage], len_sample), dtype=np_oasis_float)
    index = np.empty((storage_to_len[storage],), dtype=np.uint32)
    return loss, index


def event_computer(queue_in, queue_out, compute_queue, dependencies, storage_to_len, options, profile, sentinel):
    try:
        while True:
            event_in = queue_in.get()
            if event_in == sentinel:
                break

            event_id, input_loss, input_not_null = event_in
            input_loss = np.array(input_loss)
            input_not_null = np.array(input_not_null)
            logger.debug(f"computing {event_id}")

            try:
                output_loss, output_not_null = init_loss_variable(storage_to_len, OUTPUT_STORAGE, len_sample)
                compute_event(compute_queue, dependencies, input_loss, input_not_null, profile,
                              temp_loss, temp_not_null, losses_sum,deductibles, over_limit, under_limit,
                              output_loss, output_not_null)
            except UnboundLocalError: # initialize variable
                len_sample = input_loss.shape[1]

                temp_loss, temp_not_null, losses_sum, deductibles, over_limit, under_limit = init_intermediary_variable(storage_to_len, len_sample, options)
                output_loss, output_not_null = init_loss_variable(storage_to_len, OUTPUT_STORAGE, len_sample)

                compute_event(compute_queue, dependencies, input_loss, input_not_null, profile,
                              temp_loss, temp_not_null, losses_sum,deductibles, over_limit, under_limit,
                              output_loss, output_not_null)

            logger.debug(f"computed {event_id}")

            try:
                queue_out.put((event_id, output_loss, output_not_null))
            except QueueTerminated:
                logger.warning(f"stopped because exception was raised")
                break

        logger.info(f"compute done")
    except Exception:
        logger.exception(f"Exception in compute")
        logger.error(input_loss)
        queue_in.terminated = True
        queue_out.terminated = True
        raise


try:
    import ray
except ImportError:
    pass
else:
    from numba.typed import List, Dict


    def numba_to_python(nb_options):
        return dict(nb_options)


    def python_to_numba(py_options):
        nb_options = Dict()
        for key, val in py_options.items():
            nb_options[key] = val

        return nb_options

    @ray.remote
    def ray_event_computer(queue_in, queue_out, compute_queue, dependencies, storage_to_len, options, profile, sentinel):
        options = python_to_numba(options)
        event_computer(queue_in, queue_out, compute_queue, dependencies, storage_to_len, options, profile, sentinel)
        return
