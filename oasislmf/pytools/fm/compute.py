from .financial_structure import TEMP_STORAGE, OUTPUT_STORAGE, STORE_LOSS_SUM_OPTION,\
    PROFILE, IL_PER_GUL, IL_PER_SUB_IL, PROPORTION, COPY
from .policy import calc
from .common import float_equal_precision, np_oasis_float
from .queue import QueueTerminated

from numba import njit, boolean
import numpy as np
import logging
logger = logging.getLogger(__name__)


@njit(cache=True, fastmath=True, error_model="numpy")
def compute_event(compute_queue, dependencies, input_loss, input_not_null, profile,
                  temp_loss, temp_not_null, losses_sum, deductibles, over_limit, under_limit,
                  output_loss, output_not_null):
    temp_not_null.fill(0)
    output_not_null.fill(0)

    losses = [input_loss,
              temp_loss,
              output_loss
              ]
    not_null = [input_not_null,
                temp_not_null,
                output_not_null
                ]

    for node in compute_queue:
        if node['computation_id'] == PROFILE:
            for dependency in dependencies[node['dependencies_index_start']: node['dependencies_index_end']]:
                if not_null[dependency['storage']][dependency['index']]:
                    is_not_null = True
                    break
            else:
                is_not_null = False

            if is_not_null:
                loss_sum = losses_sum[node['index']]
                loss_sum.fill(0)
                deductibles[node['index']].fill(0)
                over_limit[node['index']].fill(0)
                under_limit[node['index']].fill(0)

                for dependency in dependencies[node['dependencies_index_start']: node['dependencies_index_end']]:
                    if not_null[dependency['storage']][dependency['index']]:
                        loss_sum += losses[dependency['storage']][dependency['index']]
                        if dependency['storage'] == TEMP_STORAGE:
                            deductibles[node['index']] += deductibles[dependency['index']]
                            over_limit[node['index']] += over_limit[dependency['index']]
                            under_limit[node['index']] += under_limit[dependency['index']]

                calc(profile[node['profile']],
                     losses[node['storage']][node['index']],
                     loss_sum,
                     deductibles[node['index']],
                     over_limit[node['index']],
                     under_limit[node['index']])

                not_null[node['storage']][node['index']] = True

        elif node['computation_id'] == IL_PER_GUL:
            node_dependencies = dependencies[node['dependencies_index_start']: node['dependencies_index_end']]
            top_node = node_dependencies[0]
            if not_null[top_node['storage']][top_node['index']]:
                node_loss = losses[node['storage']][node['index']]

                top_loss = losses[top_node['storage']][top_node['index']]
                for dependency_node in node_dependencies[1:]:
                    node_loss += losses[dependency_node['storage']][dependency_node['index']]

                for i in range(top_loss.shape[0]):
                    if top_loss[i] < float_equal_precision:
                        node_loss[i] = 0
                    else:
                        node_loss[i] = top_loss[i] / node_loss[i]
                not_null[node['storage']][node['index']] = True

        elif node['computation_id'] == IL_PER_SUB_IL:
            ba_node, il_node = dependencies[node['dependencies_index_start']: node['dependencies_index_end']]
            if not_null[ba_node['storage']][ba_node['index']]:

                node_loss = losses[node['storage']][node['index']]
                ba_loss = losses[ba_node['storage']][ba_node['index']]
                if il_node['storage'] == TEMP_STORAGE:
                    il_loss = losses_sum[il_node['index']]
                else:
                    il_loss = losses[il_node['storage']][il_node['index']]

                for i in range(node_loss.shape[0]):
                    if ba_loss[i] < float_equal_precision:
                        node_loss[i] = 0
                    else:
                        node_loss[i] = ba_loss[i] / il_loss[i]

                not_null[node['storage']][node['index']] = True

        elif node['computation_id'] == PROPORTION:
            top_node, il_node = dependencies[node['dependencies_index_start']: node['dependencies_index_end']]

            if not_null[il_node['storage']][il_node['index']] and not_null[top_node['storage']][top_node['index']]:
                losses[node['storage']][node['index']] = losses[top_node['storage']][top_node['index']] * losses[il_node['storage']][il_node['index']]
                not_null[node['storage']][node['index']] = True

        elif node['computation_id'] == COPY:
            copy_node = dependencies[node['dependencies_index_start']]
            if not_null[copy_node['storage']][copy_node['index']]:
                losses[node['storage']][node['index']] = losses[copy_node['storage']][copy_node['index']]
                not_null[node['storage']][node['index']] = True


@njit(cache=True)
def init_intermediary_variable(storage_to_len, len_sample, options):
    temp_loss = np.empty((storage_to_len[TEMP_STORAGE], len_sample), dtype=np_oasis_float)
    temp_not_null = np.empty((storage_to_len[TEMP_STORAGE],), dtype=boolean)

    if options[STORE_LOSS_SUM_OPTION]:
        losses_sum = np.empty((storage_to_len[TEMP_STORAGE], len_sample), dtype=np_oasis_float)
    else:
        losses_sum = temp_loss

    deductibles = np.empty((storage_to_len[TEMP_STORAGE], len_sample), dtype=np_oasis_float)
    over_limit = np.empty((storage_to_len[TEMP_STORAGE], len_sample), dtype=np_oasis_float)
    under_limit = np.empty((storage_to_len[TEMP_STORAGE], len_sample), dtype=np_oasis_float)

    return temp_loss, temp_not_null, losses_sum, deductibles, over_limit, under_limit


@njit(cache=True)
def init_loss_variable(storage_to_len, storage, len_sample):
    loss = np.empty((storage_to_len[storage], len_sample), dtype=np_oasis_float)
    not_null = np.empty((storage_to_len[storage],), dtype=boolean)
    return loss, not_null


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
