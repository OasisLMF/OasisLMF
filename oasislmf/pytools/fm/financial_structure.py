__all__ = [
    'load_financial_structure',
    'load_static',
    'prepare_financial_structure',
    'INPUT_STORAGE',
    'TEMP_STORAGE',
    'OUTPUT_STORAGE',
    'TOP_UP',
    'PROFILE',
    'IL_PER_GUL',
    'IL_PER_SUB_IL',
    'PROPORTION',
    'OUTPUT',
    'COPY',
    'STORE_LOSS_SUM_OPTION',
    'CALC_DEDUCTIBLE_OPTION',
]

from .common import nb_oasis_int, np_oasis_int, almost_equal

from numba import njit, types, from_dtype
from numba.typed import List, Dict
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)

INPUT_STORAGE = nb_oasis_int(0)
TEMP_STORAGE = nb_oasis_int(1)
OUTPUT_STORAGE = nb_oasis_int(2)
TOP_UP = nb_oasis_int(3)

PROFILE = nb_oasis_int(0)
IL_PER_GUL = nb_oasis_int(1)
IL_PER_SUB_IL = nb_oasis_int(2)
PROPORTION = nb_oasis_int(3)
OUTPUT = nb_oasis_int(4)
COPY = nb_oasis_int(5)

STORE_LOSS_SUM_OPTION = nb_oasis_int(0)
CALC_DEDUCTIBLE_OPTION = nb_oasis_int(1)

node_type = types.UniTuple(nb_oasis_int, 4)
storage_type = types.UniTuple(nb_oasis_int, 2)

fm_programme_dtype = np.dtype([('from_agg_id', 'i4'), ('level_id', 'i4'), ('to_agg_id', 'i4')])
fm_policytc_dtype = np.dtype([('level_id', 'i4'), ('agg_id', 'i4'), ('layer_id', 'i4'), ('policytc_id', 'i4')])
fm_profile_dtype = np.dtype([('policytc_id', 'i4'),
                             ('calcrule_id', 'i4'),
                             ('deductible_1', 'i4'),
                             ('deductible_2', 'f4'),
                             ('deductible_3', 'f4'),
                             ('attachment_1', 'f4'),
                             ('limit_1', 'f4'),
                             ('share_1', 'f4'),
                             ('share_2', 'f4'),
                             ('share_3', 'f4'),
                             ])
fm_profile_step_dtype = np.dtype([('policytc_id', 'i4'),
                                  ('calcrule_id', 'i4'),
                                  ('deductible_1', 'i4'),
                                  ('deductible_2', 'f4'),
                                  ('deductible_3', 'f4'),
                                  ('attachment_1', 'f4'),
                                  ('limit_1', 'f4'),
                                  ('share_1', 'f4'),
                                  ('share_2', 'f4'),
                                  ('share_3', 'f4'),
                                  ('step_id', 'i4'),
                                  ('trigger_start', 'f4'),
                                  ('trigger_end', 'f4'),
                                  ('payout_start', 'f4'),
                                  ('payout_end', 'f4'),
                                  ('limit_2', 'f4'),
                                  ('scale_1', 'f4'),
                                  ('scale_2', 'f4'),
                                  ])
fm_xref_dtype = np.dtype([('output_id', 'i4'), ('agg_id', 'i4'), ('layer_id', 'i4')])

output_item_index_dtype = from_dtype(np.dtype([('output_id', np.int32), ('index', np.int32)]))
compute_array_dtype = from_dtype(np.dtype([('layer_id', np.int32),
                                           ('level_id', np.int32),
                                           ('agg_id', np.int32),
                                           ('storage', np.int32),
                                           ('index', np.int32),
                                           ('computation_id', np.int32),
                                           ('dependencies_index_start', np.int32),
                                           ('dependencies_index_end', np.int32),
                                           ('profile', np.int32),
                                           ]))
dependencies_array_dtype = from_dtype(np.dtype([('storage', np.int32), ('index', np.int32)]))


def load_static(static_path):
    policytc_file_name = 'fm_policytc.bin'
    programme_file_name = 'fm_programme.bin'
    profile_file_name = 'fm_profile.bin'
    xref_file_name = 'fm_xref.bin'

    programme = np.fromfile(os.path.join(static_path, programme_file_name), dtype=fm_programme_dtype)
    policytc = np.fromfile(os.path.join(static_path, policytc_file_name), dtype=fm_policytc_dtype)
    profile = np.fromfile(os.path.join(static_path, profile_file_name), dtype=fm_profile_dtype)
    xref = np.fromfile(os.path.join(static_path, xref_file_name), dtype=fm_xref_dtype)

    return programme, policytc, profile, xref


@njit(cache=True)
def does_nothing(profile):
    return (profile['calcrule_id'] == 100 or
            profile['calcrule_id'] == 12 and almost_equal(profile['deductible_1'], 0) or
            profile['calcrule_id'] == 15 and almost_equal(profile['limit_1'], 1) or
            profile['calcrule_id'] == 16 and almost_equal(profile['deductible_1'], 0)
            )


@njit(cache=True)
def effective_node(node_indexes, layer, child):
    #if the sublayer doesn't exist we use the layer 1
    child_node = (layer, child[0], child[1], PROFILE)
    if child_node not in node_indexes:
        child_node = (nb_oasis_int(1), child[0], child[1], PROFILE)
    return child_node


@njit(cache=True)
def get_all_children(parent_to_children, node):
    children = List()

    temp = List()
    temp.extend(parent_to_children[node])
    while temp:
        parent = temp.pop()
        if parent in parent_to_children:
            temp.extend(parent_to_children[parent])
        else:
            children.append(parent)

    return children

@njit(cache=True)
def get_programe_dependency(fm_programme):
    parent_to_children = Dict()
    top_level = nb_oasis_int(0)
    programme_nodes = set()
    for i in range(fm_programme.shape[0]):
        programme = fm_programme[i]
        parent = (nb_oasis_int(programme['level_id']), nb_oasis_int(programme['to_agg_id']))
        child = (nb_oasis_int(programme['level_id'] - 1), nb_oasis_int(programme['from_agg_id']))
        programme_nodes.add(child)
        programme_nodes.add(parent)
        if parent not in parent_to_children:
            _list = List()
            _list.append(child)
            parent_to_children[parent] = _list
        else:
            parent_to_children[parent].append(child)

        if programme['level_id'] > top_level:
            top_level = nb_oasis_int(programme['level_id'])
    return parent_to_children, List(programme_nodes), top_level


@njit(cache=True)
def get_node_layers(fm_policytc, fm_profile):
    """
    get the mapping between each node in programme and their different  Layers

    :param fm_policytc: np ndarray of fm_policytc
    :param fm_profile: np ndarray of fm_profile
    :return:
        programme_node_to_layers:
        dictionary of programme Node (level_id, agg_id) to a list of layer_id and their profile index

    """

    #profile
    policytc_id_to_profile_index = Dict()
    for i in range(fm_profile.shape[0]):
        policytc_id_to_profile_index[nb_oasis_int(fm_profile[i]['policytc_id'])] = nb_oasis_int(i)

    #policytc
    programme_node_to_layers = Dict()
    for i in range(fm_policytc.shape[0]):
        policytc = fm_policytc[i]
        programme_node = (nb_oasis_int(policytc['level_id']), nb_oasis_int(policytc['agg_id']))
        layer = (nb_oasis_int(policytc['layer_id']), policytc_id_to_profile_index[nb_oasis_int(policytc['policytc_id'])])

        if programme_node not in programme_node_to_layers:
            _list = List()
            _list.append(layer)
            programme_node_to_layers[programme_node] = _list
        else:
            programme_node_to_layers[programme_node].append(layer)

    return programme_node_to_layers


@njit(cache=True)
def process_programme(allocation_rule, programme_nodes, programme_node_to_layers, parent_to_children, fm_profile, top_level):
    """
    go through the each programme node:
        - order them on the compute queue
        - give them a storage index (place to store results)
        - prepare their back allocation if necessary

    nodes are identified by (layer_id, level_id, agg_id, computation_id)

    :param allocation_rule:
    :param programme_nodes:
    :param programme_node_to_layers:
    :param parent_to_children:
    :param fm_profile:
    :param top_level:
    :return:
    """
    compute_queue = List.empty_list(node_type)
    top_nodes = List.empty_list(node_type)

    node_to_index = Dict.empty(node_type, storage_type)
    node_to_dependencies = Dict.empty(node_type, List.empty_list(node_type))
    node_to_profile = Dict.empty(node_type, nb_oasis_int)

    i_input = nb_oasis_int(0)
    i_temp = nb_oasis_int(0)
    i_output = nb_oasis_int(0)

    not_visited = set(programme_nodes)
    temp_not_visited = set(programme_nodes)
    while not_visited:
        for programme_node in not_visited:
            if programme_node in parent_to_children:# check if input node
                children = parent_to_children[programme_node]
                for child in children:# check if all dependency have been processed
                    if child in not_visited:
                        break
                else: # dependencies are satisfied
                    for layer, profile_index in programme_node_to_layers[programme_node]:
                        node = (layer, programme_node[0], programme_node[1], PROFILE)
                        node_to_profile[node] = profile_index

                        if len(children) == 1 and does_nothing(fm_profile[profile_index]):
                            # node is just a placeholder we direct its index to its child index
                            node_child = effective_node(node_to_index, layer, children[0])
                            node_to_index[node] = node_to_index[node_child]
                            _list = List()
                            _list.append(node_child)
                            node_to_dependencies[node] = _list
                        else:
                            _list = List()
                            for child in children:
                                node_child = effective_node(node_to_index, layer, child)
                                _list.append(node_child)
                            node_to_dependencies[node] = _list
                            compute_queue.append(node)

                            if programme_node[0] == top_level:
                                top_nodes.append(node)
                                node_to_index[node], i_temp = (TEMP_STORAGE, nb_oasis_int(i_temp)), i_temp + 1
                            else:
                                node_to_index[node], i_temp = (TEMP_STORAGE, nb_oasis_int(i_temp)), i_temp + 1

                    temp_not_visited.remove(programme_node)
            else:
                if programme_node[0] != 0:
                    raise Exception('non-input node have no dependencies')
                node = (nb_oasis_int(1), programme_node[0], programme_node[1], PROFILE)
                node_to_index[node] = (INPUT_STORAGE, programme_node[1] - nb_oasis_int(1))
                if i_input < programme_node[1]:
                    i_input = programme_node[1]

                temp_not_visited.remove(programme_node)
        not_visited, temp_not_visited = temp_not_visited, set(temp_not_visited)

    i_top_up = i_temp
    if allocation_rule == 0:
        for node in top_nodes:
            copy_node = (node[0], node[1], node[2], COPY)
            dependencies = List()
            dependencies.append(node)
            node_to_dependencies[copy_node] = dependencies
            compute_queue.append(copy_node)
            node_to_index[(node[0], nb_oasis_int(0), node[2], OUTPUT)] = (OUTPUT_STORAGE, nb_oasis_int(i_output))
            node_to_index[copy_node], i_output = (OUTPUT_STORAGE, nb_oasis_int(i_output)), i_output + 1

    elif allocation_rule == 1:
        for node in top_nodes:
            children = get_all_children(node_to_dependencies, node)
            node_il_per_gul = (node[0], node[1], node[2], IL_PER_GUL)
            dependencies = List()
            dependencies.append(node)
            dependencies.extend(children)
            node_to_dependencies[node_il_per_gul] = dependencies
            compute_queue.append(node_il_per_gul)
            node_to_index[node_il_per_gul], i_temp = (TEMP_STORAGE, nb_oasis_int(i_temp)), i_temp + 1

            for child in children:
                node_child = (node[0], child[1], child[2], PROPORTION)
                dependencies = List()
                dependencies.append(node_il_per_gul)
                dependencies.append(child)
                node_to_dependencies[node_child] = dependencies
                compute_queue.append(node_child)
                node_to_index[(node_child[0], node_child[1], node_child[2], OUTPUT)] = (OUTPUT_STORAGE, nb_oasis_int(i_output))
                node_to_index[node_child], i_output = (OUTPUT_STORAGE, nb_oasis_int(i_output)), i_output + 1

    if allocation_rule == 2:# in this case tu sum of il prior applying profile is already computed
        # ba, back allocation, node containing the back allocated loss
        # il, input loss, node containing the loss calculated from the input loss and profiles
        # il_per_sub_il, node containing the temporary result of ba/il
        temp = top_nodes# we are going to treat node from the end
        while temp:
            node_ba = temp.pop()
            node_il = effective_node(node_to_index, node_ba[0], (node_ba[1], node_ba[2]))
            node_il_per_sub_il = (node_ba[0], node_ba[1], node_ba[2], IL_PER_SUB_IL)

            dependencies = List()
            dependencies.append(node_ba)
            dependencies.append(node_il)
            node_to_dependencies[node_il_per_sub_il] = dependencies
            compute_queue.append(node_il_per_sub_il)
            node_to_index[node_il_per_sub_il], i_temp = (TEMP_STORAGE, nb_oasis_int(i_temp)), i_temp + 1

            if node_il in node_to_dependencies:
                children = node_to_dependencies[node_il]
                for child in children:
                    res_child = child
                    while True:
                        if (res_child[1], res_child[2]) in parent_to_children:
                            sub_childs = parent_to_children[(res_child[1], res_child[2])]
                            if len(sub_childs) == 1:
                                res_child = effective_node(node_to_index, node_ba[0], sub_childs[0])
                            else:
                                break
                        else:
                            break

                    node_child = (node_ba[0], res_child[1], res_child[2], PROPORTION)
                    dependencies = List()
                    dependencies.append(node_il_per_sub_il)
                    dependencies.append(child)
                    node_to_dependencies[node_child] = dependencies
                    compute_queue.append(node_child)
                    if node_child[1] == 0: # node is at input level
                        node_to_index[(node_child[0], node_child[1], node_child[2], OUTPUT)] = (OUTPUT_STORAGE, nb_oasis_int(i_output))
                        node_to_index[node_child], i_output = (OUTPUT_STORAGE, nb_oasis_int(i_output)), i_output + 1
                    else:
                        node_to_index[node_child], i_temp = (TEMP_STORAGE, nb_oasis_int(i_temp)), i_temp + 1
                        temp.append(node_child)
            else:
                raise Exception('missing dependencies')

    storage_to_len = np.array([i_input, i_temp, i_output, i_top_up], dtype=nb_oasis_int)

    node_to_dependencies_index = Dict.empty(node_type, List.empty_list(storage_type))
    for node, dependencies in node_to_dependencies.items():
        dependencies_index = List()
        for dependency in dependencies:
            dependencies_index.append(node_to_index[dependency])
        node_to_dependencies_index[node] = dependencies_index

    return compute_queue, top_nodes, node_to_index, node_to_dependencies_index, node_to_profile, storage_to_len


@njit(cache=True)
def get_output_item_index(node_to_index, fm_xref):
    output_item_index = np.empty(fm_xref.shape[0], dtype=output_item_index_dtype)
    for i in range(fm_xref.shape[0]):
        xref = fm_xref[i]
        output = output_item_index[i]
        output_node = (nb_oasis_int(xref['layer_id']), nb_oasis_int(0), nb_oasis_int(xref['agg_id']), OUTPUT)
        output['output_id'] = nb_oasis_int(xref['output_id'])
        output['index'] = node_to_index[output_node][1]
    return output_item_index


@njit(cache=True)
def to_arrays(compute_queue, node_to_index, node_to_dependencies, node_to_profile):
    """(layer_id, level_id, agg_id, computation_id)"""
    compute_array = np.empty(len(compute_queue), dtype=compute_array_dtype)
    dependencies_index = 0
    for i, node in enumerate(compute_queue):
        compute_line = compute_array[i]

        #node info
        compute_line['layer_id'] = node[0]
        compute_line['level_id'] = node[1]
        compute_line['agg_id'] = node[2]
        compute_line['computation_id'] = node[3]

        #storage info
        storage, index = node_to_index[node]
        compute_line['storage'] = storage
        compute_line['index'] = index

        #dependency info
        dependencies_len = len(node_to_dependencies[node])
        compute_line['dependencies_index_start'] = dependencies_index
        dependencies_index+=dependencies_len
        compute_line['dependencies_index_end'] = dependencies_index

        #profile info
        if node[3] == PROFILE:
            compute_line['profile'] = node_to_profile[node]

    # reformat dependency as np array
    dependencies_array = np.empty(dependencies_index, dtype=dependencies_array_dtype)
    dependencies_index = 0
    for node in compute_queue:
        for storage, index in node_to_dependencies[node]:
            dependency, dependencies_index = dependencies_array[dependencies_index], dependencies_index + 1
            dependency['storage'] = storage
            dependency['index'] = index

    return compute_array, dependencies_array


@njit(cache=True)
def prepare_financial_structure(allocation_rule, fm_programme, fm_policytc, fm_profile, fm_xref):
    """
    :param output_rule:
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
        compute_queue : the step of the computation to perform on each event
        node_indexes : map node to index of item in result array
        index_dependencies : map node to its dependent indexes
        node_profile : map node to profile
        output_item_index : list of item_id, index to put in the output

    """

    parent_to_children, programme_nodes, top_level = get_programe_dependency(fm_programme)
    programme_node_to_layers = get_node_layers(fm_policytc, fm_profile)

    compute_queue, top_nodes, node_to_index, node_to_dependencies, node_to_profile, storage_to_len = process_programme(
        allocation_rule,
        programme_nodes,
        programme_node_to_layers,
        parent_to_children,
        fm_profile,
        top_level)

    output_item_index = get_output_item_index(node_to_index, fm_xref)
    compute_array, dependencies_array = to_arrays(compute_queue, node_to_index, node_to_dependencies, node_to_profile)

    return compute_array, dependencies_array, output_item_index, storage_to_len


def load_financial_structure(allocation_rule, static_path):
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
    options = np.empty(1, dtype=np_oasis_int)
    options[STORE_LOSS_SUM_OPTION] = nb_oasis_int(allocation_rule == 2)

    programme, policytc, profile, xref = load_static(static_path)
    financial_structure = prepare_financial_structure(allocation_rule, programme, policytc, profile, xref)
    compute_queue, dependencies, output_item_index, storage_to_len = financial_structure
    logger.info(f'compute_queue has {len(compute_queue)} elements')
    logger.info(f'storage_to_len : {storage_to_len}')
    return compute_queue, dependencies, output_item_index, storage_to_len, options, profile
