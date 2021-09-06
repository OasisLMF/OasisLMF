__all__ = [
    'extract_financial_structure',
    'load_financial_structure',
    'create_financial_structure',
    'load_static',
]

from .common import nb_oasis_int, np_oasis_int, np_oasis_float, almost_equal, need_tiv_policy, need_extras, null_index
from .common import fm_programme_dtype, fm_policytc_dtype, fm_profile_dtype, fm_profile_step_dtype,\
    fm_profile_csv_col_map, fm_xref_dtype, fm_xref_csv_col_map, items_dtype, allowed_allocation_rule

from numba import njit, types, from_dtype
from numba.typed import List, Dict
import numpy as np
import pandas as pd
import numpy.lib.recfunctions as rfn
import os
import logging
logger = logging.getLogger(__name__)

# temp dictionary types
node_type = types.UniTuple(nb_oasis_int, 2)
output_type = types.UniTuple(nb_oasis_int, 2)
layer_type = types.UniTuple(nb_oasis_int, 3)

# finacial structure processed array
nodes_array_dtype = from_dtype(np.dtype([('node_id', np.uint64),
                                         ('level_id', np_oasis_int),
                                         ('agg_id', np_oasis_int),
                                         ('layer_len', np_oasis_int),
                                         ('profile_len', np_oasis_int),
                                         ('profiles', np_oasis_int),
                                         ('loss', np_oasis_int),
                                         ('il', np_oasis_int),
                                         ('extra', np_oasis_int),
                                         ('ba', np_oasis_int),
                                         ('realloc', np_oasis_int),
                                         ('is_reallocating', np.uint8),
                                         ('under_limit_sum', np_oasis_int),
                                         ('parent_len', np_oasis_int),
                                         ('parent', np_oasis_int),
                                         ('children', np_oasis_int),
                                         ('output_ids', np_oasis_int),
                                        ]))

compute_info_dtype = from_dtype(np.dtype([('allocation_rule', np_oasis_int),
                                          ('max_level', np_oasis_int),
                                          ('node_len', np_oasis_int),
                                          ('children_len', np_oasis_int),
                                          ('parents_len', np_oasis_int),
                                          ('profile_len', np_oasis_int),
                                          ('loss_len', np_oasis_int),
                                          ('extra_len', np_oasis_int),
                                          ('compute_len', np_oasis_int),
                                          ('start_level', np_oasis_int),
                                          ('items_len', np_oasis_int),
                                          ('output_len', np_oasis_int),
                                          ('stepped', np.bool),
                                         ]))
profile_index_dtype = from_dtype(np.dtype([('i_start', np_oasis_int),
                                           ('i_end', np_oasis_int),
                                          ]))

def load_static(static_path):
    """
    Load the raw financial data from static_path as numpy ndarray
    first check if .bin file is present then try .cvs
    try loading profile_step before falling back to normal profile,

    # TODO: handle profile_step
    # profile_step = load_file('fm_profile_step', fm_profile_step_dtype, must_exist=False)
    # profile = profile_step if profile_step else load_file('fm_profile', fm_profile_dtype)

    :param static_path: str
            static_path
    :return:
        programme : link between nodes
        policytc : info on layer
        profile : policy profile can be profile_step or profile
        xref : node to output_id
    :raise:
        FileNotFoundError if one of the static is missing
    """
    def load_file(name, _dtype, must_exist=True, col_map=None):
        if os.path.isfile(os.path.join(static_path, name + '.bin')):
            return np.fromfile(os.path.join(static_path, name + '.bin'), dtype=_dtype)
        elif must_exist or os.path.isfile(os.path.join(static_path, name + '.csv')):
            # in csv column cam be out of order and have different name,
            # we load with pandas and write each column to the ndarray
            if col_map is None:
                col_map = {}
            with open(os.path.join(static_path, name + '.csv')) as file_in:
                cvs_dtype = {col_map.get(key, key): col_dtype for key, (col_dtype, _) in _dtype.fields.items()}
                df =pd.read_csv(file_in, delimiter=',', dtype=cvs_dtype)
                res = np.empty(df.shape[0], dtype=_dtype)
                for name in _dtype.names:
                    res[name] = df[col_map.get(name, name)]
                return res

    programme = load_file('fm_programme', fm_programme_dtype)
    policytc = load_file('fm_policytc', fm_policytc_dtype)
    profile = load_file('fm_profile_step', fm_profile_step_dtype, False, col_map=fm_profile_csv_col_map)
    if profile is None:
        profile = load_file('fm_profile', fm_profile_dtype, col_map=fm_profile_csv_col_map)
        stepped = None
    else:
        stepped = True
    xref = load_file('fm_xref', fm_xref_dtype, col_map=fm_xref_csv_col_map)

    try:  # try to load items and coverage if present for TIV base policies (not used in re-insurance)
        items = load_file('items', items_dtype)[['item_id', 'coverage_id']]

        # coverage has a different structure whether it comes for the csv or the bin file
        fp = os.path.join(static_path, 'coverages.bin')
        if os.path.isfile(fp):
            coverages = np.fromfile(fp, dtype=np_oasis_float)
        else:
            fp = os.path.join(static_path, 'coverages.csv')
            with open(fp) as file_in:
                coverages = np.loadtxt(file_in, dtype=np_oasis_float, delimiter=',', skiprows=1, usecols=1)
    except FileNotFoundError:
        items = np.empty(0, dtype=items_dtype)
        coverages = np.empty(0, dtype=np_oasis_float)

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
            (profile['calcrule_id'] == 12 and almost_equal(profile['deductible_1'], 0)) or
            (profile['calcrule_id'] == 15 and almost_equal(profile['limit_1'], 1)) or
            (profile['calcrule_id'] == 16 and almost_equal(profile['deductible_1'], 0)) or
            (profile['calcrule_id'] == 34 and almost_equal(profile['deductible_1'], 0)
                and almost_equal(profile['attachment_1'], 0)
                and almost_equal(profile['share_1'], 1))
            )


@njit(cache=True)
def get_all_children(node_to_dependencies, node, items_only):
    children = List()
    temp = List()
    temp.append(node)
    # children = []
    # temp = [node]

    while temp:
        parent = temp.pop()
        if parent in node_to_dependencies:
            if not items_only:
                children.append(parent)
            temp.extend(node_to_dependencies[parent])
        else:
            children.append(parent)

    return children


@njit(cache=True)
def is_multi_peril(fm_programme):
    for i in range(fm_programme.shape[0]):
        if fm_programme[i]['level_id'] == 1 and fm_programme[i]['from_agg_id'] != fm_programme[i]['to_agg_id']:
            return True
    else:
        return False


@njit(cache=True)
def get_tiv(children, items, coverages):
    used_cov = np.zeros_like(coverages, dtype=np.uint8)
    tiv = 0
    for child_programme in children:
        coverage_i = items[child_programme[1]-1]['coverage_id']-1
        if not used_cov[coverage_i]:
            used_cov[coverage_i] = 1
            tiv += coverages[coverage_i]
    return tiv


@njit(cache=True)
def prepare_profile_simple(profile, tiv):
    # if use TIV convert calcrule to fix deductible
    if profile['calcrule_id'] == 4:
        profile['calcrule_id'] = 1
        profile['deductible_1'] *= tiv

    elif profile['calcrule_id'] == 6:
        profile['calcrule_id'] = 12
        profile['deductible_1'] *= tiv

    elif profile['calcrule_id'] == 18:
        profile['calcrule_id'] = 2
        profile['deductible_1'] *= tiv

    elif profile['calcrule_id'] == 21:
        profile['calcrule_id'] = 13
        profile['deductible_1'] *= tiv

    elif profile['calcrule_id'] == 9:
        profile['calcrule_id'] = 1
        profile['deductible_1'] *= profile['limit_1']
    elif profile['calcrule_id'] == 15:
        if profile['limit_1'] >= 1:
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
        loss = min(max(profile['payout_start'] * tiv - profile['deductible_1'], 0), profile['limit_1'])
        cond_loss = min(loss * profile['scale_2'], profile['limit_2'])
        profile['payout_start'] = (loss + cond_loss) * (1 + profile['scale_1'])

    elif profile['calcrule_id'] == 28:
        profile['trigger_start'] *= tiv
        if profile['trigger_end'] == 1:
            profile['trigger_end'] = np.inf
        else:
            profile['trigger_end'] *= tiv
        profile['scale_1'] += 1
        # special case to calculate only the conditional coverage loss (extra expenses) based on full input loss
        if profile['payout_start'] == 0:
            profile['calcrule_id'] = 281

    elif profile['calcrule_id'] == 29:
        profile['calcrule_id'] = 27
        profile['trigger_start'] *= tiv
        if profile['trigger_end'] == 1:
            profile['trigger_end'] = np.inf
        else:
            profile['trigger_end'] *= tiv
        loss = max(profile['payout_start'] * tiv - profile['deductible_1'], 0)
        cond_loss = min(loss * profile['scale_2'], profile['limit_2'])
        profile['payout_start'] = (loss + cond_loss) * (1 + profile['scale_1'])

    elif profile['calcrule_id'] == 30:
        profile['calcrule_id'] = 27
        profile['trigger_start'] *= tiv
        if profile['trigger_end'] == 1:
            profile['trigger_end'] = np.inf
        else:
            profile['trigger_end'] *= tiv
        loss = max(profile['payout_start'] * profile['limit_1'] - profile['deductible_1'], 0)
        cond_loss = min(loss * profile['scale_2'], profile['limit_2'])
        profile['payout_start'] = (loss + cond_loss) * (1 + profile['scale_1'])

    elif profile['calcrule_id'] == 31:
        profile['calcrule_id'] = 27
        profile['trigger_start'] *= tiv
        if profile['trigger_end'] == 1:
            profile['trigger_end'] = np.inf
        else:
            profile['trigger_end'] *= tiv
        loss = max(profile['payout_start'] - profile['deductible_1'], 0)
        cond_loss = min(loss * profile['scale_2'], profile['limit_2'])
        profile['payout_start'] = (loss + cond_loss) * (1 + profile['scale_1'])

    elif profile['calcrule_id'] == 32:
        profile['scale_1'] += 1
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
    # policytc_id_to_profile_index
    max_policytc_id = np.max(fm_profile['policytc_id'])
    policytc_id_to_profile_index = np.empty(max_policytc_id + 1, dtype=profile_index_dtype)
    has_tiv_policy = Dict.empty(np_oasis_int, np_oasis_int)
    last_policytc_id = 0  # real policytc_id start at 1
    for i in range(fm_profile.shape[0]):
        if fm_profile[i]['calcrule_id'] in need_tiv_policy:
            has_tiv_policy[fm_profile[i]['policytc_id']] = np_oasis_int(0)
        policytc_id_to_profile_index[fm_profile[i]['policytc_id']]['i_end'] = i + 1
        if last_policytc_id != fm_profile[i]['policytc_id']:
            policytc_id_to_profile_index[fm_profile[i]['policytc_id']]['i_start'] = i
            last_policytc_id = fm_profile[i]['policytc_id']

    # in fm_programme check if multi-peril and get size of each levels
    max_level = np.max(fm_programme['level_id'])
    level_node_len = np.zeros(max_level + 1, dtype=np_oasis_int)
    multi_peril = False
    for i in range(fm_programme.shape[0]):
        programme = fm_programme[i]
        if programme['level_id'] == 1 and programme['from_agg_id'] != programme['to_agg_id']:
            multi_peril = True
        if level_node_len[programme['level_id'] - 1] < programme['from_agg_id']:
            level_node_len[programme['level_id'] - 1] = programme['from_agg_id']

        if level_node_len[programme['level_id']] < programme['to_agg_id']:
            level_node_len[programme['level_id']] = programme['to_agg_id']

    # fm_policytc
    programme_node_to_layers = Dict.empty(node_type, List.empty_list(layer_type))
    i_new_fm_profile = fm_profile.shape[0]
    new_fm_profile_list = List.empty_list(np.int64)
    # programme_node_to_layers = {}
    for i in range(fm_policytc.shape[0]):
        policytc = fm_policytc[i]
        programme_node = (np_oasis_int(policytc['level_id']), np_oasis_int(policytc['agg_id']))
        i_start = policytc_id_to_profile_index[np_oasis_int(policytc['policytc_id'])]['i_start']
        i_end = policytc_id_to_profile_index[np_oasis_int(policytc['policytc_id'])]['i_end']

        if policytc['policytc_id'] in has_tiv_policy:
            if has_tiv_policy[policytc['policytc_id']]:
                for j in range(i_start, i_end):
                    new_fm_profile_list.append(j)
                i_start, i_end = i_new_fm_profile, i_new_fm_profile + i_end - i_start
                i_new_fm_profile = i_end
            else:
                has_tiv_policy[policytc['policytc_id']] = np_oasis_int(1)

        layer = (np_oasis_int(policytc['layer_id']), np_oasis_int(i_start), np_oasis_int(i_end))

        if programme_node not in programme_node_to_layers:
            _list = List.empty_list(layer_type)
            _list.append(layer)
            programme_node_to_layers[programme_node] = _list
            # programme_node_to_layers[programme_node] = [layer]
        else:
            programme_node_to_layers[programme_node].append(layer)

    if i_new_fm_profile - fm_profile.shape[0]:
        new_fm_profile = np.empty(i_new_fm_profile, dtype=fm_profile.dtype)
        new_fm_profile[:fm_profile.shape[0]] = fm_profile[:]
        for i in range(i_new_fm_profile - fm_profile.shape[0]):
            new_fm_profile[fm_profile.shape[0] + i] = fm_profile[new_fm_profile_list[i]]
        fm_profile = new_fm_profile

    #fm_xref
    if multi_peril:  # if single peril we can skip item level computation (level 0)
        start_level = np_oasis_int(0)
    else:
        start_level = np_oasis_int(1)

    if start_level == max_level:  # there is only one level we can switch the computation as if it were a0
        allocation_rule = 0

    if allocation_rule == 0:
        out_level = np_oasis_int(max_level)
    else:
        out_level = start_level

    node_to_output_id = Dict.empty(node_type, List.empty_list(output_type))
    # node_to_output_id = {}

    output_len = 0
    for i in range(fm_xref.shape[0]):
        xref = fm_xref[i]
        programme_node = (out_level, xref['agg_id'])
        if output_len < xref['output_id']:
            output_len = np_oasis_int(xref['output_id'])

        if programme_node in node_to_output_id:
            node_to_output_id[programme_node].append((np_oasis_int(xref['layer_id']), np_oasis_int(xref['output_id'])))
        else:
            _list = List.empty_list(output_type)
            _list.append((np_oasis_int(xref['layer_id']), np_oasis_int(xref['output_id'])))
            node_to_output_id[programme_node] = _list
            # node_to_output_id[programme_node] = [(np_oasis_int(xref['layer_id']), nb_oasis_int(xref['output_id']))]

    # programme
    parent_to_children = Dict.empty(node_type, List.empty_list(node_type))
    child_to_parents = Dict.empty(node_type, List.empty_list(node_type))
    node_layers = Dict.empty(node_type, np_oasis_int)
    # parent_to_children = {}
    # child_to_parents = {}
    # node_layers = {}
    children_len = 1
    parents_len = 0

    for programme in fm_programme:
        parent = (np_oasis_int(programme['level_id']), np_oasis_int(programme['to_agg_id']))
        if parent not in node_layers:
            node_layers[parent] = np_oasis_int(len(programme_node_to_layers[parent]))

    for level in range(max_level, start_level, -1):
        for programme in fm_programme:
            if programme['level_id'] == level:
                parent = (np_oasis_int(programme['level_id']), np_oasis_int(programme['to_agg_id']))
                child_programme = (np_oasis_int(programme['level_id'] - 1), np_oasis_int(programme['from_agg_id']))

                if parent not in parent_to_children:
                    children_len += 2
                    _list = List.empty_list(node_type)
                    _list.append(child_programme)
                    parent_to_children[parent] = _list
                    # parent_to_children[parent] = [child_programme]
                else:
                    children_len += 1
                    parent_to_children[parent].append(child_programme)

                parents_len += 1
                _list = List.empty_list(node_type)
                _list.append(parent)
                child_to_parents[child_programme] = _list
                # child_to_parents[child_programme] = [parent]
                node_layers[child_programme] = node_layers[parent]

    if allocation_rule == 0:
        node_level_start = np.zeros(level_node_len.shape[0] + 1, np_oasis_int)
        for i in range(start_level, level_node_len.shape[0]):
            node_level_start[i+1]= node_level_start[i] + level_node_len[i]
        steps = max_level + (1 - start_level)
        compute_len = node_level_start[-1] + steps + level_node_len[-1] + 1

    elif allocation_rule == 1:
        node_level_start = np.zeros(level_node_len.shape[0] + 2, np_oasis_int)
        for i in range(start_level, level_node_len.shape[0]):
            node_level_start[i+1]= node_level_start[i] + level_node_len[i]
        node_level_start[-2] = node_level_start[-3] + level_node_len[-1]
        node_level_start[-1] = node_level_start[-2] + level_node_len[start_level]

        steps = (max_level + (1 - start_level)) + 1

        # add level to do the ba
        for agg_id in range(1, level_node_len[max_level] + 1):
            parent = (np_oasis_int(max_level + 1), np_oasis_int(agg_id))
            top_node = (np_oasis_int(max_level), np_oasis_int(agg_id))
            children = get_all_children(parent_to_children, top_node, True)
            children_len += len(children) + 1
            parent_to_children[parent] = children

            parents_len += 1
            _list = List.empty_list(node_type)
            _list.append(parent)
            child_to_parents[top_node] = _list
            # child_to_parents[top_node] = [parent]
            node_layers[parent] = node_layers[top_node]
            for child_programme in children:
                parents_len += 1
                child_to_parents[child_programme].append(parent)

        compute_len = node_level_start[-1] + steps + level_node_len[-1] +1

    elif allocation_rule == 2 :
        node_level_start = np.zeros(level_node_len.shape[0] + 1, np_oasis_int)
        for i in range(start_level, max_level + 1):
            node_level_start[i+1]= node_level_start[i] + level_node_len[i]

        steps = 2 * (max_level + (1 - start_level)) - 1
        compute_len = 2 * node_level_start[-1] + steps + 1

    output_array_size = 0
    for node, layer_size in node_layers.items():
        if node[0] == out_level:
            output_array_size += layer_size

    nodes_array = np.empty(node_level_start[-1] + 1, dtype=nodes_array_dtype)
    node_parents_array = np.empty(parents_len, dtype=np_oasis_int)
    node_profiles_array = np.zeros(fm_policytc.shape[0] + 1, dtype=profile_index_dtype)
    output_array = np.zeros(output_array_size, dtype=np_oasis_int)

    node_i = 1
    children_i = 1
    parents_i = 0
    profile_i = 1
    loss_i = 0
    extra_i = 0
    output_i = 0

    for level in range(start_level, max_level+1):
        for agg_id in range(1, level_node_len[level] + 1):
            node_programme = (np_oasis_int(level), np_oasis_int(agg_id))
            node = nodes_array[node_i]
            node['node_id'] = node_i
            node_i += 1
            node['level_id'] = level
            node['agg_id'] = agg_id

            # layers
            node['layer_len'] = node_layers[node_programme]
            node['loss'], loss_i = loss_i, loss_i + node['layer_len']
            node['il'], loss_i = loss_i, loss_i + node['layer_len']
            node['ba'], loss_i = loss_i, loss_i + node['layer_len']

            node['extra'] = null_index
            node['realloc'] = null_index
            node['under_limit_sum'] = null_index
            node['is_reallocating'] = 0

            # children
            if node_programme in parent_to_children:
                children = parent_to_children[node_programme]
                node['children'], children_i = children_i, children_i + 1 + len(children)
            else:
                node['children'] = 0

            # parent
            if node_programme in child_to_parents:
                parents = child_to_parents[node_programme]
                node['parent_len'] = len(parents)
                node['parent'] = parents_i
                for parent in parents:
                    node_parents_array[parents_i], parents_i = node_level_start[parent[0]] + parent[1], np_oasis_int(parents_i + 1)
            else:
                node['parent_len'] = 0

            # profiles
            if node_programme in programme_node_to_layers:

                profiles = programme_node_to_layers[node_programme]
                node['profile_len'] = len(profiles)
                node['profiles'] = profile_i

                for layer_id, i_start, i_end in sorted(profiles):
                    node_profile, profile_i = node_profiles_array[profile_i], profile_i + 1
                    node_profile['i_start'] = i_start
                    node_profile['i_end'] = i_end

                    # if use TIV we compute it and precompute % TIV values
                    for profile_index in range(i_start, i_end):
                        if fm_profile[profile_index]['calcrule_id'] in need_tiv_policy:
                            all_children = get_all_children(parent_to_children, node_programme, True)
                            tiv = get_tiv(all_children, items, coverages)
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

                            all_children = get_all_children(parent_to_children, node_programme, False)
                            for child_programme in all_children: # include current node
                                child = nodes_array[node_level_start[child_programme[0]] + child_programme[1]]
                                if child['extra'] == null_index:
                                    child['extra'], extra_i = extra_i, extra_i + node['layer_len']
                                    if allocation_rule == 2 and fm_profile[profile_index]['calcrule_id'] != 27:
                                        child['under_limit_sum'], loss_i = loss_i, loss_i + node['layer_len']
                                        child['realloc'], loss_i = loss_i, loss_i + node['layer_len']

                            break

            else: # item level has no profile
                node['profile_len'] = 1
                node['profiles'] = 0

            if level == out_level:
                if node_programme in node_to_output_id:
                    node['output_ids'], output_i = output_i, output_i + node['layer_len']
                    for layer, output_id in node_to_output_id[node_programme]:
                        output_array[node['output_ids'] + layer - 1] = output_id
                else:
                    raise KeyError("Some output nodes are missing output_ids")

    if allocation_rule == 1:
        for agg_id in range(1, level_node_len[max_level] + 1):
            node_programme = (np_oasis_int(max_level + 1), np_oasis_int(agg_id))
            top_node = nodes_array[node_level_start[max_level] + agg_id]
            node, node_i = nodes_array[node_i], node_i + 1
            node['node_id'] = top_node['node_id']
            node['layer_len'] = top_node['layer_len']
            node['profile_len'] = 1
            node['loss'], loss_i = loss_i, loss_i + node['layer_len']
            node['il'] = node['loss']
            node['ba'] = top_node['ba']
            node['extra'] = top_node['extra']
            node['realloc'] = top_node['realloc']
            node['under_limit_sum'] = top_node['under_limit_sum']
            # children
            node['children'], children_i = children_i, children_i + 1 + len(parent_to_children[node_programme])

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
    np.save(os.path.join(static_path, f'fm_profile'), fm_profile)


def load_financial_structure(allocation_rule, static_path):
    compute_info = np.load(os.path.join(static_path, f'compute_info_{allocation_rule}.npy'), mmap_mode='r')
    nodes_array = np.load(os.path.join(static_path, f'nodes_array_{allocation_rule}.npy'), mmap_mode='r')
    node_parents_array = np.load(os.path.join(static_path, f'node_parents_array_{allocation_rule}.npy'), mmap_mode='r')
    node_profiles_array = np.load(os.path.join(static_path, f'node_profiles_array_{allocation_rule}.npy'), mmap_mode='r')
    output_array = np.load(os.path.join(static_path, f'output_array_{allocation_rule}.npy'), mmap_mode='r')
    fm_profile = np.load(os.path.join(static_path, f'fm_profile.npy'), mmap_mode='r')

    return compute_info, nodes_array, node_parents_array, node_profiles_array, output_array, fm_profile