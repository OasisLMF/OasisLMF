"""
This file contains specific functionality to read and process items files.
"""
import logging
import os

import numpy as np
import numba as nb
from numba.typed import Dict, List
from numba.types import int32 as nb_int32
from numba.types import int8 as nb_int8

from oasislmf.pytools.common.data import nb_areaperil_int, nb_oasis_float, nb_oasis_int, items_dtype
from oasislmf.pytools.gul.utils import append_to_dict_value
from oasislmf.pytools.gulmc.aggregate import gen_empty_areaperil_vuln_idx_to_weights
from oasislmf.pytools.gulmc.common import ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE


logger = logging.getLogger(__name__)


def read_items(input_path, ignore_file_type=set(), dynamic_footprint=False, legacy=False):
    """Load the items from the items file.

    Args:
        input_path (str): the path pointing to the file
        ignore_file_type (Set[str]): file extension to ignore when loading.

    Returns:
        Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]]
          vulnerability dictionary, vulnerability IDs, areaperil to vulnerability index dictionary,
          areaperil ID to vulnerability index array, areaperil ID to vulnerability array
    """
    input_files = set(os.listdir(input_path))

    if "items.bin" in input_files and "bin" not in ignore_file_type:
        items_fname = os.path.join(input_path, 'items.bin')
        logger.debug(f"loading {items_fname}")
        items = np.memmap(items_fname, dtype=items_dtype, mode='r')

    elif "items.csv" in input_files and "csv" not in ignore_file_type:
        items_fname = os.path.join(input_path, 'items.csv')
        logger.debug(f"loading {items_fname}")
        items = np.loadtxt(items_fname, dtype=items_dtype, delimiter=",", skiprows=1, ndmin=1)

    else:
        raise FileNotFoundError(f'items file not found at {input_path}')

    return items


@nb.njit(cache=True, fastmath=True)
def generate_item_map(items, coverages, valid_areaperil_id, agg_vuln_to_vulns):
    """Generate item_map; requires items to be sorted.

    Args:
        items (numpy.ndarray[int32, int32, int32]): 1-d structured array storing
            `item_id`, `coverage_id`, `group_id` for all items.
            items need to be sorted by increasing areaperil_id, vulnerability_id
            in order to output the items in correct order.
        coverages (numpy.ndarray): coverage id to information on items
        valid_areaperil_id (numpy.ndarray[int32]): list of non-filtered area_peril_id (None is no filter)
        agg_vuln_to_vulns (dict[int, list[int]]): map of aggregate vulnerability id to list of vulnerability ids.

    Returns:
        item_map (Dict[ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE]): dict storing
            the mapping between areaperil_id, vulnerability_id to item.
        areaperil_ids_map (Dict[int, Dict[int, int]]) dict storing the mapping between each
            areaperil_id and all the vulnerability ids associated with it.
        vuln id to vuln idx for each vulnerability in each areaperil, list of all used vulnerability ids.
        agg_vuln_to_vuln_idxs dict[int, list[int]]: map between aggregate vulnerability id and the list of indices where the individual vulnerability_ids
          that compose it are stored in `vuln_array`.
        areaperil_vuln_idx_to_weight dict[AGG_VULN_WEIGHTS_KEY_TYPE, AGG_VULN_WEIGHTS_VAL_TYPE]: map between the areaperil id and the index where the vulnerability function
          is stored in `vuln_array` and the vulnerability weight.

    """
    item_map = Dict.empty(ITEM_MAP_KEY_TYPE, List.empty_list(ITEM_MAP_VALUE_TYPE))
    areaperil_ids_map = Dict.empty(nb_areaperil_int, Dict.empty(nb_int32, nb_int8))
    vuln_dict = Dict()
    vuln_idx = 0
    areaperil_vuln_idx_to_weight = gen_empty_areaperil_vuln_idx_to_weights()

    agg_vuln_to_vuln_idxs = Dict.empty(nb_int32, List.empty_list(nb_int32))

    for j, item in enumerate(items):
        areaperil_id = item['areaperil_id']
        vulnerability_id = item['vulnerability_id']

        if valid_areaperil_id is not None and item['areaperil_id'] not in valid_areaperil_id:
            continue

        append_to_dict_value(item_map, tuple((areaperil_id, vulnerability_id)), j, ITEM_MAP_VALUE_TYPE)
        coverages[item['coverage_id']]['max_items'] += 1

        # Populate vuln_dict, to map all used vuln_id to vuln_i
        if item['vulnerability_id'] not in vuln_dict:
            if item['vulnerability_id'] in agg_vuln_to_vulns:  # vulnerability is an aggregate
                for sub_vuln_id in agg_vuln_to_vulns[item['vulnerability_id']]:
                    if sub_vuln_id not in vuln_dict:
                        vuln_dict[sub_vuln_id] = nb_oasis_int(vuln_idx)
                        vuln_idx += 1
                agg_vuln_to_vuln_idxs[item['vulnerability_id']] = List([vuln_dict[vuln] for vuln in agg_vuln_to_vulns[item['vulnerability_id']]])

            else:  # single vulnerability
                vuln_dict[item['vulnerability_id']] = nb_oasis_int(vuln_idx)
                vuln_idx += 1

        if item['vulnerability_id'] in agg_vuln_to_vulns:
            for sub_vuln_id in agg_vuln_to_vulns[item['vulnerability_id']]:
                areaperil_vuln_idx_to_weight[(nb_areaperil_int(areaperil_id), vuln_dict[sub_vuln_id])] = nb_oasis_float(0)
        else:
            item['vulnerability_idx'] = vuln_dict[item['vulnerability_id']]

        if areaperil_id not in areaperil_ids_map:
            areaperil_ids_map[areaperil_id] = Dict.empty(nb_int32, nb_int8)
        areaperil_ids_map[areaperil_id][vulnerability_id] = 0

    return item_map, areaperil_ids_map, vuln_dict, agg_vuln_to_vuln_idxs, areaperil_vuln_idx_to_weight
