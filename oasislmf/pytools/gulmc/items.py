"""
This file contains specific functionality to read and process items files.
"""
import logging
import os

import numpy as np
from numba import njit
from numba.typed import Dict, List
from numba.types import int32 as nb_int32
from numba.types import int64 as nb_int64

from oasislmf.pytools.common import nb_areaperil_int
from oasislmf.pytools.getmodel.common import Index_type
from oasislmf.pytools.gul.utils import append_to_dict_value
from oasislmf.pytools.gulmc.aggregate import gen_empty_agg_vuln_to_vuln_ids
from oasislmf.pytools.gulmc.common import (ITEM_MAP_KEY_TYPE,
                                           ITEM_MAP_VALUE_TYPE, Item)

logger = logging.getLogger(__name__)


def read_items(input_path, ignore_file_type=set(), legacy=False):
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
        items = np.memmap(items_fname, dtype=Item, mode='r')

    elif "items.csv" in input_files and "csv" not in ignore_file_type:
        items_fname = os.path.join(input_path, 'items.csv')
        logger.debug(f"loading {items_fname}")
        items = np.loadtxt(items_fname, dtype=Item, delimiter=",", skiprows=1, ndmin=1)

    else:
        raise FileNotFoundError(f'items file not found at {input_path}')

    return items


@njit(cache=True, fastmath=True)
def generate_item_map(items, coverages):
    """Generate item_map; requires items to be sorted.

    Args:
        items (numpy.ndarray[int32, int32, int32]): 1-d structured array storing
          `item_id`, `coverage_id`, `group_id` for all items.
          items need to be sorted by increasing areaperil_id, vulnerability_id
          in order to output the items in correct order.

    Returns:
        item_map (Dict[ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE]): dict storing
          the mapping between areaperil_id, vulnerability_id to item.
        areaperil_ids_map (Dict[int, Dict[int, int]]) dict storing the mapping between each
          areaperil_id and all the vulnerability ids associated with it.

    """
    item_map = Dict.empty(ITEM_MAP_KEY_TYPE, List.empty_list(ITEM_MAP_VALUE_TYPE))
    areaperil_ids_map = Dict.empty(nb_areaperil_int, Dict.empty(nb_int32, nb_int64))

    for j, item in enumerate(items):
        areaperil_id = item['areaperil_id']
        vulnerability_id = item['vulnerability_id']

        append_to_dict_value(item_map, tuple((areaperil_id, vulnerability_id)), j, ITEM_MAP_VALUE_TYPE)
        coverages[item['coverage_id']]['max_items'] += 1

        if areaperil_id not in areaperil_ids_map:
            areaperil_ids_map[areaperil_id] = {vulnerability_id: 0}
        else:
            areaperil_ids_map[areaperil_id][vulnerability_id] = 0

    return item_map, areaperil_ids_map


@njit(cache=True)
def process_items(items, valid_area_peril_id, agg_vuln_to_vulns=None):
    """
    Processes the Items loaded from the file extracting meta data around the vulnerability data.

    Args:
        items: (List[Item]) Data loaded from the vulnerability file
        valid_area_peril_id: array of area_peril_id to be included (if none, all are included).
        agg_vuln_to_vulns (dict[tuple[areaperil_int, int], int]): dict with aggregate vulnerability definitions.

    Returns: (Tuple[Dict[int, int],  Dict[int, int], np.array[int], np.array[int], dict[int, dict[int, int]], List[int])
             vulnerability dictionary, areaperil to vulnerability index dictionary, areaperil to vulnerability index array,
             vuln id to vuln idx for each vulnerability in each areaperil, list of all used vulnerability ids.
    """
    if not agg_vuln_to_vulns:
        agg_vuln_to_vulns = gen_empty_agg_vuln_to_vuln_ids()

    areaperil_to_vulns_size = 0
    areaperil_dict = Dict()
    vuln_dict = Dict()
    vuln_idx = 0
    used_agg_vuln_ids = List.empty_list(nb_int32)
    for i in range(items.shape[0]):
        item = items[i]

        # filter out invalid areaperil id
        if valid_area_peril_id is not None:
            if item['areaperil_id'] not in valid_area_peril_id:
                continue

        # import this vulnerability id if it has not been imported yet
        if item['vulnerability_id'] not in vuln_dict:
            if item['vulnerability_id'] in agg_vuln_to_vulns:
                # vulnerability is aggregate
                for vuln_i in agg_vuln_to_vulns[item['vulnerability_id']]:
                    if vuln_i not in vuln_dict:
                        # import this individual vulnerability_id only if it was not imported already
                        vuln_dict[vuln_i] = np.int32(vuln_idx)
                        vuln_idx += 1

                used_agg_vuln_ids.append(item['vulnerability_id'])

            else:
                # vulnerability is not aggregate
                vuln_dict[item['vulnerability_id']] = np.int32(vuln_idx)
                vuln_idx += 1

        # insert an area dictionary into areaperil_dict under the key of areaperil ID
        if item['areaperil_id'] not in areaperil_dict:
            area_vuln = Dict()

            if item['vulnerability_id'] in agg_vuln_to_vulns:
                for vuln_i in agg_vuln_to_vulns[item['vulnerability_id']]:
                    if vuln_i not in area_vuln:
                        area_vuln[vuln_i] = 0
                        areaperil_to_vulns_size += 1
            else:
                area_vuln[item['vulnerability_id']] = 0
                areaperil_to_vulns_size += 1

            areaperil_dict[item['areaperil_id']] = area_vuln
        else:
            if item['vulnerability_id'] in agg_vuln_to_vulns:
                for vuln_i in agg_vuln_to_vulns[item['vulnerability_id']]:
                    if vuln_i not in areaperil_dict[item['areaperil_id']]:
                        # import this individual vulnerability_id only if it was not imported already
                        areaperil_to_vulns_size += 1
                        areaperil_dict[item['areaperil_id']][vuln_i] = 0
            else:
                if item['vulnerability_id'] not in areaperil_dict[item['areaperil_id']]:
                    areaperil_to_vulns_size += 1
                    areaperil_dict[item['areaperil_id']][item['vulnerability_id']] = 0

    areaperil_to_vulns_idx_dict = Dict()
    areaperil_to_vulns_idx_array = np.empty(len(areaperil_dict), dtype=Index_type)
    vuln_id_to_vuln_idx_arr = np.empty(areaperil_to_vulns_size, dtype=np.int32)

    areaperil_i = 0
    vulnerability_i = 0

    for areaperil_id, vulns in areaperil_dict.items():
        areaperil_to_vulns_idx_dict[areaperil_id] = areaperil_i
        areaperil_to_vulns_idx_array[areaperil_i]['start'] = vulnerability_i

        for vuln_id in sorted(vulns):  # sorted is not necessary but doesn't impede the perf and align with cpp getmodel
            vuln_id_to_vuln_idx_arr[vulnerability_i] = vuln_dict[vuln_id]
            vulnerability_i += 1
        areaperil_to_vulns_idx_array[areaperil_i]['end'] = vulnerability_i
        areaperil_i += 1

    return vuln_dict, areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, vuln_id_to_vuln_idx_arr, areaperil_dict, used_agg_vuln_ids
