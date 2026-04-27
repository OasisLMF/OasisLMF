"""
This file is the entry point for the python get model command for the package

TODO: use selector and select for output

"""
import atexit
import json
import logging
import os
import sys
from contextlib import ExitStack

import numba as nb
import numpy as np
import pandas as pd


from oasislmf.pytools.common.hashmap import (
    init_dict as hm_init_dict, unpack as hm_unpack, rehash as hm_rehash,
    _try_add_key as hm_try_add_key, _find_key as hm_find_key,
    i_add_key_fail as hm_i_add_key_fail,
    NOT_FOUND as HM_NOT_FOUND, HM_INFO_N_VALID,
)

from oasis_data_manager.df_reader.config import get_df_reader, clean_config, InputReaderConfig
from oasis_data_manager.filestore.backends.base import BaseStorage
from oasis_data_manager.filestore.config import get_storage_from_config_path
from oasislmf.pytools.common.event_stream import PIPE_CAPACITY
from oasislmf.utils.data import validate_vulnerability_replacements, analysis_settings_loader
from oasislmf.pytools.common.data import (
    areaperil_int, areaperil_int_size, nb_areaperil_int,
    oasis_float, oasis_float_size,
    oasis_int, oasis_int_size,
    damagebin_dtype, items_dtype,
    vulnerability_dtype
)
from oasislmf.pytools.common.id_index import (
    build as id_index_build,
    get_idx as id_index_get_idx,
    NOT_FOUND as ID_INDEX_NOT_FOUND,
)
from oasislmf.pytools.data_layer.footprint_layer import FootprintLayerClient
from oasislmf.pytools.getmodel.common import Index_type
from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.utils import redirect_logging
from oasislmf.utils.ping import oasis_ping

from .vulnerability import vulnerability_dataset, parquetvulnerability_meta_filename
from ..common.data import null_index

logger = logging.getLogger(__name__)

buff_size = PIPE_CAPACITY

buff_int_size = buff_size // oasis_int_size

areaperil_int_relative_size = areaperil_int_size // oasis_int_size
oasis_float_relative_size = oasis_float_size // oasis_int_size
results_relative_size = 2 * oasis_float_relative_size

VulnerabilityIndex_dtype = np.dtype([
    ('vulnerability_id', np.int32),
    ('offset', np.int64),
    ('size', np.int64),
    ('original_size', np.int64)
])
VulnerabilityIndex = nb.from_dtype(VulnerabilityIndex_dtype)

VulnerabilityRow_dtype = np.dtype([
    ('intensity_bin_id', np.int32),
    ('damage_bin_id', np.int32),
    ('probability', oasis_float)
])
VulnerabilityRow = nb.from_dtype(VulnerabilityRow_dtype)

vuln_offset = 4


@nb.njit(cache=True)
def load_areaperil_id_u4(int32_mv, cursor, areaperil_id):
    int32_mv[cursor] = areaperil_id.view('i4')
    return cursor + 1


@nb.njit(cache=True)
def load_areaperil_id_u8(int32_mv, cursor, areaperil_id):
    int32_mv[cursor: cursor + 1] = areaperil_id.view('i4')
    return cursor + 2


if areaperil_int == 'u4':
    load_areaperil_id = load_areaperil_id_u4
elif areaperil_int == 'u8':
    load_areaperil_id = load_areaperil_id_u8
else:
    raise Exception(f"AREAPERIL_TYPE {areaperil_int} is not implemented choose u4 or u8")


@nb.njit(cache=True)
def load_items(items):
    """
    Processes pre-sorted, pre-filtered items extracting vulnerability metadata.

    Items must be sorted by (areaperil_id, vulnerability_id) before calling.

    Args:
        items: (np.ndarray[items_dtype]) sorted items array

    Returns: (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray])
             vuln_map (packed hashmap), vuln_map_keys, unique_areaperil_ids (sorted),
             areaperil_to_vulns_idx_array, areaperil_to_vulns
    """
    # Collect unique vuln_ids via hashmap
    max_vulns = max(items.shape[0], 1)
    vuln_key_table = np.empty(max_vulns, dtype=np.int32)
    vuln_table = hm_init_dict(max_vulns)
    hm_info, hm_lookup, hm_index = hm_unpack(vuln_table)

    # Pass 1: count unique areaperil_ids and unique (areaperil_id, vuln_id) pairs
    n_areaperils = 0
    n_pairs = 0
    prev_ap = nb_areaperil_int(0)
    prev_vuln = np.int32(-1)

    for i in range(items.shape[0]):
        ap = items[i]['areaperil_id']
        vuln = items[i]['vulnerability_id']

        if ap != prev_ap:
            n_areaperils += 1
            prev_ap = ap
            prev_vuln = np.int32(-1)

        if vuln != prev_vuln:
            n_pairs += 1
            prev_vuln = vuln

        # Insert vuln_id into hashmap (table pre-sized to fit all items)
        result = hm_try_add_key(hm_info, hm_lookup, hm_index, vuln_key_table, vuln)
        while result == hm_i_add_key_fail:
            vuln_table = hm_rehash(vuln_table, vuln_key_table)
            hm_info, hm_lookup, hm_index = hm_unpack(vuln_table)
            result = hm_try_add_key(hm_info, hm_lookup, hm_index, vuln_key_table, vuln)

    vuln_map_keys = vuln_key_table[:hm_info[HM_INFO_N_VALID]]

    # Pass 2: build output arrays
    unique_areaperil_ids = np.empty(n_areaperils, dtype=areaperil_int)
    areaperil_to_vulns_idx_array = np.empty(n_areaperils, dtype=Index_type)
    areaperil_to_vulns = np.empty(n_pairs, dtype=np.int32)

    ap_i = 0
    vuln_i = 0
    prev_ap = nb_areaperil_int(0)
    prev_vuln = np.int32(-1)

    for i in range(items.shape[0]):
        ap = items[i]['areaperil_id']
        vuln = items[i]['vulnerability_id']

        if ap != prev_ap:
            if prev_ap != nb_areaperil_int(0):
                areaperil_to_vulns_idx_array[ap_i - 1]['end'] = vuln_i
            unique_areaperil_ids[ap_i] = ap
            areaperil_to_vulns_idx_array[ap_i]['start'] = vuln_i
            ap_i += 1
            prev_ap = ap
            prev_vuln = np.int32(-1)

        if vuln != prev_vuln:
            areaperil_to_vulns[vuln_i] = vuln
            vuln_i += 1
            prev_vuln = vuln

    if ap_i > 0:
        areaperil_to_vulns_idx_array[ap_i - 1]['end'] = vuln_i

    return vuln_table, vuln_map_keys, unique_areaperil_ids, areaperil_to_vulns_idx_array, areaperil_to_vulns


def get_items(input_path, ignore_file_type=set(), valid_area_peril_id=None):
    """
    Loads the items from the items file.

    Args:
        input_path: (str) the path pointing to the file
        ignore_file_type: set(str) file extension to ignore when loading
        valid_area_peril_id: array of area_peril_id to include (if None, all are included)

    Returns: (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray])
             vuln_map (packed hashmap), vuln_map_keys, areaperil_id_ind (id_index),
             areaperil_to_vulns_idx_array, areaperil_to_vulns, unique_areaperil_ids
    """
    input_files = set(os.listdir(input_path))
    if "items.bin" in input_files and "bin" not in ignore_file_type:
        logger.debug(f"loading {os.path.join(input_path, 'items.bin')}")
        items = np.fromfile(os.path.join(input_path, "items.bin"), dtype=items_dtype)
    elif "items.csv" in input_files and "csv" not in ignore_file_type:
        logger.debug(f"loading {os.path.join(input_path, 'items.csv')}")
        items = np.loadtxt(os.path.join(input_path, "items.csv"), dtype=items_dtype, delimiter=",", skiprows=1, ndmin=1)
    else:
        raise FileNotFoundError(f'items file not found at {input_path}')

    if valid_area_peril_id is not None:
        items = items[np.isin(items['areaperil_id'], valid_area_peril_id)]

    items = np.sort(items, order=['areaperil_id', 'vulnerability_id'])

    vuln_table, vuln_map_keys, unique_areaperil_ids, areaperil_to_vulns_idx_array, areaperil_to_vulns = load_items(items)

    areaperil_id_ind = id_index_build(unique_areaperil_ids)

    return vuln_table, vuln_map_keys, areaperil_id_ind, areaperil_to_vulns_idx_array, areaperil_to_vulns, unique_areaperil_ids


def get_intensity_bin_dict(input_path):
    """
    Loads the intensity bin dictionary file and creates arrays to map intensities to bins.
    Used in the dynamic footprint generation as intensities can be adjusted for defences at runtime.

    Args:
        input_path: (str) the path pointing to the file

    Returns: (np.array[int32], np.array[int32, 2d])
             intensity_bin_peril_ids: 1-d array of unique encoded peril_ids (length n_perils).
             intensity_bins: 2-d array of shape (n_perils, max_intensity + 1) mapping
                 [peril_idx, intensity_value] -> intensity_bin_id.  Slots not present in the
                 CSV are pre-filled with the fallback bin for intensity=0 of that peril.
    """
    input_files = set(os.listdir(input_path))
    if "intensity_bin_dict.csv" not in input_files:
        raise FileNotFoundError(f'intensity_bin_dict file not found at {input_path}')

    logger.debug(f"loading {os.path.join(input_path, 'intensity_bin_dict.csv')}")
    data = pd.read_csv(os.path.join(input_path, "intensity_bin_dict.csv"))
    data = data[['peril_id', 'intensity', 'intensity_bin']]
    data['peril_id'] = data['peril_id'].apply(encode_peril_id)
    data = data.to_records(index=False).tolist()
    data = np.array(data, dtype=np.int32)

    # unique peril_ids (typically 1-3)
    unique_perils = np.unique(data[:, 0])
    n_perils = len(unique_perils)
    max_intensity = int(data[:, 1].max())

    # build peril_id -> peril_idx mapping (small linear scan is fine)
    intensity_bin_peril_ids = unique_perils.astype(np.int32)

    # allocate and pre-fill with fallback (bin for intensity=0 per peril)
    intensity_bins = np.zeros((n_perils, max_intensity + 1), dtype=np.int32)
    for d in data:
        peril_id, intensity_val, bin_id = d[0], d[1], d[2]
        peril_idx = np.searchsorted(intensity_bin_peril_ids, peril_id)
        intensity_bins[peril_idx, intensity_val] = bin_id

    # pre-fill missing slots with the fallback value (bin for intensity=0)
    for pi in range(n_perils):
        fallback = intensity_bins[pi, 0]
        for i in range(max_intensity + 1):
            if intensity_bins[pi, i] == 0 and i != 0:
                intensity_bins[pi, i] = fallback

    return intensity_bin_peril_ids, intensity_bins


def encode_peril_id(peril_id):
    """Encode a string to an integer.

    Args:
        peril_id (str): 3-digit Oasis peril code (also works with numeric codes).

    Returns:
        int: The encoded peril_id.
    """

    return sum(ord(c) << (8 * i) for i, c in enumerate(str(peril_id).upper()))


def get_intensity_adjustment(input_path):
    pass


@nb.njit(cache=True)
def load_vuln_probability(vuln_array, vuln, vuln_id):
    if vuln_array.shape[0] < vuln['damage_bin_id']:
        raise Exception("vulnerability_id " + str(vuln_id) + " has damage_bin_id bigger that expected maximum")
    if vuln['intensity_bin_id'] <= vuln_array.shape[1]:  # intensity in vulnerability curve but not in the footprint, we can ignore
        vuln_array[vuln['damage_bin_id'] - 1, vuln['intensity_bin_id'] - 1] = vuln['probability']


@nb.njit(cache=True)
def load_vulns_bin_idx(vulns_bin, vulns_idx_bin, vuln_map, vuln_map_keys,
                       num_damage_bins, num_intensity_bins, rowsize):
    """
    Loads the vulnerability binary index file.

    Args:
        vulns_bin: (List[VulnerabilityRow]) vulnerability data from the vulnerability file
        vulns_idx_bin: (List[VulnerabilityIndex]) vulnerability index data from the vulnerability idx file
        vuln_map: (np.ndarray[uint8]) packed hashmap table mapping vuln_id to dense index
        vuln_map_keys: (np.ndarray[int32]) array of unique vulnerability ids (hashmap keys)
        num_damage_bins: (int) number of damage bins in the data
        num_intensity_bins: (int) the number of intensity bins

    Returns: (List[List[List[floats]]]) vulnerability data grouped by intensity bin and damage bin
    """
    n_vulns = len(vuln_map_keys)
    vuln_array = np.zeros((n_vulns, num_damage_bins, num_intensity_bins), dtype=oasis_float)
    vuln_ids = np.full(n_vulns, null_index)
    hm_info, hm_lookup, hm_index = hm_unpack(vuln_map)
    for idx_i in range(vulns_idx_bin.shape[0]):
        vuln_idx = vulns_idx_bin[idx_i]
        slot = hm_find_key(hm_info, hm_lookup, hm_index, vuln_map_keys, vuln_idx['vulnerability_id'])
        if slot != HM_NOT_FOUND:
            dense_i = hm_index[slot]
            vuln_ids[dense_i] = vuln_idx['vulnerability_id']
            cur_vuln_array = vuln_array[dense_i]
            start = (vuln_idx['offset'] - vuln_offset) // rowsize
            end = start + vuln_idx['size'] // rowsize
            for vuln_i in range(start, end):
                vuln = vulns_bin[vuln_i]
                load_vuln_probability(cur_vuln_array, vuln, vuln_idx['vulnerability_id'])

    return vuln_array, vuln_ids


@nb.njit(cache=True)
def load_vulns_bin_idx_adjusted(vulns_bin, vulns_idx_bin, vuln_map, vuln_map_keys,
                                num_damage_bins, num_intensity_bins, rowsize, adj_vuln_data=None):
    """
    Loads the vulnerability binary index file, prioritizing the data in the adjustments file over the data in the
    vulnerability file.

    Args:
        vulns_bin: (List[VulnerabilityRow]) vulnerability data from the vulnerability file
        vulns_idx_bin: (List[VulnerabilityIndex]) vulnerability index data from the vulnerability idx file
        vuln_map: (np.ndarray[uint8]) packed hashmap table mapping vuln_id to dense index
        vuln_map_keys: (np.ndarray[int32]) array of unique vulnerability ids (hashmap keys)
        num_damage_bins: (int) number of damage bins in the data
        num_intensity_bins: (int) the number of intensity bins
        adj_vuln_data: (List[vulnerability_dtype]) vulnerability adjustment data, sorted by vuln_id

    Returns: (List[List[List[floats]]]) vulnerability data grouped by intensity bin and damage bin
    """
    n_vulns = len(vuln_map_keys)
    vuln_array = np.zeros((n_vulns, num_damage_bins, num_intensity_bins), dtype=oasis_float)
    vuln_ids = np.full(n_vulns, null_index)
    adj_vuln_index = 0
    hm_info, hm_lookup, hm_index = hm_unpack(vuln_map)

    for idx_i in range(vulns_idx_bin.shape[0]):
        vuln_idx = vulns_idx_bin[idx_i]
        vuln_id = vuln_idx['vulnerability_id']

        # Check if current vulnerability id is in the adjustment data
        while adj_vuln_data is not None and adj_vuln_index < len(adj_vuln_data) and adj_vuln_data[adj_vuln_index]['vulnerability_id'] < vuln_id:
            adj_vuln_index += 1

        slot = hm_find_key(hm_info, hm_lookup, hm_index, vuln_map_keys, vuln_id)
        if slot != HM_NOT_FOUND:
            dense_i = hm_index[slot]
            vuln_ids[dense_i] = vuln_id
            cur_vuln_array = vuln_array[dense_i]
            start = (vuln_idx['offset'] - vuln_offset) // rowsize
            end = start + vuln_idx['size'] // rowsize

            # Apply data from vulns_bin or adj_vuln_data
            for vuln_i in range(start, end):
                if (adj_vuln_data is not None and adj_vuln_index < len(adj_vuln_data) and adj_vuln_data[adj_vuln_index]['vulnerability_id'] == vuln_id):
                    load_vuln_probability(cur_vuln_array, adj_vuln_data[adj_vuln_index], vuln_id)
                    adj_vuln_index += 1
                else:
                    load_vuln_probability(cur_vuln_array, vulns_bin[vuln_i], vuln_id)

    # Add remaining adj_vuln_data
    while adj_vuln_data is not None and adj_vuln_index < len(adj_vuln_data):
        adj_vuln = adj_vuln_data[adj_vuln_index]
        vuln_id = adj_vuln['vulnerability_id']
        slot = hm_find_key(hm_info, hm_lookup, hm_index, vuln_map_keys, vuln_id)
        if slot != HM_NOT_FOUND:
            dense_i = hm_index[slot]
            load_vuln_probability(vuln_array[dense_i], adj_vuln, vuln_id)
        adj_vuln_index += 1

    return vuln_array, vuln_ids


@nb.njit(cache=True)
def load_vulns_bin(vulns_bin, vuln_map, vuln_map_keys, num_damage_bins, num_intensity_bins):
    """
    Loads the vulnerability data grouped by the intensity and damage bins.

    Args:
        vulns_bin: (List[Vulnerability]) vulnerability data from the vulnerability file
        vuln_map: (np.ndarray[uint8]) packed hashmap table mapping vuln_id to dense index
        vuln_map_keys: (np.ndarray[int32]) array of unique vulnerability ids (hashmap keys)
        num_damage_bins: (int) number of damage bins in the data
        num_intensity_bins: (int) the number of intensity bins

    Returns: (List[List[List[floats]]]) vulnerability data grouped by intensity bin and damage bin
    """
    n_vulns = len(vuln_map_keys)
    vuln_array = np.zeros((n_vulns, num_damage_bins, num_intensity_bins), dtype=oasis_float)
    vuln_ids = np.full(n_vulns, null_index)
    cur_vulnerability_id = -1
    hm_info, hm_lookup, hm_index = hm_unpack(vuln_map)

    for vuln_i in range(vulns_bin.shape[0]):
        vuln = vulns_bin[vuln_i]
        if vuln['vulnerability_id'] != cur_vulnerability_id:
            slot = hm_find_key(hm_info, hm_lookup, hm_index, vuln_map_keys, vuln['vulnerability_id'])
            if slot != HM_NOT_FOUND:
                dense_i = hm_index[slot]
                cur_vulnerability_id = vuln['vulnerability_id']
                vuln_ids[dense_i] = cur_vulnerability_id
                cur_vuln_array = vuln_array[dense_i]
            else:
                cur_vulnerability_id = -1
        if cur_vulnerability_id != -1:
            load_vuln_probability(cur_vuln_array, vuln, cur_vulnerability_id)

    return vuln_array, vuln_ids


@nb.njit(cache=True)
def load_vulns_bin_adjusted(vulns_bin, vuln_map, vuln_map_keys, num_damage_bins, num_intensity_bins, adj_vuln_data=None):
    """
    Loads the vulnerability data grouped by the intensity and damage bins, prioritizing the data
    in the adjustments file over the data in the vulnerability file.

    Args:
        vulns_bin: (List[Vulnerability]) vulnerability data from the vulnerability file
        vuln_map: (np.ndarray[uint8]) packed hashmap table mapping vuln_id to dense index
        vuln_map_keys: (np.ndarray[int32]) array of unique vulnerability ids (hashmap keys)
        num_damage_bins: (int) number of damage bins in the data
        num_intensity_bins: (int) the number of intensity bins
        adj_vuln_data: (List[vulnerability_dtype]) vulnerability adjustment data, sorted by vuln_id

    Returns: (List[List[List[floats]]]) vulnerability data grouped by intensity bin and damage bin
    """
    n_vulns = len(vuln_map_keys)
    vuln_array = np.zeros((n_vulns, num_damage_bins, num_intensity_bins), dtype=oasis_float)
    vuln_ids = np.full(n_vulns, null_index)
    ids_to_replace = set()
    adj_vuln_index = 0
    hm_info, hm_lookup, hm_index = hm_unpack(vuln_map)

    # Create list of ids to replace if adj_vuln_data is provided
    if adj_vuln_data is not None:
        for adj_vuln in adj_vuln_data:
            ids_to_replace.add(adj_vuln['vulnerability_id'])

    vuln_i = 0

    while vuln_i < len(vulns_bin):
        vuln = vulns_bin[vuln_i]
        vuln_id = vuln['vulnerability_id']
        slot = hm_find_key(hm_info, hm_lookup, hm_index, vuln_map_keys, vuln_id)
        if slot != HM_NOT_FOUND:
            dense_i = hm_index[slot]
            vuln_ids[dense_i] = vuln_id
            if vuln_id in ids_to_replace:
                # Advance to current vuln_id
                while adj_vuln_index < len(adj_vuln_data) and adj_vuln_data[adj_vuln_index]['vulnerability_id'] < vuln_id:
                    adj_vuln_index += 1
                # process current vuln_id
                while adj_vuln_index < len(adj_vuln_data) and adj_vuln_data[adj_vuln_index]['vulnerability_id'] == vuln_id:
                    load_vuln_probability(vuln_array[dense_i], adj_vuln_data[adj_vuln_index], vuln_id)
                    adj_vuln_index += 1
                # Skip remaining vulns_bin entries with the same vulnerability_id
                while vuln_i < len(vulns_bin) and vulns_bin[vuln_i]['vulnerability_id'] == vuln_id:
                    vuln_i += 1
                continue
            else:
                # Use data from vulns_bin
                load_vuln_probability(vuln_array[dense_i], vuln, vuln_id)
        vuln_i += 1
    return vuln_array, vuln_ids


@nb.njit(cache=True)
def update_vuln_array_with_adj_data(vuln_array, vuln_map, vuln_map_keys, adj_vuln_data):
    """
    Update the vulnerability array with adjustment data (used for parquet loading).

    Args:
        vuln_array: (3D array) The vulnerability data array.
        vuln_map: (np.ndarray[uint8]) packed hashmap table mapping vuln_id to dense index.
        vuln_map_keys: (np.ndarray[int32]) array of unique vulnerability ids (hashmap keys).
        adj_vuln_data: (List[vulnerability_dtype]) The vulnerability adjustment data.

    Returns: (3D array) The updated vulnerability data array.
    """
    hm_info, hm_lookup, hm_index = hm_unpack(vuln_map)
    for adj_vuln in adj_vuln_data:
        vuln_id = adj_vuln['vulnerability_id']
        slot = hm_find_key(hm_info, hm_lookup, hm_index, vuln_map_keys, vuln_id)
        if slot != HM_NOT_FOUND:
            dense_i = hm_index[slot]
            load_vuln_probability(vuln_array[dense_i], adj_vuln, vuln_id)
    return vuln_array


def get_vulns(
        storage: BaseStorage, run_dir, vuln_map, vuln_map_keys, num_intensity_bins,
        ignore_file_type=set(), df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader"):
    """
    Loads the vulnerabilities from the file.

    Args:
        storage: (str) the storage manager for fetching model data
        run_dir: (str) the path to the run folder (used to load the analysis settings)
        vuln_map: (np.ndarray[uint8]) packed hashmap table mapping vuln_id to dense index
        vuln_map_keys: (np.ndarray[int32]) array of unique vulnerability ids (hashmap keys)
        num_intensity_bins: (int) the number of intensity bins
        ignore_file_type: set(str) file extension to ignore when loading

    Returns: (Tuple[List[List[float]], int, np.array[int]) vulnerability data, vulnerabilities id, number of damage bins
    """
    n_vulns = len(vuln_map_keys)
    input_files = set(storage.listdir())
    vuln_ids_set = set(vuln_map_keys)
    vuln_adj = get_vulnerability_replacements(run_dir, vuln_ids_set)

    if vulnerability_dataset in input_files and "parquet" not in ignore_file_type:
        source_url = storage.get_storage_url(vulnerability_dataset, encode_params=False)[1]
        with storage.open(parquetvulnerability_meta_filename, 'r') as outfile:
            meta_data = json.load(outfile)
        logger.debug(f"loading {source_url}")

        df_reader_config = clean_config(InputReaderConfig(filepath=vulnerability_dataset, engine=df_engine))
        df_reader_config["engine"]["options"]["storage"] = storage
        reader = get_df_reader(df_reader_config, filters=[[('vulnerability_id', '==', vuln_id)] for vuln_id in vuln_map_keys])
        df = reader.as_pandas()
        num_damage_bins = meta_data['num_damage_bins']
        vuln_array_parquet = np.vstack(df['vuln_array'].to_numpy()).reshape(len(df['vuln_array']),
                                                                            num_damage_bins,
                                                                            num_intensity_bins)
        parquet_vuln_ids = df['vulnerability_id'].to_numpy()
        missing_vuln_ids = vuln_ids_set.difference(parquet_vuln_ids)
        if missing_vuln_ids:
            raise Exception(f"Vulnerability_ids {missing_vuln_ids} are missing"
                            f" from {source_url}")
        # Reindex from parquet order to dense index order
        vuln_array = np.zeros((n_vulns, num_damage_bins, num_intensity_bins), dtype=oasis_float)
        hm_info, hm_lookup, hm_index = hm_unpack(vuln_map)
        for i in range(len(parquet_vuln_ids)):
            slot = hm_find_key(hm_info, hm_lookup, hm_index, vuln_map_keys, np.int32(parquet_vuln_ids[i]))
            if slot != HM_NOT_FOUND:
                dense_i = hm_index[slot]
                vuln_array[dense_i] = vuln_array_parquet[i]
        # update vulnerability array with adjustment data if present
        if vuln_adj is not None and len(vuln_adj) > 0:
            vuln_array = update_vuln_array_with_adj_data(vuln_array, vuln_map, vuln_map_keys, vuln_adj)
        vuln_ids = vuln_map_keys.copy()

    else:
        if "vulnerability.bin" in input_files and 'bin' not in ignore_file_type:
            source_url = storage.get_storage_url('vulnerability.bin', encode_params=False)[1]
            logger.debug(f"loading {source_url}")
            with storage.open("vulnerability.bin", 'rb') as f:
                header = np.frombuffer(f.read(8), 'i4')
                num_damage_bins = header[0]

            if "vulnerability.idx" in input_files and 'idx' not in ignore_file_type:
                logger.debug(f"loading {storage.get_storage_url('vulnerability.idx', encode_params=False)[1]}")
                with storage.open("vulnerability.bin") as f:
                    vulns_bin = np.memmap(f, dtype=VulnerabilityRow, offset=4, mode='r')

                with storage.open("vulnerability.idx") as f:
                    vulns_idx_bin = np.memmap(f, dtype=VulnerabilityIndex, mode='r')

                if vuln_adj is not None and len(vuln_adj) > 0:
                    vuln_array, valid_vuln_ids = load_vulns_bin_idx_adjusted(vulns_bin, vulns_idx_bin, vuln_map, vuln_map_keys,
                                                                             num_damage_bins, num_intensity_bins, VulnerabilityRow.dtype.itemsize, vuln_adj)
                else:
                    vuln_array, valid_vuln_ids = load_vulns_bin_idx(vulns_bin, vulns_idx_bin, vuln_map, vuln_map_keys,
                                                                    num_damage_bins, num_intensity_bins, VulnerabilityRow.dtype.itemsize)
            else:
                with storage.with_fileno("vulnerability.bin") as f:
                    vulns_bin = np.memmap(f, dtype=vulnerability_dtype, offset=4, mode='r')
                if vuln_adj is not None and len(vuln_adj) > 0:
                    vuln_array, valid_vuln_ids = load_vulns_bin_adjusted(
                        vulns_bin, vuln_map, vuln_map_keys, num_damage_bins, num_intensity_bins, vuln_adj)
                else:
                    vuln_array, valid_vuln_ids = load_vulns_bin(vulns_bin, vuln_map, vuln_map_keys, num_damage_bins, num_intensity_bins)

        elif "vulnerability.csv" in input_files and "csv" not in ignore_file_type:
            source_url = storage.get_storage_url('vulnerability.csv', encode_params=False)[1]
            logger.debug(f"loading {source_url}")
            with storage.open("vulnerability.csv") as f:
                vuln_csv = np.loadtxt(f, dtype=vulnerability_dtype, delimiter=",", skiprows=1, ndmin=1)
            num_damage_bins = max(vuln_csv['damage_bin_id'])
            if vuln_adj is not None and len(vuln_adj) > 0:
                vuln_array, valid_vuln_ids = load_vulns_bin_adjusted(vuln_csv, vuln_map, vuln_map_keys, num_damage_bins, num_intensity_bins, vuln_adj)
            else:
                vuln_array, valid_vuln_ids = load_vulns_bin(vuln_csv, vuln_map, vuln_map_keys, num_damage_bins, num_intensity_bins)
        else:
            raise FileNotFoundError(f"vulnerability file not found at {storage.get_storage_url('', encode_params=False)[1]}")
        missing_vuln_ids = vuln_ids_set.difference(valid_vuln_ids)
        if missing_vuln_ids:
            raise Exception(f"Vulnerability_ids {missing_vuln_ids} are missing"
                            f" from {source_url}")
        vuln_ids = vuln_map_keys.copy()

    return vuln_array, vuln_ids, num_damage_bins


def get_vulnerability_replacements(run_dir, vuln_ids_set):
    """
    Loads the vulnerability adjustment file.

    Args:
        run_dir: (str) the path pointing to the run directory
        vuln_ids_set: (set) set of vulnerability IDs to filter by

    Returns: (List[vulnerability_dtype]) vulnerability replacement data
    """
    settings_path = os.path.join(run_dir, "analysis_settings.json")

    if not os.path.exists(settings_path):
        logger.warning(f"analysis_settings.json not found in {run_dir}.")
        return None

    if not validate_vulnerability_replacements(settings_path):
        return None

    vulnerability_replacements_key = analysis_settings_loader(settings_path).get('vulnerability_adjustments')

    # Inputting the data directly into the analysis_settings.json file takes precedence over the file path
    vulnerability_replacements_field = vulnerability_replacements_key.get('replace_data', None)
    if vulnerability_replacements_field is None:
        vulnerability_replacements_field = vulnerability_replacements_key.get('replace_file', None)

    if isinstance(vulnerability_replacements_field, dict):
        # Convert dict to flat array equivalent to csv format
        flat_data = []
        for v_id, adjustments in vulnerability_replacements_field.items():
            for adj in adjustments:
                flat_data.append((v_id, *adj))
        vuln_adj = np.array(flat_data, dtype=vulnerability_dtype)
    elif isinstance(vulnerability_replacements_field, str):
        # Load csv file
        absolute_path = os.path.abspath(vulnerability_replacements_field)
        logger.debug(f"loading {absolute_path}")
        vuln_adj = np.loadtxt(absolute_path, dtype=vulnerability_dtype, delimiter=",", skiprows=1, ndmin=1)
    vuln_adj = np.array([adj_vuln for adj_vuln in vuln_adj if adj_vuln['vulnerability_id'] in vuln_ids_set],
                        dtype=vuln_adj.dtype)
    vuln_adj.sort(order='vulnerability_id')
    logger.info("Vulnerability adjustments found in analysis settings.")
    return vuln_adj


def get_mean_damage_bins(storage: BaseStorage, ignore_file_type=set()):
    """
    Loads the mean damage bins from the damage_bin_dict file, namely, the `interpolation` value for each bin.

    Args:
        storage: (BaseStorage) the storage connector for fetching the model data
        ignore_file_type: set(str) file extension to ignore when loading

    Returns: (List[Union[damagebindictionary]]) loaded data from the damage_bin_dict file
    """
    return get_damage_bins(storage, ignore_file_type)['interpolation']


def get_damage_bins(storage: BaseStorage, ignore_file_type=set()):
    """
    Loads the damage bins from the damage_bin_dict file.

    Args:
        storage: (BaseStorage) the storage connector for fetching the model data
        ignore_file_type: set(str) file extension to ignore when loading

    Returns: (List[Union[damagebindictionary]]) loaded data from the damage_bin_dict file
    """
    input_files = set(storage.listdir())

    if "damage_bin_dict.bin" in input_files and 'bin' not in ignore_file_type:
        logger.debug(f"loading {storage.get_storage_url('damage_bin_dict.bin', encode_params=False)[1]}")
        with storage.with_fileno("damage_bin_dict.bin") as f:
            return np.fromfile(f, dtype=damagebin_dtype)
    elif "damage_bin_dict.csv" in input_files and 'csv' not in ignore_file_type:
        logger.debug(f"loading {storage.get_storage_url('damage_bin_dict.csv', encode_params=False)[1]}")
        with storage.open("damage_bin_dict.csv") as f:
            return np.loadtxt(f, dtype=damagebin_dtype, skiprows=1, delimiter=',', ndmin=1)
    else:
        raise FileNotFoundError(f"damage_bin_dict file not found at {storage.get_storage_url('', encode_params=False)[1]}")


@nb.njit(cache=True, fastmath=True)
def damage_bin_prob(p, intensities_min, intensities_max, vulns, intensities):
    """
    Calculate the probability of an event happening and then causing damage.
    Note: vulns is a 1-d array containing 1 damage bin of the damage probability distribution as a
    function of hazard intensity.

    Args:
        p: (float) the probability to be updated
        intensities_min: (int) minimum intensity bin id
        intensities_max: (int) maximum intensity bin id
        vulns: (List[float]) slice of damage probability distribution given hazard intensity
        intensities: (List[float]) intensity probability distribution

    Returns: (float) the updated probability
    """
    i = intensities_min
    while i < intensities_max:
        p += vulns[i] * intensities[i]
        i += 1
    return p


@nb.njit(cache=True, fastmath=True)
def do_result(vulns_id, vuln_array, mean_damage_bins,
              int32_mv, num_damage_bins,
              intensities_min, intensities_max, intensities,
              event_id, areaperil_id, vuln_i, cursor):
    """
    Calculate the result concerning an event ID.

    Args:
        vulns_id: (List[int]) list of vulnerability IDs
        vuln_array: (List[List[list]]) list of vulnerabilities and their data
        mean_damage_bins: (List[float]) the mean of each damage bin (len(mean_damage_bins) == num_damage_bins)
        int32_mv: (List[int]) FILL IN LATER
        num_damage_bins: (int) number of damage bins in the data
        intensities_min: (int) minimum intensity bin id
        intensities_max: (int) maximum intensity bin id
        intensities: (List[float]) intensity probability distribution
        event_id: (int) the event ID that concerns the result being calculated
        areaperil_id: (List[int]) the areaperil ID that concerns the result being calculated
        vuln_i: (int) the index concerning the vulnerability inside the vuln_array
        cursor: (int) PLEASE FILL IN

    Returns: (int) PLEASE FILL IN
    """
    int32_mv[cursor], cursor = event_id, cursor + 1
    int32_mv[cursor:cursor + areaperil_int_relative_size] = areaperil_id.view(oasis_int)
    cursor += areaperil_int_relative_size
    int32_mv[cursor], cursor = vulns_id[vuln_i], cursor + 1

    cur_vuln_mat = vuln_array[vuln_i]
    p = 0
    cursor_start = cursor
    cursor += 1
    oasis_float_mv = int32_mv[cursor: cursor + num_damage_bins * results_relative_size].view(oasis_float)
    result_cursor = 0
    damage_bin_i = 0

    while damage_bin_i < num_damage_bins:
        p = damage_bin_prob(p, intensities_min, intensities_max, cur_vuln_mat[damage_bin_i], intensities)
        oasis_float_mv[result_cursor], result_cursor = p, result_cursor + 1
        oasis_float_mv[result_cursor], result_cursor = mean_damage_bins[damage_bin_i], result_cursor + 1
        damage_bin_i += 1
        if p >= 0.999999940:
            break

    int32_mv[cursor_start] = damage_bin_i
    return cursor + (result_cursor * oasis_float_relative_size)


@nb.njit(cache=True)
def doCdf(event_id,
          num_intensity_bins, footprint,
          areaperil_id_ind, areaperil_to_vulns_idx_array, areaperil_to_vulns,
          vuln_array, vulns_id, num_damage_bins, mean_damage_bins,
          int32_mv, max_result_relative_size):
    """
    Calculates the cumulative distribution function (cdf) for an event ID.

    Args:
        event_id: (int) the event ID the the CDF is being calculated to.
        num_intensity_bins: (int) the number of intensity bins for the CDF
        footprint: (List[Tuple[int, int, float]]) information about the footprint with event_id, areaperil_id,
                                                  probability
        areaperil_id_ind: (np.array) id_index structure mapping areaperil_id to dense index
        areaperil_to_vulns_idx_array: (List[Tuple[int, int]]) the index where the areaperil ID starts and finishes
        areaperil_to_vulns: (List[int]) maps the areaperil ID to the vulnerability ID
        vuln_array: (List[list]) FILL IN LATER
        vulns_id: (List[int]) list of vulnerability IDs
        num_damage_bins: (int) number of damage bins in the data
        mean_damage_bins: (List[float]) the mean of each damage bin (len(mean_damage_bins) == num_damage_bins)
        int32_mv: (List[int]) FILL IN LATER
        max_result_relative_size: (int) the maximum result size

    Returns: (int)
    """
    if not footprint.shape[0]:
        return 0

    intensities_min = num_intensity_bins
    intensities_max = 0
    intensities = np.zeros(num_intensity_bins, dtype=oasis_float)

    areaperil_id = np.zeros(1, dtype=areaperil_int)
    has_vuln = False
    cursor = 0

    for footprint_i in range(footprint.shape[0]):
        event_row = footprint[footprint_i]
        if areaperil_id[0] != event_row['areaperil_id']:
            if has_vuln and intensities_min <= intensities_max:
                areaperil_to_vulns_idx = areaperil_to_vulns_idx_array[id_index_get_idx(areaperil_id_ind, areaperil_id[0])]
                intensities_max += 1
                for vuln_idx in range(areaperil_to_vulns_idx['start'], areaperil_to_vulns_idx['end']):
                    vuln_i = areaperil_to_vulns[vuln_idx]
                    if cursor + max_result_relative_size > buff_int_size:
                        yield cursor * oasis_int_size
                        cursor = 0

                    cursor = do_result(vulns_id, vuln_array, mean_damage_bins,
                                       int32_mv, num_damage_bins,
                                       intensities_min, intensities_max, intensities,
                                       event_id, areaperil_id, vuln_i, cursor)

            areaperil_id[0] = event_row['areaperil_id']
            has_vuln = id_index_get_idx(areaperil_id_ind, areaperil_id[0]) != ID_INDEX_NOT_FOUND

            if has_vuln:
                intensities[intensities_min: intensities_max] = 0
                intensities_min = num_intensity_bins
                intensities_max = 0
        if has_vuln:
            if event_row['probability'] > 0:
                intensity_bin_i = event_row['intensity_bin_id'] - 1
                intensities[intensity_bin_i] = event_row['probability']
                if intensity_bin_i > intensities_max:
                    intensities_max = intensity_bin_i
                if intensity_bin_i < intensities_min:
                    intensities_min = intensity_bin_i

    if has_vuln and intensities_min <= intensities_max:
        areaperil_to_vulns_idx = areaperil_to_vulns_idx_array[id_index_get_idx(areaperil_id_ind, areaperil_id[0])]
        intensities_max += 1
        for vuln_idx in range(areaperil_to_vulns_idx['start'], areaperil_to_vulns_idx['end']):
            vuln_i = areaperil_to_vulns[vuln_idx]
            if cursor + max_result_relative_size > buff_int_size:
                yield cursor * oasis_int_size
                cursor = 0

            cursor = do_result(vulns_id, vuln_array, mean_damage_bins,
                               int32_mv, num_damage_bins,
                               intensities_min, intensities_max, intensities,
                               event_id, areaperil_id, vuln_i, cursor)

    yield cursor * oasis_int_size


@nb.njit()
def convert_vuln_id_to_index(vuln_map, vuln_map_keys, areaperil_to_vulns):
    hm_info, hm_lookup, hm_index = hm_unpack(vuln_map)
    for i in range(areaperil_to_vulns.shape[0]):
        slot = hm_find_key(hm_info, hm_lookup, hm_index, vuln_map_keys, areaperil_to_vulns[i])
        if slot != HM_NOT_FOUND:
            areaperil_to_vulns[i] = np.int32(hm_index[slot])
        else:
            areaperil_to_vulns[i] = np.int32(-1)


@redirect_logging(exec_name='modelpy')
def run(
    run_dir,
    file_in,
    file_out,
    ignore_file_type,
    data_server,
    peril_filter,
    df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader",
    analysis_pk=None
):
    """
    Runs the main process of the getmodel process.

    Args:
        run_dir: (str) the directory of where the process is running
        file_in: (Optional[str]) the path to the input directory
        file_out: (Optional[str]) the path to the output directory
        ignore_file_type: set(str) file extension to ignore when loading
        data_server: (bool) if set to True runs the data server
        peril_filter (list[int]): list of perils to include in the computation (if None, all perils will be included).
        df_engine: (str) The engine to use when loading dataframes

    Returns: None
    """
    model_storage = get_storage_from_config_path(
        os.path.join(run_dir, 'model_storage.json'),
        os.path.join(run_dir, 'static'),
    )
    input_path = os.path.join(run_dir, 'input')
    ignore_file_type = set(ignore_file_type)

    if data_server:
        logger.debug("data server active")
        FootprintLayerClient.register()
        logger.debug("registered with data server")
        atexit.register(FootprintLayerClient.unregister)
    else:
        logger.debug("data server not active")

    with ExitStack() as stack:
        if file_in is None:
            streams_in = sys.stdin.buffer
        else:
            streams_in = stack.enter_context(open(file_in, 'rb'))

        if file_out is None:
            stream_out = sys.stdout.buffer
        else:
            stream_out = stack.enter_context(open(file_out, 'wb'))

        event_id_mv = memoryview(bytearray(4))
        event_ids = np.ndarray(1, buffer=event_id_mv, dtype='i4')

        # --- load or build read-only structures --------------------------------
        from oasislmf.pytools.getmodel.structure import (
            getmodel_structure_exists, load_getmodel_structure, build_structures,
        )
        if getmodel_structure_exists(run_dir):
            logger.info("loading pre-computed getmodel structures (shared memory)")
            structures = load_getmodel_structure(run_dir)
        else:
            logger.info("building getmodel structures from input files")
            structures = build_structures(run_dir, ignore_file_type, peril_filter, df_engine)

        vuln_array = structures['vuln_array']
        vulns_id = structures['vulns_id']
        areaperil_id_ind = structures['areaperil_id_ind']
        areaperil_to_vulns_idx_array = structures['areaperil_to_vulns_idx_array']
        areaperil_to_vulns = structures['areaperil_to_vulns']
        unique_areaperil_ids = structures['unique_areaperil_ids']
        mean_damage_bins = structures['mean_damage_bins']
        num_damage_bins = structures['num_damage_bins']
        num_intensity_bins = structures['num_intensity_bins']

        logger.debug('init footprint')
        footprint_obj = stack.enter_context(
            Footprint.load(model_storage, ignore_file_type, df_engine=df_engine,
                           areaperil_ids=list(unique_areaperil_ids)))

        if data_server:
            num_intensity_bins: int = FootprintLayerClient.get_number_of_intensity_bins()
            logger.info(f"got {num_intensity_bins} intensity bins from server")

        # even_id, areaperil_id, vulnerability_id, num_result, [oasis_float] * num_result
        max_result_relative_size = 1 + + areaperil_int_relative_size + 1 + 1 + num_damage_bins * results_relative_size
        mv = memoryview(bytearray(buff_size))
        int32_mv = np.ndarray(buff_size // np.int32().itemsize, buffer=mv, dtype=np.int32)

        # header
        stream_out.write(np.uint32(1).tobytes())

        logger.debug('doCdf starting')
        empty_events = 0
        while True:
            len_read = streams_in.readinto(event_id_mv)
            if len_read == 0:
                break

            # get the next event_id from the input stream
            event_id = event_ids[0]

            if data_server:
                event_footprint = FootprintLayerClient.get_event(event_id)
            else:
                event_footprint = footprint_obj.get_event(event_id)

            if event_footprint is not None:
                # compute effective damageability probability distribution
                # stream out: event_id, areaperil_id, number of damage bins, effecive damageability cdf bins (bin_mean and prob_to)
                for cursor_bytes in doCdf(event_id,
                                          num_intensity_bins, event_footprint,
                                          areaperil_id_ind, areaperil_to_vulns_idx_array, areaperil_to_vulns,
                                          vuln_array, vulns_id, num_damage_bins, mean_damage_bins,
                                          int32_mv, max_result_relative_size):

                    if cursor_bytes:
                        stream_out.write(mv[:cursor_bytes])
                    else:
                        empty_events += 1
                        break
            else:
                empty_events += 1
        oasis_ping({"events_complete": empty_events, 'analysis_pk': analysis_pk})
