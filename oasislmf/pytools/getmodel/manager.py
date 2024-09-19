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

from numba import int32 as nb_int32, float64 as nb_float64
from numba.typed import Dict

from oasis_data_manager.df_reader.config import get_df_reader, clean_config, InputReaderConfig
from oasis_data_manager.filestore.backends.base import BaseStorage
from oasis_data_manager.filestore.config import get_storage_from_config_path
from oasislmf.pytools.common.event_stream import PIPE_CAPACITY
from oasislmf.utils.data import validate_vulnerability_replacements
from oasislmf.pytools.data_layer.footprint_layer import FootprintLayerClient
from oasislmf.pytools.getmodel.common import (Index_type, Keys, areaperil_int,
                                              oasis_float)
from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.utils import redirect_logging
from ods_tools.oed import AnalysisSettingSchema

from .vulnerability import vulnerability_dataset, parquetvulnerability_meta_filename
from ..common.data import null_index

logger = logging.getLogger(__name__)

buff_size = PIPE_CAPACITY

oasis_int_dtype = np.dtype('i4')
oasis_int = np.int32
oasis_int_size = np.int32().itemsize
buff_int_size = buff_size // oasis_int_size

areaperil_int_relative_size = areaperil_int.itemsize // oasis_int_size
oasis_float_relative_size = oasis_float.itemsize // oasis_int_size
results_relative_size = 2 * oasis_float_relative_size


damagebindictionary = nb.from_dtype(np.dtype([('bin_index', np.int32),
                                              ('bin_from', oasis_float),
                                              ('bin_to', oasis_float),
                                              ('interpolation', oasis_float),
                                              ('interval_type', np.int32),
                                              ]))

EventCSV = nb.from_dtype(np.dtype([('event_id', np.int32),
                                   ('areaperil_id', areaperil_int),
                                   ('intensity_bin_id', np.int32),
                                   ('probability', oasis_float)
                                   ]))

Item = nb.from_dtype(np.dtype([('id', np.int32),
                               ('coverage_id', np.int32),
                               ('areaperil_id', areaperil_int),
                               ('vulnerability_id', np.int32),
                               ('group_id', np.int32)
                               ]))


Vulnerability = nb.from_dtype(np.dtype([('vulnerability_id', np.int32),
                                        ('intensity_bin_id', np.int32),
                                        ('damage_bin_id', np.int32),
                                        ('probability', oasis_float)
                                        ]))

VulnerabilityIndex = nb.from_dtype(np.dtype([('vulnerability_id', np.int32),
                                             ('offset', np.int64),
                                             ('size', np.int64),
                                             ('original_size', np.int64)
                                             ]))
VulnerabilityRow = nb.from_dtype(np.dtype([('intensity_bin_id', np.int32),
                                           ('damage_bin_id', np.int32),
                                           ('probability', oasis_float)
                                           ]))

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
    raise Exception(f"AREAPERIL_TYPE {areaperil_int} is not implemented chose u4 or u8")


@nb.njit(cache=True)
def load_items(items, valid_area_peril_id):
    """
    Processes the Items loaded from the file extracting meta data around the vulnerability data.

    Args:
        items: (List[Item]) Data loaded from the vulnerability file
        valid_area_peril_id: array of area_peril_id to be included (if none, all are included)

    Returns: (Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]])
             vulnerability dictionary, vulnerability IDs, areaperil to vulnerability index dictionary,
             areaperil ID to vulnerability index array, areaperil ID to vulnerability array
    """
    areaperil_to_vulns_size = 0
    areaperil_dict = Dict()
    vuln_dict = Dict()
    vuln_idx = 0
    for i in range(items.shape[0]):
        item = items[i]

        # filter areaperil_id
        if valid_area_peril_id is not None and item['areaperil_id'] not in valid_area_peril_id:
            continue

        # insert the vulnerability index if not in there
        if item['vulnerability_id'] not in vuln_dict:
            vuln_dict[item['vulnerability_id']] = np.int32(vuln_idx)
            vuln_idx += 1

        # insert an area dictionary into areaperil_dict under the key of areaperil ID
        if item['areaperil_id'] not in areaperil_dict:
            area_vuln = Dict()
            area_vuln[item['vulnerability_id']] = 0
            areaperil_dict[item['areaperil_id']] = area_vuln
            areaperil_to_vulns_size += 1
        else:
            if item['vulnerability_id'] not in areaperil_dict[item['areaperil_id']]:
                areaperil_to_vulns_size += 1
                areaperil_dict[item['areaperil_id']][item['vulnerability_id']] = 0

    areaperil_to_vulns_idx_dict = Dict()
    areaperil_to_vulns_idx_array = np.empty(len(areaperil_dict), dtype=Index_type)
    areaperil_to_vulns = np.empty(areaperil_to_vulns_size, dtype=np.int32)

    areaperil_i = 0
    vulnerability_i = 0

    for areaperil_id, vulns in areaperil_dict.items():
        areaperil_to_vulns_idx_dict[areaperil_id] = areaperil_i
        areaperil_to_vulns_idx_array[areaperil_i]['start'] = vulnerability_i

        for vuln_id in sorted(vulns):  # sorted is not necessary but doesn't impede the perf and align with cpp getmodel
            areaperil_to_vulns[vulnerability_i] = vuln_id
            vulnerability_i += 1
        areaperil_to_vulns_idx_array[areaperil_i]['end'] = vulnerability_i
        areaperil_i += 1

    return vuln_dict, areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_to_vulns


def get_items(input_path, ignore_file_type=set(), valid_area_peril_id=None):
    """
    Loads the items from the items file.

    Args:
        input_path: (str) the path pointing to the file
        ignore_file_type: set(str) file extension to ignore when loading

    Returns: (Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]])
             vulnerability dictionary, vulnerability IDs, areaperil to vulnerability index dictionary,
             areaperil ID to vulnerability index array, areaperil ID to vulnerability array
    """
    input_files = set(os.listdir(input_path))
    if "items.bin" in input_files and "bin" not in ignore_file_type:
        logger.debug(f"loading {os.path.join(input_path, 'items.csv')}")
        items = np.memmap(os.path.join(input_path, "items.bin"), dtype=Item, mode='r')
    elif "items.csv" in input_files and "csv" not in ignore_file_type:
        logger.debug(f"loading {os.path.join(input_path, 'items.csv')}")
        items = np.loadtxt(os.path.join(input_path, "items.csv"), dtype=Item, delimiter=",", skiprows=1, ndmin=1)
    else:
        raise FileNotFoundError(f'items file not found at {input_path}')

    return load_items(items, valid_area_peril_id)


def get_intensity_bin_dict(input_path):
    """
    Loads the intensity bin dictionary file and creates a dictionary to map intensities to bins
    Used in the dynamic footprint generation as intensitys can be adjusted for defences at runtime

    Args:
        input_path: (str) the path pointing to the file

    Returns: (Dict[int, int])
             intensity bin dict, with intensity value and bin index
    """
    input_files = set(os.listdir(input_path))
    intensity_bin_dict = Dict.empty(nb_int32, nb_int32)
    if "intensity_bin_dict.csv" in input_files:
        logger.debug(f"loading {os.path.join(input_path, 'intensity_bin_dict.csv')}")
        data = np.loadtxt(os.path.join(input_path, "intensity_bin_dict.csv"), dtype=np.int32, delimiter=",", skiprows=1, ndmin=1)
        for d in data:
            intensity_bin_dict[d[0]] = d[1]
    else:
        raise FileNotFoundError(f'intensity_bin_dict file not found at {input_path}')

    return intensity_bin_dict


def get_intensity_adjustment(input_path):
    pass


@nb.njit(cache=True)
def load_vuln_probability(vuln_array, vuln, vuln_id):
    if vuln_array.shape[0] < vuln['damage_bin_id']:
        raise Exception("vulnerability_id " + str(vuln_id) + " has damage_bin_id bigger that expected maximum")
    if vuln['intensity_bin_id'] <= vuln_array.shape[1]:  # intensity in vulnerability curve but not in the footprint, we can ignore
        vuln_array[vuln['damage_bin_id'] - 1, vuln['intensity_bin_id'] - 1] = vuln['probability']


@nb.njit(cache=True)
def load_vulns_bin_idx(vulns_bin, vulns_idx_bin, vuln_dict,
                       num_damage_bins, num_intensity_bins, rowsize):
    """
    Loads the vulnerability binary index file.

    Args:
        vulns_bin: (List[VulnerabilityRow]) vulnerability data from the vulnerability file
        vulns_idx_bin: (List[VulnerabilityIndex]) vulnerability index data from the vulnerability idx file
        vuln_dict: (Dict[int, int]) maps the vulnerability ID with the index in the vulnerability array
        num_damage_bins: (int) number of damage bins in the data
        num_intensity_bins: (int) the number of intensity bins

    Returns: (List[List[List[floats]]]) vulnerability data grouped by intensity bin and damage bin
    """
    vuln_array = np.zeros((len(vuln_dict), num_damage_bins, num_intensity_bins), dtype=oasis_float)
    vuln_ids = np.full(len(vuln_dict), null_index)
    for idx_i in range(vulns_idx_bin.shape[0]):
        vuln_idx = vulns_idx_bin[idx_i]
        if vuln_idx['vulnerability_id'] in vuln_dict:
            vuln_ids[vuln_dict[vuln_idx['vulnerability_id']]] = vuln_idx['vulnerability_id']
            cur_vuln_array = vuln_array[vuln_dict[vuln_idx['vulnerability_id']]]
            start = (vuln_idx['offset'] - vuln_offset) // rowsize
            end = start + vuln_idx['size'] // rowsize
            for vuln_i in range(start, end):
                vuln = vulns_bin[vuln_i]
                load_vuln_probability(cur_vuln_array, vuln, vuln_idx['vulnerability_id'])

    return vuln_array, vuln_ids


@nb.njit(cache=True)
def load_vulns_bin_idx_adjusted(vulns_bin, vulns_idx_bin, vuln_dict,
                                num_damage_bins, num_intensity_bins, rowsize, adj_vuln_data=None):
    """
    Loads the vulnerability binary index file, prioritizing the data in the adjustments file over the data in the
    vulnerability file.

    Args:
        vulns_bin: (List[VulnerabilityRow]) vulnerability data from the vulnerability file
        vulns_idx_bin: (List[VulnerabilityIndex]) vulnerability index data from the vulnerability idx file
        vuln_dict: (Dict[int, int]) maps the vulnerability ID with the index in the vulnerability array
        num_damage_bins: (int) number of damage bins in the data
        num_intensity_bins: (int) the number of intensity bins
        adj_vuln_data: (List[Vulnerability]) vulnerability adjustment data, sorted by vuln_id

    Returns: (List[List[List[floats]]]) vulnerability data grouped by intensity bin and damage bin
    """
    vuln_array = np.zeros((len(vuln_dict), num_damage_bins, num_intensity_bins), dtype=oasis_float)
    vuln_ids = np.full(len(vuln_dict), null_index)
    adj_vuln_index = 0

    for idx_i in range(vulns_idx_bin.shape[0]):
        vuln_idx = vulns_idx_bin[idx_i]
        vuln_id = vuln_idx['vulnerability_id']

        # Check if current vulnerability id is in the adjustment data
        while adj_vuln_data is not None and adj_vuln_index < len(adj_vuln_data) and adj_vuln_data[adj_vuln_index]['vulnerability_id'] < vuln_id:
            adj_vuln_index += 1

        if vuln_id in vuln_dict:
            vuln_ids[vuln_dict[vuln_id]] = vuln_id
            cur_vuln_array = vuln_array[vuln_dict[vuln_id]]
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
        if vuln_id in vuln_dict:
            load_vuln_probability(vuln_array[vuln_dict[vuln_id]], adj_vuln, vuln_id)
        adj_vuln_index += 1

    return vuln_array, vuln_ids


@nb.njit(cache=True)
def load_vulns_bin(vulns_bin, vuln_dict, num_damage_bins, num_intensity_bins):
    """
    Loads the vulnerability data grouped by the intensity and damage bins.

    Args:
        vuln_bin: (List[Vulnerability]) vulnerability data from the vulnerability file
        vuln_dict: (Dict[int, int]) maps the vulnerability ID with the index in the vulnerability array
        num_damage_bins: (int) number of damage bins in the data
        num_intensity_bins: (int) the number of intensity bins

    Returns: (List[List[List[floats]]]) vulnerability data grouped by intensity bin and damage bin
    """
    vuln_array = np.zeros((len(vuln_dict), num_damage_bins, num_intensity_bins), dtype=oasis_float)
    vuln_ids = np.full(len(vuln_dict), null_index)
    cur_vulnerability_id = -1

    for vuln_i in range(vulns_bin.shape[0]):
        vuln = vulns_bin[vuln_i]
        if vuln['vulnerability_id'] != cur_vulnerability_id:
            if vuln['vulnerability_id'] in vuln_dict:
                cur_vulnerability_id = vuln['vulnerability_id']
                vuln_ids[vuln_dict[cur_vulnerability_id]] = cur_vulnerability_id
                cur_vuln_array = vuln_array[vuln_dict[cur_vulnerability_id]]
            else:
                cur_vulnerability_id = -1
        if cur_vulnerability_id != -1:
            load_vuln_probability(cur_vuln_array, vuln, cur_vulnerability_id)

    return vuln_array, vuln_ids


@nb.njit(cache=True)
def load_vulns_bin_adjusted(vulns_bin, vuln_dict, num_damage_bins, num_intensity_bins, adj_vuln_data=None):
    """
    Loads the vulnerability data grouped by the intensity and damage bins, prioritizing the data
    in the adjustments file over the data in the vulnerability file.

    Args:
        vuln_bin: (List[Vulnerability]) vulnerability data from the vulnerability file
        vuln_dict: (Dict[int, int]) maps the vulnerability ID with the index in the vulnerability array
        num_damage_bins: (int) number of damage bins in the data
        num_intensity_bins: (int) the number of intensity bins
        adj_vuln_data: (List[Vulnerability]) vulnerability adjustment data, sorted by vuln_id

    Returns: (List[List[List[floats]]]) vulnerability data grouped by intensity bin and damage bin
    """
    vuln_array = np.zeros((len(vuln_dict), num_damage_bins, num_intensity_bins), dtype=oasis_float)
    vuln_ids = np.full(len(vuln_dict), null_index)
    ids_to_replace = set()
    adj_vuln_index = 0

    # Create list of ids to replace if adj_vuln_data is provided
    if adj_vuln_data is not None:
        for adj_vuln in adj_vuln_data:
            ids_to_replace.add(adj_vuln['vulnerability_id'])

    vuln_i = 0

    while vuln_i < len(vulns_bin):
        vuln = vulns_bin[vuln_i]
        vuln_id = vuln['vulnerability_id']
        if vuln_id in vuln_dict:
            vuln_ids[vuln_dict[vuln_id]] = vuln_id
            if vuln_id in ids_to_replace:
                # Advance to current vuln_id
                while adj_vuln_index < len(adj_vuln_data) and adj_vuln_data[adj_vuln_index]['vulnerability_id'] < vuln_id:
                    adj_vuln_index += 1
                # process current vuln_id
                while adj_vuln_index < len(adj_vuln_data) and adj_vuln_data[adj_vuln_index]['vulnerability_id'] == vuln_id:
                    load_vuln_probability(vuln_array[vuln_dict[vuln_id]], adj_vuln_data[adj_vuln_index], vuln_id)
                    adj_vuln_index += 1
                # Skip remaining vulns_bin entries with the same vulnerability_id
                while vuln_i < len(vulns_bin) and vulns_bin[vuln_i]['vulnerability_id'] == vuln_id:
                    vuln_i += 1
                continue
            else:
                # Use data from vulns_bin
                load_vuln_probability(vuln_array[vuln_dict[vuln_id]], vuln, vuln_id)
        vuln_i += 1
    return vuln_array, vuln_ids


@nb.njit()
def update_vulns_dictionary(vuln_dict, vulns_id_array):
    """
    Updates the indexes of the vulnerability IDs (usually used in loading vulnerability data from parquet file).

    Args:
        vuln_dict: (Dict[int, int]) vulnerability dict that maps the vulnerability IDs (key) with the index (value)
        vulns_id_array: (List[int]) list of vulnerability IDs loaded from the parquet file

    """
    for i in range(vulns_id_array.shape[0]):
        vuln_dict[vulns_id_array[i]] = np.int32(i)


@nb.njit()
def update_vuln_array_with_adj_data(vuln_array, vuln_dict, adj_vuln_data):
    """
    Update the vulnerability array with adjustment data (used for parquet loading).

    Args:
        vuln_array: (3D array) The vulnerability data array.
        vuln_dict: (Dict[int, int]) Maps vulnerability IDs to indices in vuln_array.
        adj_vuln_data: (List[Vulnerability]) The vulnerability adjustment data.

    Returns: (3D array) The updated vulnerability data array.
    """
    for adj_vuln in adj_vuln_data:
        vuln_id = adj_vuln['vulnerability_id']
        if vuln_id in vuln_dict:
            load_vuln_probability(vuln_array[vuln_dict[vuln_id]], adj_vuln, vuln_id)
    return vuln_array


@nb.njit()
def create_vulns_id(vuln_dict):
    """
    Creates a vulnerability array where the index of the array correlates with the index of the vulnerability.

    Args:
        vuln_dict: (Dict) maps the vulnerability of the id (key) with the vulnerability ID (value)

    Returns: (List[int]) list of vulnerability IDs
    """
    vulns_id = np.empty(len(vuln_dict), dtype=np.int32)

    for vuln_id, vuln_idx in vuln_dict.items():
        vulns_id[vuln_idx] = vuln_id

    return vulns_id


def get_vuln_rngadj_dict(run_dir, vuln_dict):
    """
    Loads vulnerability adjustments from the analysis settings file.

    Args:
        run_dir (str): path to the run directory (used to load the analysis settings)

    Returns: (Dict[nb_int32, nb_float64]) vulnerability adjustments dictionary
    """
    settings_path = os.path.join(run_dir, "analysis_settings.json")
    vuln_adj_numba_dict = Dict.empty(key_type=nb_int32, value_type=nb_float64)
    if not os.path.exists(settings_path):
        logger.debug(f"analysis_settings.json not found in {run_dir}.")
        return vuln_adj_numba_dict
    vulnerability_adjustments_field = None
    vulnerability_adjustments_field = AnalysisSettingSchema().get(settings_path, {}).get('vulnerability_adjustments', None)
    if vulnerability_adjustments_field is not None:
        adjustments = vulnerability_adjustments_field.get('adjustments', None)
    else:
        adjustments = None
    if adjustments is None:
        logger.debug(f"vulnerability_adjustments not found in {settings_path}.")
        return vuln_adj_numba_dict
    for key, value in adjustments.items():
        if nb_int32(key) in vuln_dict.keys():
            vuln_adj_numba_dict[nb_int32(key)] = nb_float64(value)
    return vuln_adj_numba_dict


def get_vulns(
        storage: BaseStorage, run_dir, vuln_dict, num_intensity_bins, ignore_file_type=set(), df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader"):
    """
    Loads the vulnerabilities from the file.

    Args:
        storage: (str) the storage manager for fetching model data
        run_dir: (str) the path to the run folder (used to load the analysis settings)
        vuln_dict: (Dict[int, int]) maps the vulnerability ID with the index in the vulnerability array
        num_intensity_bins: (int) the number of intensity bins
        ignore_file_type: set(str) file extension to ignore when loading

    Returns: (Tuple[List[List[float]], int, np.array[int]) vulnerability data, vulnerabilities id, number of damage bins
    """
    input_files = set(storage.listdir())
    vuln_adj = get_vulnerability_replacements(run_dir, vuln_dict)

    if vulnerability_dataset in input_files and "parquet" not in ignore_file_type:
        with storage.open(parquetvulnerability_meta_filename, 'r') as outfile:
            meta_data = json.load(outfile)
        logger.debug(f"loading {storage.get_storage_url(vulnerability_dataset, encode_params=False)[1]}")

        df_reader_config = clean_config(InputReaderConfig(filepath=vulnerability_dataset, engine=df_engine))
        df_reader_config["engine"]["options"]["storage"] = storage
        reader = get_df_reader(df_reader_config, filters=[[('vulnerability_id', '==', vuln_id)] for vuln_id in vuln_dict.keys()])
        df = reader.as_pandas()
        num_damage_bins = meta_data['num_damage_bins']
        vuln_array = np.vstack(df['vuln_array'].to_numpy()).reshape(len(df['vuln_array']),
                                                                    num_damage_bins,
                                                                    num_intensity_bins)
        vuln_ids = df['vulnerability_id'].to_numpy()
        missing_vuln_ids = set(vuln_dict).difference(vuln_ids)
        if missing_vuln_ids:
            raise Exception(f"Vulnerability_ids {missing_vuln_ids} are missing"
                            f" from {storage.get_storage_url(vulnerability_dataset, encode_params=False)[1]}")
        update_vulns_dictionary(vuln_dict, vuln_ids)
        # update vulnerability array with adjustment data if present
        if vuln_adj is not None and len(vuln_adj) > 0:
            vuln_array = update_vuln_array_with_adj_data(vuln_array, vuln_dict, vuln_adj)

    else:
        if "vulnerability.bin" in input_files and 'bin' not in ignore_file_type:
            logger.debug(f"loading {storage.get_storage_url('vulnerability.bin', encode_params=False)[1]}")
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
                    vuln_array, valid_vuln_ids = load_vulns_bin_idx_adjusted(vulns_bin, vulns_idx_bin, vuln_dict,
                                                                             num_damage_bins, num_intensity_bins, VulnerabilityRow.dtype.itemsize, vuln_adj)
                else:
                    vuln_array, valid_vuln_ids = load_vulns_bin_idx(vulns_bin, vulns_idx_bin, vuln_dict,
                                                                    num_damage_bins, num_intensity_bins, VulnerabilityRow.dtype.itemsize)
            else:
                with storage.with_fileno("vulnerability.bin") as f:
                    vulns_bin = np.memmap(f, dtype=Vulnerability, offset=4, mode='r')
                if vuln_adj is not None and len(vuln_adj) > 0:
                    vuln_array, valid_vuln_ids = load_vulns_bin_adjusted(vulns_bin, vuln_dict, num_damage_bins, num_intensity_bins, vuln_adj)
                else:
                    vuln_array, valid_vuln_ids = load_vulns_bin(vulns_bin, vuln_dict, num_damage_bins, num_intensity_bins)

        elif "vulnerability.csv" in input_files and "csv" not in ignore_file_type:
            logger.debug(f"loading {storage.get_storage_url('vulnerability.csv', encode_params=False)[1]}")
            with storage.open("vulnerability.csv") as f:
                vuln_csv = np.loadtxt(f, dtype=Vulnerability, delimiter=",", skiprows=1, ndmin=1)
            num_damage_bins = max(vuln_csv['damage_bin_id'])
            if vuln_adj is not None and len(vuln_adj) > 0:
                vuln_array, valid_vuln_ids = load_vulns_bin_adjusted(vuln_csv, vuln_dict, num_damage_bins, num_intensity_bins, vuln_adj)
            else:
                vuln_array, valid_vuln_ids = load_vulns_bin(vuln_csv, vuln_dict, num_damage_bins, num_intensity_bins)
        else:
            raise FileNotFoundError(f"vulnerability file not found at {storage.get_storage_url('', encode_params=False)[1]}")
        missing_vuln_ids = set(vuln_dict).difference(valid_vuln_ids)
        if missing_vuln_ids:
            raise Exception(f"Vulnerability_ids {missing_vuln_ids} are missing"
                            f" from {storage.get_storage_url(vulnerability_dataset, encode_params=False)[1]}")
        vuln_ids = create_vulns_id(vuln_dict)

    return vuln_array, vuln_ids, num_damage_bins


def get_vulnerability_replacements(run_dir, vuln_dict):
    """
    Loads the vulnerability adjustment file.

    Args:
        path: (str) the path pointing to the run directory
        vuln_dict: (Dict[int, int]) list of vulnerability IDs

    Returns: (List[Vulnerability]) vulnerability replacement data
    """
    settings_path = os.path.join(run_dir, "analysis_settings.json")

    if not os.path.exists(settings_path):
        logger.warning(f"analysis_settings.json not found in {run_dir}.")
        return None

    if not validate_vulnerability_replacements(settings_path):
        return None

    vulnerability_replacements_key = None
    vulnerability_replacements_key = AnalysisSettingSchema().get(settings_path, {}).get('vulnerability_adjustments')

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
        vuln_adj = np.array(flat_data, dtype=Vulnerability)
    elif isinstance(vulnerability_replacements_field, str):
        # Load csv file
        absolute_path = os.path.abspath(vulnerability_replacements_field)
        logger.debug(f"loading {absolute_path}")
        vuln_adj = np.loadtxt(absolute_path, dtype=Vulnerability, delimiter=",", skiprows=1, ndmin=1)
    vuln_adj = np.array([adj_vuln for adj_vuln in vuln_adj if adj_vuln['vulnerability_id'] in vuln_dict],
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
            return np.fromfile(f, dtype=damagebindictionary)
    elif "damage_bin_dict.csv" in input_files and 'csv' not in ignore_file_type:
        logger.debug(f"loading {storage.get_storage_url('damage_bin_dict.csv', encode_params=False)[1]}")
        with storage.open("damage_bin_dict.csv") as f:
            return np.loadtxt(f, dtype=damagebindictionary, skiprows=1, delimiter=',', ndmin=1)
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
    int32_mv[cursor:cursor + areaperil_int_relative_size] = areaperil_id.view(oasis_int_dtype)
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
        if p > 0.999999940:
            break

    int32_mv[cursor_start] = damage_bin_i
    return cursor + (result_cursor * oasis_float_relative_size)


@nb.njit()
def doCdf(event_id,
          num_intensity_bins, footprint,
          areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_to_vulns,
          vuln_array, vulns_id, num_damage_bins, mean_damage_bins,
          int32_mv, max_result_relative_size):
    """
    Calculates the cumulative distribution function (cdf) for an event ID.

    Args:
        event_id: (int) the event ID the the CDF is being calculated to.
        num_intensity_bins: (int) the number of intensity bins for the CDF
        footprint: (List[Tuple[int, int, float]]) information about the footprint with event_id, areaperil_id,
                                                  probability
        areaperil_to_vulns_idx_dict: (Dict[int, int]) maps the areaperil ID with the ENTER_HERE
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
                areaperil_to_vulns_idx = areaperil_to_vulns_idx_array[areaperil_to_vulns_idx_dict[areaperil_id[0]]]
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
            has_vuln = areaperil_id[0] in areaperil_to_vulns_idx_dict

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
        areaperil_to_vulns_idx = areaperil_to_vulns_idx_array[areaperil_to_vulns_idx_dict[areaperil_id[0]]]
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
def convert_vuln_id_to_index(vuln_dict, areaperil_to_vulns):
    for i in range(areaperil_to_vulns.shape[0]):
        areaperil_to_vulns[i] = vuln_dict[areaperil_to_vulns[i]]


@redirect_logging(exec_name='modelpy')
def run(
    run_dir,
    file_in,
    file_out,
    ignore_file_type,
    data_server,
    peril_filter,
    df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader",
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

        # load keys.csv to determine included AreaPerilID from peril_filter
        if peril_filter:
            keys_df = pd.read_csv(os.path.join(input_path, 'keys.csv'), dtype=Keys)
            valid_area_peril_id = keys_df.loc[keys_df['PerilID'].isin(peril_filter), 'AreaPerilID'].to_numpy()
            logger.debug(
                f'Peril specific run: ({peril_filter}), {len(valid_area_peril_id)} AreaPerilID included out of {len(keys_df)}')
        else:
            valid_area_peril_id = None

        logger.debug('init items')
        vuln_dict, areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_to_vulns = get_items(
            input_path, ignore_file_type, valid_area_peril_id)

        logger.debug('init footprint')
        footprint_obj = stack.enter_context(Footprint.load(model_storage, ignore_file_type, df_engine=df_engine))

        if data_server:
            num_intensity_bins: int = FootprintLayerClient.get_number_of_intensity_bins()
            logger.info(f"got {num_intensity_bins} intensity bins from server")
        else:
            num_intensity_bins: int = footprint_obj.num_intensity_bins

        logger.debug('init vulnerability')

        vuln_array, vulns_id, num_damage_bins = get_vulns(model_storage, run_dir, vuln_dict,
                                                          num_intensity_bins, ignore_file_type, df_engine=df_engine)
        convert_vuln_id_to_index(vuln_dict, areaperil_to_vulns)
        logger.debug('init mean_damage_bins')
        mean_damage_bins = get_mean_damage_bins(model_storage, ignore_file_type)

        # even_id, areaperil_id, vulnerability_id, num_result, [oasis_float] * num_result
        max_result_relative_size = 1 + + areaperil_int_relative_size + 1 + 1 + num_damage_bins * results_relative_size
        mv = memoryview(bytearray(buff_size))
        int32_mv = np.ndarray(buff_size // np.int32().itemsize, buffer=mv, dtype=np.int32)

        # header
        stream_out.write(np.uint32(1).tobytes())

        logger.debug('doCdf starting')
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
                                          areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_to_vulns,
                                          vuln_array, vulns_id, num_damage_bins, mean_damage_bins,
                                          int32_mv, max_result_relative_size):

                    if cursor_bytes:
                        stream_out.write(mv[:cursor_bytes])
                    else:
                        break
