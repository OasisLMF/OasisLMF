import json
import os
from contextlib import ExitStack
from typing import Dict, Union, Any

import numba as nb
import numpy as np
import pandas as pd

from oasislmf.pytools.getmodel.footprint import Footprint, Event, EventIndexBin


# reading functions
def read_zipped(static_path: str):
    """
    all should return -> Tuple[np.array[Event], Dict[str, Any], Dict[int, dict]]

    Args:
        static_path:

    Returns:

    """
    pass


def read_bin(static_path: str):
    """
    Reads the binary file for footprint data.

    Args:
        static_path: (str) path to the static file where the footprint is

    Returns: footprint data, meta data dictionary, index dictionary
    """
    with ExitStack() as stack:
        footprint_obj = stack.enter_context(Footprint.load(static_path=static_path,
                                                           ignore_file_type={'z', 'csv', 'parquet'}))
        raw_data = footprint_obj.footprint.read()
        full_data = np.frombuffer(raw_data[8:], Event)
        index_data = footprint_obj.footprint_index

        meta_data = {
            "num_intensity_bins": footprint_obj.num_intensity_bins,
            "has_intensity_uncertainty": True if footprint_obj.has_intensity_uncertainty is 1 else False
        }
    return full_data, meta_data, index_data


def read_csv(static_path: str):
    pass


# currently not for numba due to numba not supporting advanced indexing and slicing
# @nb.jit
def stitch_data(areaperil_id, intensity_bin_id, probability, buffer):
    for x in range(0, buffer.size):
        buffer[x] = (areaperil_id[x], intensity_bin_id[x], probability[x])
    return buffer


def read_parquet(static_path: str):
    """
    Reads a parquet file.

    Args:
        static_path: (str) path to the static file where the footprint is

    Returns: data read from the parquet file, meta data dictionary
    """
    with open(f'{static_path}/footprint_parquet_meta.json', 'r') as outfile:
        meta_data: Dict[str, Union[int, bool]] = json.load(outfile)

    data = pd.read_parquet(f'{static_path}/footprint.parquet')

    areaperil_id = data["areaperil_id"].to_numpy()
    intensity_bin_id = data["intensity_bin_id"].to_numpy()
    probability = data["probability"].to_numpy()
    buffer = np.empty(len(areaperil_id), dtype=Event)

    outcome = stitch_data(areaperil_id, intensity_bin_id, probability, buffer)
    data = np.array(outcome, dtype=Event)
    return data, meta_data


def read_index(static_path: str) -> Dict[int, dict]:
    pass


# writing functions
def write_zipped(data):
    pass


def write_bin(data, meta_data, static_path) -> None:
    pass


def write_csv(data):
    pass


def write_parquet(data, meta_data: Dict[str, int], static_path: str) -> None:
    """
    Writes the footprint data to a parquet file.

    Args:
        data: (np.array) data to be written to the file
        meta_data: (dict) metadata about the data to be written to the file as a dictionary
        static_path: (str) the path to the directory where the parquet file is to be written

    Returns: None
    """
    df = pd.DataFrame(data)
    df.to_parquet(f"{static_path}/footprint.parquet", index=False, compression="gzip")
    with open(f'{static_path}/footprint_parquet_meta.json', 'w') as outfile:
        json.dump(meta_data, outfile)


def write_index_bin(index_data: Dict[int, dict], static_path: str) -> None:
    buffer = []
    for key in index_data.keys():
        placeholder = index_data[key]
        package = (key, placeholder["offset"], placeholder["size"])
        buffer.append(package)

    index_array = np.array(buffer, EventIndexBin)
    with open(f"{static_path}/footprint.idx", "wb") as file:
        file.write(index_array)


# conversions
def zipped_to_parquet():
    pass


def bin_to_parquet(static_path: str) -> None:
    """
    Writes a parquet file from a binary file.

    Args:
        static_path: (str) directory where the binary is and where the parquet file is to be written

    Returns: None
    """
    data, meta_data, index_data = read_bin(static_path=static_path)
    write_parquet(data=data, meta_data=meta_data, static_path=static_path)

    if not os.path.isfile(path=f"{static_path}/footprint.idx"):
        write_index_bin(index_data=index_data, static_path=static_path)


def csv_to_parquet(static_path: str) -> None:
    pass


def parquet_to_zipped(static_path: str) -> None:
    pass


def parquet_to_bin(static_path: str) -> None:
    data, meta_data = read_parquet(static_path=static_path)
    write_bin(data=data, meta_data=meta_data, static_path=static_path)


def parquet_to_csv(static_path: str) -> None:
    pass


# utils

def _define_meta_data() -> Dict[str, Any]:
    pass

