import json
import os
from contextlib import ExitStack
from typing import Dict, Tuple, Union, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oasislmf.pytools.getmodel.footprint import Footprint, Event, EventIndexBin


# reading functions
def read_zipped(static_path: str) -> Tuple[np.array[Event], Dict[str, Any], Dict[int, dict]]:
    pass


def read_bin(static_path: str) -> Tuple[np.array[Event], Dict[str, Any], Dict[int, dict]]:
    with ExitStack() as stack:
        footprint_obj = stack.enter_context(Footprint.load(static_path=static_path,
                                                           ignore_file_type={
                                                               'footprint.bin.z',
                                                               'footprint.idx.z'
                                                           }))
        buffer = []

        for key in footprint_obj.footprint_index.keys():
            buffer.append(footprint_obj.get_event(key))

        data = np.concatenate(buffer)
        index_data = footprint_obj.footprint_index

        meta_data = {
            "num_intensity_bins": footprint_obj.num_intensity_bins,
            "has_intensity_uncertainty": True if footprint_obj.has_intensity_uncertainty is 1 else False
        }
    return data, meta_data, index_data


def read_csv(static_path: str) -> Tuple[np.array[Event], Dict[str, Any], Dict[int, dict]]:
    pass


def read_parquet(static_path: str) -> Tuple[np.array[Event], Dict[str, Any]]:
    with open(f'{static_path}/footprint_parquet_meta.json', 'w') as outfile:
        meta_data: Dict[str, Union[int, bool]] = json.load(outfile)

    data = pq.read_table(f'{static_path}/footprint.parquet').to_pandas().to_numpy(dtype=Event)
    return data, meta_data


def read_index(static_path: str) -> Dict[int, dict]:
    pass


# writing functions
def write_zipped(data):
    pass


def write_bin(data: np.array[Event], meta_data: Dict[str, Any], static_path: str) -> None:
    pass


def write_csv(data):
    pass


def write_parquet(data: np.array[Event], meta_data: Dict[str, int], static_path: str) -> None:
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)

    pq.write_table(table, f"{static_path}/footprint.parquet")
    with open(f'{static_path}/footprint_parquet_meta.json', 'r') as outfile:
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

    # if not os.path.isfile(path=f"{static_path}/footprint.idx"):
    #     write_index_bin(index_data=index_data, static_path=static_path)


def parquet_to_csv(static_path: str) -> None:
    pass


# utils

def _define_meta_data() -> Dict[str, Any]:
    pass

