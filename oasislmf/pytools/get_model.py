import argparse
import os
import struct
import sys
from io import StringIO
from typing import Optional

from pandas import read_csv, DataFrame
import numba as nb
import numpy as np
from typing import List

from .getmodel.enums import FileTypeEnum
from .getmodel.get_model_process import GetModelProcess, FileDataAccessLayer


def _process_input_data() -> Optional[DataFrame]:
    """
    Gets the input from the STDin and converts it to a DataFrame if present.

    Returns: (Optional[DataFrame]) containing event IDs
    """
    data = sys.stdin.buffer.read()

    if data == "":
        return None

    try:
        # data from the evetocsv
        eve_to_csv_data = data.decode()
        return read_csv(StringIO(eve_to_csv_data), sep=",")
    except UnicodeDecodeError:
        pass

    # data directly from eve
    eve_raw_data = [data[i:i + 4] for i in range(0, len(data), 4)]
    eve_buffer = [struct.unpack("i", i)[0] for i in eve_raw_data]
    
    return DataFrame(eve_buffer, columns=["event_id"])


def _process_file_type(file_type: str) -> FileTypeEnum:
    """
    Extracts the FileTypeEnum type based off of the string.

    Args:
        file_type: (str) the file type to be found

    Returns: (FileTypeEnum) the file type to be found
    """
    enum_map = dict()

    for i in FileTypeEnum:
        enum_map[i.value] = getattr(FileTypeEnum, i.value.upper())

    file_type_value: Optional[FileTypeEnum] = enum_map.get(file_type)
    if file_type_value is None:
        raise ValueError(
            f"file type '{file_type}' is not supported, please pick from {[i.value for i in FileTypeEnum]}"
        )
    return file_type_value


@nb.jit(cache=True, nopython=True)
def make_footprint_index_dict(footprint_index, footprint_offset, event_size):
    """
    Generates the

    :param footprint_index:
    :param footprint_offset:
    :param event_size:
    :return:
    """
    res = nb.typed.Dict()

    for i in range(footprint_index.shape[0]):
        event_index = footprint_index[i]
        res[event_index['event_id']] = ((event_index['offset'] - footprint_offset)//event_size,
                                        (event_index['offset'] - footprint_offset + event_index['size'])//event_size)

    return res


# @nb.jit(cache=True, nopython=True)
def filter_vulnerabilities(items, vulnerabilities):
    highest_group_id = items[-1][4]  # 10
    number_of_vulnerability_ids = int(len(items) / highest_group_id)

    # vulnerability IDs that are in the items
    item_vulnerability_ids = sorted([x[3] for x in items[0: number_of_vulnerability_ids]])
    # item_area_peril_ids = sorted([x[2] for x in items[0: number_of_vulnerability_ids]])

    # generate dictionary that maps the positions of the vulnerabilities
    position_map = nb.typed.Dict()
    matched = False
    vulnerability_id_pointer = 0
    vulnerability_id_end_pointer = len(item_vulnerability_ids) - 1
    start_pointer = 0

    for i in range(0, len(vulnerabilities)):
        if vulnerabilities[i][0] == item_vulnerability_ids[vulnerability_id_pointer] and matched is False:
            matched = True
            start_pointer = i
        elif matched is True and vulnerabilities[i][0] != item_vulnerability_ids[vulnerability_id_pointer]:
            finish_pointer = i
            matched = False
            position_map[item_vulnerability_ids[vulnerability_id_pointer]] = (start_pointer, finish_pointer)
            vulnerability_id_pointer += 1
        if vulnerability_id_pointer > vulnerability_id_end_pointer:
            break

    buffer = np.array([[0.0, 0.0, 0.0, 0.0]])
    for key in position_map.keys():
        start = position_map[key][0]
        finish = position_map[key][1]
        buffer = np.concatenate((buffer, vulnerabilities[start: finish]))

    return buffer[1:]
    # return position_map


def _generate_footprint_index_dict(file_type: FileTypeEnum, data_path: str) -> dict:
    """
    Reads the footprint binary file creating a footprint index dictionary.

    :return: None
    """
    # fdal is a singleton and will be used in the get_model_process
    fdal = FileDataAccessLayer(extension=file_type, data_path=data_path)
    event_index_struct = nb.from_dtype(np.dtype([('event_id', np.int32),
                                                 ('offset', np.int64),
                                                 ('size', np.int64)
                                                 ]))
    event_struct = nb.from_dtype(np.dtype([('areaperil_id', np.uint32),
                                           ('intensity_bin_id', np.int32),
                                           ('probability', np.float32)
                                           ]))
    footprint_offset = 8
    footprint_buffer: List[str] = fdal.footprint.path.split(".")
    footprint_buffer[-1] = "bin"
    # footprint_path: str = ".".join(footprint_buffer)
    footprint_buffer[-1] = "idx"
    index_path: str = ".".join(footprint_buffer)

    # footprint = np.memmap(footprint_path, dtype=event_struct, mode="r", offset=footprint_offset)
    footprint_index = np.memmap(index_path, dtype=event_index_struct, mode="r")

    return make_footprint_index_dict(footprint_index, footprint_offset, event_struct.size)


def _filter_vulnerabilities(file_type: FileTypeEnum, data_path: str):
    fdal = FileDataAccessLayer(extension=file_type, data_path=data_path)
    filtered_vulnerabilities = filter_vulnerabilities(items=fdal.items.value.to_numpy(),
                                                      vulnerabilities=fdal.vulnerabilities.value.to_numpy())
    fdal.vulnerabilities.value = filtered_vulnerabilities


def main() -> None:
    """
    Entry point of the 'getpymodel' command building the module and then piping it out as bytes.

    Returns: None
    """
    # add in argumments that accept the type of file that is being run (CSV, bin, parquet)
    parser = argparse.ArgumentParser(description="Arguments for the get model")
    parser.add_argument("-f", "--file_type", type=str, default="csv")
    args = parser.parse_args()

    data_path: str = str(os.getcwd())
    file_type: FileTypeEnum = _process_file_type(file_type=args.file_type)

    footprint_index_dictionary = _generate_footprint_index_dict(file_type=file_type, data_path=data_path)
    _filter_vulnerabilities(file_type=file_type, data_path=data_path)

    sys.stdout.buffer.write(b'\x01\x00\x00\x00')

    for i in _process_input_data().groupby(["event_id"]):
        process: GetModelProcess = GetModelProcess(data_path=data_path, events=i[1],
                                                   file_type=file_type,
                                                   footprint_index_dictionary=footprint_index_dictionary)
        if process.should_run is True:
            process.run()

        process.print_stream()
        # process.run()
