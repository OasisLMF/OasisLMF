import argparse
import os
import struct
import sys
from io import StringIO
from typing import Optional

from pandas import read_csv, DataFrame

from .getmodel.enums import FileTypeEnum
from .getmodel.get_model_process import GetModelProcess


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

    process: GetModelProcess = GetModelProcess(data_path=data_path, events=_process_input_data(),
                                               file_type=_process_file_type(file_type=args.file_type))
    process.run()
    process.print_stream()
