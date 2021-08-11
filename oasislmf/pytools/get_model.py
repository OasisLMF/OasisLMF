import argparse
import os
import sys
from io import StringIO
from typing import Optional

from pandas import read_csv, DataFrame

from .getmodel.enums import FileTypeEnum
from .getmodel.get_model_process import GetModelProcess


def _process_input_data() -> Optional[DataFrame]:
    """
    Gets the input from the STDin and converts it to

    Returns: (Optional[DataFrame])
    """
    data = sys.stdin.buffer.read()
    if data == "":
        return None
    return read_csv(StringIO(data.decode()), sep=",")


def _process_file_type(file_type: str) -> FileTypeEnum:
    """
    Extracts the type from the Enum type.

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
    Entry point of the 'new-model' command building the module and then piping it out as bytes.

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
